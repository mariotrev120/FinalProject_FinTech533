"""
Data quality dump for non-SPX tickers in data/processed/options_by_ticker/*.parquet.

OUTPUT ONLY. No interpretation. No fixes. Drives DIAGNOSIS_DATA.md.

For each of the 14 tickers:
  - delta range, mean, NaN count, NaN % by year
  - impl_volatility range, mean, NaN count, NaN % by year
  - gamma, vega, theta same
  - best_bid, best_offer ranges
  - cp_flag distribution
  - DTE (days to expiration) distribution
  - sample chain for one date

Usage:
  PYTHONPATH=. .venv/bin/python scripts/dump_data_quality.py > /tmp/data_quality_dump.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


OPTIONS_BY_TICKER_DIR = Path("data/processed/options_by_ticker")
ALL_14 = ["SPX", "RUT", "NDX", "TLT", "GLD",
          "AAPL", "MSFT", "GOOGL", "JNJ", "KO", "PG", "WMT", "JPM", "PEP"]

NUMERIC_COLUMNS = [
    "delta", "gamma", "vega", "theta", "impl_volatility",
    "best_bid", "best_offer", "strike", "volume", "open_interest",
]


def main() -> int:
    parquet_dir = OPTIONS_BY_TICKER_DIR
    if not parquet_dir.exists():
        print(f"[ERR] {parquet_dir} does not exist. Run conversion first.")
        return 2

    print("=" * 80)
    print("DATA QUALITY DUMP — data/processed/options_by_ticker/")
    print("Source: strat2.2.csv.gz (provided by user, pre-converted via DuckDB)")
    print("=" * 80)
    print()

    # First: list columns the loader actually reads
    print("Columns in each per-ticker parquet (read by load_ticker_dataframe):")
    sample = pd.read_parquet(parquet_dir / "SPX.parquet").head(1)
    for col in sample.columns:
        print(f"  - {col!r:25s} dtype={sample[col].dtype}")
    print()

    for ticker in ALL_14:
        path = parquet_dir / f"{ticker}.parquet"
        if not path.exists():
            print(f"[{ticker}] MISSING parquet at {path}")
            continue

        df = pd.read_parquet(path)
        # date and exdate may be stored as datetime.date (object dtype); coerce
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])
        if not pd.api.types.is_datetime64_any_dtype(df["exdate"]):
            df["exdate"] = pd.to_datetime(df["exdate"])
        print("=" * 80)
        print(f"[{ticker}] rows={len(df):,}  size_mb={path.stat().st_size / 1024 / 1024:.1f}")
        print(f"          date range: [{df['date'].min()}, {df['date'].max()}]")
        print(f"          n unique dates: {df['date'].nunique()}")
        print(f"          n unique exdates: {df['exdate'].nunique()}")

        # cp_flag distribution
        cp_counts = df["cp_flag"].value_counts(dropna=False).to_dict()
        print(f"          cp_flag distribution: {cp_counts}")

        # DTE
        df_dte = (df["exdate"] - df["date"]).dt.days
        print(f"          DTE (calendar days): min={int(df_dte.min())}, "
              f"max={int(df_dte.max())}, "
              f"median={int(df_dte.median())}, "
              f"mean={float(df_dte.mean()):.1f}")

        print()
        print(f"  -- Numeric columns ranges + NaN count + NaN-by-year --")
        for col in NUMERIC_COLUMNS:
            if col not in df.columns:
                print(f"    {col!r:20s}: NOT IN DATAFRAME")
                continue
            s = df[col]
            n_nan = int(s.isna().sum())
            pct_nan = 100 * n_nan / len(df) if len(df) else 0.0
            valid = s.dropna()
            if len(valid) == 0:
                print(f"    {col!r:20s}: ALL NaN ({len(df):,} rows)")
                continue
            try:
                col_min = float(valid.min())
                col_max = float(valid.max())
                col_mean = float(valid.mean())
                col_med = float(valid.median())
            except Exception as e:
                print(f"    {col!r:20s}: ERROR computing stats: {e}")
                continue
            print(f"    {col!r:20s}: range=[{col_min:>12.4f}, {col_max:>12.4f}]  "
                  f"mean={col_mean:>10.4f}  median={col_med:>10.4f}  "
                  f"NaN={n_nan:>9,} ({pct_nan:5.1f}%)")

        # NaN by year for delta + iv (the two we care about most)
        print()
        print(f"  -- delta NaN % by year --")
        df_year = df.assign(year=df["date"].dt.year)
        for year, sub in df_year.groupby("year"):
            n_nan = int(sub["delta"].isna().sum())
            pct = 100 * n_nan / len(sub) if len(sub) else 0
            print(f"    {year}: {n_nan:>9,} / {len(sub):>9,}  ({pct:5.1f}%)")

        print()
        print(f"  -- impl_volatility NaN % by year --")
        for year, sub in df_year.groupby("year"):
            n_nan = int(sub["impl_volatility"].isna().sum())
            pct = 100 * n_nan / len(sub) if len(sub) else 0
            print(f"    {year}: {n_nan:>9,} / {len(sub):>9,}  ({pct:5.1f}%)")

        # Sample chain for one liquid date
        print()
        sample_date = pd.Timestamp("2020-06-15")
        if sample_date < df["date"].min() or sample_date > df["date"].max():
            sample_date = df["date"].iloc[len(df) // 2]
        chain_today = df[df["date"] == sample_date]
        if len(chain_today) == 0:
            # fallback: nearest date
            sample_date_pos = df["date"].searchsorted(sample_date)
            sample_date = df["date"].iloc[min(sample_date_pos, len(df) - 1)]
            chain_today = df[df["date"] == sample_date]
        # limit to one expiry
        if len(chain_today):
            target_exp = chain_today["exdate"].iloc[len(chain_today) // 2]
            chain_one = chain_today[chain_today["exdate"] == target_exp].sort_values(
                ["cp_flag", "strike"]
            )
            print(f"  -- Sample chain on {sample_date.date()} for expiry {target_exp.date()} ({len(chain_one)} rows) --")
            print("    " + chain_one[
                ["cp_flag", "strike", "best_bid", "best_offer", "delta",
                 "impl_volatility", "gamma", "vega"]
            ].head(20).to_string().replace("\n", "\n    "))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
