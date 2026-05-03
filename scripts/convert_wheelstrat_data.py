"""
One-shot converter: WRDS OptionMetrics IvyDB Securities → per-ticker parquets.

Input:  wheelstrat_data.gz (CSV, columns: secid, date, ticker, low, high,
        open, close, volume, return, cfadj, cfret)
Output: data/raw/{TICKER}.parquet for each ticker, schema matching the
        existing SPX/GLD parquet convention (DatetimeIndex, columns
        open/high/low/close/volume/symbol/sec_type/exchange).

The TLT, AAPL, MSFT, GOOGL, JNJ, KO, PG, WMT, JPM, PEP files in data/raw
will be OVERWRITTEN with the WRDS-pulled history (2012-01 → 2025-12),
replacing the truncated HW5-sourced 2021+ files.

The OptionMetrics close field is RAW same-day price (already post-split
on post-split days), which matches how option chain strikes are encoded
in strat2.2.csv.gz. No cfadj/cfret adjustment is applied.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SRC = Path("wheelstrat_data.gz")
DST_DIR = Path("data/raw")


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found.", file=sys.stderr)
        return 1

    print(f"Reading {SRC}...")
    df = pd.read_csv(SRC, parse_dates=["date"])
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} tickers")

    expected = {"AAPL", "MSFT", "GOOGL", "JNJ", "KO", "PG", "WMT", "JPM", "PEP", "TLT"}
    found = set(df["ticker"].unique())
    if expected - found:
        print(f"WARNING: missing tickers in input: {sorted(expected - found)}")

    DST_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in sorted(found):
        sub = df[df["ticker"] == ticker].copy()
        sub = sub.sort_values("date").set_index("date")
        sub.index.name = None
        out = pd.DataFrame({
            "open":   sub["open"].astype(float),
            "high":   sub["high"].astype(float),
            "low":    sub["low"].astype(float),
            "close":  sub["close"].astype(float),
            "volume": sub["volume"].astype(float),
            "symbol": ticker,
            "sec_type": "STK" if ticker != "TLT" else "STK",
            "exchange": "SMART",
        })
        out_path = DST_DIR / f"{ticker}.parquet"
        out.to_parquet(out_path)
        print(f"  {ticker}: {len(out):>5d} rows  "
              f"{out.index.min().date()} → {out.index.max().date()}  "
              f"sample close (2018-04-16): ${out.loc[pd.Timestamp('2018-04-16'), 'close']:.2f}"
              if pd.Timestamp('2018-04-16') in out.index else
              f"  {ticker}: {len(out):>5d} rows  {out.index.min().date()} → {out.index.max().date()}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
