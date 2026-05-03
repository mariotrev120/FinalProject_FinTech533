"""
DIAGNOSIS_DATA.md verification dump — three additional sections:

  1. Trade-relevant delta region: puts ∈ [-0.21, -0.11], calls ∈ [0.11, 0.21].
     Sample 5 dates per ticker spanning 2018-2024. Row counts, NaN counts,
     delta-mean, IV-mean per ticker.
  2. NaN rate matrix: ticker × year × {delta, IV, gamma, vega}. Flag any
     cell where NaN rate > 25%.
  3. Strike-selection trace per ticker: one date, full chain, the row
     selected at 16-delta target, loader's delta vs BS-recomputed delta.
     Flag if |divergence| > 0.01.

If all three clean → approved for multi-instrument NAKED launch.
If any issues → document and stop, do not launch.

Output: /tmp/data_quality_v2.txt
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


OPTIONS_BY_TICKER_DIR = Path("data/processed/options_by_ticker")
ALL_14 = ["SPX", "RUT", "NDX", "TLT", "GLD",
          "AAPL", "MSFT", "GOOGL", "JNJ", "KO", "PG", "WMT", "JPM", "PEP"]

PUT_DELTA_LO, PUT_DELTA_HI = -0.21, -0.11
CALL_DELTA_LO, CALL_DELTA_HI = 0.11, 0.21
TARGET_PUT_DELTA = -0.16
TARGET_CALL_DELTA = 0.16

SAMPLE_DATES = [
    pd.Timestamp("2018-06-15"),
    pd.Timestamp("2019-09-13"),
    pd.Timestamp("2020-06-15"),
    pd.Timestamp("2022-03-15"),
    pd.Timestamp("2024-04-12"),
]


def load_ticker(t: str) -> pd.DataFrame:
    df = pd.read_parquet(OPTIONS_BY_TICKER_DIR / f"{t}.parquet")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    if not pd.api.types.is_datetime64_any_dtype(df["exdate"]):
        df["exdate"] = pd.to_datetime(df["exdate"])
    return df


def bs_delta_call(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """Black-Scholes call delta: N(d1)."""
    if T <= 0 or sigma <= 0 or K <= 0 or S <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1))


def bs_delta_put(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """Black-Scholes put delta: N(d1) − 1."""
    return bs_delta_call(S, K, T, sigma, r) - 1.0


def get_underlying_price_for_date(df: pd.DataFrame, target_date: pd.Timestamp) -> float:
    """Estimate underlying price S from the options chain on `target_date`
    by looking at the deepest-OTM put + deepest-OTM call (where bid+offer
    are tightest and ATM is bracketed). Use a smarter heuristic:
    pick the put-call pair with delta closest to ±0.5 (ATM) and average
    their strikes.

    Returns NaN if can't be determined."""
    sub = df[df["date"] == target_date]
    if len(sub) == 0:
        return float("nan")
    valid = sub.dropna(subset=["delta"])
    puts = valid[valid["cp_flag"] == "P"]
    calls = valid[valid["cp_flag"] == "C"]
    if len(puts) == 0 or len(calls) == 0:
        return float("nan")
    # Pick atm-est put (delta closest to -0.5)
    atm_put = puts.iloc[(puts["delta"] - (-0.5)).abs().argsort().head(1)]
    atm_call = calls.iloc[(calls["delta"] - 0.5).abs().argsort().head(1)]
    if len(atm_put) == 0 or len(atm_call) == 0:
        return float("nan")
    return float((atm_put["strike"].iloc[0] + atm_call["strike"].iloc[0]) / 2.0)


def find_nearest_trading_date(df: pd.DataFrame, target: pd.Timestamp) -> pd.Timestamp:
    """Find the closest date in df['date'] to target."""
    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    pos = dates.searchsorted(target)
    if pos >= len(dates):
        return dates.iloc[-1]
    return dates.iloc[pos]


def section_1(out_lines: list, df: pd.DataFrame, ticker: str):
    """Per-ticker, restricted to trade-relevant delta region, 5 dates."""
    out_lines.append("")
    out_lines.append(f"### {ticker}")
    out_lines.append("")
    out_lines.append("| Date | Side | n_rows_in_region | n_NaN_delta | n_NaN_iv | "
                     "delta_mean_in_region | iv_mean_in_region |")
    out_lines.append("|---|---|---:|---:|---:|---:|---:|")
    for tgt in SAMPLE_DATES:
        actual_date = find_nearest_trading_date(df, tgt)
        sub = df[df["date"] == actual_date]
        if len(sub) == 0:
            continue
        # Filter to 30-45 DTE (matches our entry rule)
        dte = (sub["exdate"] - sub["date"]).dt.days
        sub30 = sub[(dte >= 30) & (dte <= 45)]
        if len(sub30) == 0:
            sub30 = sub  # fall back if no 30-45 DTE on this date
        # Puts in trade region
        puts = sub30[(sub30["cp_flag"] == "P") &
                     (sub30["delta"] >= PUT_DELTA_LO) &
                     (sub30["delta"] <= PUT_DELTA_HI)]
        n_p = len(puts)
        # NaN counts must be measured against the FULL chain (puts), not the filter
        all_puts = sub30[sub30["cp_flag"] == "P"]
        n_nan_p_delta = int(all_puts["delta"].isna().sum())
        n_nan_p_iv = int(all_puts["impl_volatility"].isna().sum())
        d_mean_p = float(puts["delta"].mean()) if n_p else float("nan")
        iv_mean_p = float(puts["impl_volatility"].mean()) if n_p else float("nan")

        # Calls in trade region
        calls = sub30[(sub30["cp_flag"] == "C") &
                      (sub30["delta"] >= CALL_DELTA_LO) &
                      (sub30["delta"] <= CALL_DELTA_HI)]
        n_c = len(calls)
        all_calls = sub30[sub30["cp_flag"] == "C"]
        n_nan_c_delta = int(all_calls["delta"].isna().sum())
        n_nan_c_iv = int(all_calls["impl_volatility"].isna().sum())
        d_mean_c = float(calls["delta"].mean()) if n_c else float("nan")
        iv_mean_c = float(calls["impl_volatility"].mean()) if n_c else float("nan")

        out_lines.append(
            f"| {actual_date.date()} | P | {n_p} | {n_nan_p_delta} | "
            f"{n_nan_p_iv} | {d_mean_p:+.4f} | {iv_mean_p:.4f} |"
        )
        out_lines.append(
            f"| {actual_date.date()} | C | {n_c} | {n_nan_c_delta} | "
            f"{n_nan_c_iv} | {d_mean_c:+.4f} | {iv_mean_c:.4f} |"
        )


def section_2(out_lines: list, df: pd.DataFrame, ticker: str):
    """NaN rate by year, all 4 columns, flag if any > 25%."""
    out_lines.append("")
    out_lines.append(f"### {ticker} — NaN rate matrix (by year × column)")
    out_lines.append("")
    out_lines.append("| Year | n_rows | NaN delta % | NaN IV % | NaN gamma % | NaN vega % | FLAG |")
    out_lines.append("|---|---:|---:|---:|---:|---:|---|")
    df_year = df.assign(year=df["date"].dt.year)
    for year, sub in df_year.groupby("year"):
        n = len(sub)
        if n == 0:
            continue
        pct = {col: 100 * sub[col].isna().sum() / n for col in
               ["delta", "impl_volatility", "gamma", "vega"]}
        flag = "⚠ >25%" if any(v > 25 for v in pct.values()) else ""
        out_lines.append(
            f"| {year} | {n:,} | {pct['delta']:.1f}% | "
            f"{pct['impl_volatility']:.1f}% | {pct['gamma']:.1f}% | "
            f"{pct['vega']:.1f}% | {flag} |"
        )


def section_3(out_lines: list, df: pd.DataFrame, ticker: str):
    """Strike-selection trace: one date, find 16-delta short strike,
    compare loader delta vs BS-recomputed delta."""
    out_lines.append("")
    out_lines.append(f"### {ticker} — strike-selection trace")
    out_lines.append("")
    target_date = pd.Timestamp("2020-06-15")
    actual_date = find_nearest_trading_date(df, target_date)
    sub = df[df["date"] == actual_date]
    dte = (sub["exdate"] - sub["date"]).dt.days
    sub30 = sub[(dte >= 30) & (dte <= 45)]
    if len(sub30) == 0:
        out_lines.append(f"  No 30-45 DTE chain on {actual_date.date()}; skipping.")
        return
    target_exp = sub30["exdate"].mode().iloc[0]
    chain = sub30[sub30["exdate"] == target_exp]
    out_lines.append(f"Date: **{actual_date.date()}**, Expiry: **{target_exp.date()}** "
                     f"({(target_exp - actual_date).days} cal days), Chain rows: {len(chain)}")
    out_lines.append("")

    # Estimate underlying spot
    S = get_underlying_price_for_date(df, actual_date)
    if pd.isna(S):
        out_lines.append("  Could not estimate underlying spot S; skipping trace.")
        return
    T_yr = (target_exp - actual_date).days / 365.0
    out_lines.append(f"Estimated spot S = {S:.2f}, T = {T_yr:.4f} years")

    rf = 0.04   # use ~4% rf

    out_lines.append("")
    out_lines.append("**16-delta short strike selection (loader's delta column):**")
    out_lines.append("")
    out_lines.append("| Side | Target δ | Loader-picked strike | Loader δ | Loader IV | "
                     "BS δ (recomputed) | |Δδ| | Flag |")
    out_lines.append("|---|---|---:|---:|---:|---:|---:|---|")

    for side, target_delta, lo_d, hi_d, target_label in [
        ("P", TARGET_PUT_DELTA, PUT_DELTA_LO, PUT_DELTA_HI, "−0.16"),
        ("C", TARGET_CALL_DELTA, CALL_DELTA_LO, CALL_DELTA_HI, "+0.16"),
    ]:
        side_chain = chain[chain["cp_flag"] == side].dropna(subset=["delta"])
        if len(side_chain) == 0:
            out_lines.append(f"| {side} | {target_label} | (no valid rows) | | | | | |")
            continue
        # Pick strike whose delta is closest to target
        idx_pick = (side_chain["delta"] - target_delta).abs().argsort().iloc[0]
        row = side_chain.iloc[idx_pick]
        K = float(row["strike"])
        delta_loader = float(row["delta"])
        iv = float(row["impl_volatility"]) if not pd.isna(row["impl_volatility"]) else float("nan")
        if not pd.isna(iv):
            if side == "P":
                bs_d = bs_delta_put(S, K, T_yr, iv, rf)
            else:
                bs_d = bs_delta_call(S, K, T_yr, iv, rf)
        else:
            bs_d = float("nan")
        diff = (
            abs(delta_loader - bs_d)
            if not pd.isna(bs_d) else float("nan")
        )
        flag = "⚠ >0.01" if (not pd.isna(diff) and diff > 0.01) else ""
        out_lines.append(
            f"| {side} | {target_label} | {K:.2f} | {delta_loader:+.4f} | "
            f"{iv:.4f} | {bs_d:+.4f} | {diff:.4f} | {flag} |"
        )


def main() -> int:
    out: list[str] = []
    out.append("# DIAGNOSIS_DATA.md verification — sections 1-3")
    out.append("")
    out.append("Generated by scripts/dump_data_quality_v2.py")
    out.append("")
    out.append("=" * 80)
    out.append("## Section 1: Per-ticker delta restricted to trade-relevant region")
    out.append("Puts: δ ∈ [-0.21, -0.11]   Calls: δ ∈ [0.11, 0.21]")
    out.append("DTE filter: 30-45 calendar days (matches entry rule)")
    out.append(f"Sample dates: {[d.date() for d in SAMPLE_DATES]}")
    out.append("=" * 80)

    for ticker in ALL_14:
        try:
            df = load_ticker(ticker)
            section_1(out, df, ticker)
        except Exception as e:
            out.append(f"")
            out.append(f"### {ticker} — ERROR: {e}")
        del df

    out.append("")
    out.append("=" * 80)
    out.append("## Section 2: NaN rate matrix by ticker × year × column")
    out.append("Flag if any cell > 25%")
    out.append("=" * 80)

    for ticker in ALL_14:
        try:
            df = load_ticker(ticker)
            section_2(out, df, ticker)
        except Exception as e:
            out.append(f"")
            out.append(f"### {ticker} — ERROR: {e}")
        del df

    out.append("")
    out.append("=" * 80)
    out.append("## Section 3: Strike-selection trace, loader-δ vs BS-δ")
    out.append("=" * 80)

    for ticker in ALL_14:
        try:
            df = load_ticker(ticker)
            section_3(out, df, ticker)
        except Exception as e:
            out.append(f"")
            out.append(f"### {ticker} — ERROR: {e}")
        del df

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
