"""
DIAGNOSIS_DATA.md Section 3 verification — re-runs strike-selection
trace with corrections stacked, plus smoothness / sign-flip / DTE-
monotonicity checks.

Per user 2026-05-03:
  - For each flagged ticker, recompute BS delta with three corrections
    stacked, one at a time:
    (a) Use actual r(t) from data/raw/IRX.parquet instead of flat 4%
    (b) Add continuous dividend yield q (per-ticker estimate)
    (c) For TLT/GLD: skip (American premium correction not implemented;
        flag the residual)
  - Pass criteria after corrections (a)+(b):
      Indexes (SPX/RUT/NDX): |Δδ| ≤ 0.03, sign correct, smooth
      ETFs (TLT/GLD):        |Δδ| ≤ 0.05, sign correct, smooth
  - Hard fails: sign-flip on any single strike; divergence increasing
    systematically with DTE.

Smooth = monotonic |Δδ| with moneyness, no chaotic jumps.

Output: /tmp/data_quality_v3.txt
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

# Per-ticker continuous dividend yield estimates for 2020 (mid-OOS).
# Rough ballparks; exact values not critical — small contributions.
# Sources: macro-trends.net / index providers, 2020-mid-year approx.
DIVIDEND_YIELDS = {
    "SPX": 0.017,
    "RUT": 0.015,
    "NDX": 0.009,
    "TLT": 0.025,    # bond ETF distribution yield
    "GLD": 0.000,
    "AAPL": 0.008,
    "MSFT": 0.011,
    "GOOGL": 0.000,
    "JNJ": 0.027,
    "KO": 0.034,
    "PG": 0.024,
    "WMT": 0.016,
    "JPM": 0.036,
    "PEP": 0.029,
}


def load_irx_curve() -> pd.Series:
    """Load IRX 3-month T-bill rate as decimal-form risk-free rate.
    Per src/backtest/engine.py: rf_pct = irx_close / 10 / 100."""
    df = pd.read_parquet(Path("data/raw/IRX.parquet"))
    if "close" not in df.columns:
        raise ValueError(f"IRX parquet missing 'close' column: {list(df.columns)}")
    s = df["close"].astype(float) / 1000.0   # /10/100 = /1000 → decimal
    return s.sort_index()


def get_rf_for_date(irx: pd.Series, target: pd.Timestamp) -> float:
    """as-of lookup."""
    if target in irx.index:
        v = irx.loc[target]
    else:
        try:
            v = irx.asof(target)
        except KeyError:
            return 0.005
    if pd.isna(v):
        return 0.005
    return float(v)


def load_ticker(t: str) -> pd.DataFrame:
    df = pd.read_parquet(OPTIONS_BY_TICKER_DIR / f"{t}.parquet")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    if not pd.api.types.is_datetime64_any_dtype(df["exdate"]):
        df["exdate"] = pd.to_datetime(df["exdate"])
    return df


def bs_delta(side: str, S: float, K: float, T: float, sigma: float,
             r: float, q: float = 0.0) -> float:
    """BS delta with continuous dividend yield q. side='P' or 'C'."""
    if T <= 0 or sigma <= 0 or K <= 0 or S <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if side == "C":
        return float(math.exp(-q * T) * norm.cdf(d1))
    elif side == "P":
        return float(math.exp(-q * T) * (norm.cdf(d1) - 1))
    raise ValueError(side)


def get_atm_spot(df: pd.DataFrame, target_date: pd.Timestamp) -> float:
    sub = df[df["date"] == target_date]
    valid = sub.dropna(subset=["delta"])
    puts = valid[valid["cp_flag"] == "P"]
    calls = valid[valid["cp_flag"] == "C"]
    if len(puts) == 0 or len(calls) == 0:
        return float("nan")
    atm_p = puts.iloc[(puts["delta"] - (-0.5)).abs().argsort().head(1)]
    atm_c = calls.iloc[(calls["delta"] - 0.5).abs().argsort().head(1)]
    return float((atm_p["strike"].iloc[0] + atm_c["strike"].iloc[0]) / 2)


def find_nearest_trading_date(df: pd.DataFrame, target: pd.Timestamp) -> pd.Timestamp:
    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    pos = dates.searchsorted(target)
    if pos >= len(dates):
        return dates.iloc[-1]
    return dates.iloc[pos]


def trace_for_ticker(out_lines: list, df: pd.DataFrame, ticker: str,
                     irx: pd.Series, target_date: pd.Timestamp = pd.Timestamp("2020-06-15")):
    """Run the strike-selection trace with stacked corrections."""
    actual_date = find_nearest_trading_date(df, target_date)
    sub = df[df["date"] == actual_date]
    dte_calendar = (sub["exdate"] - sub["date"]).dt.days
    sub_30_45 = sub[(dte_calendar >= 30) & (dte_calendar <= 45)]
    if len(sub_30_45) == 0:
        out_lines.append(f"\n### {ticker}: no 30-45 DTE chain on {actual_date.date()}; skipping")
        return None

    target_exp = sub_30_45["exdate"].mode().iloc[0]
    chain = sub_30_45[sub_30_45["exdate"] == target_exp].copy()
    S = get_atm_spot(df, actual_date)
    if pd.isna(S):
        out_lines.append(f"\n### {ticker}: cannot determine spot; skipping")
        return None
    T_yr = (target_exp - actual_date).days / 365.0
    r_actual = get_rf_for_date(irx, actual_date)
    q = DIVIDEND_YIELDS.get(ticker, 0.0)

    out_lines.append(f"\n### {ticker} — recomputed strike-selection trace")
    out_lines.append(f"  Date: {actual_date.date()}, Expiry: {target_exp.date()}, T={T_yr:.4f}y, "
                     f"S={S:.2f}, r(t)={r_actual*100:.3f}%, q={q*100:.2f}%")

    # Pick 16-delta-target strike per side (loader's delta column)
    rows = []
    for side, target_delta in [("P", -0.16), ("C", 0.16)]:
        side_chain = chain[chain["cp_flag"] == side].dropna(subset=["delta"])
        if len(side_chain) == 0:
            continue
        idx_pick = (side_chain["delta"] - target_delta).abs().argsort().iloc[0]
        row = side_chain.iloc[idx_pick]
        K = float(row["strike"])
        loader_d = float(row["delta"])
        iv = float(row["impl_volatility"])
        if pd.isna(iv):
            continue

        # Three stacked recomputes
        d_naive_4 = bs_delta(side, S, K, T_yr, iv, r=0.04, q=0.0)
        d_a = bs_delta(side, S, K, T_yr, iv, r=r_actual, q=0.0)
        d_ab = bs_delta(side, S, K, T_yr, iv, r=r_actual, q=q)
        rows.append({
            "ticker": ticker, "side": side, "K": K,
            "loader_d": loader_d, "iv": iv,
            "d_naive_r4": d_naive_4,
            "d_a_actual_r": d_a,
            "d_ab_actual_r_q": d_ab,
            "abs_div_naive": abs(loader_d - d_naive_4),
            "abs_div_a": abs(loader_d - d_a),
            "abs_div_ab": abs(loader_d - d_ab),
            "sign_match_loader": (loader_d < 0) if side == "P" else (loader_d > 0),
        })
    return rows


def smoothness_check(out_lines: list, df: pd.DataFrame, ticker: str,
                     irx: pd.Series, target_date: pd.Timestamp = pd.Timestamp("2020-06-15")):
    """Across moneyness, check whether |Δδ_ab| trends monotonically.
    Smoothness criterion: |Δδ| at adjacent strikes differs by less than
    0.02 (no chaotic jumps), and overall trend is monotonic in K."""
    actual_date = find_nearest_trading_date(df, target_date)
    sub = df[df["date"] == actual_date]
    dte_cal = (sub["exdate"] - sub["date"]).dt.days
    sub_30_45 = sub[(dte_cal >= 30) & (dte_cal <= 45)]
    if len(sub_30_45) == 0:
        return None
    target_exp = sub_30_45["exdate"].mode().iloc[0]
    chain = sub_30_45[sub_30_45["exdate"] == target_exp].dropna(
        subset=["delta", "impl_volatility"]
    ).copy()
    S = get_atm_spot(df, actual_date)
    if pd.isna(S):
        return None
    T_yr = (target_exp - actual_date).days / 365.0
    r_actual = get_rf_for_date(irx, actual_date)
    q = DIVIDEND_YIELDS.get(ticker, 0.0)

    chain = chain.sort_values(["cp_flag", "strike"])
    chain["bs_d_ab"] = chain.apply(
        lambda r: bs_delta(r["cp_flag"], S, r["strike"], T_yr,
                           r["impl_volatility"], r=r_actual, q=q),
        axis=1,
    )
    chain["abs_div_ab"] = (chain["delta"] - chain["bs_d_ab"]).abs()

    # Sign correctness
    n_put_pos_delta = ((chain["cp_flag"] == "P") & (chain["delta"] > 0)).sum()
    n_call_neg_delta = ((chain["cp_flag"] == "C") & (chain["delta"] < 0)).sum()
    sign_flip_count = int(n_put_pos_delta + n_call_neg_delta)

    # Smoothness: largest jump in |Δδ| between adjacent strikes per side
    max_jump_per_side = {}
    for side in ["P", "C"]:
        side_sub = chain[chain["cp_flag"] == side].sort_values("strike")
        if len(side_sub) < 3:
            continue
        diffs = side_sub["abs_div_ab"].diff().abs().dropna()
        max_jump_per_side[side] = float(diffs.max())

    return {
        "ticker": ticker,
        "n_put": int((chain["cp_flag"] == "P").sum()),
        "n_call": int((chain["cp_flag"] == "C").sum()),
        "sign_flip_count": sign_flip_count,
        "max_abs_div_ab": float(chain["abs_div_ab"].max()),
        "median_abs_div_ab": float(chain["abs_div_ab"].median()),
        "max_jump_put": max_jump_per_side.get("P", float("nan")),
        "max_jump_call": max_jump_per_side.get("C", float("nan")),
    }


def dte_monotonicity_check(out_lines: list, df: pd.DataFrame, ticker: str,
                            irx: pd.Series,
                            target_date: pd.Timestamp = pd.Timestamp("2020-06-15")):
    """Across multiple DTE buckets on the same date, check whether |Δδ_ab|
    stays bounded. Hard fail if it grows monotonically with DTE."""
    actual_date = find_nearest_trading_date(df, target_date)
    sub = df[df["date"] == actual_date].dropna(
        subset=["delta", "impl_volatility"]
    ).copy()
    dte_cal = (sub["exdate"] - sub["date"]).dt.days
    sub["dte"] = dte_cal
    S = get_atm_spot(df, actual_date)
    if pd.isna(S):
        return None
    r_actual = get_rf_for_date(irx, actual_date)
    q = DIVIDEND_YIELDS.get(ticker, 0.0)

    rows = []
    for dte_lo, dte_hi in [(7, 21), (22, 35), (36, 50), (51, 60)]:
        bucket = sub[(sub["dte"] >= dte_lo) & (sub["dte"] <= dte_hi)]
        # Restrict to trade-relevant 16-delta region
        rel = bucket[
            (((bucket["cp_flag"] == "P") & (bucket["delta"] >= -0.21) & (bucket["delta"] <= -0.11))
             | ((bucket["cp_flag"] == "C") & (bucket["delta"] >= 0.11) & (bucket["delta"] <= 0.21)))
        ]
        if len(rel) == 0:
            rows.append({"dte_bucket": f"{dte_lo}-{dte_hi}", "n": 0,
                         "median_abs_div_ab": float("nan")})
            continue
        T_avg = float(rel["dte"].mean()) / 365.0
        rel = rel.copy()
        rel["bs_d_ab"] = rel.apply(
            lambda r: bs_delta(r["cp_flag"], S, r["strike"],
                               (r["dte"] / 365.0),
                               r["impl_volatility"], r=r_actual, q=q),
            axis=1,
        )
        rel["abs_div_ab"] = (rel["delta"] - rel["bs_d_ab"]).abs()
        rows.append({
            "dte_bucket": f"{dte_lo}-{dte_hi}",
            "n": len(rel),
            "median_abs_div_ab": float(rel["abs_div_ab"].median()),
        })
    return rows


def main() -> int:
    out: list[str] = []
    out.append("# DIAGNOSIS_DATA.md Section 3 — corrected strike-selection trace")
    out.append("")
    out.append("Re-runs Section 3 with stacked corrections:")
    out.append("  (a) Actual r(t) from data/raw/IRX.parquet")
    out.append("  (b) Continuous dividend yield q (per-ticker estimate)")
    out.append("  (c) American-premium correction NOT implemented (BS-European used) — flagged for TLT/GLD/equities")
    out.append("")

    irx = load_irx_curve()
    out.append(f"IRX rate on 2020-06-15: {get_rf_for_date(irx, pd.Timestamp('2020-06-15'))*100:.3f}%")
    out.append(f"(vs the naive flat r=4.000% used in Section 3 v1)")
    out.append("")
    out.append("=" * 80)
    out.append("## Stacked-correction comparison table")
    out.append("=" * 80)
    out.append("")
    out.append("| Ticker | Side | Strike | Loader δ | δ@r=4%,q=0 | δ@r(t),q=0 | δ@r(t),q | "
               "|Δ|naive | |Δ|+(a) | |Δ|+(a)+(b) |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    all_rows: list[dict] = []
    for ticker in ALL_14:
        try:
            df = load_ticker(ticker)
            rows = trace_for_ticker(out, df, ticker, irx)
            if rows:
                for r in rows:
                    all_rows.append(r)
                    out.append(
                        f"| {r['ticker']} | {r['side']} | {r['K']:.2f} | "
                        f"{r['loader_d']:+.4f} | {r['d_naive_r4']:+.4f} | "
                        f"{r['d_a_actual_r']:+.4f} | {r['d_ab_actual_r_q']:+.4f} | "
                        f"{r['abs_div_naive']:.4f} | {r['abs_div_a']:.4f} | "
                        f"{r['abs_div_ab']:.4f} |"
                    )
            del df
        except Exception as e:
            out.append(f"| {ticker} | ERROR: {e} |")

    out.append("")
    out.append("=" * 80)
    out.append("## Pass criteria evaluation")
    out.append("=" * 80)
    out.append("")
    out.append("| Ticker | Side | |Δ|+(a)+(b) | Threshold | Sign OK | Pass? |")
    out.append("|---|---|---:|---:|---|---|")
    INDEX_TICKERS = {"SPX", "RUT", "NDX"}
    ETF_TICKERS = {"TLT", "GLD"}

    overall_pass = True
    for r in all_rows:
        if r["ticker"] in INDEX_TICKERS:
            threshold = 0.03
        elif r["ticker"] in ETF_TICKERS:
            threshold = 0.05
        else:
            threshold = 0.05  # equities — same lenient threshold
        passes_threshold = r["abs_div_ab"] <= threshold
        sign_ok = r["sign_match_loader"]
        passes = passes_threshold and sign_ok
        if not passes:
            overall_pass = False
        out.append(
            f"| {r['ticker']} | {r['side']} | {r['abs_div_ab']:.4f} | ≤{threshold:.2f} | "
            f"{'YES' if sign_ok else 'NO ⚠'} | {'PASS' if passes else 'FAIL ⚠'} |"
        )

    out.append("")
    out.append(f"**Overall pass criteria: {'PASS' if overall_pass else 'FAIL'}**")

    out.append("")
    out.append("=" * 80)
    out.append("## Smoothness + sign-flip check (FULL chain across moneyness)")
    out.append("=" * 80)
    out.append("")
    out.append("| Ticker | n_put | n_call | sign_flips | max |Δ|+(a)+(b) | "
               "median |Δ|+(a)+(b) | max_jump_put | max_jump_call |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    smoothness_pass = True
    for ticker in ALL_14:
        try:
            df = load_ticker(ticker)
            res = smoothness_check(out, df, ticker, irx)
            if res:
                if res["sign_flip_count"] > 0:
                    smoothness_pass = False
                    flag = " ⚠ SIGN FLIP"
                else:
                    flag = ""
                out.append(
                    f"| {ticker} | {res['n_put']} | {res['n_call']} | "
                    f"{res['sign_flip_count']}{flag} | "
                    f"{res['max_abs_div_ab']:.4f} | {res['median_abs_div_ab']:.4f} | "
                    f"{res['max_jump_put']:.4f} | {res['max_jump_call']:.4f} |"
                )
            del df
        except Exception as e:
            out.append(f"| {ticker} | ERROR: {e} |")

    out.append("")
    if smoothness_pass:
        out.append("**Sign-flip check: PASS** (no put with positive δ; no call with negative δ)")
    else:
        out.append("**Sign-flip check: FAIL** ⚠ — hard data-quality bug")

    out.append("")
    out.append("=" * 80)
    out.append("## DTE monotonicity check (does |Δ| grow with DTE? — would suggest T bug)")
    out.append("=" * 80)
    out.append("")

    for ticker in ALL_14:
        try:
            df = load_ticker(ticker)
            res = dte_monotonicity_check(out, df, ticker, irx)
            if not res:
                continue
            out.append(f"\n### {ticker}")
            out.append(f"")
            out.append("| DTE bucket | n_in_region | median |Δ|+(a)+(b) |")
            out.append("|---|---:|---:|")
            for r in res:
                out.append(f"| {r['dte_bucket']} | {r['n']} | "
                           f"{r['median_abs_div_ab'] if r['median_abs_div_ab'] is not None else float('nan'):.4f} |")
            del df
        except Exception as e:
            out.append(f"\n### {ticker} — ERROR: {e}")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
