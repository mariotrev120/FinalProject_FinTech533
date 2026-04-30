"""
Tier 2 audit: halt-firing precision and recall over 2018-2024.

For every halt activation in the OOS halts_only run, capture:
  - date
  - state (hard / soft / drawdown / slow)
  - triggers (which conditions fired)
  - SPX 30-day forward drawdown from that date
  - Whether SPX experienced a >=5% drawdown in the next 30 days (true positive)

Also computes the BASE RATE: fraction of all OOS days where SPX had a >=5%
drawdown in the next 30 days. Compares halt-conditional drawdown to base rate.

Outputs:
  - data/processed/audit_halt_firing.parquet  (every halt firing date)
  - precision/recall summary in stdout
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_default_pricer, load_inputs
from src.config import DATA_RAW_DIR, OOS_END, OOS_START


log = logging.getLogger(__name__)

DRAWDOWN_THRESHOLD = 0.05   # 5% SPX drawdown in next 30 days = "real stress"
FORWARD_WINDOW_DAYS = 30


def forward_max_drawdown(spx_close: pd.Series, start: pd.Timestamp,
                          window_days: int = FORWARD_WINDOW_DAYS) -> float:
    """SPX max drawdown over the next `window_days` calendar days starting `start`."""
    end = start + pd.Timedelta(days=window_days)
    win = spx_close.loc[start:end]
    if len(win) < 2:
        return 0.0
    peak = win.cummax()
    dd = (peak - win) / peak
    return float(dd.max())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    inputs = load_inputs()
    pricer = load_default_pricer()
    log.info("running halts_only mode for halt-log over 2018-2024...")
    result = run_backtest(inputs, pricer, mode="halts_only", start=OOS_START, end=OOS_END)
    halt_log = result.halt_log
    log.info("halt log has %d rows", len(halt_log))

    spx_close = pd.read_parquet(DATA_RAW_DIR / "SPX.parquet")["close"]
    spx_close.index = pd.to_datetime(spx_close.index)
    spx_close = spx_close.loc[OOS_START:OOS_END]

    # === Halt activation events ===
    halt_log = halt_log.copy()
    halt_log.index = pd.to_datetime(halt_log.index)
    halt_log["state_change"] = halt_log["state"] != halt_log["state"].shift(1)
    activations = halt_log[(halt_log["state"] != "active") & halt_log["state_change"]]
    log.info("distinct halt activation events: %d", len(activations))

    rows = []
    for date, row in activations.iterrows():
        fdd = forward_max_drawdown(spx_close, date)
        is_true_positive = fdd >= DRAWDOWN_THRESHOLD
        rows.append({
            "date": date,
            "state": row["state"],
            "triggers": row["triggers"],
            "fwd_30d_max_dd_pct": fdd * 100,
            "true_positive_5pct": is_true_positive,
        })
    df = pd.DataFrame(rows)
    df.to_parquet("data/processed/audit_halt_firing.parquet")

    print("\n" + "=" * 70)
    print(f"HALT FIRING AUDIT — OOS 2018-2024")
    print("=" * 70)
    if len(df) == 0:
        print("No halt activations — nothing to audit.")
        return 0
    print(f"Total activation events: {len(df)}")
    print()
    print("By state:")
    print(df["state"].value_counts().to_string())
    print()
    print("Top trigger combinations:")
    print(df["triggers"].value_counts().head(10).to_string())
    print()

    # Conditional drawdown vs base rate
    n_true_pos = df["true_positive_5pct"].sum()
    print(f"Halt activations followed by >= {DRAWDOWN_THRESHOLD*100:.0f}% SPX drawdown within 30 days: {n_true_pos}/{len(df)} = {n_true_pos/len(df)*100:.1f}%")
    print(f"Mean fwd 30d drawdown after a halt: {df['fwd_30d_max_dd_pct'].mean():.2f}%")
    print(f"Median fwd 30d drawdown after a halt: {df['fwd_30d_max_dd_pct'].median():.2f}%")
    print()

    # Base rate: of all trading days, what fraction had >= 5% drawdown in next 30d?
    base_rate_dds = []
    for d in spx_close.index:
        base_rate_dds.append(forward_max_drawdown(spx_close, d))
    base_rate_dds = pd.Series(base_rate_dds, index=spx_close.index)
    n_base_pos = (base_rate_dds >= DRAWDOWN_THRESHOLD).sum()
    base_rate = n_base_pos / len(base_rate_dds)
    print(f"BASE RATE: {n_base_pos}/{len(base_rate_dds)} = {base_rate*100:.1f}% of all OOS days had >= 5% SPX drawdown in next 30d")
    print(f"Halt-conditional rate: {n_true_pos/len(df)*100:.1f}%")
    print(f"PRECISION LIFT vs base rate: {(n_true_pos/len(df) - base_rate)*100:+.1f} pct points")
    print()

    if n_true_pos / len(df) > base_rate * 1.5:
        print("CONCLUSION: halts have meaningful precision (>= 1.5x base rate)")
    elif n_true_pos / len(df) > base_rate:
        print("CONCLUSION: halts have modest precision (above base rate)")
    else:
        print("CONCLUSION: halts have NO precision over base rate. They are firing")
        print("            as often as random days predict drawdowns. False positives dominate.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
