"""
Tier 2 audit: strike snap distance distribution.

For every trade in the naked-mode blotter, compare:
  - target_delta:    -0.16 (the strategy's stated 16-delta short strike)
  - realized_delta:  the actual delta at the traded strike (from OptionMetrics)

Reports the distribution of |target - realized| so we can confirm whether
the strategy is truly trading "16-delta puts" vs a band around 16-delta.

Also reports the distribution of trade outcomes by realized-delta bucket,
which helps quantify whether the strike snap introduces selection bias.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_default_pricer, load_inputs
from src.config import IS_START, OOS_END
from src.strategy.optionmetrics_pricer import make_optionmetrics_pricer


log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    inputs = load_inputs()
    pricer = load_default_pricer()
    log.info("running naked backtest to populate blotter for audit...")
    result = run_backtest(inputs, pricer, mode="naked", start=IS_START, end=OOS_END)
    log.info("blotter has %d trades", len(result.trades))

    # For each trade, look up the realized short-leg delta at entry from
    # the pricer and compare to target -0.16
    rows = []
    for t in result.trades:
        as_of = pd.Timestamp(t.entry_date)
        q = pricer.quote_put(as_of, t.spread.underlying,
                              t.spread.short_leg.strike, t.spread.short_leg.expiry)
        if q is None:
            continue
        realized_delta = q.delta
        diff = abs((-0.16) - realized_delta)
        rows.append({
            "entry_date": pd.Timestamp(t.entry_date),
            "year": as_of.year,
            "vix": t.entry_vix,
            "target_delta": -0.16,
            "realized_delta": realized_delta,
            "abs_diff": diff,
            "win": int(t.pnl_per_spread > 0) if t.pnl_per_spread is not None else 0,
            "snapped_strike": q.strike,
            "requested_strike": t.spread.short_leg.strike,
        })
    df = pd.DataFrame(rows)
    print(f"\n=== Strike snap audit ({len(df)} trades) ===\n")
    print("Realized short-leg delta (signed):")
    print(df["realized_delta"].describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).round(4).to_string())
    print()
    print("|target_delta - realized_delta|:")
    print(df["abs_diff"].describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).round(4).to_string())
    print()
    print("Win rate by realized-delta bucket:")
    df["delta_bucket"] = pd.cut(df["realized_delta"],
                                 bins=[-1.0, -0.25, -0.20, -0.18, -0.16, -0.14, -0.12, -0.10, 0.0])
    by_bucket = df.groupby("delta_bucket", observed=True).agg(
        trades=("win", "count"), win_rate=("win", "mean"))
    print(by_bucket.to_string())
    print()
    print(f"=== Conclusion ===")
    p95 = df["abs_diff"].quantile(0.95)
    if p95 < 0.02:
        print(f"  95th pct |delta diff| = {p95:.4f}: strategy is genuinely 16-delta")
    elif p95 < 0.04:
        print(f"  95th pct |delta diff| = {p95:.4f}: strategy is ~16-delta within reasonable bounds")
    else:
        print(f"  95th pct |delta diff| = {p95:.4f}: WARNING the strategy trades a wide delta band")
        print(f"  README description should reflect actual delta range, not just '16-delta'")

    # Drop Interval-typed delta_bucket col before parquet write (pyarrow
    # doesn't support categorical-of-interval directly).
    df.drop(columns=["delta_bucket"], errors="ignore").to_parquet(
        "data/processed/audit_strike_snap.parquet"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
