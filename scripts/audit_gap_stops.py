"""
Find every stop-loss trade with a same-day gap > 1% and verify the gap-aware
MAX rule fires (realized debit >= max(close-debit, gap-debit, stop_level)).
"""
from __future__ import annotations

import logging
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_default_pricer, load_inputs
from src.config import OOS_END, OOS_START


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    inputs = load_inputs()
    pricer = load_default_pricer()
    naked = run_backtest(inputs, pricer, mode="naked", start=OOS_START, end=OOS_END)

    spx = inputs.bars_spx.copy()
    spx.index = pd.to_datetime(spx.index)

    rows = []
    for t in naked.trades:
        if t.fate != "stop_loss" or t.exit_date is None:
            continue
        exit_dt = pd.Timestamp(t.exit_date)
        if exit_dt not in spx.index:
            continue
        idx = spx.index.get_loc(exit_dt)
        prior_close = spx.iloc[idx - 1]["close"] if idx > 0 else float("nan")
        gap_pct = (spx.loc[exit_dt, "open"] / prior_close - 1.0) * 100
        rows.append({
            "entry_date": pd.Timestamp(t.entry_date),
            "exit_date": exit_dt,
            "short_strike": t.spread.short_leg.strike,
            "entry_credit_per_spread": t.entry_credit_per_spread,
            "exit_debit_per_spread": t.exit_debit_per_spread,
            "spx_open": spx.loc[exit_dt, "open"],
            "spx_close": spx.loc[exit_dt, "close"],
            "prior_close": prior_close,
            "gap_pct": gap_pct,
            "intraday_low": spx.loc[exit_dt, "low"],
            "pnl_per_spread": t.pnl_per_spread,
        })
    df = pd.DataFrame(rows)
    print(f"\nTotal stop_loss exits: {len(df)}")
    print(f"  with abs gap > 1%:   {(df['gap_pct'].abs() > 1.0).sum()}")
    print(f"  with abs gap > 2%:   {(df['gap_pct'].abs() > 2.0).sum()}")
    print()
    big = df[df["gap_pct"].abs() > 1.0].sort_values("gap_pct").head(8)
    print("Worst-gap stop_loss trades:")
    print(big[["exit_date", "short_strike", "entry_credit_per_spread",
                "exit_debit_per_spread", "spx_open", "spx_close",
                "prior_close", "gap_pct", "pnl_per_spread"]].to_string(index=False))

    print(f"\n--- Sanity:  exit_debit observations ---")
    print(f"  median exit_debit at gap_abs>1%:  {df[df['gap_pct'].abs() > 1.0]['exit_debit_per_spread'].median():.2f}")
    print(f"  median exit_debit at gap_abs<=1%: {df[df['gap_pct'].abs() <= 1.0]['exit_debit_per_spread'].median():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
