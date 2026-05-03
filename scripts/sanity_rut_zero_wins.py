"""
Sanity check: spot-check 10 random RUT naked-mode trades.

The Head 1 v1 (per-side) labels reported RUT win rate 0.0% over 591 trades.
0/591 wins is implausible — even structurally bad regimes typically have a
small positive tail. This script runs the RUT naked backtest and prints
detailed records for 10 random trades to verify the result is real
(stop_loss exits at reasonable entry credits) and not a pricing or
strike-selection bug specific to RUT.

Print fields (per directive):
  entry_date | side (P/C) | entry_credit | exit_price | fate | pnl_per_spread

Plus diagnostic: distribution of entry credits, per-side win counts,
distribution of exit fates, sample of a few trades that ARE wins (if any).
"""
from __future__ import annotations

import logging
import random
import sys

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_inputs
from src.config import OOS_END, OOS_START, SEED
from src.strategy.optionmetrics_pricer import make_pricer_v2


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("RUT naked-mode sanity check: 10-trade spot-check + summaries")
    log.info("=" * 70)

    inputs = load_inputs()
    log.info("Loading RUT pricer...")
    pricer = make_pricer_v2("RUT")

    log.info("Running RUT naked backtest (OOS: %s to %s)...", OOS_START, OOS_END)
    result = run_backtest(
        inputs, pricer, mode="naked",
        start=OOS_START, end=OOS_END, use_iron_condor=True,
    )
    trades = result.trades
    n = len(trades)
    log.info("RUT naked backtest: %d trades, %d skipped entries",
             n, len(result.skipped_entries))

    if result.skipped_entries:
        sk_df = pd.DataFrame(result.skipped_entries)
        log.info("\n=== Skip-reason counts ===")
        print(sk_df["reason"].value_counts().to_string())

    if n == 0:
        log.error("Zero trades — cannot diagnose. Check engine config / pricer.")
        return 1

    # Build a per-trade DataFrame
    rows = []
    for t in trades:
        side = "P" if (t.spread is not None and t.spread.right == "P") else "C"
        short_k = t.spread.short_leg.strike if t.spread is not None else None
        long_k = t.spread.long_leg.strike if t.spread is not None else None
        rows.append({
            "trade_id": t.trade_id,
            "iron_condor_id": t.iron_condor_id,
            "entry_date": pd.Timestamp(t.entry_date),
            "exit_date": pd.Timestamp(t.exit_date) if t.exit_date else pd.NaT,
            "side": side,
            "short_strike": short_k,
            "long_strike": long_k,
            "entry_credit": t.entry_credit_per_spread,
            "exit_debit": t.exit_debit_per_spread,
            "fate": t.fate,
            "pnl_per_spread": t.pnl_per_spread,
            "entry_dte": t.entry_dte,
            "entry_spx": t.entry_spx,
        })
    df = pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)

    # 1. Per-side summary
    log.info("\n=== Per-side counts and win rates ===")
    print(df.groupby("side").agg(
        n_trades=("trade_id", "count"),
        wins=("pnl_per_spread", lambda x: int((x > 0).sum())),
        win_rate=("pnl_per_spread", lambda x: float((x > 0).mean())),
        mean_pnl=("pnl_per_spread", "mean"),
        median_pnl=("pnl_per_spread", "median"),
    ).round(4).to_string())

    # 2. Fate distribution
    log.info("\n=== Fate distribution (per side) ===")
    print(df.groupby(["side", "fate"]).size().unstack(fill_value=0).to_string())

    # 3. Entry credit distribution per side
    log.info("\n=== Entry credit distribution (per side) ===")
    print(df.groupby("side")["entry_credit"].describe().round(3).to_string())

    # 4. Spot-check 10 random trades
    log.info("\n=== Spot check: 10 random trades ===")
    rng = random.Random(SEED)
    sample_idx = rng.sample(range(len(df)), min(10, len(df)))
    sample = df.iloc[sample_idx].sort_values("entry_date")
    print(sample[[
        "entry_date", "side", "short_strike", "long_strike",
        "entry_credit", "exit_debit", "fate", "pnl_per_spread",
    ]].round(3).to_string(index=False))

    # 5. If any wins exist, show 5 of them for context
    wins = df[df["pnl_per_spread"] > 0]
    if len(wins):
        log.info("\n=== Sample of WINS (n=%d) ===", len(wins))
        ws = wins.sample(min(5, len(wins)), random_state=SEED)
        print(ws[[
            "entry_date", "side", "short_strike", "long_strike",
            "entry_credit", "exit_debit", "fate", "pnl_per_spread",
        ]].round(3).to_string(index=False))
    else:
        log.warning("\n=== ZERO WINS across %d trades — confirming Head 1 finding ===", n)

    # 6. Sanity: max profit anyway (fate=profit_target should never appear if 0 wins)
    log.info("\n=== Max profitable trade (regardless of fate) ===")
    top5 = df.nlargest(5, "pnl_per_spread")
    print(top5[[
        "entry_date", "side", "short_strike", "long_strike",
        "entry_credit", "exit_debit", "fate", "pnl_per_spread",
    ]].round(3).to_string(index=False))

    log.info("\n=== Min profit trade (worst losses) ===")
    bot5 = df.nsmallest(5, "pnl_per_spread")
    print(bot5[[
        "entry_date", "side", "short_strike", "long_strike",
        "entry_credit", "exit_debit", "fate", "pnl_per_spread",
    ]].round(3).to_string(index=False))

    # 7. Sanity check: how many trades had positive entry_credit
    n_pos_credit = int((df["entry_credit"] > 0).sum())
    log.info("\nEntry credits > 0: %d / %d (= %.1f%%)",
             n_pos_credit, n, 100 * n_pos_credit / n)
    log.info("Entry credits == 0: %d", int((df["entry_credit"] == 0).sum()))
    log.info("Entry credits  < 0: %d", int((df["entry_credit"] < 0).sum()))

    return 0


if __name__ == "__main__":
    sys.exit(main())
