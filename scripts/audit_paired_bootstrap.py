"""
Tier 3 audit: paired block-bootstrap on (ml_only - naked) trade returns.

The standard component attribution reports independent CIs per mode. This is
the wrong test for "does ML add value" because trade-by-trade outcomes are
correlated across modes (same dates → same VRP regime). The principled
test is a paired bootstrap on the per-trade DIFFERENCE series.

Procedure:
  1. Run naked and ml_only over OOS.
  2. Align by entry_date (intersection of trades present in both blotters).
  3. Compute diff_return[t] = ret_ml[t] - ret_naked[t].
  4. Block-bootstrap CI on mean(diff) and on Sharpe(diff).
  5. If CI(mean diff) crosses zero, ML adds no significant edge.

Outputs the CIs and an interpretation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_default_pricer, load_inputs
from src.config import OOS_END, OOS_START, SEED
from src.metrics.bootstrap import bootstrap_ci, mean_return_stat, sharpe_stat


log = logging.getLogger(__name__)


def trade_returns_by_date(trades, starting_equity=100_000.0) -> pd.Series:
    rows = {}
    for t in trades:
        if t.total_pnl is None:
            continue
        rows[pd.Timestamp(t.entry_date)] = t.total_pnl / starting_equity
    return pd.Series(rows).sort_index()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    inputs = load_inputs()
    pricer = load_default_pricer()

    log.info("Running naked OOS...")
    naked = run_backtest(inputs, pricer, mode="naked", start=OOS_START, end=OOS_END)
    log.info("Running ml_only OOS...")
    ml = run_backtest(inputs, pricer, mode="ml_only", start=OOS_START, end=OOS_END)

    rn = trade_returns_by_date(naked.trades)
    rm = trade_returns_by_date(ml.trades)
    common = rn.index.intersection(rm.index)
    rn_c, rm_c = rn.loc[common], rm.loc[common]
    diff = rm_c - rn_c

    print(f"\n=== Paired Bootstrap (ml_only - naked) ===")
    print(f"  naked trades:                {len(rn)}")
    print(f"  ml_only trades:              {len(rm)}")
    print(f"  trades in both blotters:     {len(common)}")
    print(f"  ml-only-only (filtered out by naked? impossible): "
          f"{len(rm) - len(common)}")
    print(f"  naked-only (gated out by ML): {len(rn) - len(common)}")
    print()
    print(f"  diff series (ml_ret - naked_ret) on common dates:")
    print(f"    n:           {len(diff)}")
    print(f"    mean diff:   {diff.mean():+.6f}")
    print(f"    std diff:    {diff.std():.6f}")
    print(f"    nonzero rows: {(diff != 0).sum()}")
    print()

    if (diff != 0).sum() == 0:
        print(f"All paired diffs are exactly zero — ML and naked produce identical")
        print(f"outcomes on every shared date. ML adds NO economic value.")
        return 0

    mean_ci = bootstrap_ci(diff, mean_return_stat, block_size=3, n_resamples=10_000, seed=SEED)
    sharpe_ci = bootstrap_ci(diff, sharpe_stat, block_size=3, n_resamples=10_000, seed=SEED)

    print(f"  Block-bootstrap CIs (block=3, n=10000):")
    print(f"    mean(diff)   point={mean_ci['point']:+.6f}  90% CI [{mean_ci['lower']:+.6f}, {mean_ci['upper']:+.6f}]")
    print(f"    Sharpe(diff) point={sharpe_ci['point']:+.4f}  90% CI [{sharpe_ci['lower']:+.4f}, {sharpe_ci['upper']:+.4f}]")
    print()

    crosses_zero = mean_ci["lower"] < 0 < mean_ci["upper"]
    if crosses_zero:
        print(f"  CI for mean(diff) CROSSES ZERO at 90%.")
        print(f"  -> Cannot reject H0: ML adds no per-trade edge over naked.")
    else:
        print(f"  CI for mean(diff) is one-sided.")
        if mean_ci["lower"] > 0:
            print(f"  -> ML adds significantly positive per-trade edge.")
        else:
            print(f"  -> ML adds significantly NEGATIVE per-trade edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
