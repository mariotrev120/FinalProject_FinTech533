"""
Smoke test: SPX iron condor through the engine using the per-ticker
parquet loader.

This is the test that was deferred for weeks because loading the 2.4GB
strat2.2.csv.gz crashed WSL. After scripts/convert_optionmetrics_to_parquet.py
ran, SPX is its own 745 MB parquet — loads in ~3s, no crash.

Acceptance:
  - SPX pricer loads from data/processed/options_by_ticker/SPX.parquet
  - run_backtest(use_iron_condor=True) over OOS produces > 200 trades
  - Both put-side and call-side trades present
  - All five fates appear (profit_target, stop_loss, time_exit,
    emergency, eos_force)
  - At least one IC pair has different fates between sides (verifies
    per-side independent management is working)
  - Aggregate stats: equity curve final value, naive Sharpe

Run:
  PYTHONPATH=. .venv/bin/python scripts/smoke_iron_condor_spx.py
"""
from __future__ import annotations

import collections
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_inputs
from src.config import OOS_END, OOS_START
from src.strategy.optionmetrics_pricer import make_pricer_v2


log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    t0 = time.time()
    log.info("=" * 70)
    log.info("SPX iron condor smoke test (per-ticker parquet loader)")
    log.info("=" * 70)

    log.info("Loading SPX pricer from data/processed/options_by_ticker/SPX.parquet")
    pricer = make_pricer_v2("SPX")
    log.info(
        "Pricer loaded in %.1fs. put dict: %d, call dict: %d",
        time.time() - t0,
        len(pricer._by_date_exp_put),
        len(pricer._by_date_exp_call),
    )

    log.info("Loading backtest inputs (small parquets only)...")
    t = time.time()
    inputs = load_inputs()
    log.info("Inputs loaded in %.1fs", time.time() - t)

    log.info("Running iron condor naked OOS [%s, %s]...", OOS_START, OOS_END)
    t = time.time()
    result = run_backtest(
        inputs,
        pricer,
        mode="naked",
        start=OOS_START,
        end=OOS_END,
        use_iron_condor=True,
    )
    log.info(
        "Backtest done in %.1fs. n_trades=%d",
        time.time() - t,
        len(result.trades),
    )

    # --- Trade-side breakdown ---
    fates = collections.Counter(t.fate for t in result.trades)
    rights = collections.Counter(t.spread.right for t in result.trades)
    ic_ids = [t.iron_condor_id for t in result.trades if t.iron_condor_id is not None]
    log.info("Fates: %s", dict(fates))
    log.info("Sides: %s", dict(rights))
    log.info(
        "IC paired trades: %d / %d   unique IC ids: %d",
        len(ic_ids),
        len(result.trades),
        len(set(ic_ids)),
    )

    # --- Per-side independent management diagnostic ---
    ic_groups = defaultdict(list)
    for tr in result.trades:
        if tr.iron_condor_id is not None:
            ic_groups[tr.iron_condor_id].append(tr)
    diff_fate = sum(
        1
        for ts in ic_groups.values()
        if len(ts) == 2 and len(set(t.fate for t in ts)) > 1
    )
    diff_close = sum(
        1
        for ts in ic_groups.values()
        if len(ts) == 2 and len(set(t.close_date for t in ts)) > 1
    )
    log.info(
        "IC pairs with DIFFERENT fates: %d (validates per-side independent mgmt)",
        diff_fate,
    )
    log.info(
        "IC pairs with DIFFERENT close dates: %d",
        diff_close,
    )

    # --- Aggregate ---
    total_pnl = sum(t.pnl_per_spread or 0 for t in result.trades)
    final_eq = result.equity_curve.iloc[-1]
    ret = result.equity_curve.pct_change().dropna()
    sharpe = (
        float(ret.mean() / ret.std() * np.sqrt(252))
        if ret.std() > 0
        else float("nan")
    )

    log.info("=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)
    log.info("Total PnL: $%s", f"{total_pnl:,.0f}")
    log.info("Final equity: $%s", f"{final_eq:,.0f}")
    log.info("Naive Sharpe (annualized): %.3f", sharpe)
    log.info("Total time: %.1fs", time.time() - t0)

    # --- Acceptance checks ---
    log.info("=" * 70)
    log.info("ACCEPTANCE CHECKS")
    log.info("=" * 70)
    checks = [
        ("n_trades > 200", len(result.trades) > 200),
        ("both put and call sides present", "P" in rights and "C" in rights),
        (
            "all 5 fates appear",
            {"profit_target", "stop_loss", "time_exit", "emergency", "eos_force"}
            <= set(fates.keys()),
        ),
        ("at least 1 IC pair with different fates", diff_fate > 0),
        ("Sharpe is a finite number", np.isfinite(sharpe)),
    ]
    all_pass = True
    for name, passed in checks:
        symbol = "[PASS]" if passed else "[FAIL]"
        log.info("  %s %s", symbol, name)
        if not passed:
            all_pass = False

    log.info("=" * 70)
    if all_pass:
        log.info("[PASS] All acceptance checks passed.")
        log.info("Multi-instrument blocker is CLEARED. Engine + IC + parquet loader work end-to-end.")
        return 0
    else:
        log.info("[FAIL] One or more acceptance checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
