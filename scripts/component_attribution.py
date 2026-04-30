"""
Component Attribution table — the ablation deliverable.

Runs all 4 modes (naked, ml_only, halts_only, full) on both IS and OOS,
computes per-mode metrics with block-bootstrap 90% CIs on Sharpe, and
emits the comparison table that lands in the writeup.

Pre-committed interpretation rule (PRE_COMMITMENT.md):
  If `ml_only` Sharpe is within 0.1 of `naked` Sharpe in OOS, the ML filter
  is decorative. The writeup states this explicitly.

Usage:
    PYTHONPATH=. .venv/bin/python -m src.backtest.component_attribution
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.loader import load_inputs
from src.config import (
    DATA_PROCESSED_DIR, IS_END, IS_START, ML_DECISION_THRESHOLD, OOS_END, OOS_START,
)
from src.metrics.bootstrap import bootstrap_ci, sharpe_stat, winrate_stat
from src.metrics.performance import (
    combined_metrics, equity_curve_metrics, summarize_blotter, trade_returns,
)
from src.backtest.loader import load_default_pricer as make_default_pricer
from src.strategy.types import Mode


log = logging.getLogger(__name__)


def run_one(mode: Mode, start: str, end: str) -> BacktestResult:
    inputs = load_inputs()
    pricer = make_default_pricer()
    return run_backtest(inputs, pricer, mode=mode, start=start, end=end)


def attribution_row(result: BacktestResult, mode: Mode, label: str) -> dict:
    metrics = combined_metrics(result.trades, result.equity_curve, starting_equity=100_000.0)
    rets = trade_returns(result.trades, starting_equity=100_000.0)
    sharpe_ci = bootstrap_ci(rets, sharpe_stat, block_size=3, n_resamples=5_000)
    wr_ci = bootstrap_ci(rets, winrate_stat, block_size=3, n_resamples=5_000)
    return {
        "label": label,
        "mode": mode,
        "n_trades": metrics["n_trades"],
        "win_rate": metrics["win_rate"],
        "win_rate_ci_lo": wr_ci["lower"],
        "win_rate_ci_hi": wr_ci["upper"],
        "expected_return_per_trade_pct": metrics["expected_return_per_trade_pct"],
        "annualized_return_pct": metrics["annualized_return_pct"],
        "sharpe_daily": metrics["sharpe_daily"],
        "sharpe_trade": metrics["sharpe_trade_level"],
        "sharpe_trade_ci_lo": sharpe_ci["lower"],
        "sharpe_trade_ci_hi": sharpe_ci["upper"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "max_dd_duration_days": metrics["max_dd_duration_days"],
        "fates": metrics["fate_distribution"],
        "skipped_entries": len(result.skipped_entries),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = []
    windows = [
        ("IS",  IS_START,  IS_END),
        ("OOS", OOS_START, OOS_END),
    ]
    modes: list[Mode] = ["naked", "ml_only", "halts_only", "full"]

    for win_label, start, end in windows:
        for mode in modes:
            log.info("[%s] mode=%s on %s -> %s", win_label, mode, start, end)
            try:
                r = run_one(mode, start, end)
            except Exception as e:
                log.error("[%s] mode=%s failed: %s", win_label, mode, e)
                continue
            rows.append(attribution_row(r, mode, label=win_label))

    df = pd.DataFrame(rows)
    out_path = DATA_PROCESSED_DIR / "component_attribution.parquet"
    df.to_parquet(out_path)
    csv_path = DATA_PROCESSED_DIR / "component_attribution.csv"
    df.drop(columns="fates").to_csv(csv_path, index=False)
    log.info("wrote %s and %s", out_path, csv_path)

    # --- Pre-committed interpretation check ---
    print("\n" + "=" * 80)
    print(f"COMPONENT ATTRIBUTION — ML decision threshold p >= {ML_DECISION_THRESHOLD}")
    print("=" * 80)
    cols = ["label", "mode", "n_trades", "win_rate",
            "annualized_return_pct", "sharpe_trade", "sharpe_trade_ci_lo",
            "sharpe_trade_ci_hi", "max_drawdown_pct"]
    print(df[cols].round(3).to_string(index=False))

    # OOS-specific interpretation
    oos = df[df.label == "OOS"].set_index("mode")
    if "naked" in oos.index and "ml_only" in oos.index:
        delta = oos.loc["ml_only", "sharpe_trade"] - oos.loc["naked", "sharpe_trade"]
        print(f"\nOOS Sharpe[ml_only] - Sharpe[naked] = {delta:+.3f}")
        if abs(delta) < 0.1:
            print("  >>> ML filter is DECORATIVE (within 0.1 of naked).")
            print("  >>> Writeup must state this explicitly per PRE_COMMITMENT.md.")
        else:
            print(f"  >>> ML filter contributes {delta:+.3f} Sharpe over naked.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
