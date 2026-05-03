"""
Layer 4 catch-22 diagnosis confirmation.

Per user-specified criteria (2026-05-03):
  Diagnosis LOCKED if BOTH:
    (a) Average per-trade P&L within ±0.1% of starting capital
    (b) Median trade duration < 5 trading days

If a different pattern emerges, STOP and document — do NOT propose a
fix. Wait for morning review.

Outputs:
  - logs/layer4_confirmation.log
  - data/processed/layer4_confirmation/spx_halts_only_equity_vs_hwm.parquet
  - data/processed/layer4_confirmation/spx_halts_only_trades_summary.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_inputs
from src.config import DATA_PROCESSED_DIR, OOS_END, OOS_START
from src.strategy.optionmetrics_pricer import make_pricer_v2


log = logging.getLogger(__name__)


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    out_dir = Path(DATA_PROCESSED_DIR) / "layer4_confirmation"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/layer4_confirmation.log", mode="w"),
        ],
    )

    log.info("=" * 70)
    log.info("Layer 4 catch-22 — diagnostic confirmation (SPX halts_only OOS)")
    log.info("=" * 70)

    inputs = load_inputs()
    pricer = make_pricer_v2("SPX")

    log.info("Running SPX halts_only IC over OOS [%s, %s]...", OOS_START, OOS_END)
    result = run_backtest(
        inputs, pricer, mode="halts_only",
        start=OOS_START, end=OOS_END, use_iron_condor=True,
    )

    n_trades = len(result.trades)
    starting_capital = inputs.initial_equity
    log.info("n_trades=%d, starting_capital=$%s", n_trades, f"{starting_capital:,.0f}")

    # --- Per-trade PnL stats ---
    pnls = np.array([
        t.pnl_per_spread for t in result.trades
        if t.pnl_per_spread is not None
    ])
    pnl_mean = float(pnls.mean()) if len(pnls) else float("nan")
    pnl_median = float(np.median(pnls)) if len(pnls) else float("nan")
    pnl_pct_of_starting = pnl_mean / starting_capital * 100

    # --- Trade duration stats (in trading days) ---
    # entry_date and exit_date are date objects; convert to pandas
    durations_calendar = []
    for t in result.trades:
        if t.exit_date is None or t.entry_date is None:
            continue
        d = (pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days
        durations_calendar.append(d)
    durations_calendar = np.array(durations_calendar)
    duration_mean = float(durations_calendar.mean()) if len(durations_calendar) else float("nan")
    duration_median = float(np.median(durations_calendar)) if len(durations_calendar) else float("nan")

    log.info("=" * 70)
    log.info("PER-TRADE STATS")
    log.info("=" * 70)
    log.info("  Mean PnL per trade:          $%.2f", pnl_mean)
    log.info("  Median PnL per trade:        $%.2f", pnl_median)
    log.info("  Mean PnL as %% of starting:   %+.4f%%", pnl_pct_of_starting)
    log.info("  Mean trade duration (cal d): %.1f", duration_mean)
    log.info("  Median trade duration (cal d):%.0f", duration_median)
    log.info("  ±0.1%% of starting:           [%+.4f%%, %+.4f%%]", -0.10, 0.10)

    # --- Lock criteria ---
    crit_a_pnl_near_zero = abs(pnl_pct_of_starting) <= 0.10
    crit_b_short_duration = duration_median < 5
    diagnosis_locked = crit_a_pnl_near_zero and crit_b_short_duration

    log.info("=" * 70)
    log.info("DIAGNOSIS CONFIRMATION")
    log.info("=" * 70)
    log.info("  (a) avg per-trade PnL within ±0.1%% of start: %s (got %+.4f%%)",
             "PASS" if crit_a_pnl_near_zero else "FAIL", pnl_pct_of_starting)
    log.info("  (b) median trade duration < 5 days:         %s (got %.0f)",
             "PASS" if crit_b_short_duration else "FAIL", duration_median)
    log.info("  → Layer 4 catch-22 diagnosis: %s",
             "LOCKED" if diagnosis_locked else "DIFFERENT PATTERN — STOP")

    # --- Equity vs HWM in 2020-02-24 → 2022-10-21 ---
    eq = result.equity_curve
    eq_window = eq.loc["2020-02-24":"2022-10-21"]
    hwm_window = eq_window.cummax()
    underwater = eq_window < hwm_window
    days_underwater = int(underwater.sum())
    pct_underwater = 100 * days_underwater / len(eq_window) if len(eq_window) else 0.0

    log.info("")
    log.info("EQUITY vs HWM in 2020-02-24 → 2022-10-21 (%d trading days):", len(eq_window))
    log.info("  start equity:    $%s", f"{float(eq_window.iloc[0]):,.2f}")
    log.info("  end equity:      $%s", f"{float(eq_window.iloc[-1]):,.2f}")
    log.info("  HWM at end:      $%s", f"{float(hwm_window.iloc[-1]):,.2f}")
    log.info("  days underwater: %d  (%.1f%% of window)", days_underwater, pct_underwater)
    log.info("  max DD in window: %+.2f%%",
             100 * float((eq_window / hwm_window - 1).min()))

    # Save artifacts
    eq_window.to_frame("equity").assign(hwm=hwm_window).to_parquet(
        out_dir / "spx_halts_only_equity_vs_hwm.parquet"
    )

    trades_summary = pd.DataFrame({
        "n_trades": [n_trades],
        "pnl_mean": [pnl_mean],
        "pnl_median": [pnl_median],
        "pnl_pct_of_starting": [pnl_pct_of_starting],
        "duration_mean_cal_days": [duration_mean],
        "duration_median_cal_days": [duration_median],
        "criterion_a_pnl_near_zero": [crit_a_pnl_near_zero],
        "criterion_b_short_duration": [crit_b_short_duration],
        "diagnosis_locked": [diagnosis_locked],
    })
    trades_summary.to_csv(out_dir / "spx_halts_only_trades_summary.csv", index=False)

    log.info("=" * 70)
    log.info("Saved artifacts to %s", out_dir)
    return 0 if diagnosis_locked else 1


if __name__ == "__main__":
    sys.exit(main())
