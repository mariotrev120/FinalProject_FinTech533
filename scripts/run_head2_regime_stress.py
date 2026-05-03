"""
Real-data integration runner for Head 2 regime-stress classifier.

Runs the full Head 2 pipeline on REAL data:
  - data/processed/features.parquet — cross-asset feature matrix
  - 5 LABELED_STRESS_EVENT_PEAKS from scripts/stress_events.py (v1 audit)

Reports actual Brier numbers vs the 22-day naive baseline, walk-forward
folds, and the §5.1 5%-Brier-reduction acceptance gate.

Output:
  - stdout: full report (also written to logs/head2_regime_stress.log)
  - data/processed/p_stress_head2.parquet — calibrated p_stress series
  - data/processed/forward_stress_label.parquet — supervised target

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_head2_regime_stress.py

This is NOT a unit test. It's an end-to-end run on the real data and
the headline numbers it produces will be reported in the writeup
results page.

Hodrick (1992) SEs for predictive significance: TODO until C7 lands.
For now we report the Brier numbers without Hodrick CI.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED_DIR, IS_END, OOS_END
from src.models.regime_stress import (
    DEFAULT_FORWARD_HORIZON_TRADING_DAYS,
    LABELED_STRESS_EVENT_PEAKS,
    NAIVE_BASELINE_LOOKBACK_DAYS,
    forward_stress_from_peaks,
    stress_today_from_peaks,
    train_regime_stress_walkforward,
)


log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/head2_regime_stress.log", mode="w"),
        ],
    )

    log.info("=" * 70)
    log.info("Head 2 — Regime Stress Classifier — Real-Data Integration Run")
    log.info("=" * 70)

    features_path = Path(DATA_PROCESSED_DIR) / "features.parquet"
    log.info("Loading features from %s", features_path)
    features = pd.read_parquet(features_path)
    log.info(
        "Features: shape=%s  range=[%s, %s]  cols=%s",
        features.shape,
        features.index.min().date(),
        features.index.max().date(),
        list(features.columns),
    )

    log.info("Building stress-day series and 63-day forward label from %d peaks:",
             len(LABELED_STRESS_EVENT_PEAKS))
    for pk in LABELED_STRESS_EVENT_PEAKS:
        log.info("  - %s", pk.date())

    stress_today = stress_today_from_peaks(features.index)
    forward_label = forward_stress_from_peaks(
        features.index,
        horizon_trading_days=DEFAULT_FORWARD_HORIZON_TRADING_DAYS,
    )
    n_pos = int(forward_label.sum())
    n_total = len(forward_label)
    log.info(
        "Forward label: %d positive / %d total (%.1f%% base rate)",
        n_pos, n_total, 100 * n_pos / n_total,
    )
    log.info(
        "Stress-today indicator: %d trading days flagged",
        int(stress_today.sum()),
    )

    # Walk-forward folds: re-train annually starting OOS_START.
    # Use the IS/OOS split locked in src/config.py.
    is_end = pd.Timestamp(IS_END)
    oos_end = pd.Timestamp(OOS_END)
    log.info(
        "IS/OOS split: IS up to %s, OOS [%s, %s]",
        is_end.date(), (is_end + pd.Timedelta(days=1)).date(), oos_end.date(),
    )

    fold_starts = []
    fs = pd.Timestamp("2018-01-01")
    while fs <= oos_end:
        fold_starts.append(fs)
        fs = fs + pd.DateOffset(years=1)
    log.info("Annual fold starts: %s", [fs.date() for fs in fold_starts])

    log.info("=" * 70)
    log.info("Training Head 2 walk-forward...")
    log.info("=" * 70)

    rep = train_regime_stress_walkforward(
        features=features,
        forward_label=forward_label,
        stress_today=stress_today,
        fold_starts=fold_starts,
        fold_end=oos_end,
        horizon_trading_days=DEFAULT_FORWARD_HORIZON_TRADING_DAYS,
    )

    log.info("=" * 70)
    log.info("RESULTS (per PRE_COMMITMENT_VRP §5.1 acceptance gate):")
    log.info("=" * 70)
    for line in rep.headline().splitlines():
        log.info(line)

    # Write artifacts
    p_path = Path(DATA_PROCESSED_DIR) / "p_stress_head2.parquet"
    label_path = Path(DATA_PROCESSED_DIR) / "forward_stress_label.parquet"
    if len(rep.p_stress) > 0:
        rep.p_stress.to_frame().to_parquet(p_path)
        log.info("Saved p_stress series to %s", p_path)
    rep.forward_label.to_frame().to_parquet(label_path)
    log.info("Saved forward_stress_label to %s", label_path)

    # Final pass/fail
    log.info("=" * 70)
    if rep.passed_5pct_threshold:
        log.info(
            "[PASS] Head 2 OOS Brier reduction = %+.1f%% ≥ 5%% threshold. "
            "Activate Head 2 with calibrated p_stress book scaler.",
            rep.brier_reduction_pct * 100,
        )
        return 0
    else:
        log.info(
            "[FAIL/NA] Head 2 OOS Brier reduction = %+.1f%% < 5%% threshold. "
            "Per §5.1 fallback rule: drop Head 2; book scaler defaults to "
            "1.0; report negative finding in writeup.",
            rep.brier_reduction_pct * 100
            if not pd.isna(rep.brier_reduction_pct)
            else float("nan"),
        )
        return 0   # negative finding is a valid finding, not a script failure


if __name__ == "__main__":
    sys.exit(main())
