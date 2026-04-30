"""
End-to-end ML training pipeline.

  1. Load processed features.
  2. Run naked-mode backtest over the IS+OOS window to generate trade labels.
  3. Walk-forward refit XGBoost (annual cadence, harness in
     src/backtest/walkforward.py) using IS labels through each fold boundary.
  4. Save the calibrated probability series for use in ml_only / full modes.

Output: data/processed/ml_probabilities.parquet — a Series indexed by
Monday date, value = calibrated XGBoost p(win).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/train_ml.py
"""
from __future__ import annotations

import logging

import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_inputs
from src.backtest.walkforward import annual_walk_forward_xgb
from src.config import DATA_PROCESSED_DIR, IS_START, OOS_END, SEED
from src.models.label_trades import label_trades_dataframe
from src.strategy.black_scholes import make_default_pricer


log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    features = pd.read_parquet(DATA_PROCESSED_DIR / "features.parquet")
    log.info("loaded features: %d rows %s -> %s",
             len(features), features.index.min().date(), features.index.max().date())

    # Step 1: naked backtest -> trade labels
    inputs = load_inputs()
    pricer = make_default_pricer()
    log.info("running naked backtest to label trades over %s -> %s", IS_START, OOS_END)
    naked_result = run_backtest(inputs, pricer, mode="naked", start=IS_START, end=OOS_END)
    labels = label_trades_dataframe(naked_result)
    log.info("labelled %d trades, win rate %.1f%%",
             len(labels), 100 * labels["win"].mean())

    # Step 2: annual walk-forward (harness in src/backtest/walkforward.py)
    probs = annual_walk_forward_xgb(
        features=features, labels=labels,
        is_start=IS_START, oos_end=OOS_END, seed=SEED,
    )
    log.info("predictions: %d Mondays, mean=%.3f, median=%.3f",
             len(probs), probs.mean(), probs.median())

    out_path = DATA_PROCESSED_DIR / "ml_probabilities.parquet"
    probs.to_frame("p_calibrated").to_parquet(out_path)
    log.info("saved probabilities to %s", out_path)

    labels_path = DATA_PROCESSED_DIR / "naked_labels.parquet"
    labels.to_parquet(labels_path)
    log.info("saved labels to %s", labels_path)

    print(f"\n=== ML training complete ===")
    print(f"  trades labelled: {len(labels)} (win rate {100 * labels['win'].mean():.1f}%)")
    print(f"  Mondays predicted: {len(probs)}")
    print(f"  prob distribution:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"    {int(q*100)}th pct: {probs.quantile(q):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
