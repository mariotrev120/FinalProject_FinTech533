"""
End-to-end ML training pipeline.

  1. Load processed features.
  2. Run naked-mode backtest over the IS+OOS window to generate trade labels.
  3. Walk-forward refit XGBoost (annual cadence) using IS labels through
     each fold boundary.
  4. Save the calibrated probability series for use in ml_only / full modes.

Output: data/processed/ml_probabilities.parquet — a Series indexed by
Monday date, value = calibrated XGBoost p(win).

Usage:
    PYTHONPATH=. .venv/bin/python -m src.models.train
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.run import load_inputs
from src.config import (
    DATA_PROCESSED_DIR, IS_END, IS_START, OOS_END, OOS_START, SEED,
)
from src.models.label_trades import label_trades_dataframe
from src.models.xgboost_primary import walk_forward_predict
from src.strategy.black_scholes import make_default_pricer


log = logging.getLogger(__name__)


def annual_fold_starts(start: str, end: str) -> list[pd.Timestamp]:
    """Year-end boundaries between [start, end]. Each fold predicts the year
    after a boundary."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = []
    yr = s.year + 1   # first fold predicts year (s.year+1)
    while pd.Timestamp(f"{yr}-01-01") <= e:
        out.append(pd.Timestamp(f"{yr}-01-01"))
        yr += 1
    return out


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

    # Step 2: annual walk-forward, predicting OOS years (2018+)
    fold_starts = annual_fold_starts(IS_START, OOS_END)
    log.info("annual fold boundaries: %s", [f.date().isoformat() for f in fold_starts])

    probs = walk_forward_predict(
        features=features, labels=labels,
        fold_starts=fold_starts,
        fold_end=pd.Timestamp(OOS_END),
        seed=SEED,
    )
    log.info("predictions: %d Mondays, mean=%.3f, median=%.3f",
             len(probs), probs.mean(), probs.median())

    out_path = DATA_PROCESSED_DIR / "ml_probabilities.parquet"
    probs.to_frame("p_calibrated").to_parquet(out_path)
    log.info("saved probabilities to %s", out_path)

    # Also save labels
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
