"""
Walk-forward refit harness.

Annual cadence: each fold trains on all labelled trades before the
fold-start date and predicts every Monday in the fold window.

Pulled out of `scripts/train_ml.py` per Robert's structure so the harness is
reusable for any (features, labels, fold-starts) configuration — not just
XGBoost training. Future extensions (re-fitting halt thresholds quarterly,
re-fitting calibration on new windows) should call into this module.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.config import SEED
from src.models.xgboost_primary import walk_forward_predict


log = logging.getLogger(__name__)


def annual_fold_starts(start: str, end: str) -> list[pd.Timestamp]:
    """Year-end boundaries between [start, end]. Each fold predicts the year
    after a boundary. So fold_starts=['2018-01-01', '2019-01-01', ...] yields
    one fold per OOS year, training on all labelled trades up to fold start."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = []
    yr = s.year + 1
    while pd.Timestamp(f"{yr}-01-01") <= e:
        out.append(pd.Timestamp(f"{yr}-01-01"))
        yr += 1
    return out


def annual_walk_forward_xgb(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    is_start: str,
    oos_end: str,
    seed: int = SEED,
) -> pd.Series:
    """Convenience wrapper: build annual fold starts and call the XGBoost
    walk-forward predictor.

    Returns a Series of calibrated probabilities indexed by Monday date,
    covering all years from is_start.year+1 through oos_end.year.
    """
    fold_starts = annual_fold_starts(is_start, oos_end)
    log.info("annual fold starts: %s", [f.date().isoformat() for f in fold_starts])
    return walk_forward_predict(
        features=features, labels=labels,
        fold_starts=fold_starts,
        fold_end=pd.Timestamp(oos_end),
        seed=seed,
    )
