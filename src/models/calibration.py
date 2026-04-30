"""
Probability calibration helpers.

Isotonic regression maps raw classifier scores onto observed empirical
probabilities. Critical when downstream consumers (Kelly sizing, decision
thresholds) interpret the score as P(win) — without calibration, XGBoost
scores tend to be over-extreme near 0 and 1 and the threshold-based gate
is mis-calibrated.

Pulled out of `xgboost_primary.py` per Robert's structure (one module per
concern, easier per-function tests). The ElasticNet benchmark uses Platt
scaling instead; that's wired up directly in `elastic_net_bench.py` via
sklearn's CalibratedClassifierCV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def fit_isotonic_on_oof(
    oof_predictions: np.ndarray, y_true: pd.Series,
) -> IsotonicRegression:
    """Fit an isotonic regression mapping raw scores to empirical win rates.

    oof_predictions: out-of-fold predicted probabilities (NaN where the fold
        wasn't trained — those rows are skipped).
    y_true: observed binary labels aligned with oof_predictions.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    valid = ~np.isnan(oof_predictions)
    if valid.sum() < 20 or len(np.unique(y_true.iloc[np.where(valid)[0]])) < 2:
        # Degenerate fallback: identity calibration
        iso.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    else:
        iso.fit(oof_predictions[valid], y_true.iloc[np.where(valid)[0]])
    return iso


def apply_isotonic(iso: IsotonicRegression, raw_scores: np.ndarray) -> np.ndarray:
    return iso.predict(raw_scores)
