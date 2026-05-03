"""
Probability calibration helpers.

Isotonic regression and Platt scaling map raw classifier scores onto
observed empirical probabilities. Critical when downstream consumers
(Kelly sizing, decision thresholds, sizing curves) interpret the score
as P(win) — without calibration, XGBoost scores tend to be over-extreme
near 0 and 1 and the threshold-based gate is mis-calibrated.

Per Niculescu-Mizil & Caruana 2005 §5 (learning-curve analysis):
  - n < ~200:  Platt scaling outperforms isotonic on every learning method
                (isotonic overfits with too few calibration points).
  - n ~ 200-1000:  Mixed; Platt is more robust.
  - n >= 1000: Isotonic equal-or-better; uses its extra flexibility.

`pick_calibration_method(n_pos, n_neg)` implements the auto-select
following these thresholds. Either method's output is duck-typed by
having a `predict()` (isotonic) or `predict_proba()` (sklearn LR for
Platt) method; `apply_calibrator()` dispatches transparently.
"""
from __future__ import annotations

from typing import Literal, Union

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


CalibrationMethod = Literal["isotonic", "platt", "auto"]
Calibrator = Union[IsotonicRegression, LogisticRegression]

PLATT_THRESHOLD: int = 200   # Niculescu-Mizil §5: Platt strictly better below this


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


def fit_platt_on_oof(
    oof_predictions: np.ndarray, y_true: pd.Series,
) -> LogisticRegression:
    """Platt scaling per Niculescu-Mizil & Caruana 2005 §2.1.

    Fits a sigmoid `P(y=1|f) = 1 / (1 + exp(A·f + B))` on out-of-fold
    XGBoost scores via sklearn LogisticRegression. Niculescu-Mizil's
    original formulation uses target adjustment `y+ = (N+ + 1)/(N+ + 2)`,
    `y- = 1/(N- + 2)` to avoid overfitting; sklearn's plain LR with binary
    labels is the standard practical approximation and dominates isotonic
    when n < ~200 per the paper's learning-curve analysis (§5).
    """
    valid = ~np.isnan(oof_predictions)
    if valid.sum() < 20 or len(np.unique(y_true.iloc[np.where(valid)[0]])) < 2:
        # Degenerate fallback: identity-ish (no transformation possible)
        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
        return lr
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(
        oof_predictions[valid].reshape(-1, 1),
        y_true.iloc[np.where(valid)[0]].astype(int).to_numpy(),
    )
    return lr


def apply_platt(lr: LogisticRegression, raw_scores: np.ndarray) -> np.ndarray:
    return lr.predict_proba(raw_scores.reshape(-1, 1))[:, 1]


def pick_calibration_method(n_pos: int, n_neg: int) -> Literal["isotonic", "platt"]:
    """Auto-select per Niculescu-Mizil & Caruana 2005 §5.

    Threshold uses the smaller class count (positive class for our rare
    stress events) since isotonic's overfit pathology is driven by the
    minority-class sample size, not the total."""
    n_minority = min(n_pos, n_neg)
    return "platt" if n_minority < PLATT_THRESHOLD else "isotonic"


def fit_calibrator_on_oof(
    oof_predictions: np.ndarray, y_true: pd.Series, method: CalibrationMethod = "auto",
) -> tuple[Calibrator, Literal["isotonic", "platt"]]:
    """Fit a calibrator and return (calibrator, method_used).

    method='auto' picks isotonic vs Platt per Niculescu-Mizil §5
    threshold (PLATT_THRESHOLD positives in minority class).
    """
    valid = ~np.isnan(oof_predictions)
    y_valid = y_true.iloc[np.where(valid)[0]].astype(int)
    n_pos = int((y_valid == 1).sum())
    n_neg = int((y_valid == 0).sum())

    if method == "auto":
        chosen = pick_calibration_method(n_pos, n_neg)
    else:
        chosen = method

    if chosen == "platt":
        return fit_platt_on_oof(oof_predictions, y_true), "platt"
    return fit_isotonic_on_oof(oof_predictions, y_true), "isotonic"


def apply_calibrator(calibrator: Calibrator, raw_scores: np.ndarray) -> np.ndarray:
    """Dispatch to the right apply_* based on calibrator type."""
    if isinstance(calibrator, LogisticRegression):
        return apply_platt(calibrator, raw_scores)
    if isinstance(calibrator, IsotonicRegression):
        return apply_isotonic(calibrator, raw_scores)
    raise TypeError(f"Unknown calibrator type: {type(calibrator)}")


# === Compatibility aliases for tests/test_models.py (Robby's TDD spec) ===
fit_isotonic_calibration = fit_isotonic_on_oof
apply_calibration = apply_calibrator
