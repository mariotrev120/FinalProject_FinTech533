"""
XGBoost primary classifier with isotonic calibration.

Walk-forward refit:
  - Each fold trains on all data with entry_date < fold_start
  - Out-of-fold predictions on the fold's data form the predict series
  - Isotonic calibration is fit on training-fold OOF preds (helpers live in
    src/models/calibration.py), then applied to fold predictions

Pre-committed per src/config.py SEED. Hyperparameters use sensible defaults
(not Optuna-tuned in this minimal build — the README's nested-CV + 50 Optuna
trials is the institutional version we surface as a known scope cut). The
elastic-net benchmark in elastic_net_bench.py provides the simpler-model
fallback when Optuna isn't run.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.config import SEED
from src.models.calibration import apply_isotonic, fit_isotonic_on_oof


log = logging.getLogger(__name__)


def _xgb_params(seed: int = SEED) -> dict:
    return dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=seed,
        eval_metric="logloss",
    )


def fit_xgb_with_isotonic(
    X_train: pd.DataFrame, y_train: pd.Series, seed: int = SEED,
) -> tuple[XGBClassifier, IsotonicRegression]:
    """Train XGBoost, then fit isotonic calibration on OOF predictions
    obtained via TimeSeriesSplit (5 folds)."""
    if len(X_train) < 50:
        log.warning("training set has only %d rows; isotonic calibration may be unstable",
                    len(X_train))
    tscv = TimeSeriesSplit(n_splits=5)
    oof = np.full(len(X_train), np.nan)
    for tr_idx, val_idx in tscv.split(X_train):
        if len(np.unique(y_train.iloc[tr_idx])) < 2:
            continue
        m = XGBClassifier(**_xgb_params(seed))
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]

    # Fit isotonic on OOF preds (helper in src/models/calibration.py)
    iso = fit_isotonic_on_oof(oof, y_train)

    # Refit XGBoost on all data
    final = XGBClassifier(**_xgb_params(seed))
    final.fit(X_train, y_train)
    return final, iso


def predict_calibrated(
    model: XGBClassifier, iso: IsotonicRegression, X: pd.DataFrame,
) -> pd.Series:
    raw = model.predict_proba(X)[:, 1]
    cal = apply_isotonic(iso, raw)
    return pd.Series(cal, index=X.index, name="p_calibrated")


def _prior_trading_day_features(
    features: pd.DataFrame, target_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """For each target date, return the feature row at the most recent
    trading day STRICTLY BEFORE that date.

    Implements the Bug #1 fix: features used to predict for Monday-morning
    entry must come from Friday close, not from Monday close (which would
    leak Monday's intraday market activity into the decision).

    Uses the features.index as the trading-day calendar, so weekends and
    holidays are handled automatically. The result is indexed by the
    *target* dates (Mondays) with values from prior-trading-day features —
    so downstream callers can still key by target date but consume safely
    point-in-time inputs.
    """
    out_rows = []
    out_index = []
    for d in target_dates:
        prior_pos = features.index.searchsorted(d, side="left") - 1
        if prior_pos < 0:
            continue
        prior_date = features.index[prior_pos]
        out_rows.append(features.iloc[prior_pos].values)
        out_index.append(d)
    if not out_rows:
        return pd.DataFrame(columns=features.columns)
    return pd.DataFrame(out_rows, index=pd.DatetimeIndex(out_index), columns=features.columns)


def walk_forward_predict(
    features: pd.DataFrame, labels: pd.DataFrame,
    fold_starts: list[pd.Timestamp],
    fold_end: Optional[pd.Timestamp] = None,
    seed: int = SEED,
) -> pd.Series:
    """For each annual fold, train on labelled trades up to fold_start,
    predict probabilities for every Monday in the fold using PRIOR-TRADING-
    DAY feature values (Bug #1 fix).

    Both training and prediction use features at the prior trading day:
      - Training: each label's features are from the trading day before its
        entry_date. Pairs (Friday-close-features, Monday-trade-outcome).
      - Prediction: for each Monday in the fold, use features from the
        most recent trading day strictly before that Monday.

    Returns a single concatenated Series of calibrated probabilities
    indexed by Monday date covering the union of all folds.
    """
    out = []
    for i, fs in enumerate(fold_starts):
        fe = fold_starts[i + 1] if i + 1 < len(fold_starts) else (fold_end or features.index.max())

        # Training data: trades whose ENTRY AND EXIT both happened before
        # fs. Filtering by entry_date alone would leak post-fs price data
        # via labels of trades that entered just before fs but exited after.
        if "exit_date" in labels.columns:
            mask = (labels.index < fs) & (labels["exit_date"] < fs)
            train_labels = labels[mask]
        else:
            train_labels = labels.loc[:fs - pd.Timedelta(days=1)]
        if len(train_labels) < 30:
            log.warning("fold %s: only %d training labels; skipping",
                        fs.date(), len(train_labels))
            continue

        # Bug #1 fix: pair each Monday-entry label with PRIOR-TRADING-DAY
        # features (Friday close). features.reindex(train_labels.index)
        # would have used Monday-close features — a 1-trading-day leak.
        X_train = _prior_trading_day_features(features, train_labels.index)
        # Align labels to whatever training rows survived (some Mondays may
        # be missing prior-day features at the very start of the index)
        common = X_train.index.intersection(train_labels.index)
        X_train = X_train.loc[common]
        y_train = train_labels.loc[common]["win"]

        # Predict set: every Monday in [fs, fe), using prior-trading-day features
        mondays_in_fold = features.loc[fs:fe - pd.Timedelta(days=1)]
        mondays_in_fold = mondays_in_fold[mondays_in_fold.index.dayofweek == 0]
        if len(mondays_in_fold) == 0:
            continue
        X_pred = _prior_trading_day_features(features, mondays_in_fold.index)
        if len(X_pred) == 0:
            continue

        try:
            model, iso = fit_xgb_with_isotonic(X_train, y_train, seed=seed)
        except Exception as e:
            log.error("fold %s training failed: %s", fs.date(), e)
            continue
        preds = predict_calibrated(model, iso, X_pred)
        out.append(preds)
        log.info("fold %s -> %s: trained on %d (prior-day features), predicted %d Mondays",
                 fs.date(), fe.date(), len(X_train), len(preds))

    if not out:
        return pd.Series(dtype=float, name="p_calibrated")
    return pd.concat(out).sort_index()
