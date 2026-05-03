"""
Head 2: regime-stress classifier.

Source-of-truth for the methodology: PRE_COMMITMENT_VRP §5, §5.1, §5.2, §5.3.

Architecture summary:
  - **Stress label source**: 5 vetted historical events from
    `scripts/stress_events.py` EVENTS constant. Each event has a peak date
    (Aug2015 China devaluation, Feb2018 Volmageddon, Q4-2018 selloff,
    Mar2020 COVID, Mar2023 banking crisis). NO programmatic thresholds.
  - **Forward-window horizon**: 63 trading days (one quarter), per Bollerslev/
    Tauchen/Zhou 2009 §3.1 Table 2 — the empirically optimal forecast
    horizon for VRP-based predictability (R²=6.82% alone, 19.74% with
    controls).
  - **Labeling logic**: For each trading day D, label TRUE if any of the 5
    peak dates has trading-day position in (pos(D), pos(D)+63] — i.e., the
    63 trading days BEFORE peak. Peak day itself is negative (warning is
    for the future, not the present).
  - **Naive baseline**: trailing 22-day mean of stress-event indicator (the
    indicator-on-peak-day series, NOT the forward label). ~1 trading month.
  - **Acceptance**: Head 2 OOS Brier ≥ 5% lower than naive baseline.
  - **Calibration**: XGBoost + isotonic via TimeSeriesSplit. Per Niculescu-
    Mizil & Caruana 2005, switch to Platt scaling if any per-fold cal-set
    has < 200 cases (reuses fit_xgb_with_isotonic from xgboost_primary.py;
    that switch is a future task).
  - **Standard errors for evaluation**: Hodrick (1992) per BTZ §3 fn21
    (TODO until C7 lands).

This module trains on REAL data — features.parquet + the labeled events.
No synthetic fixtures here; the training pipeline takes whatever is passed
in. Tests in tests/test_regime_stress.py use the real features.parquet and
real event peak dates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.config import SEED
from src.models.xgboost_primary import (
    fit_xgb_calibrated,
    predict_calibrated_auto,
)


log = logging.getLogger(__name__)


# --- Locked constants per PRE_COMMITMENT_VRP §5.1 -------------------------

DEFAULT_FORWARD_HORIZON_TRADING_DAYS: int = 63
NAIVE_BASELINE_LOOKBACK_DAYS: int = 22

# 5 vetted historical stress event peak dates, sourced from
# scripts/stress_events.py EVENTS constant (v1 audit work).
LABELED_STRESS_EVENT_PEAKS: list[pd.Timestamp] = [
    pd.Timestamp("2015-08-24"),  # Aug2015 China devaluation
    pd.Timestamp("2018-02-05"),  # Feb2018 Volmageddon
    pd.Timestamp("2018-12-24"),  # Q4-2018 selloff
    pd.Timestamp("2020-03-23"),  # Mar2020 COVID crash
    pd.Timestamp("2023-03-13"),  # Mar2023 banking crisis
]


# --- Stress-day series and forward-window label ---------------------------

def stress_today_from_peaks(
    trading_calendar: pd.DatetimeIndex,
    peak_dates: Optional[list[pd.Timestamp]] = None,
) -> pd.Series:
    """Boolean per trading day: True ONLY on a labeled peak date itself.

    For dates not in the trading calendar (e.g., a peak that falls on a
    holiday), the label is shifted forward to the next trading day via
    `searchsorted(side="left")`. All 5 of our locked peak dates are
    Mondays which are trading days, so this is a non-issue today; the
    helper handles the edge case for completeness.

    Used as the input to:
      - forward_stress_from_peaks(): the per-date forward-window label
        (the supervised training target).
      - naive_baseline_p_stress(): trailing 22-day mean baseline forecast.
    """
    if peak_dates is None:
        peak_dates = LABELED_STRESS_EVENT_PEAKS
    cal = pd.DatetimeIndex(trading_calendar)
    out = pd.Series(False, index=cal, name="stress_today")
    for pk in peak_dates:
        pk_ts = pd.Timestamp(pk)
        pos = cal.searchsorted(pk_ts, side="left")
        if pos < len(cal):
            out.iloc[pos] = True
    return out


def forward_stress_from_peaks(
    trading_calendar: pd.DatetimeIndex,
    peak_dates: Optional[list[pd.Timestamp]] = None,
    horizon_trading_days: int = DEFAULT_FORWARD_HORIZON_TRADING_DAYS,
) -> pd.Series:
    """For each trading day D, label TRUE if any peak in `peak_dates` has
    trading-day position in (pos(D), pos(D)+horizon_trading_days].

    Construction:
      For each peak P at trading-day position pp, mark trading-day
      positions [pp-H, pp-1] as positive (slice end exclusive — peak day
      itself excluded). This implements the user-confirmed semantic:
      "the 63 days BEFORE the peak are labeled positive; the peak day
      itself and days after are labeled negative — the warning is for
      the future, not the present."
    """
    if peak_dates is None:
        peak_dates = LABELED_STRESS_EVENT_PEAKS
    cal = pd.DatetimeIndex(trading_calendar)
    H = int(horizon_trading_days)
    label = pd.Series(False, index=cal, name="forward_stress")
    for pk in peak_dates:
        pk_ts = pd.Timestamp(pk)
        pp = cal.searchsorted(pk_ts, side="left")
        if pp >= len(cal):
            # Peak after end of calendar — no positive labels can be set
            continue
        start = max(0, pp - H)
        end = pp  # exclusive — peak day itself is NOT marked positive
        if end > start:
            label.iloc[start:end] = True
    return label


# --- Naive baseline + Brier ------------------------------------------------

def naive_baseline_p_stress(
    stress_today: pd.Series,
    lookback_days: int = NAIVE_BASELINE_LOOKBACK_DAYS,
) -> pd.Series:
    """Trailing-lookback mean of stress_today as the naive forecast.

    Per PRE_COMMITMENT_VRP §5.1, lookback = 22 trading days (~1 month).
    """
    return (
        stress_today.astype(float)
        .rolling(lookback_days, min_periods=1)
        .mean()
        .rename("naive_p_stress")
    )


def brier_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    """Mean squared error between binary truth and probability forecast."""
    y_t, y_p = y_true.align(y_prob, join="inner")
    if len(y_t) == 0:
        return float("nan")
    return float(((y_t.astype(float) - y_p.astype(float)) ** 2).mean())


# --- Training pipeline ----------------------------------------------------

@dataclass
class RegimeStressReport:
    """Bundle of Head 2 OOS results."""
    p_stress: pd.Series                                # calibrated OOS predictions
    forward_label: pd.Series                           # the binary target (post-hoc)
    brier_model: float                                 # OOS Brier
    brier_baseline: float                              # naive baseline OOS Brier
    brier_reduction_pct: float                         # 1 - brier_model/brier_baseline
    passed_5pct_threshold: bool                        # acceptance gate per §5
    fold_boundaries: list[pd.Timestamp] = field(default_factory=list)
    n_train_total: int = 0
    n_eval: int = 0

    def headline(self) -> str:
        return (
            f"Head 2 — Regime Stress (Brier-score acceptance):\n"
            f"  Model OOS Brier:                   {self.brier_model:.4f}\n"
            f"  Naive baseline (1mo mean) Brier:   {self.brier_baseline:.4f}\n"
            f"  Reduction (model vs baseline):     {self.brier_reduction_pct*100:+.1f}%\n"
            f"  Passes 5% threshold:               {self.passed_5pct_threshold}\n"
            f"  Folds:                             {len(self.fold_boundaries)}\n"
            f"  Final-fold training samples:       {self.n_train_total}\n"
            f"  Evaluation rows:                   {self.n_eval}\n"
        )


def train_regime_stress_walkforward(
    features: pd.DataFrame,
    forward_label: pd.Series,
    stress_today: pd.Series,
    fold_starts: list[pd.Timestamp],
    fold_end: Optional[pd.Timestamp] = None,
    horizon_trading_days: int = DEFAULT_FORWARD_HORIZON_TRADING_DAYS,
    seed: int = SEED,
) -> RegimeStressReport:
    """Walk-forward training of Head 2.

    For each fold start `fs`:
      1. Training data = rows of `features` with index strictly before
         the train_cutoff. We use train_cutoff = fs - 1.5 * horizon to
         ensure all training labels' forward windows are fully observed
         before fs (no leakage).
      2. Train XGBoost + isotonic on (X_train, y_train) where y_train is
         the forward-stress label.
      3. Predict calibrated p_stress for every date in [fs, next_fs).

    Concatenate predictions across folds → OOS p_stress series.

    Compute Brier scores on the OOS predictions vs forward_label, and
    against the naive baseline (trailing 22d mean of stress_today).

    Returns: RegimeStressReport bundling p_stress, label, both Brier
    scores, the reduction %, and the pass/fail gate.
    """
    feats = features.dropna(how="all").copy()
    forward_label = forward_label.reindex(feats.index, fill_value=False)
    stress_today = stress_today.reindex(feats.index, fill_value=False)

    fold_starts = sorted(pd.Timestamp(fs) for fs in fold_starts)
    end_date = pd.Timestamp(fold_end) if fold_end is not None else feats.index.max()

    out_preds: list[pd.Series] = []
    last_train_size = 0

    for i, fs in enumerate(fold_starts):
        fe = fold_starts[i + 1] if i + 1 < len(fold_starts) else end_date

        # Training cutoff: strictly before fs by 1.5x horizon to ensure
        # training labels' forward windows are fully observed pre-fold.
        train_cutoff = fs - pd.Timedelta(days=int(horizon_trading_days * 1.5))
        train_idx = feats.index[feats.index < train_cutoff]
        if len(train_idx) < 100:
            log.warning(
                "regime_stress fold %s: only %d training rows; skipping",
                fs.date(), len(train_idx),
            )
            continue

        X_train = feats.loc[train_idx]
        y_train = forward_label.loc[train_idx].astype(int)

        # Need both classes for isotonic to be meaningful.
        if y_train.nunique() < 2:
            log.warning(
                "regime_stress fold %s: training labels are single-class; skipping",
                fs.date(),
            )
            continue
        last_train_size = len(X_train)

        try:
            model, calibrator, cal_method = fit_xgb_calibrated(
                X_train, y_train, seed=seed, method="auto",
                use_scale_pos_weight=True,
            )
        except Exception as e:
            log.error("regime_stress fold %s training failed: %s", fs.date(), e)
            continue
        log.info(
            "regime_stress fold %s: trained on %d (%d pos), calibration=%s",
            fs.date(), len(X_train), int((y_train == 1).sum()), cal_method,
        )

        pred_idx = feats.index[(feats.index >= fs) & (feats.index < fe)]
        if len(pred_idx) == 0:
            continue
        X_pred = feats.loc[pred_idx]
        preds = predict_calibrated_auto(model, calibrator, X_pred)
        out_preds.append(preds)

    if not out_preds:
        return RegimeStressReport(
            p_stress=pd.Series(dtype=float, name="p_stress"),
            forward_label=forward_label,
            brier_model=float("nan"),
            brier_baseline=float("nan"),
            brier_reduction_pct=float("nan"),
            passed_5pct_threshold=False,
            fold_boundaries=fold_starts,
            n_train_total=last_train_size,
            n_eval=0,
        )

    p_stress = pd.concat(out_preds).sort_index().rename("p_stress")

    # Truncate the comparison to dates where the forward window is fully
    # observable (else the label is biased toward "no stress in future"
    # because the forward window extends past end_date).
    cutoff = end_date - pd.Timedelta(days=int(horizon_trading_days * 1.5))
    eval_idx = p_stress.index[p_stress.index < cutoff]
    if len(eval_idx) == 0:
        eval_idx = p_stress.index

    y_oos = forward_label.loc[eval_idx].astype(float)
    p_model_oos = p_stress.loc[eval_idx]
    p_naive_oos = naive_baseline_p_stress(stress_today).loc[eval_idx]

    bm = brier_score(y_oos, p_model_oos)
    bb = brier_score(y_oos, p_naive_oos)
    reduction = (
        float("nan")
        if (np.isnan(bm) or np.isnan(bb) or bb == 0)
        else 1.0 - bm / bb
    )
    passed = (not np.isnan(reduction)) and (reduction >= 0.05)

    return RegimeStressReport(
        p_stress=p_stress,
        forward_label=forward_label,
        brier_model=bm,
        brier_baseline=bb,
        brier_reduction_pct=reduction,
        passed_5pct_threshold=passed,
        fold_boundaries=fold_starts,
        n_train_total=last_train_size,
        n_eval=len(eval_idx),
    )
