"""
Unit tests for src/models/regime_stress.py — Head 2 regime-stress classifier.

Real data only. Reads:
  - data/processed/features.parquet (~150KB, ~3500 rows × 15 cols)
  - 5 LABELED_STRESS_EVENT_PEAKS hardcoded in src/models/regime_stress.py
    (sourced from scripts/stress_events.py EVENTS — vetted in v1 audit)

No synthetic features, no synthetic labels. Tests verify the labeling
logic, the naive baseline, the Brier-score function, and the locked
constants. The walk-forward training itself is exercised in the
integration script (scripts/run_head2_regime_stress.py), not the test
suite — that's where actual Brier numbers get reported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.regime_stress import (
    DEFAULT_FORWARD_HORIZON_TRADING_DAYS,
    LABELED_STRESS_EVENT_PEAKS,
    NAIVE_BASELINE_LOOKBACK_DAYS,
    brier_score,
    forward_stress_from_peaks,
    naive_baseline_p_stress,
    stress_today_from_peaks,
)


FEATURES_PATH = Path("data/processed/features.parquet")
PEAKS_DOC = pd.DatetimeIndex(LABELED_STRESS_EVENT_PEAKS)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    """Load features.parquet once per test module."""
    if not FEATURES_PATH.exists():
        pytest.skip(f"{FEATURES_PATH} not present — skipping real-data tests")
    return pd.read_parquet(FEATURES_PATH)


@pytest.fixture(scope="module")
def trading_calendar(features: pd.DataFrame) -> pd.DatetimeIndex:
    return features.index


# --- Locked-constants verification ---------------------------------------

def test_locked_constants_per_pre_commitment_5_1():
    """PRE_COMMITMENT_VRP §5.1 locks these values. They must not drift."""
    assert DEFAULT_FORWARD_HORIZON_TRADING_DAYS == 63
    assert NAIVE_BASELINE_LOOKBACK_DAYS == 22


def test_labeled_event_peaks_match_v1_audit_source():
    """5 peak dates from scripts/stress_events.py EVENTS constant must
    match the regime_stress module's locked list exactly."""
    expected = [
        pd.Timestamp("2015-08-24"),
        pd.Timestamp("2018-02-05"),
        pd.Timestamp("2018-12-24"),
        pd.Timestamp("2020-03-23"),
        pd.Timestamp("2023-03-13"),
    ]
    assert LABELED_STRESS_EVENT_PEAKS == expected


def test_all_peak_dates_are_within_features_calendar(features: pd.DataFrame):
    """Each labeled peak must fall within the trading calendar
    of features.parquet (else forward_stress_from_peaks silently
    produces all-False and we'd never notice)."""
    for pk in PEAKS_DOC:
        assert features.index.min() <= pk <= features.index.max(), (
            f"Peak {pk.date()} not in features range "
            f"[{features.index.min().date()}, {features.index.max().date()}]"
        )


def _effective_peak(pk: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the trading day on which `pk` lands after holiday-shift.
    Mirrors `stress_today_from_peaks` searchsorted(side='left') logic.

    Some original peaks may not be directly in `calendar` if features.parquet
    has gaps (e.g., 2020-03-23 missing → shifts to 2020-03-24)."""
    pos = calendar.searchsorted(pd.Timestamp(pk), side="left")
    if pos >= len(calendar):
        raise ValueError(f"Peak {pk} after end of calendar")
    return calendar[pos]


def test_peak_dates_within_at_most_3_trading_days_of_original(trading_calendar: pd.DatetimeIndex):
    """Every effective (post-shift) peak day must be at most a few
    trading days after the original locked peak. Detects accidental
    huge-gap shifts that would silently change the labeling semantics."""
    for pk in PEAKS_DOC:
        eff = _effective_peak(pk, trading_calendar)
        # eff is the next trading day on/after pk; it cannot be before pk
        assert eff >= pk
        # Cap the gap: original peaks were chosen on/near actual trading
        # days. Allow up to 5 calendar-day shift (handles a 3-day weekend).
        assert (eff - pk).days <= 5, (
            f"Peak {pk.date()} effective shift to {eff.date()} "
            f"is more than 5 calendar days — likely upstream data gap"
        )


# --- stress_today_from_peaks ---------------------------------------------

def test_stress_today_marks_exactly_5_days(trading_calendar: pd.DatetimeIndex):
    s = stress_today_from_peaks(trading_calendar)
    assert s.sum() == 5


def test_stress_today_marks_effective_peak_position(trading_calendar: pd.DatetimeIndex):
    """The True positions must be exactly the searchsorted-shifted
    trading-day positions of the 5 original peaks."""
    s = stress_today_from_peaks(trading_calendar)
    for pk in PEAKS_DOC:
        eff = _effective_peak(pk, trading_calendar)
        assert s.loc[eff] == True   # noqa: E712


def test_stress_today_is_all_false_outside_effective_peaks(trading_calendar: pd.DatetimeIndex):
    s = stress_today_from_peaks(trading_calendar)
    eff_peaks = pd.DatetimeIndex([_effective_peak(pk, trading_calendar) for pk in PEAKS_DOC])
    non_peak = s.drop(eff_peaks)
    assert non_peak.sum() == 0


# --- forward_stress_from_peaks -------------------------------------------

def test_forward_label_peak_day_itself_is_negative(trading_calendar: pd.DatetimeIndex):
    """Per user-confirmed labeling logic: peak day at position pp is at
    the END of the slice [pp-63, pp] (exclusive). Peak day → False."""
    fwd = forward_stress_from_peaks(trading_calendar)
    for pk in PEAKS_DOC:
        eff = _effective_peak(pk, trading_calendar)
        assert fwd.loc[eff] == False, (   # noqa: E712
            f"Peak day {eff.date()} (orig {pk.date()}) should be negative "
            f"(warning is for the future, not the present)"
        )


def test_forward_label_day_before_peak_is_positive(trading_calendar: pd.DatetimeIndex):
    fwd = forward_stress_from_peaks(trading_calendar)
    for pk in PEAKS_DOC:
        pp = trading_calendar.searchsorted(pk, side="left")
        assert pp > 0, f"Peak {pk.date()} at start of calendar"
        day_before = trading_calendar[pp - 1]
        assert fwd.loc[day_before] == True, (   # noqa: E712
            f"Trading day before peak {pk.date()} ({day_before.date()}) "
            f"should be positive (forward window includes peak)"
        )


def test_forward_label_63_days_before_peak_is_positive(trading_calendar: pd.DatetimeIndex):
    """Position pp-63 is the EARLIEST positive label for peak at pp.
    Slice [pp-63 : pp] is inclusive of pp-63, exclusive of pp."""
    fwd = forward_stress_from_peaks(trading_calendar)
    for pk in PEAKS_DOC:
        pp = trading_calendar.searchsorted(pk, side="left")
        if pp < DEFAULT_FORWARD_HORIZON_TRADING_DAYS:
            continue  # peak too early in calendar
        day_63_before = trading_calendar[pp - DEFAULT_FORWARD_HORIZON_TRADING_DAYS]
        assert fwd.loc[day_63_before] == True, (   # noqa: E712
            f"63 trading days before peak {pk.date()} should be positive"
        )


def test_forward_label_64_days_before_peak_is_negative(trading_calendar: pd.DatetimeIndex):
    """Position pp-64 is OUTSIDE the forward window of peak pp; its
    forward window (pp-64+1 .. pp-64+63] = (pp-63 .. pp-1] which does
    NOT include pp. Should be False unless ANOTHER peak falls in
    (pp-64+1 .. pp-64+63]."""
    fwd = forward_stress_from_peaks(trading_calendar)
    for pk in PEAKS_DOC:
        pp = trading_calendar.searchsorted(pk, side="left")
        if pp < DEFAULT_FORWARD_HORIZON_TRADING_DAYS + 1:
            continue
        day_64_before = trading_calendar[pp - DEFAULT_FORWARD_HORIZON_TRADING_DAYS - 1]
        # Compute whether any OTHER peak's window covers day_64_before
        date_idx = day_64_before
        date_pos = pp - DEFAULT_FORWARD_HORIZON_TRADING_DAYS - 1
        covered_by_other = any(
            (other_pp := trading_calendar.searchsorted(other_pk, side="left")) - DEFAULT_FORWARD_HORIZON_TRADING_DAYS
            <= date_pos < other_pp
            for other_pk in PEAKS_DOC
            if other_pk != pk
        )
        if covered_by_other:
            continue   # another peak's window includes this date — skip
        assert fwd.loc[day_64_before] == False, (   # noqa: E712
            f"64 trading days before peak {pk.date()} ({day_64_before.date()}) "
            f"should be negative (outside its forward window AND outside "
            f"every other peak's window)"
        )


def test_forward_label_total_count_bounded(trading_calendar: pd.DatetimeIndex):
    """Total positive labels ≤ 5 × 63 = 315 (less if any windows overlap).
    Empirically the 5 peaks are separated by 18+ months minimum, so no
    overlap; expect exactly 315 minus any windows that get truncated at
    the start of the calendar."""
    fwd = forward_stress_from_peaks(trading_calendar)
    n_pos = fwd.sum()
    assert n_pos <= 5 * DEFAULT_FORWARD_HORIZON_TRADING_DAYS
    # Should be very close to 315 — only the first peak's window may be
    # truncated if it lands within 63 days of the calendar start.
    assert n_pos >= 5 * DEFAULT_FORWARD_HORIZON_TRADING_DAYS - 63


# --- naive baseline + Brier ----------------------------------------------

def test_naive_baseline_at_peak_window_is_nonzero(trading_calendar: pd.DatetimeIndex):
    """In a 22-day window containing an effective peak day, the
    trailing-22d mean of stress_today must be > 0 (specifically
    1/22 ≈ 0.0455 if peak just happened)."""
    s = stress_today_from_peaks(trading_calendar)
    base = naive_baseline_p_stress(s, lookback_days=NAIVE_BASELINE_LOOKBACK_DAYS)
    for pk in PEAKS_DOC:
        eff = _effective_peak(pk, trading_calendar)
        # Day of effective peak: trailing-22d mean includes the peak itself
        assert base.loc[eff] > 0
        # Day 22+ AFTER peak: peak has rolled out → mean = 0 unless
        # another peak is in window
        pp = trading_calendar.searchsorted(eff, side="left")
        far_idx = pp + NAIVE_BASELINE_LOOKBACK_DAYS + 5
        if far_idx >= len(trading_calendar):
            continue
        far_date = trading_calendar[far_idx]
        any_peak_in_window = any(
            0 <= (far_idx - trading_calendar.searchsorted(_effective_peak(other_pk, trading_calendar), side="left"))
            < NAIVE_BASELINE_LOOKBACK_DAYS
            for other_pk in PEAKS_DOC
        )
        if any_peak_in_window:
            continue
        assert base.loc[far_date] == 0.0


def test_brier_score_perfect_forecast_is_zero(trading_calendar: pd.DatetimeIndex):
    fwd = forward_stress_from_peaks(trading_calendar)
    p_perfect = fwd.astype(float)
    assert brier_score(fwd, p_perfect) == 0.0


def test_brier_score_zero_forecast_equals_positive_rate(trading_calendar: pd.DatetimeIndex):
    """Constant-zero forecast has Brier = positive base rate of label.
    For our forward_label with ~315 positives in ~3500 days, base ≈ 0.09.
    Brier of zero forecast = mean((y - 0)²) = mean(y) since y in {0,1}."""
    fwd = forward_stress_from_peaks(trading_calendar)
    p_zero = pd.Series(0.0, index=fwd.index)
    bs = brier_score(fwd, p_zero)
    # Should equal the positive base rate
    base_rate = float(fwd.astype(float).mean())
    assert abs(bs - base_rate) < 1e-9
