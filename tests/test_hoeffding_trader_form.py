"""
Tests for the Egger & Vestal (2025) trader-form Hoeffding helpers added
to src/metrics/hoeffding.py.

The pre-existing helpers (hoeffding_lower_bound, rolling_winrate,
hoeffding_breach_dates) are covered indirectly by the audit scripts;
this file tests the new trader-form additions specifically.

Synthetic data only.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.metrics.hoeffding import (
    regime_change_probability_bound,
    regime_change_signal,
    rolling_regime_signal,
)


# --- regime_change_probability_bound -------------------------------------

def test_no_underperformance_returns_one():
    """If observed >= committed μ, no underperformance → bound = 1.0."""
    assert regime_change_probability_bound(0.65, 0.65, 60) == 1.0
    assert regime_change_probability_bound(0.65, 0.70, 60) == 1.0


def test_zero_n_returns_one():
    assert regime_change_probability_bound(0.65, 0.40, 0) == 1.0


def test_bound_decreases_with_n():
    """Same underperformance, more trades → tighter bound (lower p)."""
    p_30 = regime_change_probability_bound(0.65, 0.50, 30)
    p_60 = regime_change_probability_bound(0.65, 0.50, 60)
    p_120 = regime_change_probability_bound(0.65, 0.50, 120)
    assert p_30 > p_60 > p_120
    # Sanity floor: still positive
    assert p_120 > 0


def test_bound_decreases_with_underperformance():
    """Same N, larger underperformance → tighter bound."""
    p_small = regime_change_probability_bound(0.65, 0.60, 60)
    p_med = regime_change_probability_bound(0.65, 0.50, 60)
    p_big = regime_change_probability_bound(0.65, 0.40, 60)
    assert p_small > p_med > p_big


def test_bound_matches_formula():
    """Spot-check: t=0.10, N=60 → e^(-2*0.01*60) = e^(-1.2) ≈ 0.301."""
    bound = regime_change_probability_bound(0.65, 0.55, 60)
    assert abs(bound - math.exp(-2 * 0.10**2 * 60)) < 1e-9


# --- regime_change_signal threshold semantics ----------------------------

def test_signal_thresholds():
    # Bound = 1.0 (no underperformance) → green
    assert regime_change_signal(0.65, 0.65, 60) == "green"
    # Compute t such that bound is exactly 0.50, 0.25, 0.10
    # bound = exp(-2 t² N)  ⇒  t = sqrt(-ln(bound) / (2N))
    N = 60
    t_50 = math.sqrt(-math.log(0.50) / (2 * N))   # bound = 0.50
    t_25 = math.sqrt(-math.log(0.25) / (2 * N))
    t_10 = math.sqrt(-math.log(0.10) / (2 * N))
    mu = 0.65
    # Just below 50% threshold
    assert regime_change_signal(mu, mu - t_50 - 1e-4, N) == "yellow"
    # Just below 25%
    assert regime_change_signal(mu, mu - t_25 - 1e-4, N) == "red"
    # Just below 10%
    assert regime_change_signal(mu, mu - t_10 - 1e-4, N) == "critical"


# --- rolling_regime_signal -----------------------------------------------

def test_rolling_signal_warmup_is_nan():
    idx = pd.date_range("2018-01-01", periods=100, freq="B")
    outcomes = pd.Series(np.ones(100, dtype=int), index=idx)
    df = rolling_regime_signal(outcomes, mu_committed=0.65, window=60)
    # First 59 rows are NaN (insufficient window)
    assert df["rolling_winrate"].iloc[:59].isna().all()
    # Row 60 onwards has values
    assert not df["rolling_winrate"].iloc[59:].isna().any()


def test_rolling_signal_perfect_winrate_is_green():
    idx = pd.date_range("2018-01-01", periods=200, freq="B")
    outcomes = pd.Series(np.ones(200, dtype=int), index=idx)
    df = rolling_regime_signal(outcomes, mu_committed=0.65, window=60)
    # Mature window: rolling = 1.0, X̄ > μ → no underperformance → green
    assert df["signal"].iloc[100] == "green"
    assert df["hoeffding_bound"].iloc[100] == 1.0


def test_rolling_signal_zero_winrate_is_critical():
    idx = pd.date_range("2018-01-01", periods=200, freq="B")
    outcomes = pd.Series(np.zeros(200, dtype=int), index=idx)
    df = rolling_regime_signal(outcomes, mu_committed=0.65, window=60)
    # Rolling winrate = 0, μ = 0.65, t = 0.65, N = 60
    # bound = e^(-2 * 0.65² * 60) ≈ e^(-50.7) ≈ 1e-22 → "critical"
    assert df["signal"].iloc[100] == "critical"


def test_rolling_signal_dataframe_columns():
    idx = pd.date_range("2018-01-01", periods=80, freq="B")
    outcomes = pd.Series(np.array([1, 0] * 40), index=idx)
    df = rolling_regime_signal(outcomes, mu_committed=0.55, window=20)
    assert list(df.columns) == ["rolling_winrate", "hoeffding_bound", "signal"]
    assert len(df) == 80
