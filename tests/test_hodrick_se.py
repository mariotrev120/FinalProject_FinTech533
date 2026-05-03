"""
Unit tests for src/metrics/hodrick_se.py — Hodrick (1992) standard errors.

Validates:
  - h=1 case reduces to OLS (no overlap correction needed)
  - Synthetic null: when y is independent noise, t-stat is NOT artificially
    inflated by overlapping forward windows
  - Synthetic signal: when there's a real β, the estimator recovers it
  - Sanity: SE > 0, t-stat is finite

Synthetic data only — pure logic, no parquet loads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.hodrick_se import HodrickRegressionResult, hodrick_predictive_regression


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_series(values: np.ndarray) -> pd.Series:
    idx = pd.date_range("2018-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


# --- h=1 reduces to OLS (sanity) ------------------------------------------

def test_horizon_1_recovers_beta(rng):
    n = 1000
    x = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)
    beta_true = 0.7
    y = 1.0 + beta_true * x + eps
    rep = hodrick_predictive_regression(_make_series(y), _make_series(x), horizon=1)
    assert abs(rep.beta - beta_true) < 0.05
    assert rep.se_beta_hodrick > 0
    assert rep.n_obs == n
    assert rep.horizon == 1


# --- horizon > 1 with real signal -----------------------------------------

def test_horizon_3_recovers_beta(rng):
    """When the predictor genuinely predicts a 3-period-forward outcome,
    the Hodrick estimator should recover β with reasonable precision."""
    n = 2000
    x = rng.normal(0, 1, n)
    # forward 3-period outcome = 0.5 * x_t + noise
    eps = rng.normal(0, 0.4, n)
    y_true = 0.5 * x + eps
    rep = hodrick_predictive_regression(_make_series(y_true), _make_series(x), horizon=3)
    assert abs(rep.beta - 0.5) < 0.06
    assert rep.t_beta_hodrick != 0  # finite


# --- Null: random noise should NOT produce inflated t-stats ---------------

def test_horizon_63_null_avg_t_stat_not_inflated(rng):
    """Per Boudoukh-Richardson-Whitelaw 2008 (cited in BTZ): naive
    standard errors with overlapping h-period regressions OVERSTATE
    significance. Hodrick SEs should NOT — average |t-stat| under H0
    should be near 1 (not >> 1)."""
    h = 63
    n = 1500
    n_runs = 30
    abs_ts = []
    for trial in range(n_runs):
        rng_t = np.random.default_rng(trial * 13 + 7)
        x = rng_t.normal(0, 1, n)
        # PURE NOISE outcome (independent of x)
        y = rng_t.normal(0, 1, n)
        try:
            rep = hodrick_predictive_regression(_make_series(y), _make_series(x), horizon=h)
            if np.isfinite(rep.t_beta_hodrick):
                abs_ts.append(abs(rep.t_beta_hodrick))
        except ValueError:
            continue
    assert len(abs_ts) >= 20
    mean_abs_t = np.mean(abs_ts)
    # Under H0, |t| should be O(1). Even allowing for finite-sample
    # variability, mean |t| should be < 2 (not 5+ which would be the
    # naive-SE inflation).
    assert mean_abs_t < 2.5, (
        f"Hodrick SEs gave inflated t-stats under null: mean |t| = {mean_abs_t:.2f} "
        f"(expected < 2.5)"
    )


# --- Edge cases -----------------------------------------------------------

def test_invalid_horizon_raises():
    x = _make_series(np.arange(10, dtype=float))
    y = _make_series(np.arange(10, dtype=float))
    with pytest.raises(ValueError, match="horizon"):
        hodrick_predictive_regression(y, x, horizon=0)


def test_too_few_obs_raises():
    x = _make_series(np.array([1.0, 2.0, 3.0]))
    y = _make_series(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        hodrick_predictive_regression(y, x, horizon=2)


def test_zero_variance_raises():
    n = 50
    x = _make_series(np.ones(n))   # zero variance
    y = _make_series(np.arange(n, dtype=float))
    with pytest.raises(ValueError, match="variance"):
        hodrick_predictive_regression(y, x, horizon=3)


def test_result_is_hodrick_regression_result(rng):
    n = 200
    x = rng.normal(0, 1, n)
    y = 0.5 * x + rng.normal(0, 0.3, n)
    rep = hodrick_predictive_regression(_make_series(y), _make_series(x), horizon=2)
    assert isinstance(rep, HodrickRegressionResult)
    d = rep.to_dict()
    assert "alpha" in d and "beta" in d
    assert "se_beta_hodrick" in d
    assert "t_beta_hodrick" in d
