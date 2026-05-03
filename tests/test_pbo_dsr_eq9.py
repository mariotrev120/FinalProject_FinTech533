"""
Unit tests for:
  - DSR Eq. (9): n_eff_from_matrix (Bailey & López de Prado 2014)
  - CSCV PBO: cscv_pbo (Bailey/Borwein/López de Prado/Zhu 2015 Algorithm 2.3)

All synthetic in-memory data — no CSV loads. Validates:
  1. n_eff degrades correctly with correlation: ρ̄=0 → N̂=M; ρ̄=1 → N̂=1
  2. PBO is high (≈ 0.5+) when "trials" are pure noise (no real edge)
  3. PBO is low (< 0.3) when one trial has a real signal that holds OOS
  4. Performance-degradation slope is negative when overfit, near-zero
     when not overfit
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.deflated_sharpe import n_eff_from_matrix
from src.metrics.pbo import cscv_pbo, sharpe_per_column


# --- DSR Eq. (9) tests -----------------------------------------------------

def test_n_eff_independent_columns_returns_M():
    rng = np.random.default_rng(42)
    M = pd.DataFrame(rng.normal(0, 1, (500, 20)))
    n_hat, rho_bar = n_eff_from_matrix(M)
    # 20 truly independent columns → ρ̄ near 0, N̂ near 20
    assert rho_bar < 0.05
    assert 18 <= n_hat <= 20


def test_n_eff_perfectly_correlated_columns_returns_1():
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, 500)
    # 10 columns all identical to base → ρ̄ = 1, N̂ = 1
    M = pd.DataFrame(np.tile(base[:, None], (1, 10)))
    n_hat, rho_bar = n_eff_from_matrix(M)
    assert rho_bar > 0.99
    assert n_hat == 1


def test_n_eff_partially_correlated_columns():
    """ρ̄ ≈ 0.5 should give N̂ ≈ 0.5 + 0.5 · M."""
    rng = np.random.default_rng(11)
    base = rng.normal(0, 1, 1000)
    # Each column is 50% common + 50% idiosyncratic noise
    M = 10
    cols = []
    for _ in range(M):
        idio = rng.normal(0, 1, 1000)
        cols.append(base + idio)
    df = pd.DataFrame(np.column_stack(cols))
    n_hat, rho_bar = n_eff_from_matrix(df)
    # cov(base+idio_i, base+idio_j) = var(base) = 1
    # var(base+idio) = 1+1 = 2
    # ρ̄ ≈ 1/2 = 0.5
    assert 0.4 <= rho_bar <= 0.6
    # N̂ = 0.5 + 0.5 · 10 = 5.5 → rounds to 6 (or 5)
    assert 4 <= n_hat <= 7


def test_n_eff_single_column_returns_1():
    M = pd.DataFrame(np.zeros((100, 1)))
    n_hat, rho_bar = n_eff_from_matrix(M)
    assert n_hat == 1
    assert rho_bar == 0.0


def test_n_eff_short_sample_no_crash():
    M = pd.DataFrame(np.zeros((2, 5)))
    n_hat, _ = n_eff_from_matrix(M)
    assert n_hat == 5  # falls through to column count when T too short


# --- CSCV / PBO tests ------------------------------------------------------

def test_pbo_high_when_all_trials_are_noise():
    """When every column is pure noise (no skill), the IS-best is overfit
    to noise and OOS rank should be no better than median ⇒ PBO ≈ 0.5+."""
    rng = np.random.default_rng(42)
    T, N = 256, 30
    M = pd.DataFrame(rng.normal(0, 0.01, (T, N)))
    rep = cscv_pbo(M, n_partitions=8)
    # Pure noise → PBO is high (≈ 0.5)
    assert rep.pbo > 0.35
    # No real signal → performance-degradation slope is roughly random
    # but mean-reverting (β often near 0 or negative)
    assert rep.n_combinations == 70  # C(8,4) = 70
    assert rep.n_observations == T


def test_pbo_low_when_one_trial_has_real_signal():
    """If one column has a genuine positive-mean signal that persists in
    every subsample (not noise-correlated), it remains best OOS too →
    PBO is low."""
    rng = np.random.default_rng(13)
    T, N = 512, 20
    # Background: pure noise
    noise = rng.normal(0, 0.01, (T, N))
    # Inject genuine signal in column 0: persistent positive drift
    noise[:, 0] += 0.005  # +0.005/period drift, with vol 0.01 → SR ≈ 0.5/period
    M = pd.DataFrame(noise)
    rep = cscv_pbo(M, n_partitions=8)
    # Real signal in 1 column → IS-best (col 0) keeps winning OOS → low PBO
    assert rep.pbo < 0.3
    # Median logit should be strongly positive
    assert rep.median_logit > 0


def test_pbo_with_anti_signal_perfect_overfitting():
    """A column with mean-reverting structure across folds (perfect
    overfitting case) — IS-best columns should systematically rank
    LOW OOS because their high IS came from in-sample noise that
    reverses out-of-sample."""
    rng = np.random.default_rng(99)
    T, N = 256, 16
    # Build a matrix where row-wise the "best IS" columns flip OOS:
    # half-period 1: cols 0..N/2 high, cols N/2..N low
    # half-period 2: cols 0..N/2 low,  cols N/2..N high
    half = T // 2
    block1 = np.zeros((half, N))
    block2 = np.zeros((half, N))
    block1[:, : N // 2] = rng.normal(0.005, 0.01, (half, N // 2))
    block1[:, N // 2 :] = rng.normal(-0.005, 0.01, (half, N // 2))
    block2[:, : N // 2] = rng.normal(-0.005, 0.01, (half, N // 2))
    block2[:, N // 2 :] = rng.normal(0.005, 0.01, (half, N // 2))
    M = pd.DataFrame(np.vstack([block1, block2]))
    rep = cscv_pbo(M, n_partitions=8)
    # By construction this is a worst-case anti-signal: IS-best columns
    # are guaranteed losers OOS → PBO should be very high.
    assert rep.pbo > 0.7
    # Performance degradation: β should be strongly negative
    assert rep.perf_degradation_slope < 0


def test_pbo_reports_combination_counts_correctly():
    rng = np.random.default_rng(5)
    M = pd.DataFrame(rng.normal(0, 1, (160, 5)))
    rep = cscv_pbo(M, n_partitions=16)
    # C(16, 8) = 12870
    assert rep.n_combinations == 12870
    assert rep.n_partitions == 16


def test_pbo_rejects_odd_partition_count():
    M = pd.DataFrame(np.zeros((100, 5)))
    with pytest.raises(ValueError, match="even"):
        cscv_pbo(M, n_partitions=15)


def test_pbo_rejects_T_less_than_S():
    M = pd.DataFrame(np.zeros((10, 5)))
    with pytest.raises(ValueError, match="cannot partition"):
        cscv_pbo(M, n_partitions=16)


def test_pbo_truncates_remainder_rows_cleanly():
    """When T is not exactly divisible by S, the implementation truncates
    to the largest multiple of S. Verify no crash and report.n_observations
    reflects the original T."""
    rng = np.random.default_rng(1)
    M = pd.DataFrame(rng.normal(0, 1, (165, 8)))  # 165 rows, S=16 → 10 per block, 5 truncated
    rep = cscv_pbo(M, n_partitions=16)
    assert rep.n_observations == 165
    # PBO is still well-defined
    assert 0.0 <= rep.pbo <= 1.0


def test_sharpe_per_column_matches_manual():
    """Vectorized Sharpe matches naive per-column computation."""
    rng = np.random.default_rng(3)
    block = rng.normal(0.001, 0.02, (100, 4))
    fast = sharpe_per_column(block)
    slow = np.array([
        block[:, j].mean() / block[:, j].std(ddof=1) for j in range(4)
    ])
    np.testing.assert_allclose(fast, slow, rtol=1e-10)


def test_sharpe_per_column_handles_zero_std():
    block = np.tile(np.array([[1.0, 2.0, 3.0]]), (50, 1))
    sr = sharpe_per_column(block)
    assert np.all(np.isnan(sr))
