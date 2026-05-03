"""
Deflated Sharpe Ratio (DSR) per Bailey & López de Prado (2014).

DSR adjusts observed Sharpe for three sources of inflation:
  1. Selection bias from multiple-trials testing (sensitivity grid, mode
     comparisons, hyperparameter search). With N trials, the expected
     maximum Sharpe across trials is positive even under H0 of zero skill.
  2. Non-normality of returns (skew, kurtosis affect Sharpe variance).
  3. Sample length (T enters via sqrt(T-1)).

Equation references match the paper:

  Eq. (1): Expected max Sharpe under N independent trials, H0: SR=0
    E[max{SR̂_n}] ≈ √V · [(1-γ)·Z⁻¹(1-1/N) + γ·Z⁻¹(1-1/(N·e))]

  Eq. (2): Deflated Sharpe Ratio
    DSR = Z[ (SR̂ - SR̂_0)·√(T-1)
              / √(1 - γ̂_3·SR̂ + ((γ̂_4-1)/4)·SR̂²) ]

Where:
  γ ≈ 0.5772 (Euler-Mascheroni constant)
  Z = standard normal CDF; Z⁻¹ = inverse
  V = variance of {SR̂_n} across trials
  N = number of independent trials
  SR̂ = observed Sharpe (PER-PERIOD, not annualized) of selected strategy
  T = sample length (number of return observations)
  γ̂_3 = sample skewness
  γ̂_4 = sample raw kurtosis (4th-moment ratio; = 3 for Normal)

Reference:
  Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
  Journal of Portfolio Management, 40(5), 94-107.
  SSRN: https://ssrn.com/abstract=2460551

Reference implementation crosscheck: pypbo (https://github.com/esvhd/pypbo).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import DATA_PROCESSED_DIR


EULER_MASCHERONI: float = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sharpe_var_per_period: float) -> float:
    """Expected maximum Sharpe across N independent trials under H0: true SR = 0.

    Equation (1) from Bailey & López de Prado (2014). Returns a per-period
    Sharpe (multiply by √periods_per_year for annualized).

    Inputs:
      n_trials: N — number of trials (e.g. parameter combinations tested)
      sharpe_var_per_period: V[{SR̂_n}] in per-period units, NOT annualized.
        If you have annualized trial Sharpes, divide their sample variance
        by periods_per_year before passing here.
    """
    if n_trials <= 1 or sharpe_var_per_period <= 0:
        return 0.0
    g = EULER_MASCHERONI
    a = (1.0 - g) * norm.ppf(1.0 - 1.0 / n_trials)
    b = g * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_var_per_period) * (a + b)


def probabilistic_sharpe_ratio(
    sharpe_obs_per_period: float,
    sharpe_benchmark_per_period: float,
    n_returns: int,
    returns_skew: float,
    returns_kurt_raw: float,
) -> float:
    """Probability that true Sharpe > benchmark, accounting for sample
    length and higher moments (Bailey & López de Prado 2012).

    PSR = Z[ (SR̂ - SR̂_threshold) · √(T-1)
              / √(1 - γ̂_3·SR̂ + ((γ̂_4-1)/4)·SR̂²) ]

    Returns a probability in [0, 1].

    NOTE: returns_kurt_raw is the RAW 4th-moment ratio (= 3 for Normal),
    NOT excess kurtosis. Pandas .kurt() returns excess; convert via
    raw = excess + 3.
    """
    if n_returns < 2:
        return float("nan")
    var_term = (
        1.0
        - returns_skew * sharpe_obs_per_period
        + (returns_kurt_raw - 1.0) / 4.0 * sharpe_obs_per_period**2
    )
    if var_term <= 0:
        return float("nan")
    z = (
        (sharpe_obs_per_period - sharpe_benchmark_per_period)
        * math.sqrt(n_returns - 1)
        / math.sqrt(var_term)
    )
    return float(norm.cdf(z))


@dataclass
class DSRReport:
    """Per-strategy DSR + supporting moments."""
    sharpe_obs_per_period: float
    sharpe_obs_annualized: float
    sharpe_benchmark_per_period: float
    sharpe_benchmark_annualized: float
    n_trials: int
    sharpe_var_across_trials_per_period: float
    sample_length: int
    skew: float
    kurt_raw: float
    psr: float
    is_significant_at_95: bool

    def to_dict(self) -> dict:
        return {
            "sharpe_obs_per_period": self.sharpe_obs_per_period,
            "sharpe_obs_annualized": self.sharpe_obs_annualized,
            "sharpe_benchmark_per_period": self.sharpe_benchmark_per_period,
            "sharpe_benchmark_annualized": self.sharpe_benchmark_annualized,
            "n_trials": self.n_trials,
            "sharpe_var_across_trials_per_period": self.sharpe_var_across_trials_per_period,
            "sample_length": self.sample_length,
            "skew": self.skew,
            "kurt_raw": self.kurt_raw,
            "deflated_sharpe_psr": self.psr,
            "is_significant_at_95": self.is_significant_at_95,
        }

    def headline(self) -> str:
        return (
            f"Observed Sharpe (annualized):        {self.sharpe_obs_annualized:>+8.4f}\n"
            f"Benchmark Sharpe (annualized):       {self.sharpe_benchmark_annualized:>+8.4f}\n"
            f"  (expected max under N={self.n_trials} trials, V_pp={self.sharpe_var_across_trials_per_period:.6g})\n"
            f"Sample moments: skew={self.skew:+.3f}  kurt_raw={self.kurt_raw:.3f}  T={self.sample_length}\n"
            f"Deflated Sharpe (PSR):               {self.psr:>+8.4f}\n"
            f"Significant at 95% (PSR > 0.95):     {self.is_significant_at_95}\n"
        )


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    sharpe_var_across_trials_per_period: float,
    periods_per_year: int = 252,
) -> DSRReport:
    """Compute DSR for the given return series.

    Inputs:
      returns: per-period (e.g. daily) return series
      n_trials: N from sensitivity grid / mode comparisons
      sharpe_var_across_trials_per_period: V in per-period units (see
        trial_sharpe_variance_from_grid for converting from annualized)
      periods_per_year: conversion factor for annualization display only
    """
    r = returns.dropna()
    T = len(r)
    if T < 5:
        return DSRReport(
            sharpe_obs_per_period=float("nan"),
            sharpe_obs_annualized=float("nan"),
            sharpe_benchmark_per_period=float("nan"),
            sharpe_benchmark_annualized=float("nan"),
            n_trials=n_trials,
            sharpe_var_across_trials_per_period=sharpe_var_across_trials_per_period,
            sample_length=T, skew=float("nan"), kurt_raw=float("nan"),
            psr=float("nan"), is_significant_at_95=False,
        )
    mean_r = float(r.mean())
    std_r = float(r.std(ddof=1))
    if std_r == 0:
        sr_pp = 0.0
    else:
        sr_pp = mean_r / std_r
    sr_ann = sr_pp * math.sqrt(periods_per_year)

    sr_bench_pp = expected_max_sharpe(n_trials, sharpe_var_across_trials_per_period)
    sr_bench_ann = sr_bench_pp * math.sqrt(periods_per_year)

    skew = float(r.skew())
    # pandas .kurt() returns EXCESS kurtosis; convert to raw 4th-moment ratio
    kurt_raw = float(r.kurt() + 3.0)

    psr = probabilistic_sharpe_ratio(sr_pp, sr_bench_pp, T, skew, kurt_raw)

    return DSRReport(
        sharpe_obs_per_period=sr_pp,
        sharpe_obs_annualized=sr_ann,
        sharpe_benchmark_per_period=sr_bench_pp,
        sharpe_benchmark_annualized=sr_bench_ann,
        n_trials=n_trials,
        sharpe_var_across_trials_per_period=sharpe_var_across_trials_per_period,
        sample_length=T,
        skew=skew,
        kurt_raw=kurt_raw,
        psr=psr,
        is_significant_at_95=psr > 0.95,
    )


def n_eff_from_matrix(M: pd.DataFrame) -> tuple[int, float]:
    """Implied number of independent trials N̂ from a (T × M) returns matrix
    of M correlated trials, via Eq. (9) of Bailey & López de Prado 2014:

        N̂ = ρ̄ + (1 − ρ̄) · M

    where ρ̄ is the equal-weighted average pairwise correlation between
    trial-return columns. Eq. (8) of the paper:

        ρ̄ = (Σ Σ ρ_{i,j} − M) / (M · (M − 1))

    Inputs:
      M: pandas DataFrame, shape (T, M_trials). Each column is one trial's
         per-period return series. Rows are aligned observations.

    Returns:
      (N_hat, rho_bar) where N_hat is rounded to int (≥ 1).

    Why this matters: a sensitivity grid produces highly correlated trials
    (adjacent delta values, adjacent DTE windows), so the raw column count
    M overstates the number of independent trials. Eq. (9) deflates M
    toward 1 as ρ̄ → 1 and toward M as ρ̄ → 0. The headline DSR should
    use N̂, not raw M; otherwise SR̂_0 (the noise-floor Sharpe under H0)
    is overstated and DSR is needlessly conservative.

    Per the paper's caveats: for short samples T < M·(M−1)/2 the corr
    matrix is ill-conditioned. We guard with a minimum of 2 columns and
    no further dimension reduction (kept simple — caller should ensure T
    is long relative to M).
    """
    if M.shape[1] < 2:
        return max(M.shape[1], 1), 0.0
    if M.shape[0] < 3:
        return M.shape[1], 0.0
    corr = M.corr().to_numpy(copy=False)
    M_count = corr.shape[0]
    # Sum of off-diagonal pairwise correlations (counted once each direction)
    off_diag_sum = float(corr.sum() - np.trace(corr))
    rho_bar = off_diag_sum / (M_count * (M_count - 1))
    # Clamp ρ̄ to [-1/(M-1), 1] per the paper; for our use case, clip to [0, 1]
    # because negative-correlated trials don't make sense for a sensitivity grid
    # over a single strategy.
    rho_bar = max(0.0, min(rho_bar, 1.0))
    n_hat = rho_bar + (1.0 - rho_bar) * M_count
    return max(int(round(n_hat)), 1), float(rho_bar)


def trial_sharpe_variance_from_grid(
    grid_csv_path: str | None = None,
    sharpe_col: str = "sharpe_ann",
    periods_per_year: int = 252,
) -> tuple[int, float]:
    """Load sensitivity grid and compute (N_trials, V_per_period) for DSR.

    The grid CSV is expected to have one row per parameter combo with a
    column of ANNUALIZED Sharpe ratios. Variance is converted to per-period
    units by dividing by periods_per_year (since SR_ann = SR_pp · √PPy and
    therefore Var(SR_ann) = PPy · Var(SR_pp)).
    """
    if grid_csv_path is None:
        grid_csv_path = str(DATA_PROCESSED_DIR / "audit_sensitivity_grid.csv")
    df = pd.read_csv(grid_csv_path)
    if sharpe_col not in df.columns:
        raise ValueError(
            f"Column '{sharpe_col}' not in {grid_csv_path}; "
            f"available: {list(df.columns)}"
        )
    sr_ann = df[sharpe_col].dropna()
    n = len(sr_ann)
    var_ann = float(sr_ann.var(ddof=1)) if n > 1 else 0.0
    var_pp = var_ann / periods_per_year
    return n, var_pp
