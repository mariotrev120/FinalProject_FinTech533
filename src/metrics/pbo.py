"""
Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
Cross-Validation (CSCV) per Bailey, Borwein, López de Prado, Zhu (2015).

Companion to the Deflated Sharpe Ratio (`deflated_sharpe.py`). DSR
controls for selection-bias inflation of the headline Sharpe; PBO
answers the complementary question: "given the full sensitivity grid,
how often does the IS-best strategy underperform OOS?"

Algorithm 2.3 from the paper:
  1. Form a (T × N) performance matrix M where each column is one
     trial's per-period return series.
  2. Partition rows into S disjoint submatrices of size T/S × N each.
     S must be EVEN (paper recommends S=16 for daily backtests with
     a few years of data → 12,780 combinations, σ[f(λ)] < 0.005).
  3. Form all C(S, S/2) row-combinations c. Each c picks half of the
     S submatrices for training (J), the other half for testing (J̄).
  4. For each c:
     a. Concatenate training submatrices in original row order → J.
     b. Compute IS performance vector R^c on each of the N columns
        (Sharpe by default; pluggable).
     c. Identify n* = argmax R^c (best strategy IS).
     d. Compute OOS performance vector R̄^c on J̄.
     e. Compute relative rank ω̄_c = rank(R̄^c[n*]) / (N + 1) ∈ (0,1).
     f. Compute logit λ_c = ln(ω̄_c / (1 − ω̄_c)).
  5. PBO = fraction of {c} with λ_c < 0 (equivalently, ω̄_c < 0.5).
     PBO ≈ 0 → no overfitting; PBO ≈ 1 → severe overfitting.

The framework also produces:
  - Performance-degradation slope (β from regressing R̄_n* on R_n*)
  - Probability of OOS loss: Prob[R̄_n* < 0]
  - Stochastic-dominance check (R_n* CDF vs Mean(R̄) CDF)

Reproducible: no random component. Same (T × N) matrix → same PBO.

Reference:
  Bailey, D., Borwein, J., López de Prado, M., & Zhu, Q. J. (2015).
  The Probability of Backtest Overfitting.
  Journal of Computational Finance.
  SSRN: https://ssrn.com/abstract=2326253
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Optional

import numpy as np
import pandas as pd


# --- Performance metrics on a returns matrix ------------------------------

def sharpe_per_column(returns_block: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Per-column Sharpe ratio (per-period, NOT annualized) on a (T × N)
    block of returns. Vectorized over columns.

    Returns NaN for columns with zero std. NaN treated as worst (will not
    be selected as IS-best, will land at the bottom of the OOS rank)."""
    if returns_block.shape[0] < 2:
        return np.full(returns_block.shape[1], np.nan)
    mean = np.nanmean(returns_block, axis=0)
    std = np.nanstd(returns_block, axis=0, ddof=ddof)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(std > 0, mean / std, np.nan)
    return out


PerformanceFn = Callable[[np.ndarray], np.ndarray]


# --- CSCV core algorithm --------------------------------------------------

def _ranks_with_nan_last(values: np.ndarray) -> np.ndarray:
    """Return ranks (1 = worst, N = best) of values, with NaN ranked
    LAST (i.e. assigned the smallest rank). Stable order preserves
    column index for ties."""
    n = len(values)
    not_nan = ~np.isnan(values)
    ranks = np.zeros(n, dtype=float)
    if not_nan.any():
        # rankdata-equivalent: assign 1..k to the k non-nan values
        order = np.argsort(values[not_nan], kind="stable")
        nan_idx = np.where(not_nan)[0][order]
        for i, idx in enumerate(nan_idx):
            ranks[idx] = i + 1.0
    # NaN columns get rank 0 (below worst non-nan) — they will never be
    # IS-best (n* = argmax) and will land at the worst OOS position.
    return ranks


@dataclass
class PBOReport:
    """CSCV-derived statistics over a sensitivity grid."""
    n_trials: int                              # N (columns in M)
    n_observations: int                        # T (rows in M)
    n_partitions: int                          # S
    n_combinations: int                        # C(S, S/2)
    pbo: float                                 # fraction of c with λ_c < 0
    median_logit: float
    perf_degradation_slope: float              # β from R̄_n* = α + β·R_n*
    perf_degradation_intercept: float          # α
    prob_oos_loss: float                       # Prob[R̄_n* < 0]
    stoch_dom_first_order: bool                # R_n* dominates Mean(R̄)?
    is_best_perf: list[float] = field(default_factory=list)   # R^c[n*] per c
    oos_at_is_best_perf: list[float] = field(default_factory=list)  # R̄^c[n*]
    logits: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_trials": self.n_trials,
            "n_observations": self.n_observations,
            "n_partitions": self.n_partitions,
            "n_combinations": self.n_combinations,
            "pbo": self.pbo,
            "median_logit": self.median_logit,
            "perf_degradation_slope": self.perf_degradation_slope,
            "perf_degradation_intercept": self.perf_degradation_intercept,
            "prob_oos_loss": self.prob_oos_loss,
            "stoch_dom_first_order": self.stoch_dom_first_order,
        }

    def headline(self) -> str:
        return (
            f"CSCV PBO Report:\n"
            f"  Trials (N):                {self.n_trials}\n"
            f"  Observations (T):          {self.n_observations}\n"
            f"  Partitions (S):            {self.n_partitions}\n"
            f"  Combinations C(S, S/2):    {self.n_combinations}\n"
            f"  --- Headline ---\n"
            f"  PBO (φ):                   {self.pbo:>+.4f}\n"
            f"  Median logit:              {self.median_logit:>+.4f}\n"
            f"  --- Performance degradation ---\n"
            f"  R̄_n* = α + β · R_n*\n"
            f"     α (intercept):          {self.perf_degradation_intercept:>+.6f}\n"
            f"     β (slope):              {self.perf_degradation_slope:>+.4f}\n"
            f"  --- Loss / dominance ---\n"
            f"  Prob[OOS R̄_n* < 0]:       {self.prob_oos_loss:>+.4f}\n"
            f"  R_n* first-order dominates Mean(R̄): {self.stoch_dom_first_order}\n"
        )


def cscv_pbo(
    M: pd.DataFrame | np.ndarray,
    n_partitions: int = 16,
    performance_fn: PerformanceFn = sharpe_per_column,
) -> PBOReport:
    """Compute PBO and supporting statistics on (T × N) returns matrix M.

    Inputs:
      M: shape (T, N). Each column is one trial's per-period return series.
         Rows must be aligned synchronous observations across trials.
      n_partitions: S (even). Default 16 per BBLP recommendation.
      performance_fn: function taking (block × N) array → length-N perf
         vector. Default: per-period Sharpe. Pluggable for other metrics.

    Returns:
      PBOReport with PBO, performance-degradation regression, prob-of-loss,
      first-order stochastic-dominance flag, and full per-combination
      logit/IS/OOS arrays for downstream visualization.

    Raises ValueError if S is odd or T is not divisible by S.
    """
    if isinstance(M, pd.DataFrame):
        M_arr = M.to_numpy()
    else:
        M_arr = np.asarray(M)
    T, N = M_arr.shape
    S = int(n_partitions)
    if S < 2 or S % 2 != 0:
        raise ValueError(f"S must be even and ≥ 2; got {S}")
    if T < S:
        raise ValueError(f"T={T} < S={S}; cannot partition")
    block_len = T // S
    # Truncate any remainder rows so each submatrix is exactly block_len
    used_T = block_len * S
    M_used = M_arr[:used_T]
    submatrices = [M_used[s * block_len : (s + 1) * block_len] for s in range(S)]

    half = S // 2
    combos = list(combinations(range(S), half))
    n_combos = len(combos)

    is_best_perf: list[float] = []
    oos_at_is_best_perf: list[float] = []
    logits: list[float] = []
    overall_oos: list[float] = []  # Mean OOS performance across all N (for stoch dom)

    for combo in combos:
        train_idx = list(combo)
        test_idx = [s for s in range(S) if s not in combo]
        # Concatenate in original row order (sorting indices ensures order)
        J = np.concatenate([submatrices[s] for s in sorted(train_idx)], axis=0)
        J_bar = np.concatenate([submatrices[s] for s in sorted(test_idx)], axis=0)

        R = performance_fn(J)
        R_bar = performance_fn(J_bar)

        # n* = argmax R (IS best). NaN columns get rank 0 in our impl,
        # so finite-only argmax via nanargmax.
        if np.all(np.isnan(R)):
            continue
        n_star = int(np.nanargmax(R))
        is_best_perf.append(float(R[n_star]))
        r_bar_at_star = float(R_bar[n_star])
        oos_at_is_best_perf.append(r_bar_at_star)
        overall_oos.extend(R_bar.tolist())

        # Relative rank of R̄[n*] within R̄: rank=1 → worst, rank=N → best
        ranks_bar = _ranks_with_nan_last(R_bar)
        rank_at_star = ranks_bar[n_star]
        # ω̄ ∈ (0, 1)
        omega_bar = rank_at_star / (N + 1)
        # Clip away from {0, 1} to avoid logit(±∞)
        omega_bar = min(max(omega_bar, 1e-6), 1.0 - 1e-6)
        lam = math.log(omega_bar / (1.0 - omega_bar))
        logits.append(lam)

    if not logits:
        return PBOReport(
            n_trials=N, n_observations=T, n_partitions=S,
            n_combinations=n_combos, pbo=float("nan"),
            median_logit=float("nan"),
            perf_degradation_slope=float("nan"),
            perf_degradation_intercept=float("nan"),
            prob_oos_loss=float("nan"),
            stoch_dom_first_order=False,
        )

    logits_arr = np.array(logits)
    is_arr = np.array(is_best_perf)
    oos_arr = np.array(oos_at_is_best_perf)
    overall_oos_arr = np.array(overall_oos)

    pbo = float((logits_arr < 0).mean())
    median_logit = float(np.median(logits_arr))

    # Performance-degradation regression: R̄_n* = α + β · R_n*
    if len(is_arr) >= 2 and np.var(is_arr) > 0:
        slope, intercept = np.polyfit(is_arr, oos_arr, 1)
    else:
        slope, intercept = float("nan"), float("nan")

    prob_oos_loss = float((oos_arr < 0).mean())

    # First-order stochastic dominance: R_n* CDF ≤ Mean(R̄) CDF for all x?
    # Equivalent: at every x, fraction of R_n* ≥ x must be ≥ fraction of
    # overall OOS ≥ x.
    if len(overall_oos_arr) > 0:
        check_xs = np.linspace(
            min(oos_arr.min(), overall_oos_arr.min()),
            max(oos_arr.max(), overall_oos_arr.max()),
            50,
        )
        dominates = True
        any_strict = False
        for x in check_xs:
            p_select = float((oos_arr >= x).mean())
            p_overall = float((overall_oos_arr >= x).mean())
            if p_select < p_overall - 1e-9:
                dominates = False
                break
            if p_select > p_overall + 1e-9:
                any_strict = True
        stoch_dom = dominates and any_strict
    else:
        stoch_dom = False

    return PBOReport(
        n_trials=N, n_observations=T, n_partitions=S,
        n_combinations=n_combos,
        pbo=pbo,
        median_logit=median_logit,
        perf_degradation_slope=float(slope),
        perf_degradation_intercept=float(intercept),
        prob_oos_loss=prob_oos_loss,
        stoch_dom_first_order=stoch_dom,
        is_best_perf=is_arr.tolist(),
        oos_at_is_best_perf=oos_arr.tolist(),
        logits=logits_arr.tolist(),
    )
