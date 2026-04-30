"""
Block bootstrap confidence intervals.

Non-overlapping block bootstrap on the trade-level return series. Block size
should reflect the auto-correlation horizon of trade outcomes; for weekly
trades on overlapping spreads, blocks of 3-4 trades preserve the local
correlation structure without overspecifying.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.config import SEED


def block_indices(n: int, block_size: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Return a (n_resamples, n) array of integer indices into the trade
    series, drawn by sampling whole non-overlapping blocks with replacement."""
    n_blocks = n // block_size
    if n_blocks <= 0:
        # Fall back to per-trade resampling
        return rng.integers(0, n, size=(n_resamples, n))
    block_starts = np.arange(n_blocks) * block_size
    out = np.empty((n_resamples, n_blocks * block_size), dtype=np.int64)
    for r in range(n_resamples):
        chosen = rng.choice(block_starts, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block_size) for s in chosen])
        out[r] = idx
    return out


def bootstrap_ci(
    series: pd.Series,
    statistic: Callable[[np.ndarray], float],
    block_size: int = 3,
    n_resamples: int = 10_000,
    alpha: float = 0.10,
    seed: int = SEED,
) -> dict:
    """Block bootstrap CI for a scalar statistic of a 1D return series.

    Returns dict with keys: point, lower, upper, n_resamples.
    alpha=0.10 -> 90% CI.
    """
    arr = series.dropna().values
    if len(arr) < 2:
        return {"point": float("nan"), "lower": float("nan"),
                "upper": float("nan"), "n_resamples": 0}
    rng = np.random.default_rng(seed)
    idx = block_indices(len(arr), block_size, n_resamples, rng)
    samples = arr[idx]    # shape (n_resamples, n)
    stats = np.array([statistic(s) for s in samples])
    return {
        "point": float(statistic(arr)),
        "lower": float(np.quantile(stats, alpha / 2)),
        "upper": float(np.quantile(stats, 1 - alpha / 2)),
        "n_resamples": n_resamples,
        "block_size": block_size,
    }


# --- Common statistics ---------------------------------------------------

def sharpe_stat(returns: np.ndarray, periods_per_year: float = 52) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)


def winrate_stat(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def mean_return_stat(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float(returns.mean())
