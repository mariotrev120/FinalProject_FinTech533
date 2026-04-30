"""
Hoeffding bound for live win-rate monitoring.

Under H0 (the strategy's in-sample regime persists), Hoeffding's Inequality
bounds the probability that observed underperformance is due to chance:

    P(|p_hat - p_baseline| >= eps) <= 2 * exp(-2 * n * eps^2)

For an alpha-level lower bound on win rate after n trades, solve for eps:
    eps = sqrt(log(2/alpha) / (2n))

If the realized rolling win rate falls below baseline - eps, the in-sample
regime is statistically rejected at the alpha level.

This is the formal answer to Vestal's "how do you know it stopped working?"
question. The slow halt's Trigger A in the halt framework references this
inequality.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def hoeffding_lower_bound(baseline: float, n: int, alpha: float = 0.10) -> float:
    """One-sided lower bound on win rate at significance alpha.

    Returns the win rate threshold below which the regime is rejected.
    For alpha=0.10 (90% confidence), eps ~ sqrt(log(20)/(2n))
    """
    if n <= 0:
        return 0.0
    eps = math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
    return max(baseline - eps, 0.0)


def rolling_winrate(trade_outcomes: pd.Series, window: int = 60) -> pd.Series:
    """Rolling fraction of wins over the last `window` trades.

    trade_outcomes: pd.Series of 0/1 (loss/win) indexed by trade entry date.
    """
    return trade_outcomes.rolling(window).mean()


def hoeffding_breach_dates(
    trade_outcomes: pd.Series,
    baseline: float,
    window: int = 60,
    alpha: float = 0.10,
) -> pd.Series:
    """For each trade index, return True if the rolling win rate is below
    the Hoeffding lower bound at that point. The first `window-1` trades
    return NaN (insufficient data)."""
    rolling = rolling_winrate(trade_outcomes, window)
    bound = hoeffding_lower_bound(baseline, window, alpha)
    return rolling < bound
