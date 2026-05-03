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


# --- Egger & Vestal (2025) trader-application form ------------------------

def regime_change_probability_bound(
    mu_committed: float,
    x_bar_realized: float,
    n_trades: int,
) -> float:
    """Compute the Hoeffding upper bound on the probability of observing
    the realized underperformance, given the committed regime hypothesis H0.

    Per Egger & Vestal (2025) Eq. 1.4 — trader-application form, distinct
    from the statistical-learning-theory application.

        P[X̄ − μ ≥ t | H0] ≤ e^(−2·t²·N)

    where t = μ − X̄  (realized underperformance), N = n_trades.

    Outputs the upper bound (a probability ∈ (0, 1]). Lower values =
    decreasing plausibility of H0, increasing plausibility of regime
    change H1.

    Inputs:
      mu_committed: pre-committed μ baseline (e.g. IS-baseline win rate),
        sourced from PRE_COMMITMENT_VRP/WHEEL — NOT inferred from the
        observed series.
      x_bar_realized: observed mean of bounded RV (e.g. rolling win rate).
      n_trades: number of observations underlying x_bar.

    Returns 1.0 (no underperformance) when X̄ ≥ μ. Otherwise returns the
    upper-bound probability, which decreases monotonically with both
    realized underperformance magnitude and sample size.
    """
    if n_trades <= 0:
        return 1.0
    t = float(mu_committed) - float(x_bar_realized)
    if t <= 0:
        return 1.0
    return float(math.exp(-2.0 * (t ** 2) * n_trades))


def regime_change_signal(
    mu_committed: float, x_bar_realized: float, n_trades: int,
) -> str:
    """Map Hoeffding probability bound to a 4-state regime signal per
    Egger/Vestal threshold semantics:

      "green"   : bound ≥ 50% — observed underperformance plausibly
                  due to chance under H0
      "yellow"  : 25% ≤ bound < 50% — beliefs about regime probably
                  no longer fully right; consider stake reduction
      "red"     : 10% ≤ bound < 25% — significant regime risk;
                  substantial concern
      "critical": bound < 10% — almost certain regime change;
                  halt and rethink
    """
    p = regime_change_probability_bound(mu_committed, x_bar_realized, n_trades)
    if p >= 0.50:
        return "green"
    if p >= 0.25:
        return "yellow"
    if p >= 0.10:
        return "red"
    return "critical"


def rolling_regime_signal(
    trade_outcomes: pd.Series,
    mu_committed: float,
    window: int = 60,
) -> pd.DataFrame:
    """For each trade index, compute (rolling win rate, Hoeffding bound,
    signal) using the committed μ and a rolling window.

    Returns a DataFrame with columns:
      - rolling_winrate
      - hoeffding_bound  (upper bound on P[X̄−μ≥t | H0])
      - signal           ("green"/"yellow"/"red"/"critical")
    """
    win = rolling_winrate(trade_outcomes, window)
    bounds = pd.Series(index=win.index, dtype=float)
    signals = pd.Series(index=win.index, dtype=object)
    for idx, x_bar in win.items():
        if pd.isna(x_bar):
            bounds.loc[idx] = float("nan")
            signals.loc[idx] = pd.NA
            continue
        b = regime_change_probability_bound(mu_committed, x_bar, window)
        bounds.loc[idx] = b
        signals.loc[idx] = regime_change_signal(mu_committed, x_bar, window)
    return pd.DataFrame({
        "rolling_winrate": win,
        "hoeffding_bound": bounds,
        "signal": signals,
    })
