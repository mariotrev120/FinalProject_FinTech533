"""
Cross-asset stress / regime features built from rolling correlations and
spreads.

Pulled out of `exogenous.py` per Robert's structure for cleaner per-feature
testing. All functions take pre-cleaned price Series and return the derived
feature Series.

Note on README deviation: the original spec was SPY-TLT 20d correlation. TWS
paper accounts cap TLT history at 2016-02-03, which would force the IS window
to start in 2016. To preserve a pre-2016 IS window, the implementation uses
SPY-TNX 20d correlation as a Treasury proxy (TNX = 10Y yield, available back
to 2011-05). Same regime signal: positive correlation between SPY returns
and 10Y yield changes indicates risk-off (stocks down, yields up = bonds
selling off too). Disclosed in the writeup.
"""
from __future__ import annotations

import pandas as pd


def rolling_returns_correlation(
    spy: pd.Series, other: pd.Series, window: int = 20,
) -> pd.Series:
    """Rolling correlation between SPY pct returns and another series'
    pct returns. Other is expected to be a price Series (gets pct-changed)."""
    return spy.pct_change().rolling(window).corr(other.pct_change())


def spy_tnx_correlation_20d(spy: pd.Series, tnx: pd.Series) -> pd.Series:
    """SPY pct return correlated with TNX (10Y yield) raw change. Yields
    are level series, not prices, so we use diff() not pct_change()."""
    return spy.pct_change().rolling(20).corr(tnx.diff())


def spy_gld_correlation_20d(spy: pd.Series, gld: pd.Series) -> pd.Series:
    return rolling_returns_correlation(spy, gld, window=20)


def hyg_lqd_spread(hyg: pd.Series, lqd: pd.Series) -> pd.Series:
    """High-yield minus investment-grade credit price spread. Widening spread
    = credit-market stress. Used as a Layer-2 soft halt trigger and as a
    feature for the ML gate."""
    return hyg - lqd
