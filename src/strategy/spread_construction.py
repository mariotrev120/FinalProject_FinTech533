"""
Entry-side spread construction.

Given an entry date, the underlying spot, current VIX, and a pricing provider,
build the put credit spread we want to sell that day:

  - Short strike: nearest 5-pt strike whose BS delta is closest to -0.16
                  (1-sigma OTM)
  - Long strike: short_strike - 5
  - Expiry: first Friday at least DTE_MIN days out, capped at DTE_MAX

Returns a fully-formed `Spread` plus the per-share credit and the short-leg
delta at entry. Friction is applied separately in the engine, so the credit
returned here is the "mid" theoretical credit.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from src.config import DTE_MAX, DTE_MIN, ENTRY_DELTA_TARGET, SPREAD_WIDTH_PTS
from src.strategy.pricer import PricingProvider
from src.strategy.types import OptionContract, Spread


STRIKE_INCREMENT: int = 5      # SPX strikes near the money are typically every 5 pts


def _next_friday(d: date) -> date:
    days_ahead = (4 - d.weekday()) % 7
    return d + timedelta(days=days_ahead or 7)


def select_expiry(entry_date: date, dte_min: int = DTE_MIN, dte_max: int = DTE_MAX) -> date:
    """Pick the first Friday at least dte_min days out, no later than dte_max.

    Returns the first valid Friday. If none fits, returns the closest Friday
    inside the window."""
    candidate = _next_friday(entry_date + timedelta(days=dte_min))
    if (candidate - entry_date).days <= dte_max:
        return candidate
    # Step back one week if we overshot
    return candidate - timedelta(days=7)


def _round_to_strike(price: float, increment: int = STRIKE_INCREMENT) -> int:
    return int(round(price / increment) * increment)


def select_short_strike(
    pricer: PricingProvider,
    as_of: pd.Timestamp,
    underlying: str,
    spot: float,
    expiry: date,
    vix: float,
    target_delta: float = ENTRY_DELTA_TARGET,
    search_pct: float = 0.20,
) -> tuple[int, float]:
    """Find the strike (rounded to STRIKE_INCREMENT) whose BS put delta
    magnitude is closest to target_delta. Search a band of `search_pct`
    below spot.

    Returns (strike, achieved_abs_delta).
    """
    lo = _round_to_strike(spot * (1 - search_pct), STRIKE_INCREMENT)
    hi = _round_to_strike(spot * 0.999, STRIKE_INCREMENT)   # stay OTM
    best_strike = hi
    best_diff = float("inf")
    best_abs_delta = 0.0
    for k in range(lo, hi + 1, STRIKE_INCREMENT):
        d = abs(pricer.implied_delta(as_of, underlying, spot, k, expiry, vix))
        diff = abs(d - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_strike = k
            best_abs_delta = d
    return best_strike, best_abs_delta


def build_spread(
    pricer: PricingProvider,
    as_of: pd.Timestamp,
    underlying: str,
    spot: float,
    vix: float,
    entry_date: Optional[date] = None,
) -> tuple[Spread, float, float]:
    """Build a complete entry spread.

    Returns (spread, theoretical_credit_per_share, short_leg_delta_abs).
    """
    if entry_date is None:
        entry_date = as_of.date()
    expiry = select_expiry(entry_date)
    short_strike, abs_delta = select_short_strike(
        pricer, as_of, underlying, spot, expiry, vix
    )
    long_strike = short_strike - SPREAD_WIDTH_PTS
    short_leg = OptionContract(underlying, expiry, short_strike, "P")
    long_leg = OptionContract(underlying, expiry, long_strike, "P")
    spread = Spread(short_leg=short_leg, long_leg=long_leg)
    short_px = pricer.price_put(as_of, underlying, spot, short_strike, expiry, vix)
    long_px = pricer.price_put(as_of, underlying, spot, long_strike, expiry, vix)
    credit = max(short_px - long_px, 0.0)
    return spread, credit, abs_delta
