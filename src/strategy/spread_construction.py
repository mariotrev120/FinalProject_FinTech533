"""
Entry-side spread construction for credit spreads (puts, calls, iron condors).

Given an entry date, the underlying spot, current VIX, and a pricing provider,
build the credit spread we want to sell that day:

  - Short strike: nearest grid strike whose BS delta is closest to ±0.16
                  (1-sigma OTM, 16-delta convention)
  - Long strike: short_strike ± wing_width, where wing_width is per-instrument
                 (max(strike_increment, 0.5% × spot), snapped to listed grid)
  - Expiry: first Friday at least DTE_MIN days out, capped at DTE_MAX

Per-instrument strike grids and wing-width policy are defined below; each
underlying's wing scales with spot so absolute wing widths track the price
range of the instrument (1-pt wings on TLT, ~25-pt wings on NDX).

Returns a fully-formed `Spread` plus the per-share credit and the short-leg
delta at entry. Friction is applied separately in the engine, so the credit
returned here is the "mid" theoretical credit.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

import src.config as cfg
from src.strategy.pricer import PricingProvider
from src.strategy.types import OptionContract, Spread


STRIKE_INCREMENT_BY_UNDERLYING: dict[str, int] = {
    # Indices
    "SPX": 5,
    "RUT": 5,
    "NDX": 25,
    "XSP": 1,
    # ETFs
    "TLT": 1,
    "GLD": 1,
    # Wheel basket — single-stock options trade in $1 strike grid OTM at
    # the deltas we target (16-25); narrower than that for ATM but our
    # wing-snap clamps to the 0.5%-of-spot rule.
    "AAPL": 1, "MSFT": 1, "GOOGL": 1, "JNJ": 1, "KO": 1,
    "PG": 1, "WMT": 1, "JPM": 1, "PEP": 1,
}

# Wing width is the constant cfg.SPREAD_WIDTH_PTS (currently 5). The per-
# ticker strike grid (above) handles strike snap; the wing width is fixed
# in points and applies symmetrically to all tickers in the universe.
WING_PCT_OF_SPOT: float = 0.0   # retained for diagnostic compatibility


def compute_wing_width(underlying: str, spot: float) -> int:    # noqa: ARG001
    """Wing width in strike points. Fixed at cfg.SPREAD_WIDTH_PTS for all
    tickers — the strike grid (per STRIKE_INCREMENT_BY_UNDERLYING) handles
    chain-grid heterogeneity at strike-selection time.
    """
    return int(cfg.SPREAD_WIDTH_PTS)


def _next_friday(d: date) -> date:
    days_ahead = (4 - d.weekday()) % 7
    return d + timedelta(days=days_ahead or 7)


def select_expiry(entry_date: date, dte_min: Optional[int] = None, dte_max: Optional[int] = None) -> date:
    """Pick the first Friday at least dte_min days out, no later than dte_max.

    Returns the first valid Friday. If none fits, returns the closest Friday
    inside the window. Defaults pulled from config at call time so sensitivity
    sweeps that monkey-patch cfg.DTE_MIN/DTE_MAX take effect."""
    if dte_min is None:
        dte_min = cfg.DTE_MIN
    if dte_max is None:
        dte_max = cfg.DTE_MAX
    candidate = _next_friday(entry_date + timedelta(days=dte_min))
    if (candidate - entry_date).days <= dte_max:
        return candidate
    return candidate - timedelta(days=7)


def _round_to_strike(price: float, increment: int) -> int:
    return int(round(price / increment) * increment)


def select_short_strike(
    pricer: PricingProvider,
    as_of: pd.Timestamp,
    underlying: str,
    spot: float,
    expiry: date,
    vix: float,
    target_delta: Optional[float] = None,
    search_pct: float = 0.20,
    right: str = "P",
) -> tuple[int, float]:
    """Find the strike whose BS option delta magnitude is closest to target_delta.

    For puts (right='P'): searches the band `search_pct` BELOW spot. Uses
    pricer.implied_delta (which returns negative put deltas).
    For calls (right='C'): searches the band `search_pct` ABOVE spot. Uses
    pricer.implied_call_delta (which returns positive call deltas).

    Returns (strike, achieved_abs_delta).
    """
    if target_delta is None:
        target_delta = cfg.ENTRY_DELTA_TARGET
    inc = STRIKE_INCREMENT_BY_UNDERLYING.get(underlying, 5)

    if right == "P":
        lo = _round_to_strike(spot * (1 - search_pct), inc)
        hi = _round_to_strike(spot * 0.999, inc)   # stay OTM (below spot)
        delta_fn = pricer.implied_delta
    else:   # "C"
        lo = _round_to_strike(spot * 1.001, inc)   # stay OTM (above spot)
        hi = _round_to_strike(spot * (1 + search_pct), inc)
        delta_fn = pricer.implied_call_delta

    best_strike = lo
    best_diff = float("inf")
    best_abs_delta = 0.0
    for k in range(lo, hi + 1, inc):
        d = abs(delta_fn(as_of, underlying, spot, k, expiry, vix))
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
    """Build a put credit spread (the original / v1 behavior).

    Returns (spread, theoretical_credit_per_share, short_leg_delta_abs).
    """
    if entry_date is None:
        entry_date = as_of.date()
    expiry = select_expiry(entry_date)
    short_strike, abs_delta = select_short_strike(
        pricer, as_of, underlying, spot, expiry, vix, right="P",
    )
    wing = compute_wing_width(underlying, spot)
    long_strike = short_strike - wing
    short_leg = OptionContract(underlying, expiry, short_strike, "P")
    long_leg = OptionContract(underlying, expiry, long_strike, "P")
    spread = Spread(short_leg=short_leg, long_leg=long_leg)
    short_px = pricer.price_put(as_of, underlying, spot, short_strike, expiry, vix)
    long_px = pricer.price_put(as_of, underlying, spot, long_strike, expiry, vix)
    credit = max(short_px - long_px, 0.0)
    return spread, credit, abs_delta


def build_call_credit_spread(
    pricer: PricingProvider,
    as_of: pd.Timestamp,
    underlying: str,
    spot: float,
    vix: float,
    entry_date: Optional[date] = None,
) -> tuple[Spread, float, float]:
    """Build a call credit spread (the call wing of an iron condor).

    Short the lower-strike call (closer to spot), long the higher-strike call
    (further wing). Profit if underlying stays below short strike.

    Returns (spread, theoretical_credit_per_share, short_leg_delta_abs).
    """
    if entry_date is None:
        entry_date = as_of.date()
    expiry = select_expiry(entry_date)
    short_strike, abs_delta = select_short_strike(
        pricer, as_of, underlying, spot, expiry, vix, right="C",
    )
    wing = compute_wing_width(underlying, spot)
    long_strike = short_strike + wing
    short_leg = OptionContract(underlying, expiry, short_strike, "C")
    long_leg = OptionContract(underlying, expiry, long_strike, "C")
    spread = Spread(short_leg=short_leg, long_leg=long_leg)
    short_px = pricer.price_call(as_of, underlying, spot, short_strike, expiry, vix)
    long_px = pricer.price_call(as_of, underlying, spot, long_strike, expiry, vix)
    credit = max(short_px - long_px, 0.0)
    return spread, credit, abs_delta


def build_iron_condor(
    pricer: PricingProvider,
    as_of: pd.Timestamp,
    underlying: str,
    spot: float,
    vix: float,
    entry_date: Optional[date] = None,
) -> tuple[Spread, Spread, float, float, float, float]:
    """Build a complete 16-delta iron condor (put credit spread + call credit spread).

    Both wings same expiry, both 5-pt wide. Each side built independently with
    the same target delta on its own side of spot.

    Returns (put_spread, call_spread,
             put_credit_per_share, call_credit_per_share,
             put_short_abs_delta, call_short_abs_delta).
    """
    put_spread, put_credit, put_delta = build_spread(
        pricer, as_of, underlying, spot, vix, entry_date,
    )
    call_spread, call_credit, call_delta = build_call_credit_spread(
        pricer, as_of, underlying, spot, vix, entry_date,
    )
    return put_spread, call_spread, put_credit, call_credit, put_delta, call_delta
