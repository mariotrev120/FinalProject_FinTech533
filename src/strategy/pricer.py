"""
Option pricing interface.

A `PricingProvider` is anything that can answer "what was the mid price of
this put on this date?". The backtest engine calls into a pricer instance;
nothing else cares about HOW the price was sourced.

Concrete implementations live in:
  - black_scholes.py — derives IV from VIX (with optional SKEW adjustment),
    closed-form BS price. Used as the default fallback when no real option
    quote source is wired up.
  - wrds.py (TBD) — looks up real OptionMetrics IvyDB EOD quotes.
  - polygon.py (TBD) — Polygon.io historical OPRA.

Swapping between providers is a one-line change in the engine instantiation.
"""
from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PricingProvider(Protocol):
    """Anything that can answer mid price for an SPX/XSP put on a given date."""

    def price_put(
        self,
        as_of: pd.Timestamp,
        underlying: str,
        spot: float,
        strike: float,
        expiry: date,
        vix: float,
    ) -> float:
        """Return the mid price of one put contract (per share, not per
        100-share contract — i.e. the quoted option premium)."""
        ...

    def implied_delta(
        self,
        as_of: pd.Timestamp,
        underlying: str,
        spot: float,
        strike: float,
        expiry: date,
        vix: float,
    ) -> float:
        """Return the put delta (negative for puts) for the same contract.
        Used for emergency-exit and entry-strike selection."""
        ...
