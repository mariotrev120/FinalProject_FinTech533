"""
Exit logic. Three fates plus emergency, evaluated daily on each open trade.

  Fate 1 (profit_target): close at 50% of max profit
  Fate 2 (stop_loss):     close at 200% of credit received
                          executed at min(stop_level, opening_gap_price)
  Fate 3 (time_exit):     hard close at 21 DTE
  Emergency:              short leg delta > 0.50

Returns the exit `fate` and the exit debit (per share, mid-price; friction is
applied in the engine, not here).
"""
from __future__ import annotations

from datetime import date
from typing import Optional, TypedDict

import pandas as pd

import src.config as cfg
from src.strategy.pricer import PricingProvider
from src.strategy.types import Fate, Spread


class ExitDecision(TypedDict, total=False):
    fate: Fate
    exit_debit: float
    exit_short_delta: float


def _spread_debit(
    pricer: PricingProvider,
    as_of: pd.Timestamp,
    spot: float,
    spread: Spread,
    vix: float,
) -> float:
    """Mid-price debit needed to close (buy back) the spread, per share."""
    sp = pricer.price_put(
        as_of, spread.underlying, spot,
        spread.short_leg.strike, spread.short_leg.expiry, vix,
    )
    lp = pricer.price_put(
        as_of, spread.underlying, spot,
        spread.long_leg.strike, spread.long_leg.expiry, vix,
    )
    return max(sp - lp, 0.0)


def evaluate_exit(
    pricer: PricingProvider,
    as_of: pd.Timestamp,
    spot: float,
    open_gap_spot: Optional[float],     # for gap-aware execution
    spread: Spread,
    vix: float,
    entry_credit_per_share: float,
    entry_date: date,
) -> Optional[ExitDecision]:
    """Decide whether to close this trade today. Returns None to keep open.

    Order of checks: emergency > profit_target > stop_loss > time_exit.
    """
    today_debit = _spread_debit(pricer, as_of, spot, spread, vix)
    short_delta_abs = abs(pricer.implied_delta(
        as_of, spread.underlying, spot,
        spread.short_leg.strike, spread.short_leg.expiry, vix,
    ))

    # Emergency
    if short_delta_abs > cfg.EMERGENCY_DELTA:
        return {"fate": "emergency", "exit_debit": today_debit,
                "exit_short_delta": short_delta_abs}

    pt_level = entry_credit_per_share * (1.0 - cfg.PROFIT_TARGET_FRAC)
    if today_debit <= pt_level:
        return {"fate": "profit_target", "exit_debit": today_debit,
                "exit_short_delta": short_delta_abs}

    stop_level = entry_credit_per_share * cfg.STOP_LOSS_MULT
    if today_debit >= stop_level:
        # Gap-aware: realized exit is whichever is worse for the seller
        # (i.e. higher debit) between the stop level and the morning gap.
        if open_gap_spot is not None and open_gap_spot != spot:
            gap_debit = _spread_debit(pricer, as_of, open_gap_spot, spread, vix)
            realized_debit = max(today_debit, gap_debit, stop_level)
        else:
            realized_debit = max(today_debit, stop_level)
        return {"fate": "stop_loss", "exit_debit": realized_debit,
                "exit_short_delta": short_delta_abs}

    dte = (spread.short_leg.expiry - as_of.date()).days
    if dte <= cfg.TIME_EXIT_DTE:
        return {"fate": "time_exit", "exit_debit": today_debit,
                "exit_short_delta": short_delta_abs}

    return None
