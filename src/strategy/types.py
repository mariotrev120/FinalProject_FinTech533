"""
Core data types for the trade lifecycle.

Designed so the trade record is fully self-describing for the blotter, equity
curve, Hoeffding monitor, and ablation table — every downstream artifact can
reconstruct what happened from a list of Trade dicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional


Right = Literal["P", "C"]
Fate = Literal["profit_target", "stop_loss", "time_exit", "emergency", "eos_force", "open"]
Mode = Literal["naked", "ml_only", "halts_only", "full"]


@dataclass(frozen=True)
class OptionContract:
    underlying: str          # "SPX" or "XSP"
    expiry: date
    strike: float
    right: Right
    multiplier: int = 100


@dataclass(frozen=True)
class Spread:
    """Defined-risk credit spread. Generalized to either side of the surface:

      - Put credit spread (right="P"): short the higher-strike put,
        long the lower-strike put. Profit if underlying stays above short strike.
      - Call credit spread (right="C"): short the lower-strike call,
        long the higher-strike call. Profit if underlying stays below short strike.

    For an iron condor, two Spread instances are opened simultaneously (one P,
    one C) and linked at the Trade level via `iron_condor_id`.
    """
    short_leg: OptionContract
    long_leg: OptionContract

    def __post_init__(self) -> None:
        if self.short_leg.right != self.long_leg.right:
            raise ValueError(
                f"Spread legs must share right; got short={self.short_leg.right} "
                f"long={self.long_leg.right}"
            )
        if self.short_leg.right == "P":
            if self.short_leg.strike <= self.long_leg.strike:
                raise ValueError(
                    f"put credit spread requires short_strike > long_strike "
                    f"(got short={self.short_leg.strike}, long={self.long_leg.strike})"
                )
        else:  # "C"
            if self.short_leg.strike >= self.long_leg.strike:
                raise ValueError(
                    f"call credit spread requires short_strike < long_strike "
                    f"(got short={self.short_leg.strike}, long={self.long_leg.strike})"
                )
        if self.short_leg.expiry != self.long_leg.expiry:
            raise ValueError("legs must share an expiry")
        if self.short_leg.underlying != self.long_leg.underlying:
            raise ValueError("legs must share an underlying")

    @property
    def width(self) -> float:
        return abs(self.short_leg.strike - self.long_leg.strike)

    @property
    def underlying(self) -> str:
        return self.short_leg.underlying

    @property
    def right(self) -> Right:
        return self.short_leg.right


@dataclass
class Trade:
    """One trade record from entry through close (or still-open).

    Every field that drives a metric or plot is captured here. The blotter is
    just `pd.DataFrame([trade.__dict__ for trade in trades])`.
    """
    # Identity
    trade_id: int
    mode: Mode

    # Entry
    entry_date: date
    spread: Spread
    contracts: int
    entry_credit_per_spread: float       # net credit received per spread, post-friction

    # Snapshots at entry (for diagnostics / regime tagging)
    entry_spx: float
    entry_vix: float
    entry_iv: float                       # IV used to price entry (e.g. VIX/100 + skew)
    entry_dte: int

    # Decision audit (for "no silent filtering" rule)
    ml_probability: Optional[float] = None        # calibrated p, None in naked mode
    halt_state_at_entry: Optional[str] = None     # "active" or which halt layer was firing

    # Exit (None until closed)
    exit_date: Optional[date] = None
    exit_debit_per_spread: Optional[float] = None   # net debit paid per spread, post-friction
    fate: Fate = "open"
    exit_spx: Optional[float] = None
    exit_vix: Optional[float] = None
    exit_short_delta: Optional[float] = None        # for emergency-exit bookkeeping

    # Realized P&L per spread, in dollars per contract * multiplier
    pnl_per_spread: Optional[float] = None

    # Sizing diagnostics
    kelly_fraction: Optional[float] = None
    vol_multiplier: Optional[float] = None
    stress_multiplier: Optional[float] = None

    # Iron-condor pairing: when set, this trade is one side of an IC opened on
    # the same date with another trade sharing this id. Per-side fates are
    # managed independently; aggregate IC P&L = sum of per-side P&L grouped
    # by iron_condor_id. None means the trade is a standalone (put-only)
    # credit spread.
    iron_condor_id: Optional[int] = None

    @property
    def is_open(self) -> bool:
        return self.fate == "open"

    @property
    def total_pnl(self) -> Optional[float]:
        """Total P&L in dollars. `pnl_per_spread` is already dollars per
        contract (multiplier baked in by the engine), so we just scale by
        contract count — no second multiplier hit."""
        if self.pnl_per_spread is None:
            return None
        return self.pnl_per_spread * self.contracts
