"""
Position sizing.

Three multiplicative components plus a hard cap:
  1. Kelly-fractional from calibrated probability (capped at quarter Kelly)
  2. Volatility scaling: VOL_SCALE_PIVOT / VIX_t
  3. Stress multiplier: 0.5 if SPY-TLT 20d corr > threshold else 1.0
  Hard cap: max loss per trade <= TRADE_RISK_CAP_FRAC of equity

Returns the integer number of contracts to trade, plus the diagnostic
multipliers for the Trade record.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import (
    KELLY_CAP, STRESS_CORR_THRESHOLD, STRESS_MULTIPLIER, TRADE_RISK_CAP_FRAC,
    VOL_SCALE_PIVOT,
)


def kelly_fraction(p: float, max_win: float, max_loss: float) -> float:
    """Standard Kelly for a binary bet:  f = p/L - (1-p)/W
    where W and L are the win and loss magnitudes (in $).
    """
    if max_win <= 0 or max_loss <= 0:
        return 0.0
    return p / max_loss - (1.0 - p) / max_win


def vol_multiplier(vix: float) -> float:
    if vix <= 0:
        return 1.0
    return VOL_SCALE_PIVOT / vix


def stress_multiplier(spy_treasury_corr: float) -> float:
    return STRESS_MULTIPLIER if spy_treasury_corr > STRESS_CORR_THRESHOLD else 1.0


@dataclass
class SizeDecision:
    contracts: int
    kelly_fraction: float
    vol_multiplier: float
    stress_multiplier: float
    notional_at_risk: float


def size_position(
    equity: float,
    p_calibrated: float,
    max_win_per_spread: float,
    max_loss_per_spread: float,
    vix: float,
    spy_treasury_corr: float,
    use_kelly: bool = False,
) -> SizeDecision:
    """Compute the number of spreads to trade.

    Base sizing is the hard cap (1% of equity at risk per trade), scaled by
    vol_multiplier and stress_multiplier. Kelly is *additionally* applied
    only when `use_kelly=True` (i.e. ml_only and full modes — when we have a
    calibrated probability worth weighting size by). Naked / halts_only modes
    pass use_kelly=False and trade the cap-scaled size.

    For credit spreads with typical loss/win ratios, raw Kelly is negative
    at p<=0.85, which would zero out every trade. The interpretation is:
    Kelly thinks this is an unfavorable bet even at modest edge. We respect
    that signal in ML modes (sizing scales with bounded Kelly, which can be
    0 = skip), and ignore it in non-ML modes where the strategy claim is
    that the halt framework + structural premium provide the edge instead.

    All dollar inputs are PER SPREAD (after multiplier x100), not per share.
    """
    vm = vol_multiplier(vix)
    sm = stress_multiplier(spy_treasury_corr)

    raw_kelly = kelly_fraction(p_calibrated, max_win_per_spread, max_loss_per_spread)
    bounded_kelly = max(min(raw_kelly, KELLY_CAP), 0.0)

    # Hard cap is the primary risk budget
    hard_cap = equity * TRADE_RISK_CAP_FRAC

    if use_kelly:
        # Kelly multiplies into the cap (so 0 Kelly => 0 trade, full Kelly => full cap)
        risk_budget = hard_cap * (bounded_kelly / KELLY_CAP) * vm * sm
    else:
        # Cap scaled by vol and stress, no Kelly
        risk_budget = hard_cap * vm * sm

    risk_budget = min(risk_budget, hard_cap)

    contracts = int(risk_budget // max_loss_per_spread) if max_loss_per_spread > 0 else 0
    notional = contracts * max_loss_per_spread
    return SizeDecision(
        contracts=contracts,
        kelly_fraction=bounded_kelly,
        vol_multiplier=vm,
        stress_multiplier=sm,
        notional_at_risk=notional,
    )
