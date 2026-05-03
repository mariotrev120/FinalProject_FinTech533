"""
Friction model — commissions, VIX-scaled bid-ask slippage, gap-aware
execution rules, Section 1256 tax overlay.

Each function takes positional args (no hidden state) and is unit-testable
against the values in src/config.py.
"""
from __future__ import annotations

from src.config import (
    COMMISSION_MIN_PER_ORDER, COMMISSION_PER_LEG, GAP_SKIP_THRESHOLD_PCT,
    GAP_SLIPPAGE_THRESHOLD_PCT, REG_FEES_PER_LEG, SLIPPAGE_PCT_BY_VIX,
    TAX_LTCG_FRAC, TAX_STCG_FRAC,
)


# --- Commissions ---------------------------------------------------------

def commission_per_side(contracts: int, legs: int = 2) -> float:
    """IBKR Pro tiered: $0.65/contract/leg, $1 minimum per ORDER, plus
    regulatory fees ~$0.05/contract/leg."""
    base = legs * contracts * COMMISSION_PER_LEG
    base = max(base, COMMISSION_MIN_PER_ORDER)
    reg = legs * contracts * REG_FEES_PER_LEG
    return base + reg


def round_trip_commissions(contracts: int) -> float:
    """Entry + exit, both with 2 legs each."""
    return 2 * commission_per_side(contracts, legs=2)


# --- VIX-scaled bid-ask slippage ----------------------------------------

def slippage_pct_at_vix(vix: float) -> float:
    """Return the fraction of the bid-ask spread paid per side."""
    for (lo, hi), pct in SLIPPAGE_PCT_BY_VIX.items():
        if lo <= vix < hi:
            return pct
    return 1.0   # if vix is somehow not in any bucket


def slippage_dollars_per_share(
    bid_ask_spread_per_share: float, vix: float, gap_pct: float = 0.0,
) -> float:
    """Per-share slippage cost. `gap_pct` is the absolute Friday-close to
    Monday-open SPX move; if it crosses GAP_SLIPPAGE_THRESHOLD_PCT we add
    50% extra."""
    base = bid_ask_spread_per_share * slippage_pct_at_vix(vix)
    if abs(gap_pct) >= GAP_SLIPPAGE_THRESHOLD_PCT:
        base *= 1.5
    return base


def should_skip_entry_due_to_gap(gap_pct: float) -> bool:
    return abs(gap_pct) >= GAP_SKIP_THRESHOLD_PCT


# --- Section 1256 tax overlay -------------------------------------------

def section_1256_tax(
    annual_pnl: float,
    ltcg_rate: float,
    stcg_rate: float,
    ltcg_frac: float = TAX_LTCG_FRAC,
    stcg_frac: float = TAX_STCG_FRAC,
) -> float:
    """Apply 60/40 split to aggregate annual P&L. Negative P&L produces
    negative tax (a refund / deduction)."""
    return annual_pnl * (ltcg_frac * ltcg_rate + stcg_frac * stcg_rate)


# === Compatibility aliases for tests/test_friction.py (Robby's TDD spec) ===
compute_slippage_pct = slippage_pct_at_vix


# === Compatibility shim for tests/test_friction.py (Robby's TDD spec) ===
def compute_entry_cost(
    spread_credit: float,
    spread_width: float,
    vix: float,
    n_contracts: int = 1,
    gap_pct: float = 0.0,
):
    """Adapter that returns (entry_cost_dollars, skip_due_to_gap) tuple,
    matching Robby's spec interface. Wraps the per-share slippage
    helpers and adds round-trip commissions.

    entry_cost = slippage_per_share * 100 * n_contracts + commissions
    """
    skip = should_skip_entry_due_to_gap(gap_pct)
    if skip:
        return 0.0, True
    ba_per_share = max(0.10, 0.02 * float(vix))
    slip_dollars = slippage_dollars_per_share(ba_per_share, vix, gap_pct=gap_pct)
    cost = slip_dollars * 100.0 * n_contracts + round_trip_commissions(n_contracts) / 2.0
    return cost, False


def compute_exit_cost(
    exit_debit_per_share: float,
    spread_width: float,
    vix: float,
    n_contracts: int = 1,
):
    """Adapter for Robby's spec interface. Returns total exit cost in
    dollars (slippage + commissions)."""
    ba = max(0.10, 0.02 * float(vix))
    slip = slippage_dollars_per_share(ba, vix)
    capped_debit = min(exit_debit_per_share, float(spread_width))
    return (capped_debit + slip) * 100.0 * n_contracts + round_trip_commissions(n_contracts) / 2.0


compute_commission = commission_per_side


apply_section_1256_tax = section_1256_tax
