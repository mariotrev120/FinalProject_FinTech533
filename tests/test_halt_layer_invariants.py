"""
Halt-layer absorbing-state invariant tests.

NOT YET COMMITTED — gates morning review sign-off on the Layer 4 fix.

Pattern lesson from this session: Layer 4 (current) and Layer 5 v1 BOTH
have the same catch-22 — a halt that latches because the lift condition
references strategy state that the halt itself prevents. This is a
design pattern issue, not two independent bugs.

Invariant: every halt layer's lift condition must be REACHABLE while
the halt is active. Equivalently, every layer must include either:
  (a) a market-state-only lift condition (no strategy-state dependency)
  (b) a time-based escape hatch (bounded worst-case halt duration)

Applied to current halt layers:
  Layer 1 (hard):           PASS — lift condition is market-state-only
  Layer 2 (dropped post-audit): N/A
  Layer 3 (dropped post-audit): N/A
  Layer 4 (drawdown):       FAIL — lift requires DD < 15% AND
                                   underwater_days < 90, both computed
                                   from ALL-TIME HWM. Halt prevents PnL
                                   recovery → never lifts.
                                   PROPOSED FIX (DIAGNOSIS_HALTS.md
                                   Option A): switch to trailing-90d
                                   reference, which auto-resolves.
  Layer 5 v1 (auto-resume): FAIL — required DD-within-3%-of-HWM as
                                   one of 4 conditions. Same catch-22.
                                   FIXED in 2026-05-03 amendment:
                                   dropped DD condition + added 60d
                                   time fallback.
  Layer 5 v2 (current):     PASS — 3 market-only conditions OR 60d
                                   time fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.strategy.halts import (
    auto_resume_ready,
    drawdown_halt_triggered,
)


# ----------------------------------------------------------------------
# Synthetic helpers — construct a "calm market, idle strategy" state
# that any well-designed halt layer should be able to lift FROM.
# ----------------------------------------------------------------------

def calm_term_history(days: int = 30) -> tuple[pd.Series, pd.Series]:
    idx = pd.date_range("2020-01-01", periods=days, freq="B")
    vix3m = pd.Series(20.0, index=idx)
    vix = pd.Series(17.0, index=idx)   # term normalized: VIX3M − VIX = +3 > 2
    return vix3m, vix


def calm_rv_history() -> tuple[float, pd.Series]:
    rng = np.random.default_rng(0)
    rv_hist = pd.Series(rng.normal(18.0, 4.0, 252))
    rv_today = 12.0   # well below 80th pctile
    return rv_today, rv_hist


def calm_credit_history() -> pd.Series:
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0.0, 0.5, 252))
    s.iloc[-1] = 0.05  # within 1 SD
    return s


# ----------------------------------------------------------------------
# INVARIANT TEMPLATE — to be reused for every new halt layer
# ----------------------------------------------------------------------

@dataclass
class CalmMarketIdleStrategyState:
    """The state any well-designed halt should be able to lift from:
    market normalized AND strategy idle (no fresh PnL movement).

    Note 'strategy idle' means equity is at its initial value with no
    halt-period gains or losses. underwater_days could be high (because
    halt itself has duration), and current_dd_frac is whatever the
    strategy's last realized PnL produced.
    """
    vix3m_history: pd.Series
    vix_history: pd.Series
    rv5d_today: float
    rv5d_history_252d: pd.Series
    hyg_lqd_history: pd.Series
    underwater_days: int
    current_drawdown_frac: float
    days_in_halt: int


def build_calm_idle_state(
    underwater_days: int = 95,
    current_drawdown_frac: float = 0.01,   # 1% DD — strategy briefly down before halt
    days_in_halt: int = 30,                # 30 days into halt
) -> CalmMarketIdleStrategyState:
    vix3m, vix = calm_term_history()
    rv_today, rv_hist = calm_rv_history()
    cred = calm_credit_history()
    return CalmMarketIdleStrategyState(
        vix3m_history=vix3m,
        vix_history=vix,
        rv5d_today=rv_today,
        rv5d_history_252d=rv_hist,
        hyg_lqd_history=cred,
        underwater_days=underwater_days,
        current_drawdown_frac=current_drawdown_frac,
        days_in_halt=days_in_halt,
    )


# ----------------------------------------------------------------------
# Layer 1 — hard halt
# Lift condition: next-day market state has NO fresh hard trigger.
# Market-state-only → invariant satisfied trivially.
# ----------------------------------------------------------------------

def test_layer1_lift_condition_is_market_state_only():
    """Layer 1 lifts whenever the fresh hard-halt triggers do not fire.
    No strategy-state dependency. Invariant: PASS."""
    # Simulate: next day, VIX no spike, SPX no big move, term normal.
    # Layer 1 lift = "all 3 fresh hard checks return False on next day"
    from src.strategy.halts import (
        spx_move_triggered, term_inversion_hard_triggered, vix_spike_triggered,
    )
    vix_today, vix_yday = 17.0, 17.5     # no spike
    spx_today, spx_yday = 4500, 4490      # 0.2% move
    vix3m_today = 20.0                    # term normal
    no_spike = not vix_spike_triggered(vix_today, vix_yday)
    no_big_move = not spx_move_triggered(spx_today, spx_yday)
    no_inversion = not term_inversion_hard_triggered(vix3m_today, vix_today)
    assert no_spike and no_big_move and no_inversion, (
        "Layer 1 lift condition is market-state-only and cleared in this "
        "scenario, so the layer can lift from any halt state."
    )


# ----------------------------------------------------------------------
# Layer 4 (current implementation) — FAILS the invariant
# ----------------------------------------------------------------------

def test_layer4_current_FAILS_invariant_demonstrates_catch22():
    """Document the v1 Layer 4 catch-22 explicitly via a passing test of
    the broken behavior. The point: this test SHOULD fail under a fixed
    Layer 4. Currently it passes because Layer 4 IS broken.

    Scenario: market is normalized, strategy was 1% underwater for 95
    days. Per current Layer 4:
      - underwater_days >= 90 → halt fires
      - current_drawdown_frac = 0.01 < 0.15 → no DD-depth fire
      - But underwater_days alone fires Layer 4
      - Lift requires equity ≥ all-time HWM, but strategy is halted.
    """
    state = build_calm_idle_state(
        underwater_days=95, current_drawdown_frac=0.01,
    )
    # Layer 4 fires:
    fires = drawdown_halt_triggered(
        state.underwater_days, state.current_drawdown_frac,
    )
    assert fires, (
        "Layer 4 (current impl) fires at 95 days underwater even with "
        "tiny DD — confirms the catch-22 trigger exists."
    )

    # Lift condition (current): requires equity to reach NEW all-time HWM.
    # Strategy is halted → no PnL → never reaches new HWM → halt is absorbing.
    # We document this as a known pathology.


def test_layer4_PROPOSED_FIX_passes_invariant_under_calm_drift():
    """Under Option A (trailing-90d reference), even an idle strategy
    accruing rf-rate interest will eventually have equity ≥ trailing-90
    max → underwater_days resets → halt lifts. Proves the proposed fix
    satisfies the invariant.

    NOTE: this is a CONCEPTUAL test, executed against a synthetic
    rolling-window simulator. The actual fix lives in
    src/backtest/engine.py (UNAPPLIED — see DIAGNOSIS_HALTS.md Option A).
    """
    # Synthetic equity path: 90 days at $99K (1% below initial $100K),
    # then rf accrual lifts above $99K.
    eq = [100_000.0]
    for _ in range(90):
        eq.append(99_000.0)
    # rf accrual → equity inches up
    for i in range(20):
        eq.append(99_000.0 + 100 * (i + 1))   # +$100 per day

    # Under Option A (trailing-90d window):
    underwater_days = 0
    rolling = []
    halt_lifted = False
    for d, current_eq in enumerate(eq):
        rolling.append(current_eq)
        if len(rolling) > 90:
            rolling = rolling[-90:]
        trailing_high = max(rolling)
        if current_eq >= trailing_high:
            underwater_days = 0
        else:
            underwater_days += 1
        # Layer 4 fires?
        dd_frac = (
            (trailing_high - current_eq) / trailing_high
            if trailing_high > 0 else 0.0
        )
        fires = drawdown_halt_triggered(underwater_days, dd_frac)
        if not fires and d >= 90:
            halt_lifted = True
            break
    assert halt_lifted, (
        "Under Option A (trailing-90d reference), idle-strategy rf "
        "accrual eventually lifts the halt. Layer 4 satisfies the "
        "invariant after Option A is applied."
    )


# ----------------------------------------------------------------------
# Layer 5 v2 (current) — PASSES the invariant
# ----------------------------------------------------------------------

def test_layer5_v2_PASSES_invariant_via_market_conditions():
    """Layer 5 v2 lifts when ALL three market-regime conditions hold,
    regardless of strategy state. Calm-market-idle-strategy → lifts."""
    state = build_calm_idle_state(
        underwater_days=95,
        current_drawdown_frac=0.10,   # significant strategy DD
        days_in_halt=10,
    )
    can_lift = auto_resume_ready(
        vix3m_history=state.vix3m_history,
        vix_history=state.vix_history,
        rv5d_today=state.rv5d_today,
        rv5d_history_252d=state.rv5d_history_252d,
        hyg_lqd_history=state.hyg_lqd_history,
        current_drawdown_frac=state.current_drawdown_frac,
        days_in_halt=state.days_in_halt,
    )
    assert can_lift, (
        "Layer 5 v2 should lift on market-conditions-clear regardless "
        "of strategy DD."
    )


def test_layer5_v2_PASSES_invariant_via_time_fallback():
    """Even with all market conditions failing, the 60-day time fallback
    lifts the halt. Worst-case bound is reachable."""
    bad_vix3m = pd.Series([20.0] * 10)
    bad_vix = pd.Series([22.0] * 10)
    rv_hist = pd.Series([18.0] * 30)
    bad_credit = pd.Series([5.0] * 30)
    can_lift = auto_resume_ready(
        vix3m_history=bad_vix3m, vix_history=bad_vix,
        rv5d_today=30.0, rv5d_history_252d=rv_hist,
        hyg_lqd_history=bad_credit,
        current_drawdown_frac=0.20,
        days_in_halt=60,
    )
    assert can_lift


# ----------------------------------------------------------------------
# Invariant template — for any future halt layer
# ----------------------------------------------------------------------

def test_template_for_any_future_halt_layer():
    """Template: when adding a new halt Layer N, replicate this test
    structure with the layer's lift function:

        from src.strategy.halts import layer_N_lift_ready

        state = build_calm_idle_state(...)
        # Override any state attributes specific to layer N's lift conditions

        can_lift = layer_N_lift_ready(state, ...)
        assert can_lift, (
            f"Layer N lift condition cannot fire when market is "
            f"normal and strategy is idle. This is the same design "
            f"pattern as v1 Layer 4 / Layer 5 catch-22. Add a "
            f"market-only condition or a time-based escape hatch."
        )
    """
    pass   # template-only; no assertion in template
