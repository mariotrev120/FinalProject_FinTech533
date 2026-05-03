"""
Unit tests for Layer 5 auto-resume in evaluate_halts.

These run with synthetic in-memory pandas Series only — no CSV loads, no
strat2.2.csv.gz, no engine roundtrip. They verify the state-machine
behavior of evaluate_halts in isolation.

Acceptance scope (from CAPSTONE_BACKLOG task A2):
  - When state enters halt and auto_resume_ready returns True next session,
    state transitions to active.
  - All four Layer 5 resume conditions tested individually and combined.
  - Fresh Layer 1 / Layer 4 triggers re-arm a halt regardless of prior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    RESUME_DD_RECOVERY_FRAC,
    RESUME_HYG_LQD_SD,
    RESUME_REALIZED_VOL_PCT,
    RESUME_TERM_BUFFER_PTS,
    RESUME_TERM_DAYS,
)
from src.strategy.halts import (
    auto_resume_ready,
    evaluate_halts,
    resume_drawdown_recovered,
    resume_hyg_lqd_within_band,
    resume_realized_vol_calmed,
    resume_term_recovered,
)


# --- Fixtures --------------------------------------------------------------

def _calm_term_history(days: int = RESUME_TERM_DAYS, n: int = 30) -> tuple[pd.Series, pd.Series]:
    """VIX3M − VIX > 2 pts for the last `days` closes. Earlier sessions
    are inverted to test that only the trailing window matters."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    vix3m = pd.Series(20.0, index=idx)
    vix = pd.Series(25.0, index=idx)  # inverted by default
    # Last `days` flip to non-inverted with 3pt buffer
    vix.iloc[-days:] = 17.0
    vix3m.iloc[-days:] = 20.0
    return vix3m, vix


def _calm_rv_history(rv5d_today: float = 12.0, n: int = 252) -> tuple[float, pd.Series]:
    """Trailing 252d distribution centered around 18% annualized; today's
    rv5d at 12% sits well below the 80th percentile."""
    rng = np.random.default_rng(42)
    hist = pd.Series(rng.normal(18.0, 4.0, n))
    return rv5d_today, hist


def _calm_credit_history(n: int = 252) -> pd.Series:
    """HYG-LQD spread anchored around 0; latest within 1 SD."""
    rng = np.random.default_rng(7)
    s = pd.Series(rng.normal(0.0, 0.5, n))
    s.iloc[-1] = 0.1  # within 1 SD of mean
    return s


# --- Individual resume condition tests ------------------------------------

def test_resume_term_recovered_pass():
    vix3m, vix = _calm_term_history()
    assert resume_term_recovered(vix3m, vix)


def test_resume_term_recovered_fail_short_history():
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    vix3m = pd.Series([20.0, 20.0], index=idx)
    vix = pd.Series([17.0, 17.0], index=idx)
    assert not resume_term_recovered(vix3m, vix)


def test_resume_term_recovered_fail_buffer_too_small():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    vix3m = pd.Series(20.0, index=idx)
    vix = pd.Series(19.0, index=idx)
    assert not resume_term_recovered(vix3m, vix)


def test_resume_realized_vol_calmed_pass():
    rv5d_today, hist = _calm_rv_history(rv5d_today=12.0)
    assert resume_realized_vol_calmed(rv5d_today, hist)


def test_resume_realized_vol_calmed_fail_above_pct():
    rv5d_today, hist = _calm_rv_history(rv5d_today=30.0)
    assert not resume_realized_vol_calmed(rv5d_today, hist)


def test_resume_realized_vol_calmed_fail_nan_today():
    _, hist = _calm_rv_history()
    assert not resume_realized_vol_calmed(float("nan"), hist)


def test_resume_hyg_lqd_within_band_pass():
    s = _calm_credit_history()
    assert resume_hyg_lqd_within_band(s)


def test_resume_hyg_lqd_within_band_fail_too_wide():
    s = _calm_credit_history()
    s.iloc[-1] = 5.0
    assert not resume_hyg_lqd_within_band(s)


def test_resume_drawdown_recovered_pass():
    assert resume_drawdown_recovered(0.01)


def test_resume_drawdown_recovered_fail():
    assert not resume_drawdown_recovered(0.10)


# --- Composite auto_resume_ready ------------------------------------------

def test_auto_resume_ready_all_four_pass():
    vix3m, vix = _calm_term_history()
    rv5d_today, rv_hist = _calm_rv_history()
    credit_hist = _calm_credit_history()
    assert auto_resume_ready(
        vix3m_history=vix3m,
        vix_history=vix,
        rv5d_today=rv5d_today,
        rv5d_history_252d=rv_hist,
        hyg_lqd_history=credit_hist,
        current_drawdown_frac=0.01,
    )


def test_auto_resume_ready_dd_failure_blocks():
    vix3m, vix = _calm_term_history()
    rv5d_today, rv_hist = _calm_rv_history()
    credit_hist = _calm_credit_history()
    assert not auto_resume_ready(
        vix3m_history=vix3m, vix_history=vix,
        rv5d_today=rv5d_today, rv5d_history_252d=rv_hist,
        hyg_lqd_history=credit_hist,
        current_drawdown_frac=0.10,
    )


def test_auto_resume_ready_rv_failure_blocks():
    vix3m, vix = _calm_term_history()
    _, rv_hist = _calm_rv_history()
    credit_hist = _calm_credit_history()
    assert not auto_resume_ready(
        vix3m_history=vix3m, vix_history=vix,
        rv5d_today=30.0, rv5d_history_252d=rv_hist,
        hyg_lqd_history=credit_hist,
        current_drawdown_frac=0.01,
    )


def test_auto_resume_ready_term_failure_blocks():
    rv5d_today, rv_hist = _calm_rv_history()
    credit_hist = _calm_credit_history()
    bad_vix3m = pd.Series([20.0] * 10)
    bad_vix = pd.Series([22.0] * 10)
    assert not auto_resume_ready(
        vix3m_history=bad_vix3m, vix_history=bad_vix,
        rv5d_today=rv5d_today, rv5d_history_252d=rv_hist,
        hyg_lqd_history=credit_hist,
        current_drawdown_frac=0.01,
    )


def test_auto_resume_ready_credit_failure_blocks():
    vix3m, vix = _calm_term_history()
    rv5d_today, rv_hist = _calm_rv_history()
    bad_credit = _calm_credit_history()
    bad_credit.iloc[-1] = 5.0
    assert not auto_resume_ready(
        vix3m_history=vix3m, vix_history=vix,
        rv5d_today=rv5d_today, rv5d_history_252d=rv_hist,
        hyg_lqd_history=bad_credit,
        current_drawdown_frac=0.01,
    )

# --- evaluate_halts state-machine tests -----------------------------------

def _baseline_call_kwargs():
    """A 'calm' market state where no fresh Layer 1 or Layer 4 trigger fires.
    Used as the base for state-machine transition tests."""
    vix3m, vix = _calm_term_history()
    rv5d_today, rv_hist = _calm_rv_history()
    credit_hist = _calm_credit_history()
    return dict(
        vix_today=18.0, vix_yday=18.5,            # no spike (< +40%)
        spx_today=4500.0, spx_yday=4490.0,        # +0.2% intraday move
        vix3m_today=20.0,                          # term not inverted hard
        vix3m_history=vix3m, vix_history=vix,
        hyg_lqd_history=credit_hist,
        prob_history=None,
        rolling_winrate=None,
        is_winrate_baseline=0.65,
        n_trades_in_rolling=0,
        underwater_days=0,
        current_drawdown_frac=0.01,                # not in DD halt
        rv5d_today=rv5d_today, rv5d_history_252d=rv_hist,
    )


def test_state_machine_active_to_active_no_change():
    kw = _baseline_call_kwargs()
    d = evaluate_halts(prior_state="active", **kw)
    assert d.state == "active"
    assert d.triggers == []


def test_state_machine_halt_to_active_when_resume_ready():
    """Was hard-halted yesterday; today calm and all 4 resume conditions
    pass → lift to active with auto_resumed trigger."""
    kw = _baseline_call_kwargs()
    d = evaluate_halts(prior_state="hard_halt", **kw)
    assert d.state == "active"
    assert "auto_resumed" in d.triggers


def test_state_machine_halt_holds_when_one_condition_fails():
    """Was drawdown-halted yesterday; today calm except DD still elevated
    → hold drawdown_halt state with awaiting_resume trigger."""
    kw = _baseline_call_kwargs()
    kw["current_drawdown_frac"] = 0.10  # still in DD; fresh Layer 4 also fires though
    # current_dd_frac=0.10 is below HALT_DD_DEPTH_FRAC=0.15 and underwater_days=0
    # so Layer 4 does NOT fire fresh. But Layer 5 dd_recovered also fails.
    d = evaluate_halts(prior_state="drawdown_halt", **kw)
    assert d.state == "drawdown_halt"
    assert "awaiting_resume" in d.triggers


def test_state_machine_fresh_hard_trigger_overrides_prior():
    """Fresh VIX spike re-arms hard_halt regardless of prior_state being
    a halt or active. Auto-resume cannot fire on a day with a fresh trigger."""
    kw = _baseline_call_kwargs()
    kw["vix_today"] = 30.0   # +50%+ from 18.5
    kw["vix_yday"] = 18.0
    d = evaluate_halts(prior_state="hard_halt", **kw)
    assert d.state == "hard_halt"
    assert "vix_spike" in d.triggers
    assert "auto_resumed" not in d.triggers


def test_state_machine_fresh_drawdown_overrides_active():
    """Was active; today drawdown threshold breached → drawdown_halt."""
    kw = _baseline_call_kwargs()
    kw["underwater_days"] = 100  # > 90
    kw["current_drawdown_frac"] = 0.20
    d = evaluate_halts(prior_state="active", **kw)
    assert d.state == "drawdown_halt"
    assert "drawdown" in d.triggers


def test_state_machine_default_prior_state_legacy_behavior():
    """When prior_state is omitted (defaults to 'active'), behavior matches
    the v1.5 stateless semantics: calm market returns 'active' without any
    Layer 5 evaluation."""
    kw = _baseline_call_kwargs()
    d = evaluate_halts(**kw)
    assert d.state == "active"
    assert d.triggers == []
