"""
Tests for src/strategy/halts.py — the five-layer halt framework.

Expected interface
------------------
src.strategy.halts.HaltFramework

    __init__(
        self,
        in_sample_win_rate: float,      # baseline win rate from 2010-2017 IS period
        in_sample_sharpe: float,        # baseline Sharpe from IS period
        se_multiplier: float = 2.0,     # Hoeffding SE threshold for Layer 3
        seed: int = 42,
    )

    evaluate(
        self,
        market_state: dict,
        trade_history: list[dict],
    ) -> str
        Returns one of: "active", "soft_halt", "hard_halt"

    check_resume(
        self,
        market_state: dict,
        trade_history: list[dict],
    ) -> bool
        Returns True only when ALL four Layer 5 conditions are simultaneously met.

    reset(self) -> None
        Clears any accumulated streak counters or state.

market_state keys (used across layers):
    vix               : float   — current VIX level
    vix_prev_close    : float   — prior day VIX close (for spike %)
    vix3m             : float   — VIX3M level
    spx_intraday_pct  : float   — SPX intraday move as fraction (e.g. -0.06)
    hyg_lqd_spread_zscore : float — HYG-LQD spread vs 252-day rolling mean (in SD)
    model_probability : float   — latest XGBoost calibrated probability
    consecutive_low_prob_days : int — days in a row with model_probability < 0.55
    consecutive_vix_inversion_days : int — days in a row with VIX3M < VIX
    vix3m_above_vix_by2_consecutive : int — days in a row with VIX3M > VIX + 2 (for resume)
    realized_vol_5d   : float   — 5-day realized vol (annualized)
    realized_vol_252d_pct80 : float — 80th percentile of trailing 252-day realized vol
    hwm_drawdown_pct  : float   — drawdown from high-water mark (negative fraction)
    underwater_days   : int     — consecutive trading days below HWM

trade_history keys (each trade dict):
    outcome : int  — 1=win, 0=loss
    pnl     : float
"""
import pytest

from src.strategy.halts import HaltFramework


# ===========================================================================
# Fixtures — pre-built HaltFramework instances
# ===========================================================================

@pytest.fixture
def halt_framework():
    """Standard halt framework with realistic in-sample baseline."""
    return HaltFramework(
        in_sample_win_rate=0.75,
        in_sample_sharpe=0.90,
        se_multiplier=2.0,
        seed=42,
    )


# ===========================================================================
# Layer 1 — Hard halt (immediate, all positions closed)
# ===========================================================================

class TestLayer1HardHalt:
    def test_vix_spike_40pct_triggers_hard_halt(self, halt_framework, vix_spike_state, strong_trade_history):
        """VIX spikes 50% intraday from prior close → hard halt."""
        result = halt_framework.evaluate(vix_spike_state, strong_trade_history)
        assert result == "hard_halt", (
            f"A 50% VIX spike should trigger hard_halt, got '{result}'"
        )

    def test_vix_spike_below_40pct_does_not_hard_halt(self, halt_framework, normal_market_state, strong_trade_history):
        """VIX spike of 25% (below 40% threshold) must not trigger hard halt."""
        state = dict(normal_market_state)
        state["vix"] = 19.5
        state["vix_prev_close"] = 15.6  # +25% — below threshold
        result = halt_framework.evaluate(state, strong_trade_history)
        assert result != "hard_halt", (
            "A 25% VIX spike (below 40% threshold) should not trigger hard_halt"
        )

    def test_vix_spike_exactly_at_threshold_is_consistent(self, halt_framework, strong_trade_history):
        """VIX spike of exactly 40% — boundary behavior must be deterministic."""
        state = {
            "vix": 21.0,
            "vix_prev_close": 15.0,         # exactly +40%
            "vix3m": 22.0,
            "spx_intraday_pct": 0.001,
            "hyg_lqd_spread_zscore": 0.5,
            "model_probability": 0.62,
            "consecutive_low_prob_days": 0,
            "consecutive_vix_inversion_days": 0,
            "vix3m_above_vix_by2_consecutive": 0,
            "realized_vol_5d": 0.10,
            "realized_vol_252d_pct80": 0.14,
            "hwm_drawdown_pct": -0.02,
            "underwater_days": 3,
        }
        result1 = halt_framework.evaluate(state, strong_trade_history)
        result2 = halt_framework.evaluate(state, strong_trade_history)
        assert result1 == result2, "Boundary evaluation must be deterministic"

    def test_spx_5pct_intraday_drop_triggers_hard_halt(self, halt_framework, spx_crash_state, strong_trade_history):
        """SPX drops 6% intraday — hard halt."""
        result = halt_framework.evaluate(spx_crash_state, strong_trade_history)
        assert result == "hard_halt", (
            f"A -6% SPX move should trigger hard_halt, got '{result}'"
        )

    def test_spx_5pct_intraday_rally_also_triggers_hard_halt(self, halt_framework, strong_trade_history):
        """SPX +6% rally (either direction triggers) → hard halt."""
        state = {
            "vix": 16.0, "vix_prev_close": 15.5, "vix3m": 18.0,
            "spx_intraday_pct": 0.06,           # +6% rally
            "hyg_lqd_spread_zscore": 0.5,
            "model_probability": 0.62,
            "consecutive_low_prob_days": 0, "consecutive_vix_inversion_days": 0,
            "vix3m_above_vix_by2_consecutive": 0,
            "realized_vol_5d": 0.10, "realized_vol_252d_pct80": 0.14,
            "hwm_drawdown_pct": -0.02, "underwater_days": 3,
        }
        result = halt_framework.evaluate(state, strong_trade_history)
        assert result == "hard_halt", "SPX +6% intraday move should also trigger hard_halt"

    def test_spx_move_below_5pct_does_not_hard_halt(self, halt_framework, strong_trade_history):
        """SPX -4.5% intraday — below 5% threshold, should not trigger hard halt."""
        state = {
            "vix": 16.0, "vix_prev_close": 15.5, "vix3m": 18.0,
            "spx_intraday_pct": -0.045,
            "hyg_lqd_spread_zscore": 0.5,
            "model_probability": 0.62,
            "consecutive_low_prob_days": 0, "consecutive_vix_inversion_days": 0,
            "vix3m_above_vix_by2_consecutive": 0,
            "realized_vol_5d": 0.10, "realized_vol_252d_pct80": 0.14,
            "hwm_drawdown_pct": -0.02, "underwater_days": 3,
        }
        result = halt_framework.evaluate(state, strong_trade_history)
        assert result != "hard_halt", (
            "A -4.5% SPX move (below 5% threshold) must not trigger hard_halt"
        )

    def test_severe_term_structure_inversion_triggers_hard_halt(self, halt_framework, term_structure_inversion_state, strong_trade_history):
        """VIX > VIX3M by 3 points → hard halt (threshold is 2 points)."""
        result = halt_framework.evaluate(term_structure_inversion_state, strong_trade_history)
        assert result == "hard_halt", (
            f"VIX > VIX3M by 3pts should trigger hard_halt, got '{result}'"
        )

    def test_mild_term_inversion_does_not_hard_halt(self, halt_framework, strong_trade_history):
        """VIX > VIX3M by 1 point — below the 2-point hard halt threshold."""
        state = {
            "vix": 21.0, "vix_prev_close": 20.5, "vix3m": 20.0,  # only 1pt inversion
            "spx_intraday_pct": 0.001,
            "hyg_lqd_spread_zscore": 0.5,
            "model_probability": 0.62,
            "consecutive_low_prob_days": 0,
            "consecutive_vix_inversion_days": 1,
            "vix3m_above_vix_by2_consecutive": 0,
            "realized_vol_5d": 0.10, "realized_vol_252d_pct80": 0.14,
            "hwm_drawdown_pct": -0.02, "underwater_days": 3,
        }
        result = halt_framework.evaluate(state, strong_trade_history)
        assert result != "hard_halt", (
            "VIX > VIX3M by only 1 point should not trigger hard_halt"
        )


# ===========================================================================
# Layer 2 — Soft halt (no new entries; existing trades managed to natural exit)
# ===========================================================================

class TestLayer2SoftHalt:
    def test_credit_stress_triggers_soft_halt(self, halt_framework, soft_halt_credit_stress_state, strong_trade_history):
        """HYG-LQD spread > 2 SD → soft halt."""
        result = halt_framework.evaluate(soft_halt_credit_stress_state, strong_trade_history)
        assert result == "soft_halt", (
            f"HYG-LQD spread > 2SD should trigger soft_halt, got '{result}'"
        )

    def test_credit_stress_below_2sd_does_not_soft_halt(self, halt_framework, normal_market_state, strong_trade_history):
        """HYG-LQD spread at 1.8 SD — below threshold, no soft halt."""
        state = dict(normal_market_state)
        state["hyg_lqd_spread_zscore"] = 1.8
        result = halt_framework.evaluate(state, strong_trade_history)
        assert result != "soft_halt"

    def test_5_consecutive_low_prob_days_triggers_soft_halt(self, halt_framework, soft_halt_low_prob_state, strong_trade_history):
        """5 consecutive days with model_probability < 0.55 → soft halt."""
        result = halt_framework.evaluate(soft_halt_low_prob_state, strong_trade_history)
        assert result == "soft_halt", (
            f"5 consecutive low-prob days should trigger soft_halt, got '{result}'"
        )

    def test_4_consecutive_low_prob_days_does_not_soft_halt(self, halt_framework, strong_trade_history):
        """4 consecutive low-prob days — one short of threshold, no soft halt."""
        state = {
            "vix": 16.0, "vix_prev_close": 15.5, "vix3m": 18.0,
            "spx_intraday_pct": 0.001,
            "hyg_lqd_spread_zscore": 0.5,
            "model_probability": 0.52,
            "consecutive_low_prob_days": 4,  # one day short
            "consecutive_vix_inversion_days": 0,
            "vix3m_above_vix_by2_consecutive": 0,
            "realized_vol_5d": 0.10, "realized_vol_252d_pct80": 0.14,
            "hwm_drawdown_pct": -0.02, "underwater_days": 3,
        }
        result = halt_framework.evaluate(state, strong_trade_history)
        assert result != "soft_halt", (
            "4 consecutive low-prob days (below 5-day threshold) should not trigger soft_halt"
        )

    def test_vix_inversion_for_2_consecutive_closes_triggers_soft_halt(self, halt_framework, strong_trade_history):
        """VIX3M < VIX for 2 consecutive closes → Layer 2 soft halt."""
        state = {
            "vix": 20.0, "vix_prev_close": 19.5, "vix3m": 19.0,  # VIX3M < VIX
            "spx_intraday_pct": 0.001,
            "hyg_lqd_spread_zscore": 0.8,
            "model_probability": 0.60,
            "consecutive_low_prob_days": 0,
            "consecutive_vix_inversion_days": 2,   # 2 consecutive closes
            "vix3m_above_vix_by2_consecutive": 0,
            "realized_vol_5d": 0.12, "realized_vol_252d_pct80": 0.15,
            "hwm_drawdown_pct": -0.03, "underwater_days": 5,
        }
        result = halt_framework.evaluate(state, strong_trade_history)
        assert result == "soft_halt", (
            f"VIX3M < VIX for 2 consecutive closes should trigger soft_halt, got '{result}'"
        )

    def test_vix_inversion_for_1_day_does_not_soft_halt(self, halt_framework, strong_trade_history):
        """VIX3M < VIX for only 1 close — not yet persistent enough for soft halt."""
        state = {
            "vix": 20.0, "vix_prev_close": 19.5, "vix3m": 19.0,
            "spx_intraday_pct": 0.001,
            "hyg_lqd_spread_zscore": 0.8,
            "model_probability": 0.60,
            "consecutive_low_prob_days": 0,
            "consecutive_vix_inversion_days": 1,   # only 1 day — below threshold
            "vix3m_above_vix_by2_consecutive": 0,
            "realized_vol_5d": 0.12, "realized_vol_252d_pct80": 0.15,
            "hwm_drawdown_pct": -0.03, "underwater_days": 5,
        }
        result = halt_framework.evaluate(state, strong_trade_history)
        assert result != "soft_halt", (
            "VIX inversion for only 1 day should not trigger soft_halt"
        )

    def test_soft_halt_does_not_close_existing_positions(self, halt_framework, soft_halt_credit_stress_state, strong_trade_history):
        """
        Soft halt must NOT force-close existing positions.
        The framework should signal 'soft_halt' (block new entries only),
        not change any trade state.
        evaluate() must return "soft_halt" and not raise or modify trade_history.
        """
        original_len = len(strong_trade_history)
        result = halt_framework.evaluate(soft_halt_credit_stress_state, strong_trade_history)
        assert result == "soft_halt"
        assert len(strong_trade_history) == original_len, (
            "Soft halt must not modify trade_history — existing trades run to natural exit"
        )


# ===========================================================================
# Layer 3 — Slow halt (Hoeffding-driven, statistical)
# ===========================================================================

class TestLayer3HoeffdingHalt:
    def test_low_win_rate_triggers_slow_halt(self, halt_framework, normal_market_state, trade_history_with_losses):
        """
        60 trades with 55% win rate when in-sample baseline is 75%.
        The Hoeffding bound should reject the in-sample hypothesis → halt.
        """
        result = halt_framework.evaluate(normal_market_state, trade_history_with_losses)
        assert result in ("soft_halt", "hard_halt"), (
            "Win rate far below in-sample baseline should trigger a halt via Hoeffding bound"
        )

    def test_strong_win_rate_does_not_trigger_halt(self, halt_framework, normal_market_state, strong_trade_history):
        """75% win rate matching in-sample baseline → strategy active, no halt."""
        result = halt_framework.evaluate(normal_market_state, strong_trade_history)
        assert result == "active", (
            f"Win rate matching in-sample baseline should leave strategy active, got '{result}'"
        )

    def test_insufficient_trades_for_hoeffding_skips_layer3(self, halt_framework, normal_market_state):
        """
        Fewer than ~30 trades is insufficient for the Hoeffding bound to be meaningful.
        With only 5 trades and a poor win rate, Layer 3 should NOT trigger (not enough data).
        Layer 3 trigger requires a minimum trade count.
        """
        small_history = [
            {"outcome": 0, "pnl": -300} for _ in range(5)  # 5 losses — tiny sample
        ]
        result = halt_framework.evaluate(normal_market_state, small_history)
        # With only 5 trades, the Hoeffding bound cannot be statistically meaningful
        # The result should not be "hard_halt" based solely on Layer 3
        # (Other layers may still trigger if other conditions are met, but with benign state
        # and tiny history, the result should be "active")
        assert result == "active", (
            "Layer 3 should not trigger on only 5 trades — sample too small for Hoeffding"
        )


# ===========================================================================
# Layer 4 — Drawdown halt
# ===========================================================================

class TestLayer4DrawdownHalt:
    def test_drawdown_depth_triggers_halt(self, halt_framework, drawdown_halt_state, strong_trade_history):
        """Portfolio 16% below HWM → drawdown depth halt."""
        result = halt_framework.evaluate(drawdown_halt_state, strong_trade_history)
        assert result in ("soft_halt", "hard_halt"), (
            f"16% drawdown should trigger a halt, got '{result}'"
        )

    def test_drawdown_below_threshold_does_not_halt(self, halt_framework, normal_market_state, strong_trade_history):
        """14% drawdown — below 15% threshold, no drawdown halt."""
        state = dict(normal_market_state)
        state["hwm_drawdown_pct"] = -0.14
        state["underwater_days"] = 40
        result = halt_framework.evaluate(state, strong_trade_history)
        assert result == "active", (
            "A 14% drawdown (below 15% threshold) should not trigger a halt"
        )

    def test_drawdown_duration_triggers_halt(self, halt_framework, duration_halt_state, strong_trade_history):
        """Portfolio underwater for 92 trading days → duration halt."""
        result = halt_framework.evaluate(duration_halt_state, strong_trade_history)
        assert result in ("soft_halt", "hard_halt"), (
            f"92 days underwater should trigger a halt, got '{result}'"
        )

    def test_drawdown_duration_below_threshold_does_not_halt(self, halt_framework, normal_market_state, strong_trade_history):
        """85 trading days underwater — below 90-day threshold, no duration halt."""
        state = dict(normal_market_state)
        state["underwater_days"] = 85
        state["hwm_drawdown_pct"] = -0.08
        result = halt_framework.evaluate(state, strong_trade_history)
        assert result == "active", (
            "85 days underwater (below 90-day threshold) should not trigger a halt"
        )

    def test_exact_depth_boundary_is_consistent(self, halt_framework, strong_trade_history):
        """At exactly -15% drawdown, behavior is deterministic."""
        state = {
            "vix": 16.0, "vix_prev_close": 15.5, "vix3m": 18.0,
            "spx_intraday_pct": 0.001,
            "hyg_lqd_spread_zscore": 0.5,
            "model_probability": 0.62,
            "consecutive_low_prob_days": 0, "consecutive_vix_inversion_days": 0,
            "vix3m_above_vix_by2_consecutive": 0,
            "realized_vol_5d": 0.10, "realized_vol_252d_pct80": 0.14,
            "hwm_drawdown_pct": -0.15,   # exactly at threshold
            "underwater_days": 30,
        }
        r1 = halt_framework.evaluate(state, strong_trade_history)
        r2 = halt_framework.evaluate(state, strong_trade_history)
        assert r1 == r2, "Evaluation at exact drawdown boundary must be deterministic"


# ===========================================================================
# Layer 5 — Auto-resume
# ===========================================================================

class TestLayer5AutoResume:
    def test_all_four_conditions_required_for_resume(self, halt_framework, full_resume_state, strong_trade_history):
        """Strategy resumes ONLY when all four conditions are simultaneously satisfied."""
        assert halt_framework.check_resume(full_resume_state, strong_trade_history), (
            "All four Layer 5 conditions met — strategy should resume"
        )

    @pytest.mark.parametrize("missing_condition", [
        "vix3m_above_vix_by2_consecutive",   # needs ≥5 consecutive days
        "realized_vol_5d_ok",                 # realized vol < 80th pct
        "hyg_lqd_spread_ok",                  # spread within 1 SD
        "drawdown_recovered",                 # within 3% of HWM
    ])
    def test_missing_one_condition_prevents_resume(self, halt_framework, full_resume_state, strong_trade_history, missing_condition):
        """
        Removing any single resume condition must prevent resumption.
        Tests all four combinations of "missing exactly one condition."
        """
        state = dict(full_resume_state)

        # Break each condition in turn
        if missing_condition == "vix3m_above_vix_by2_consecutive":
            state["vix3m_above_vix_by2_consecutive"] = 3  # only 3 days, need 5
        elif missing_condition == "realized_vol_5d_ok":
            state["realized_vol_5d"] = 0.20              # above 80th pct (0.14)
        elif missing_condition == "hyg_lqd_spread_ok":
            state["hyg_lqd_spread_zscore"] = 1.5          # above 1 SD threshold
        elif missing_condition == "drawdown_recovered":
            state["hwm_drawdown_pct"] = -0.04             # still 4% below HWM (need <3%)

        assert not halt_framework.check_resume(state, strong_trade_history), (
            f"Resume should be blocked when condition '{missing_condition}' is not met"
        )

    def test_no_resume_during_dead_cat_bounce(self, halt_framework, strong_trade_history):
        """
        Market briefly satisfies 3 of 4 conditions (classic dead-cat bounce pattern).
        All four are still required simultaneously.
        """
        partial_recovery = {
            "vix": 15.0, "vix_prev_close": 14.5, "vix3m": 18.0,
            "vix3m_above_vix_by2_consecutive": 5,    # ✓ term structure recovered
            "spx_intraday_pct": 0.001,
            "hyg_lqd_spread_zscore": 0.9,            # ✓ credit stress resolved (within 1 SD)
            "model_probability": 0.65,
            "consecutive_low_prob_days": 0, "consecutive_vix_inversion_days": 0,
            "realized_vol_5d": 0.09,
            "realized_vol_252d_pct80": 0.14,         # ✓ vol regime calmed
            "hwm_drawdown_pct": -0.05,               # ✗ still 5% below HWM (need <3%)
            "underwater_days": 0,
        }
        assert not halt_framework.check_resume(partial_recovery, strong_trade_history), (
            "3-of-4 resume conditions should NOT trigger resume (all four required)"
        )

    def test_persistent_term_structure_recovery_required(self, halt_framework, strong_trade_history):
        """
        VIX3M > VIX by 2+ points for only 4 days — one short of the 5-day requirement.
        """
        state = {
            "vix": 14.0, "vix_prev_close": 13.8, "vix3m": 17.0,
            "vix3m_above_vix_by2_consecutive": 4,    # only 4 days, need 5
            "spx_intraday_pct": 0.001,
            "hyg_lqd_spread_zscore": 0.8,
            "model_probability": 0.63,
            "consecutive_low_prob_days": 0, "consecutive_vix_inversion_days": 0,
            "realized_vol_5d": 0.08,
            "realized_vol_252d_pct80": 0.14,
            "hwm_drawdown_pct": -0.025,
            "underwater_days": 0,
        }
        assert not halt_framework.check_resume(state, strong_trade_history), (
            "4 days of term structure recovery (need 5) should not trigger resume"
        )


# ===========================================================================
# State machine correctness
# ===========================================================================

class TestHaltStateMachine:
    def test_hard_halt_overrides_soft_halt(self, halt_framework, strong_trade_history):
        """
        If a soft halt condition AND a hard halt condition are both present,
        hard halt must win. The most severe classification applies.
        """
        combined_state = {
            "vix": 30.0,
            "vix_prev_close": 20.0,         # +50% spike → hard halt (Layer 1)
            "vix3m": 22.0,
            "spx_intraday_pct": 0.001,
            "hyg_lqd_spread_zscore": 2.5,   # also triggers soft halt (Layer 2)
            "model_probability": 0.62,
            "consecutive_low_prob_days": 0, "consecutive_vix_inversion_days": 0,
            "vix3m_above_vix_by2_consecutive": 0,
            "realized_vol_5d": 0.10, "realized_vol_252d_pct80": 0.14,
            "hwm_drawdown_pct": -0.02, "underwater_days": 3,
        }
        result = halt_framework.evaluate(combined_state, strong_trade_history)
        assert result == "hard_halt", (
            "When both hard and soft halt conditions are met, hard_halt must win"
        )

    def test_normal_conditions_return_active(self, halt_framework, normal_market_state, strong_trade_history):
        """No halt conditions triggered → strategy is 'active'."""
        result = halt_framework.evaluate(normal_market_state, strong_trade_history)
        assert result == "active"

    def test_evaluate_is_deterministic(self, halt_framework, normal_market_state, strong_trade_history):
        """Same inputs always produce the same output — no hidden random state."""
        r1 = halt_framework.evaluate(normal_market_state, strong_trade_history)
        r2 = halt_framework.evaluate(normal_market_state, strong_trade_history)
        assert r1 == r2, "evaluate() must be deterministic for identical inputs"

    def test_check_resume_returns_false_when_not_halted(self, halt_framework, full_resume_state, strong_trade_history):
        """
        check_resume checks Layer 5 conditions only.
        Even if all four conditions are met, the function simply reports the condition status.
        It must be callable without prior halt state.
        """
        # This tests that check_resume doesn't raise when called on a fresh framework
        result = halt_framework.check_resume(full_resume_state, strong_trade_history)
        assert isinstance(result, bool), "check_resume must return a bool"

    def test_reset_clears_streak_counters(self, halt_framework, strong_trade_history):
        """
        After reset(), any streak-based halt (e.g., Layer 2 consecutive days)
        should not carry state from the previous evaluation.
        """
        # Trigger a soft halt to build up state
        soft_state = {
            "vix": 16.0, "vix_prev_close": 15.5, "vix3m": 18.0,
            "spx_intraday_pct": 0.001,
            "hyg_lqd_spread_zscore": 0.5,
            "model_probability": 0.52,
            "consecutive_low_prob_days": 5,
            "consecutive_vix_inversion_days": 2,
            "vix3m_above_vix_by2_consecutive": 0,
            "realized_vol_5d": 0.10, "realized_vol_252d_pct80": 0.14,
            "hwm_drawdown_pct": -0.02, "underwater_days": 3,
        }
        halt_framework.evaluate(soft_state, strong_trade_history)
        halt_framework.reset()

        # After reset, normal state should return "active"
        result_after_reset = halt_framework.evaluate(
            {
                "vix": 16.0, "vix_prev_close": 15.5, "vix3m": 18.0,
                "spx_intraday_pct": 0.001,
                "hyg_lqd_spread_zscore": 0.5,
                "model_probability": 0.62,
                "consecutive_low_prob_days": 0,
                "consecutive_vix_inversion_days": 0,
                "vix3m_above_vix_by2_consecutive": 0,
                "realized_vol_5d": 0.10, "realized_vol_252d_pct80": 0.14,
                "hwm_drawdown_pct": -0.01, "underwater_days": 0,
            },
            strong_trade_history,
        )
        assert result_after_reset == "active", (
            "After reset(), a benign market state should return 'active' — reset did not clear state"
        )

    def test_evaluate_returns_valid_string(self, halt_framework, normal_market_state, strong_trade_history):
        """evaluate() must return one of the three documented strings."""
        valid_states = {"active", "soft_halt", "hard_halt"}
        result = halt_framework.evaluate(normal_market_state, strong_trade_history)
        assert result in valid_states, (
            f"evaluate() returned '{result}', which is not in {valid_states}"
        )

    def test_consecutive_halt_resume_does_not_corrupt_state(self, halt_framework, strong_trade_history):
        """
        Halt → resume → halt again: the second halt should fire correctly
        even after a resume cycle. Checks for state corruption.
        """
        halt_state = {
            "vix": 30.0, "vix_prev_close": 20.0,  # +50% spike
            "vix3m": 22.0, "spx_intraday_pct": 0.001,
            "hyg_lqd_spread_zscore": 0.5, "model_probability": 0.62,
            "consecutive_low_prob_days": 0, "consecutive_vix_inversion_days": 0,
            "vix3m_above_vix_by2_consecutive": 0,
            "realized_vol_5d": 0.10, "realized_vol_252d_pct80": 0.14,
            "hwm_drawdown_pct": -0.02, "underwater_days": 3,
        }
        r1 = halt_framework.evaluate(halt_state, strong_trade_history)
        assert r1 == "hard_halt"

        halt_framework.reset()

        r2 = halt_framework.evaluate(halt_state, strong_trade_history)
        assert r2 == "hard_halt", (
            "Second halt after reset should produce the same result as the first"
        )
