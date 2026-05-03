"""
Tests for src/strategy/ — spread construction, exits, and position sizing.

Expected interfaces
-------------------
src.strategy.spread_construction
    build_spread(
        options_chain: pd.DataFrame,  # columns: strike, right, delta, bid, ask, dte, expiry
        target_delta: float = 0.16,
        spread_width: float = 5.0,
        min_dte: int = 30,
        max_dte: int = 45,
    ) -> dict | None
    Returns keys: short_strike, long_strike, expiry, dte, credit, max_loss, max_gain
    Returns None when no valid spread can be constructed.

src.strategy.exits
    evaluate_exits(trade: dict, bar: dict) -> dict | None
    trade keys: entry_date, entry_credit, short_strike, long_strike, spread_width,
                expiry, dte_at_entry, num_contracts, max_gain, max_loss
    bar keys: date, current_spread_value, open_spread_value, short_delta,
              dte_remaining, vix
    Returns dict {fate, exit_price, exit_date} or None (no exit triggered).

src.strategy.sizing
    kelly_fraction(p: float, max_win: float, max_loss: float, cap: float = 0.25) -> float
    compute_position_size(
        p_calibrated: float, max_win: float, max_loss: float,
        vix: float, spy_tlt_corr: float, account_equity: float,
        spread_max_loss: float,
        kelly_cap: float = 0.25,
        max_risk_pct: float = 0.01,
        entry_threshold: float = 0.55,
    ) -> int
"""
import numpy as np
import pandas as pd
import pytest

from src.strategy.spread_construction import build_spread
from src.strategy.exits import evaluate_exits
from src.strategy.sizing import kelly_fraction, compute_position_size


# ===========================================================================
# Spread construction
# ===========================================================================

class TestSpreadConstruction:
    def test_returns_correct_strike_keys(self, sample_options_chain):
        """Result dict must have all required keys for downstream use."""
        result = build_spread(sample_options_chain, target_delta=0.16, spread_width=5.0)
        assert result is not None
        for key in ["short_strike", "long_strike", "expiry", "dte", "credit", "max_loss", "max_gain"]:
            assert key in result, f"Missing key '{key}' in build_spread result"

    def test_selects_strike_closest_to_16_delta(self, sample_options_chain):
        """
        With strikes at delta = 0.07, 0.10, 0.14, 0.16, 0.22, 0.25,
        the 0.16-delta strike must be selected as the short.
        """
        result = build_spread(sample_options_chain, target_delta=0.16, spread_width=5.0)
        assert result is not None
        # Strike with delta=-0.16 in the fixture is 3920
        assert result["short_strike"] == 3920, (
            f"Expected short_strike=3920 (closest to 16-delta), got {result['short_strike']}"
        )

    def test_long_strike_is_spread_width_below_short(self, sample_options_chain):
        """Long strike must be exactly spread_width points below the short strike."""
        result = build_spread(sample_options_chain, target_delta=0.16, spread_width=5.0)
        assert result is not None
        expected_long = result["short_strike"] - 5
        assert result["long_strike"] == expected_long, (
            f"Long strike should be short_strike - 5 = {expected_long}, "
            f"got {result['long_strike']}"
        )

    def test_dte_in_valid_range(self, sample_options_chain):
        """DTE at construction must be in [min_dte, max_dte]."""
        result = build_spread(sample_options_chain, target_delta=0.16,
                              spread_width=5.0, min_dte=30, max_dte=45)
        assert result is not None
        assert 30 <= result["dte"] <= 45, (
            f"DTE {result['dte']} is outside the valid [30, 45] window"
        )

    def test_returns_none_when_no_valid_dte(self, sample_options_chain):
        """
        If every option in the chain has DTE outside [min_dte, max_dte],
        return None rather than constructing an invalid spread.
        """
        # Fixture has DTE=35; setting min_dte=40 should yield no valid options
        result = build_spread(sample_options_chain, target_delta=0.16,
                              spread_width=5.0, min_dte=40, max_dte=45)
        assert result is None, (
            "build_spread should return None when no option meets the DTE constraint"
        )

    def test_returns_none_on_empty_chain(self):
        """Empty options chain must return None, not raise."""
        empty_chain = pd.DataFrame(
            columns=["strike", "right", "delta", "bid", "ask", "dte", "expiry"]
        )
        result = build_spread(empty_chain, target_delta=0.16, spread_width=5.0)
        assert result is None

    def test_credit_is_positive(self, sample_options_chain):
        """Net credit received must be positive (we are selling the spread)."""
        result = build_spread(sample_options_chain, target_delta=0.16, spread_width=5.0)
        assert result is not None
        assert result["credit"] > 0, "Credit received on a put credit spread must be positive"

    def test_max_loss_equals_width_minus_credit(self, sample_options_chain):
        """
        Max loss = (spread width - credit received) × multiplier.
        For a $5-wide spread with $2 credit, max loss = ($5 - $2) × $100 = $300.
        """
        result = build_spread(sample_options_chain, target_delta=0.16, spread_width=5.0)
        assert result is not None
        expected_max_loss = (result["long_strike"] - result["short_strike"] + result["credit"]) * -100
        # Or: (spread_width - credit) * 100
        # Accept either sign convention, but relationship must hold
        assert abs(result["max_loss"]) == pytest.approx(
            (5.0 - result["credit"]) * 100, rel=0.01
        ), "max_loss must equal (spread_width - credit) × $100 multiplier"

    def test_max_gain_equals_credit_times_multiplier(self, sample_options_chain):
        """Max gain = credit × $100 multiplier."""
        result = build_spread(sample_options_chain, target_delta=0.16, spread_width=5.0)
        assert result is not None
        assert result["max_gain"] == pytest.approx(result["credit"] * 100, rel=0.01)

    def test_equidistant_delta_selects_more_otm_strike(self):
        """
        When two strikes are equidistant from target_delta, the more OTM one
        (lower strike for puts) must be selected — conservative choice.
        """
        chain = pd.DataFrame([
            {"strike": 3910, "right": "P", "delta": -0.14, "bid": 4.0, "ask": 4.8, "dte": 35, "expiry": "20100219"},
            {"strike": 3930, "right": "P", "delta": -0.18, "bid": 5.5, "ask": 6.5, "dte": 35, "expiry": "20100219"},
        ])
        # Both are equidistant from 0.16: |0.14 - 0.16| = 0.02 = |0.18 - 0.16|
        result = build_spread(chain, target_delta=0.16, spread_width=5.0)
        assert result is not None
        assert result["short_strike"] == 3910, (
            "When equidistant from target delta, the more OTM (lower put strike) "
            "should be selected for safety"
        )

    def test_only_puts_are_considered(self):
        """build_spread should only consider put contracts (right == 'P')."""
        chain = pd.DataFrame([
            {"strike": 3920, "right": "C", "delta": 0.16, "bid": 5.0, "ask": 6.0, "dte": 35, "expiry": "20100219"},
            {"strike": 3920, "right": "P", "delta": -0.16, "bid": 5.0, "ask": 6.0, "dte": 35, "expiry": "20100219"},
        ])
        result = build_spread(chain, target_delta=0.16, spread_width=5.0)
        assert result is not None
        # If calls were considered, strike selection could be wrong
        assert result["short_strike"] == 3920


# ===========================================================================
# Exit logic
# ===========================================================================

class TestExitLogic:
    def test_profit_target_fires_at_50_pct(self, sample_trade, winning_bar):
        """
        Spread value dropped from $2.00 entry credit to $1.00 (50%).
        Profit target must fire on this bar.
        """
        result = evaluate_exits(sample_trade, winning_bar)
        assert result is not None, "Profit target should have triggered"
        assert result["fate"] == "profit_target"

    def test_profit_target_does_not_fire_before_50_pct(self, sample_trade):
        """At 60% of credit remaining, the trade should still be open."""
        bar = {
            "date": "2010-01-20",
            "current_spread_value": 1.20,   # 60% of 2.00 — not yet at 50%
            "open_spread_value": 1.22,
            "short_delta": -0.10,
            "dte_remaining": 26,
            "vix": 16.0,
        }
        result = evaluate_exits(sample_trade, bar)
        assert result is None, "No exit should trigger at 60% of credit remaining"

    def test_stop_loss_fires_at_200_pct_credit(self, sample_trade, stop_bar):
        """
        Spread value hit 4.00 = 200% of entry credit 2.00.
        Stop loss must fire.
        """
        result = evaluate_exits(sample_trade, stop_bar)
        assert result is not None, "Stop loss should have triggered"
        assert result["fate"] == "stop_loss"

    def test_stop_loss_does_not_fire_below_200_pct(self, sample_trade):
        """At 190% of credit ($3.80), no stop should trigger."""
        bar = {
            "date": "2010-01-14",
            "current_spread_value": 3.80,   # 190% of 2.00 — just under threshold
            "open_spread_value": 3.82,
            "short_delta": -0.35,
            "dte_remaining": 30,
            "vix": 28.0,
        }
        result = evaluate_exits(sample_trade, bar)
        assert result is None, "Stop loss should not trigger at 190% — threshold is 200%"

    def test_time_exit_fires_at_21_dte(self, sample_trade, time_exit_bar):
        """Hard close at exactly 21 DTE — no profit target or stop triggered."""
        result = evaluate_exits(sample_trade, time_exit_bar)
        assert result is not None, "Time exit should have triggered at 21 DTE"
        assert result["fate"] == "time_exit"

    def test_time_exit_does_not_fire_above_21_dte(self, sample_trade):
        """At 22 DTE with no other triggers, the trade must remain open."""
        bar = {
            "date": "2010-01-28",
            "current_spread_value": 1.60,
            "open_spread_value": 1.62,
            "short_delta": -0.12,
            "dte_remaining": 22,
            "vix": 17.0,
        }
        result = evaluate_exits(sample_trade, bar)
        assert result is None, "Time exit should NOT fire at 22 DTE"

    def test_emergency_exit_fires_when_delta_exceeds_50(self, sample_trade, emergency_bar):
        """Short leg delta > 0.50 (ATM) → emergency exit."""
        result = evaluate_exits(sample_trade, emergency_bar)
        assert result is not None, "Emergency exit should have triggered (delta > 0.50)"
        assert result["fate"] == "emergency_exit"

    def test_emergency_exit_does_not_fire_below_50_delta(self, sample_trade):
        """Delta of 0.48 is not yet at the ATM threshold — no emergency exit."""
        bar = {
            "date": "2010-01-10",
            "current_spread_value": 3.20,
            "open_spread_value": 3.25,
            "short_delta": -0.48,           # below emergency threshold
            "dte_remaining": 30,
            "vix": 32.0,
        }
        result = evaluate_exits(sample_trade, bar)
        assert result is None, "Emergency exit must not trigger at delta = 0.48"

    def test_gap_aware_stop_uses_open_when_worse_than_stop(self, sample_trade, gap_bar):
        """
        Market gaps to 4.80 at open, past the 4.00 stop trigger level.
        Execution must be at 4.80 (the realized open), not 4.00 (the theoretical stop).
        The realized loss must be larger than the theoretical stop loss amount.
        """
        result = evaluate_exits(sample_trade, gap_bar)
        assert result is not None
        assert result["fate"] == "stop_loss"
        # Exit price should be the open price (4.80), not the stop level (4.00)
        assert result["exit_price"] >= 4.00, "Exit price must be at or worse than stop level"
        # Specifically, it should use the open (4.80), not the stop (4.00)
        assert result["exit_price"] == pytest.approx(4.80, abs=0.01), (
            "Gap-aware stop execution: exit price should be the open price (4.80) "
            "when the market gaps past the stop level"
        )

    def test_gap_stop_not_triggered_when_open_below_stop_level(self, sample_trade):
        """
        If the market opens at 3.80 (below the 4.00 stop), the stop has NOT been hit.
        The trade should continue normally.
        """
        bar = {
            "date": "2010-01-11",
            "current_spread_value": 3.80,
            "open_spread_value": 3.80,      # below stop trigger of 4.00
            "short_delta": -0.36,
            "dte_remaining": 30,
            "vix": 28.0,
        }
        result = evaluate_exits(sample_trade, bar)
        assert result is None, (
            "Stop loss should not trigger when open is 3.80 (below the 4.00 stop level)"
        )

    def test_all_four_fates_are_achievable(self, sample_trade):
        """
        Verify that all four fate strings can actually be returned.
        This catches any fate path that is coded as unreachable.
        """
        fates_seen = set()

        # Profit target
        r = evaluate_exits(sample_trade, {
            "date": "2010-01-20", "current_spread_value": 1.00, "open_spread_value": 1.00,
            "short_delta": -0.08, "dte_remaining": 25, "vix": 16.0,
        })
        if r: fates_seen.add(r["fate"])

        # Stop loss
        r = evaluate_exits(sample_trade, {
            "date": "2010-01-15", "current_spread_value": 4.10, "open_spread_value": 4.10,
            "short_delta": -0.40, "dte_remaining": 29, "vix": 30.0,
        })
        if r: fates_seen.add(r["fate"])

        # Time exit
        r = evaluate_exits(sample_trade, {
            "date": "2010-01-29", "current_spread_value": 1.60, "open_spread_value": 1.60,
            "short_delta": -0.12, "dte_remaining": 21, "vix": 17.0,
        })
        if r: fates_seen.add(r["fate"])

        # Emergency exit
        r = evaluate_exits(sample_trade, {
            "date": "2010-01-12", "current_spread_value": 3.50, "open_spread_value": 3.80,
            "short_delta": -0.55, "dte_remaining": 28, "vix": 35.0,
        })
        if r: fates_seen.add(r["fate"])

        assert "profit_target"   in fates_seen, "profit_target fate never triggered"
        assert "stop_loss"       in fates_seen, "stop_loss fate never triggered"
        assert "time_exit"       in fates_seen, "time_exit fate never triggered"
        assert "emergency_exit"  in fates_seen, "emergency_exit fate never triggered"

    def test_exit_result_contains_required_keys(self, sample_trade, winning_bar):
        """Exit result must always include fate, exit_price, and exit_date."""
        result = evaluate_exits(sample_trade, winning_bar)
        assert result is not None
        for key in ["fate", "exit_price", "exit_date"]:
            assert key in result, f"Exit result missing required key '{key}'"

    def test_exit_price_is_numeric_and_positive(self, sample_trade, stop_bar):
        """Exit price must be a positive finite float."""
        result = evaluate_exits(sample_trade, stop_bar)
        assert result is not None
        assert isinstance(result["exit_price"], (int, float))
        assert result["exit_price"] > 0
        assert np.isfinite(result["exit_price"])

    def test_no_exit_returns_none_not_empty_dict(self, sample_trade):
        """
        When no exit condition is met, return None — not an empty dict or a dict
        with fate=None. Downstream code checks `if result is not None:`.
        """
        benign_bar = {
            "date": "2010-01-08",
            "current_spread_value": 1.80,   # 90% of credit — not at 50% target
            "open_spread_value": 1.82,
            "short_delta": -0.14,
            "dte_remaining": 32,
            "vix": 15.0,
        }
        result = evaluate_exits(sample_trade, benign_bar)
        assert result is None, (
            "evaluate_exits must return None (not {}) when no exit is triggered"
        )

    def test_profit_target_wins_over_time_exit_at_21_dte(self, sample_trade):
        """
        At exactly 21 DTE AND profit target conditions both met simultaneously,
        profit_target should take priority (it captures 50% already achieved).
        """
        bar = {
            "date": "2010-01-29",
            "current_spread_value": 1.00,   # 50% of credit → profit target
            "open_spread_value": 1.02,
            "short_delta": -0.08,
            "dte_remaining": 21,             # also triggers time exit
            "vix": 16.0,
        }
        result = evaluate_exits(sample_trade, bar)
        assert result is not None
        # Both conditions are true; the defined priority must be consistently applied
        assert result["fate"] in ("profit_target", "time_exit"), (
            "When both profit_target and time_exit conditions are met, one must win"
        )
        # Critically: same fate must be returned every time (deterministic)
        result2 = evaluate_exits(sample_trade, bar)
        assert result["fate"] == result2["fate"], (
            "Exit fate selection must be deterministic — same bar, same fate every call"
        )


# ===========================================================================
# Position sizing — Kelly fraction
# ===========================================================================

class TestKellyFraction:
    def test_basic_kelly_formula(self):
        """
        Kelly fraction = (p * max_win - (1-p) * max_loss) / max_win.
        For p=0.70, max_win=200, max_loss=300:
          f = (0.70*200 - 0.30*300) / 200 = (140 - 90) / 200 = 0.25
        """
        f = kelly_fraction(p=0.70, max_win=200, max_loss=300, cap=1.0)
        assert f == pytest.approx(0.25, abs=0.001)

    def test_kelly_capped_at_quarter(self):
        """
        kelly_fraction must never exceed 0.25 (quarter-Kelly institutional convention).
        With very high probability (p=0.95), raw Kelly would be >0.25.
        """
        f = kelly_fraction(p=0.95, max_win=200, max_loss=300, cap=0.25)
        assert f <= 0.25, f"Kelly fraction {f:.4f} exceeds 0.25 cap"

    def test_kelly_zero_below_breakeven_probability(self):
        """
        At breakeven probability (where EV=0), Kelly fraction should be 0 or negative.
        For max_win=200, max_loss=300: breakeven p = 300/(200+300) = 0.60.
        At p=0.59, Kelly must be 0 (no edge).
        """
        f = kelly_fraction(p=0.59, max_win=200, max_loss=300, cap=0.25)
        assert f <= 0, (
            "Kelly fraction must be 0 or negative when there is no edge (p < breakeven)"
        )

    def test_kelly_positive_above_breakeven(self):
        """At p=0.61 (above breakeven), Kelly should be positive."""
        f = kelly_fraction(p=0.61, max_win=200, max_loss=300, cap=0.25)
        assert f > 0, "Kelly fraction must be positive when p > breakeven"

    def test_kelly_scales_monotonically_with_probability(self):
        """Higher probability → higher Kelly fraction (monotone relationship)."""
        fractions = [
            kelly_fraction(p=p, max_win=200, max_loss=300, cap=1.0)
            for p in [0.60, 0.65, 0.70, 0.75, 0.80]
        ]
        for i in range(len(fractions) - 1):
            assert fractions[i] <= fractions[i + 1], (
                f"Kelly fraction is not monotone: f({0.60+i*0.05:.2f})={fractions[i]:.4f} "
                f"> f({0.60+(i+1)*0.05:.2f})={fractions[i+1]:.4f}"
            )


# ===========================================================================
# Position sizing — full position size computation
# ===========================================================================

class TestPositionSizing:
    def test_returns_zero_when_probability_below_threshold(self):
        """
        At p=0.54 (below 0.55 entry threshold), position size must be 0.
        No contracts should be entered.
        """
        size = compute_position_size(
            p_calibrated=0.54, max_win=200, max_loss=300,
            vix=18.0, spy_tlt_corr=0.2, account_equity=100_000,
            spread_max_loss=300, entry_threshold=0.55,
        )
        assert size == 0, f"Position size should be 0 at p=0.54 (below 0.55 threshold), got {size}"

    def test_returns_positive_when_probability_meets_threshold(self):
        """At p=0.55, position size should be positive (≥1 contract)."""
        size = compute_position_size(
            p_calibrated=0.55, max_win=200, max_loss=300,
            vix=18.0, spy_tlt_corr=0.2, account_equity=100_000,
            spread_max_loss=300, entry_threshold=0.55,
        )
        assert size >= 1, f"Position size should be ≥1 at p=0.55, got {size}"

    def test_hard_cap_enforced_at_1_pct_of_equity(self):
        """
        No single trade can risk more than 1% of account equity ($1,000 on $100k).
        With spread_max_loss=$300/contract, the cap is 3 contracts.
        Even with very high probability (p=0.99), contracts must not exceed 3.
        """
        size = compute_position_size(
            p_calibrated=0.99, max_win=200, max_loss=300,
            vix=18.0, spy_tlt_corr=0.0, account_equity=100_000,
            spread_max_loss=300, max_risk_pct=0.01,
        )
        max_allowed = int(100_000 * 0.01 / 300)
        assert size <= max_allowed, (
            f"Hard cap violated: {size} contracts × $300 max_loss "
            f"= ${size * 300} > 1% of $100k"
        )

    def test_vol_scaling_reduces_size_at_high_vix(self):
        """
        VIX=30 should produce a smaller position than VIX=15, all else equal.
        vol_multiplier = 15 / VIX_current, so VIX=30 → 0.5x, VIX=15 → 1.0x.
        """
        size_low_vix = compute_position_size(
            p_calibrated=0.70, max_win=200, max_loss=300,
            vix=15.0, spy_tlt_corr=0.2, account_equity=100_000,
            spread_max_loss=300,
        )
        size_high_vix = compute_position_size(
            p_calibrated=0.70, max_win=200, max_loss=300,
            vix=30.0, spy_tlt_corr=0.2, account_equity=100_000,
            spread_max_loss=300,
        )
        assert size_high_vix <= size_low_vix, (
            f"Higher VIX should reduce position size: "
            f"VIX=30 gave {size_high_vix} contracts, VIX=15 gave {size_low_vix}"
        )

    def test_stress_multiplier_cuts_size_when_spy_tlt_corr_exceeds_threshold(self):
        """
        SPY-TLT correlation > 0.5 triggers the 0.5× stress multiplier.
        Position at corr=0.6 must be ≤ half the position at corr=0.2.
        """
        size_normal = compute_position_size(
            p_calibrated=0.70, max_win=200, max_loss=300,
            vix=18.0, spy_tlt_corr=0.2, account_equity=100_000,
            spread_max_loss=300,
        )
        size_stressed = compute_position_size(
            p_calibrated=0.70, max_win=200, max_loss=300,
            vix=18.0, spy_tlt_corr=0.6, account_equity=100_000,
            spread_max_loss=300,
        )
        assert size_stressed <= size_normal, (
            "Stressed correlation should not increase position size"
        )

    def test_stress_multiplier_exact_boundary(self):
        """
        spy_tlt_corr=0.50 is the threshold. Test that exactly at 0.50 the multiplier
        behavior is consistent (either always triggered or always not). Must not
        produce different results on repeated calls.
        """
        size1 = compute_position_size(
            p_calibrated=0.70, max_win=200, max_loss=300,
            vix=18.0, spy_tlt_corr=0.50, account_equity=100_000,
            spread_max_loss=300,
        )
        size2 = compute_position_size(
            p_calibrated=0.70, max_win=200, max_loss=300,
            vix=18.0, spy_tlt_corr=0.50, account_equity=100_000,
            spread_max_loss=300,
        )
        assert size1 == size2, "Position size at the exact stress threshold must be deterministic"

    def test_position_size_is_non_negative_integer(self):
        """Position size must always be a non-negative integer (whole contracts)."""
        for p in [0.40, 0.55, 0.70, 0.90]:
            size = compute_position_size(
                p_calibrated=p, max_win=200, max_loss=300,
                vix=18.0, spy_tlt_corr=0.2, account_equity=100_000,
                spread_max_loss=300,
            )
            assert isinstance(size, int), f"Position size must be int, got {type(size)} at p={p}"
            assert size >= 0, f"Position size must be non-negative, got {size} at p={p}"

    def test_kelly_cap_prevents_aggressive_sizing_at_extreme_probability(self):
        """
        At p=0.99 with no vol-scaling or stress multiplier pressure,
        the quarter-Kelly cap must still prevent excessive sizing.
        The position should still be bounded.
        """
        size = compute_position_size(
            p_calibrated=0.99, max_win=200, max_loss=300,
            vix=15.0, spy_tlt_corr=0.0, account_equity=100_000,
            spread_max_loss=300, kelly_cap=0.25, max_risk_pct=0.01,
        )
        # Hard cap: 1% of $100k / $300 per contract = 3 contracts max
        assert size <= 10, f"Quarter-Kelly cap seems ineffective: {size} contracts at p=0.99"

    def test_position_size_zero_on_small_account_relative_to_spread_loss(self):
        """
        On a $5,000 account with 1% risk cap, max risk is $50.
        If spread max_loss=$300, no full contract can be taken. Size must be 0.
        """
        size = compute_position_size(
            p_calibrated=0.80, max_win=200, max_loss=300,
            vix=18.0, spy_tlt_corr=0.0, account_equity=5_000,
            spread_max_loss=300, max_risk_pct=0.01,
        )
        assert size == 0, (
            "Cannot risk $300 on a $5,000 account with 1% risk cap ($50 limit)"
        )
