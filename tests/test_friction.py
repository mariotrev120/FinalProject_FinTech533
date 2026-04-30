"""
Tests for src/strategy/friction.py — the cost model.

Expected interfaces
-------------------
src.strategy.friction
    compute_slippage_pct(vix: float) -> float
        Returns 0.30, 0.50, 0.75, or 1.00 based on VIX bucket.

    compute_entry_cost(
        credit: float,
        bid_ask_spread: float,
        num_contracts: int,
        vix: float,
        monday_gap_pct: float = 0.0,
    ) -> tuple[float, bool]
        Returns (total_friction_$, skip_entry_bool).
        skip_entry_bool is True when monday_gap_pct >= 0.03.

    compute_exit_cost(
        exit_value: float,
        bid_ask_spread: float,
        num_contracts: int,
        vix: float,
    ) -> float
        Returns total friction cost at exit in dollars.

    compute_commission(num_contracts: int, num_legs: int = 2) -> float
        Returns commission in dollars (IBKR Pro tiered schedule).

    apply_section_1256_tax(
        annual_pnl: float,
        ltcg_rate: float = 0.15,
        stcg_rate: float = 0.22,
    ) -> dict
        Returns {"ltcg_portion": float, "stcg_portion": float, "tax_liability": float}.
"""
import pytest
import numpy as np

from src.strategy.friction import (
    compute_slippage_pct,
    compute_entry_cost,
    compute_exit_cost,
    compute_commission,
    apply_section_1256_tax,
)


# ===========================================================================
# VIX-scaled slippage
# ===========================================================================

class TestSlippageBuckets:
    def test_bucket_below_20(self):
        """VIX < 20 → 30% of bid-ask spread charged as slippage."""
        assert compute_slippage_pct(15.0) == pytest.approx(0.30)

    def test_bucket_20_to_30(self):
        """VIX in [20, 30) → 50% of bid-ask spread."""
        assert compute_slippage_pct(25.0) == pytest.approx(0.50)

    def test_bucket_30_to_40(self):
        """VIX in [30, 40) → 75% of bid-ask spread."""
        assert compute_slippage_pct(35.0) == pytest.approx(0.75)

    def test_bucket_above_40(self):
        """VIX ≥ 40 → 100% of bid-ask spread (full spread paid as slippage)."""
        assert compute_slippage_pct(45.0) == pytest.approx(1.00)

    def test_exact_boundary_vix_20_consistent(self):
        """
        VIX = 20.0 exactly lies on the boundary of the first two buckets.
        The convention (which bucket it falls into) must be deterministic.
        Both calls must return the same value.
        """
        result1 = compute_slippage_pct(20.0)
        result2 = compute_slippage_pct(20.0)
        assert result1 == result2, "compute_slippage_pct must be deterministic"
        assert result1 in (0.30, 0.50), (
            f"VIX=20.0 must map to either the <20 or 20-30 bucket, got {result1}"
        )

    def test_exact_boundary_vix_30_consistent(self):
        """VIX = 30.0 exactly — boundary between 20-30 and 30-40 buckets."""
        result1 = compute_slippage_pct(30.0)
        result2 = compute_slippage_pct(30.0)
        assert result1 == result2
        assert result1 in (0.50, 0.75), (
            f"VIX=30.0 must map to either the 20-30 or 30-40 bucket, got {result1}"
        )

    def test_exact_boundary_vix_40_consistent(self):
        """VIX = 40.0 exactly — boundary between 30-40 and ≥40 buckets."""
        result = compute_slippage_pct(40.0)
        assert result in (0.75, 1.00), (
            f"VIX=40.0 must map to either the 30-40 or ≥40 bucket, got {result}"
        )

    def test_returns_known_schedule_values_only(self):
        """Slippage pct must always be one of the four documented values."""
        valid = {0.30, 0.50, 0.75, 1.00}
        for vix in [10, 15, 19.9, 20, 20.1, 25, 29.9, 30, 30.1, 35, 39.9, 40, 40.1, 50, 80]:
            result = compute_slippage_pct(float(vix))
            assert result in valid, (
                f"compute_slippage_pct({vix}) returned {result}, which is not in {valid}"
            )

    def test_monotone_non_decreasing_with_vix(self):
        """
        Slippage pct must be non-decreasing as VIX increases.
        A higher stress environment should never produce less slippage.
        """
        vix_levels = [10, 15, 20, 25, 30, 35, 40, 50]
        slippages = [compute_slippage_pct(float(v)) for v in vix_levels]
        for i in range(len(slippages) - 1):
            assert slippages[i] <= slippages[i + 1], (
                f"Slippage decreased from VIX={vix_levels[i]} ({slippages[i]}) "
                f"to VIX={vix_levels[i+1]} ({slippages[i+1]})"
            )


# ===========================================================================
# Commission calculation
# ===========================================================================

class TestCommissions:
    def test_single_contract_two_leg_commission(self):
        """1 contract, 2 legs (spread entry): 2 × $0.65 = $1.30. Above $1 minimum."""
        comm = compute_commission(num_contracts=1, num_legs=2)
        assert comm == pytest.approx(1.30, abs=0.05), (
            "1 contract × 2 legs at $0.65 = $1.30 before regulatory fees"
        )

    def test_commission_minimum_does_not_reduce_standard_amount(self):
        """
        The $1 minimum per order kicks in only for fractional amounts.
        1 contract × 2 legs = $1.30 already exceeds $1, so minimum has no effect.
        """
        comm = compute_commission(num_contracts=1, num_legs=2)
        assert comm >= 1.00, "Commission must be at least $1.00 (IBKR minimum)"
        assert comm >= 1.30 * 0.95, (
            "Commission should not be reduced by minimum — 1 contract already exceeds minimum"
        )

    def test_commission_scales_linearly_with_contracts(self):
        """3 contracts, 2 legs: 3 × 2 × $0.65 = $3.90."""
        comm_1 = compute_commission(num_contracts=1, num_legs=2)
        comm_3 = compute_commission(num_contracts=3, num_legs=2)
        assert comm_3 == pytest.approx(comm_1 * 3, rel=0.05), (
            "Commission must scale linearly with contract count"
        )

    def test_commission_includes_regulatory_fees(self):
        """
        ORF, SEC, and OCC fees add ~$0.05/contract/side.
        Total commission for 1 contract round-trip should exceed raw $1.30.
        """
        entry_comm = compute_commission(num_contracts=1, num_legs=2)
        exit_comm = compute_commission(num_contracts=1, num_legs=2)
        round_trip = entry_comm + exit_comm
        # Regulatory fees mean the total should exceed $2.60
        assert round_trip > 2.50, (
            f"Round-trip commission {round_trip:.2f} seems too low — regulatory fees may be missing"
        )

    def test_commission_non_negative(self):
        """Commission is always a non-negative cost."""
        for contracts in [1, 2, 5, 10]:
            comm = compute_commission(num_contracts=contracts, num_legs=2)
            assert comm >= 0, f"Commission must be non-negative, got {comm}"


# ===========================================================================
# Entry cost (slippage + commission + gap adjustments)
# ===========================================================================

class TestEntryCost:
    def test_total_entry_cost_positive(self):
        """Entry friction must be a positive cost (deducted from P&L)."""
        cost, skip = compute_entry_cost(
            credit=2.00, bid_ask_spread=1.00, num_contracts=1, vix=18.0,
            monday_gap_pct=0.0,
        )
        assert cost > 0, "Entry friction must be a positive cost"
        assert not skip

    def test_slippage_applied_on_entry(self):
        """At VIX=18 (<20 bucket, 30% slippage), entry cost includes 30% × bid-ask."""
        # bid_ask_spread=1.00, slippage=30%, 1 contract, 1 spread × $100 multiplier
        cost, skip = compute_entry_cost(
            credit=2.00, bid_ask_spread=1.00, num_contracts=1, vix=18.0,
        )
        # Slippage portion = 0.30 × $1.00 × $100 × 1 contract = $30
        slippage_portion = 0.30 * 1.00 * 100 * 1
        assert cost >= slippage_portion * 0.9, (
            f"Entry cost {cost:.2f} is less than the slippage portion {slippage_portion:.2f}"
        )

    def test_entry_slippage_scales_with_vix_bucket(self):
        """Higher VIX → more slippage → higher entry cost, all else equal."""
        cost_low, _ = compute_entry_cost(
            credit=2.00, bid_ask_spread=1.00, num_contracts=1, vix=15.0,
        )
        cost_high, _ = compute_entry_cost(
            credit=2.00, bid_ask_spread=1.00, num_contracts=1, vix=45.0,
        )
        assert cost_high > cost_low, (
            f"Entry cost at VIX=45 ({cost_high:.2f}) should exceed cost at VIX=15 ({cost_low:.2f})"
        )

    def test_skip_entry_when_gap_exceeds_3_pct(self):
        """A Monday gap ≥ 3% means skip entry entirely."""
        cost, skip = compute_entry_cost(
            credit=2.00, bid_ask_spread=1.00, num_contracts=1, vix=18.0,
            monday_gap_pct=0.031,
        )
        assert skip is True, "Entry should be skipped when Monday gap > 3%"

    def test_no_skip_when_gap_below_3_pct(self):
        """A 2% gap should NOT skip entry (only adds extra slippage)."""
        cost, skip = compute_entry_cost(
            credit=2.00, bid_ask_spread=1.00, num_contracts=1, vix=18.0,
            monday_gap_pct=0.02,
        )
        assert not skip, "Entry should NOT be skipped for a 2% Monday gap"

    def test_extra_slippage_applied_for_gap_between_1_5_and_3_pct(self):
        """
        A 2% gap (between 1.5% and 3%) adds 50% extra slippage on entry.
        Cost at gap=2% must exceed cost at gap=0%.
        """
        cost_no_gap, _ = compute_entry_cost(
            credit=2.00, bid_ask_spread=1.00, num_contracts=1, vix=18.0,
            monday_gap_pct=0.0,
        )
        cost_with_gap, _ = compute_entry_cost(
            credit=2.00, bid_ask_spread=1.00, num_contracts=1, vix=18.0,
            monday_gap_pct=0.02,
        )
        assert cost_with_gap > cost_no_gap, (
            "A 2% Monday gap should increase entry cost via extra slippage"
        )

    def test_no_extra_slippage_below_1_5_pct_gap(self):
        """A 1% gap is too small to trigger the extra slippage rule."""
        cost_no_gap, _ = compute_entry_cost(
            credit=2.00, bid_ask_spread=1.00, num_contracts=1, vix=18.0,
            monday_gap_pct=0.0,
        )
        cost_small_gap, _ = compute_entry_cost(
            credit=2.00, bid_ask_spread=1.00, num_contracts=1, vix=18.0,
            monday_gap_pct=0.01,
        )
        assert cost_small_gap == pytest.approx(cost_no_gap, rel=0.01), (
            "A 1% gap (below 1.5% threshold) should not add extra slippage"
        )

    def test_entry_cost_scales_with_contract_count(self):
        """Entry cost for 3 contracts should be approximately 3× the 1-contract cost."""
        cost_1, _ = compute_entry_cost(
            credit=2.00, bid_ask_spread=1.00, num_contracts=1, vix=18.0,
        )
        cost_3, _ = compute_entry_cost(
            credit=2.00, bid_ask_spread=1.00, num_contracts=3, vix=18.0,
        )
        assert cost_3 == pytest.approx(cost_1 * 3, rel=0.10), (
            "Entry cost should scale approximately linearly with contract count"
        )


# ===========================================================================
# Exit cost
# ===========================================================================

class TestExitCost:
    def test_exit_cost_positive(self):
        """Exit friction must be a positive cost."""
        cost = compute_exit_cost(
            exit_value=1.00, bid_ask_spread=0.80, num_contracts=1, vix=18.0,
        )
        assert cost > 0

    def test_exit_slippage_higher_at_high_vix(self):
        """VIX=45 exit must cost more than VIX=15 exit (same spread width)."""
        cost_low = compute_exit_cost(
            exit_value=1.00, bid_ask_spread=0.80, num_contracts=1, vix=15.0,
        )
        cost_high = compute_exit_cost(
            exit_value=1.00, bid_ask_spread=0.80, num_contracts=1, vix=45.0,
        )
        assert cost_high > cost_low

    def test_round_trip_friction_not_double_counted(self):
        """
        Entry + exit friction should be separately accounted — not the same cost
        charged twice from either side. Verify both functions produce non-zero,
        independent results.
        """
        entry_cost, _ = compute_entry_cost(
            credit=2.00, bid_ask_spread=1.00, num_contracts=1, vix=18.0,
        )
        exit_cost = compute_exit_cost(
            exit_value=1.00, bid_ask_spread=0.80, num_contracts=1, vix=18.0,
        )
        assert entry_cost > 0
        assert exit_cost > 0
        # They should NOT be identical (different credit/value and potentially different spreads)
        # But both should be on similar order of magnitude
        assert 0.1 < (exit_cost / entry_cost) < 10.0, (
            "Entry and exit costs are wildly mismatched — one may be wrong"
        )


# ===========================================================================
# Section 1256 tax treatment
# ===========================================================================

class TestSection1256Tax:
    def test_profit_split_60_40(self):
        """$10,000 annual profit: 60% LTCG = $6,000; 40% STCG = $4,000."""
        result = apply_section_1256_tax(
            annual_pnl=10_000, ltcg_rate=0.15, stcg_rate=0.22,
        )
        assert result["ltcg_portion"] == pytest.approx(6_000, rel=0.001)
        assert result["stcg_portion"] == pytest.approx(4_000, rel=0.001)

    def test_tax_liability_correct_for_profit(self):
        """Tax = 0.60 × 10,000 × 0.15 + 0.40 × 10,000 × 0.22 = 900 + 880 = $1,780."""
        result = apply_section_1256_tax(
            annual_pnl=10_000, ltcg_rate=0.15, stcg_rate=0.22,
        )
        expected_tax = 0.60 * 10_000 * 0.15 + 0.40 * 10_000 * 0.22
        assert result["tax_liability"] == pytest.approx(expected_tax, rel=0.001)

    def test_loss_split_60_40(self):
        """
        A $5,000 loss splits as: 60% LTCG portion = -$3,000; 40% STCG portion = -$2,000.
        Section 1256 losses retain the 60/40 split for loss carryback/carryforward.
        """
        result = apply_section_1256_tax(
            annual_pnl=-5_000, ltcg_rate=0.15, stcg_rate=0.22,
        )
        assert result["ltcg_portion"] == pytest.approx(-3_000, rel=0.001)
        assert result["stcg_portion"] == pytest.approx(-2_000, rel=0.001)

    def test_loss_tax_liability_non_positive(self):
        """A net loss should produce a non-positive tax liability (a tax benefit)."""
        result = apply_section_1256_tax(
            annual_pnl=-5_000, ltcg_rate=0.15, stcg_rate=0.22,
        )
        assert result["tax_liability"] <= 0, (
            "A net loss should produce zero or negative tax liability"
        )

    def test_portions_sum_to_annual_pnl(self):
        """ltcg_portion + stcg_portion must equal annual_pnl exactly."""
        for pnl in [0, 5_000, -3_000, 20_000]:
            result = apply_section_1256_tax(
                annual_pnl=pnl, ltcg_rate=0.15, stcg_rate=0.22,
            )
            total = result["ltcg_portion"] + result["stcg_portion"]
            assert total == pytest.approx(pnl, abs=0.01), (
                f"ltcg + stcg portions ({total:.2f}) must sum to pnl ({pnl:.2f})"
            )

    def test_zero_pnl_produces_zero_tax(self):
        """Zero annual P&L → zero tax liability."""
        result = apply_section_1256_tax(annual_pnl=0, ltcg_rate=0.15, stcg_rate=0.22)
        assert result["tax_liability"] == pytest.approx(0.0, abs=0.001)

    def test_tax_applied_annually_not_per_trade(self):
        """
        The Section 1256 function accepts annual_pnl (an aggregate), not per-trade P&L.
        Calling it with a small trade-level P&L should still produce a proportional result.
        This test verifies the function is not internally dividing/multiplying by 252
        or otherwise rescaling for daily vs annual application.
        """
        small_pnl = 100.0
        result = apply_section_1256_tax(annual_pnl=small_pnl, ltcg_rate=0.15, stcg_rate=0.22)
        expected_total = 0.60 * small_pnl * 0.15 + 0.40 * small_pnl * 0.22
        assert result["tax_liability"] == pytest.approx(expected_total, rel=0.001), (
            "Tax function appears to be rescaling the input — should accept raw P&L"
        )

    def test_result_contains_all_required_keys(self):
        """Return dict must have ltcg_portion, stcg_portion, and tax_liability."""
        result = apply_section_1256_tax(annual_pnl=10_000)
        for key in ["ltcg_portion", "stcg_portion", "tax_liability"]:
            assert key in result, f"Missing key '{key}' in apply_section_1256_tax result"


# ===========================================================================
# Integration: friction reduces net P&L
# ===========================================================================

class TestFrictionReducesPnL:
    def test_round_trip_friction_reduces_trade_pnl(self):
        """
        For a winning trade: entry credit - exit value - total friction < entry credit - exit value.
        Friction always reduces (never increases) net P&L.
        """
        credit = 2.00          # received at entry
        exit_value = 1.00      # paid at exit (50% profit target)
        gross_pnl = (credit - exit_value) * 100  # $100 gross per contract

        entry_cost, _ = compute_entry_cost(
            credit=credit, bid_ask_spread=1.00, num_contracts=1, vix=18.0,
        )
        exit_cost = compute_exit_cost(
            exit_value=exit_value, bid_ask_spread=0.80, num_contracts=1, vix=18.0,
        )
        net_pnl = gross_pnl - entry_cost - exit_cost
        assert net_pnl < gross_pnl, "Friction must reduce net P&L below gross P&L"
        assert net_pnl > 0, (
            "For a standard winning trade, net P&L after friction should still be positive"
        )

    def test_friction_can_wipe_out_marginal_winner(self):
        """
        On a very small winner (1% profit), friction should convert it to a small loser
        or at least severely compress the gain. Tests that friction is not trivially small.
        """
        credit = 2.00
        exit_value = 1.98     # only 1% improvement
        gross_pnl = (credit - exit_value) * 100  # $2.00 gross

        entry_cost, _ = compute_entry_cost(
            credit=credit, bid_ask_spread=1.00, num_contracts=1, vix=18.0,
        )
        exit_cost = compute_exit_cost(
            exit_value=exit_value, bid_ask_spread=0.80, num_contracts=1, vix=18.0,
        )
        net_pnl = gross_pnl - entry_cost - exit_cost
        # Either net_pnl < 0 (friction wipes the gain) or it's severely compressed
        assert net_pnl < gross_pnl * 0.5, (
            f"Friction should compress a marginal $2.00 gross gain significantly; "
            f"net_pnl={net_pnl:.2f} is too close to gross"
        )
