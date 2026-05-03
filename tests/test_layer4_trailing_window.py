"""
Tests for Layer 4 Option A — drawdown halt uses trailing-90-day window
per PRE_COMMITMENT_VRP §4 spec text "in trailing 90 days."

Four scenarios:
  1. Equity drifts 1% below the all-time HWM for 95 days, then recovers.
     Trailing-90 max rolls forward → drawdown is small AND underwater
     resets when equity hits trailing-window max → halt should NOT fire.
  2. Equity drops 16% in 30 days. Drawdown > 15% in trailing 90 days
     → halt SHOULD fire on the day the drawdown crosses 15%.
  3. Equity drifts 5% below HWM for 95 days, then recovers to a NEW
     all-time high. During the 95-day dip: trailing-90 max rolls forward
     and underwater_days resets when the $100K baseline rolls out of
     the deque (specifically on day 90, when deque = [$95K]*90).
     Halt should NOT fire. After the new ATH: counter at 0; trailing-90
     max is the new ATH.
  4. Monotonically rising equity: underwater_days stays 0 throughout,
     current_dd_frac stays 0, all-time HWM tracks equity exactly.
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

from src.strategy.halts import drawdown_halt_triggered


DRAWDOWN_LOOKBACK_DAYS = 90


def _simulate_trailing_window(
    equity_path: List[float], lookback: int = DRAWDOWN_LOOKBACK_DAYS,
) -> List[Tuple[int, float, bool, float, float]]:
    """Replicate the engine's trailing-window book-keeping outside the
    engine. Returns per-day:
      (underwater_days, current_dd_frac, halted, trailing_high, all_time_hwm)
    """
    equity_trailing: deque = deque(maxlen=lookback)
    underwater_days = 0
    all_time_hwm = float("-inf")
    out: List[Tuple[int, float, bool, float, float]] = []
    for eq in equity_path:
        equity_trailing.append(eq)
        trailing_high = max(equity_trailing)
        if eq >= trailing_high:
            underwater_days = 0
            dd = 0.0
        else:
            underwater_days += 1
            dd = (trailing_high - eq) / trailing_high if trailing_high > 0 else 0.0
        if eq > all_time_hwm:
            all_time_hwm = eq
        halted = drawdown_halt_triggered(underwater_days, dd)
        out.append((underwater_days, dd, halted, trailing_high, all_time_hwm))
    return out


# ------------------------------------------------------------
# Test 1 — 1% sustained drift below all-time HWM
# ------------------------------------------------------------

def test_one_pct_drift_for_95_days_does_not_halt():
    """1% sustained drift below all-time HWM. Trailing-window resets
    each day after $100K rolls out of the deque (day 90+). No halt fires."""
    eq = [100_000.0]                  # ATH
    for _ in range(95):
        eq.append(99_000.0)           # 1% below ATH for 95 days
    for _ in range(10):
        eq.append(101_000.0)          # recovery above prior ATH

    seq = _simulate_trailing_window(eq, lookback=90)
    halted_any = any(h for _, _, h, *_ in seq)
    assert not halted_any, (
        "Layer 4 should NOT fire on a 1% drift sustained for 95 days. "
        "After day 90+ the trailing-90 max IS the dip level, equity == "
        "trailing_high, underwater_days resets each day, drawdown_frac=0."
    )

    # Sanity: max drawdown observed under trailing-window logic is 1%.
    max_dd = max(dd for _, dd, *_ in seq)
    assert max_dd <= 0.011, f"Max DD should be ~1%, got {max_dd:.4f}"


# ------------------------------------------------------------
# Test 2 — 16% drop in 30 days (halt fires within trailing 90d)
# ------------------------------------------------------------

def test_sixteen_pct_drop_in_30_days_does_halt():
    """16% decline in 30 days. Drawdown > 15% within trailing 90d window
    → halt fires."""
    eq = [100_000.0]
    for i in range(30):
        eq.append(100_000.0 * (1 - 0.16 * (i + 1) / 30))

    seq = _simulate_trailing_window(eq, lookback=90)
    halted_any = any(h for _, _, h, *_ in seq)
    assert halted_any, "Layer 4 should fire when DD exceeds 15% in trailing 90d"

    # Day the halt first fires
    first_halt_idx = next(i for i, (_, _, h, *_) in enumerate(seq) if h)
    assert first_halt_idx < 30, (
        f"Halt should fire within the 30-day decline (got idx {first_halt_idx})"
    )
    _, dd_at_halt, _, _, _ = seq[first_halt_idx]
    assert dd_at_halt >= 0.15 - 1e-6


# ------------------------------------------------------------
# Test 3 — 5% dip 95 days, then new ATH
# Day-by-day verification on specific deque-roll-out boundaries
# ------------------------------------------------------------

def test_five_pct_dip_95_days_then_new_ath_resets_correctly():
    """5% dip below HWM for 95 days, then recovery to a NEW all-time high.

    Verified day-by-day at the specific deque-roll-out boundary:
      eq[0]    = $100K (initial ATH)
      eq[1..95]= $95K  (5% dip, 95 days)
      eq[96]   = $105K (NEW ATH)
      eq[97..]= $105K plateau

    Deque(maxlen=90) state:
      Day 0:  deque = [$100K]            → trailing_high = $100K, eq==trail → underwater=0
      Day 1:  deque = [$100K, $95K]      → trailing_high = $100K, eq<trail → underwater=1, dd=5%
      Day 89: deque = [$100K, $95K×89]   → trailing_high = $100K, underwater=89, dd=5%
      Day 90: deque = [$95K × 90]        → trailing_high = $95K = eq → underwater=0, dd=0
      Day 91-95: deque = [$95K × 90]     → trailing_high = $95K = eq → underwater=0, dd=0
      Day 96: deque = [$95K × 89, $105K] → trailing_high = $105K = eq → underwater=0, dd=0
    """
    eq = [100_000.0]                  # initial ATH
    for _ in range(95):
        eq.append(95_000.0)           # 5% below ATH for 95 days
    eq.append(105_000.0)              # NEW ATH (5% above prior)
    for _ in range(20):
        eq.append(105_000.0)          # plateau at new ATH

    seq = _simulate_trailing_window(eq, lookback=90)
    halted_any = any(h for _, _, h, *_ in seq)
    assert not halted_any, (
        "Layer 4 should NOT fire on a 5% dip for 95 days followed by recovery."
    )

    # Day-by-day specific assertions at the deque-roll-out boundary

    # Day 0: $100K = trailing_high → underwater=0
    underwater_d0, dd_d0, _, trail_d0, hwm_d0 = seq[0]
    assert underwater_d0 == 0
    assert dd_d0 == 0.0
    assert trail_d0 == 100_000.0
    assert hwm_d0 == 100_000.0

    # Day 1: $95K, deque = [$100K, $95K], trailing_high = $100K
    underwater_d1, dd_d1, _, trail_d1, _ = seq[1]
    assert underwater_d1 == 1
    assert abs(dd_d1 - 0.05) < 1e-9
    assert trail_d1 == 100_000.0

    # Day 89: deque has $100K + 89×$95K, trailing_high still $100K, underwater=89
    underwater_d89, dd_d89, _, trail_d89, _ = seq[89]
    assert underwater_d89 == 89, (
        f"Day 89 should still see $100K in deque (last day before roll-out), "
        f"underwater should be 89; got {underwater_d89}"
    )
    assert abs(dd_d89 - 0.05) < 1e-9
    assert trail_d89 == 100_000.0

    # Day 90: $100K rolls out of deque, deque = 90×$95K, trailing_high=$95K=eq
    # underwater RESETS to 0, dd=0
    underwater_d90, dd_d90, _, trail_d90, _ = seq[90]
    assert underwater_d90 == 0, (
        f"Day 90 is the deque-roll-out day: $100K should be gone, "
        f"trailing_high=$95K=eq, underwater MUST reset to 0; got {underwater_d90}"
    )
    assert dd_d90 == 0.0
    assert trail_d90 == 95_000.0

    # Days 91-95: same condition — underwater stays 0, dd stays 0
    for i in range(91, 96):
        u, d, _, t, _ = seq[i]
        assert u == 0, f"Day {i}: underwater should be 0, got {u}"
        assert d == 0.0
        assert t == 95_000.0

    # Day 96: NEW ATH. deque = [89×$95K, $105K], trailing_high=$105K=eq
    underwater_d96, dd_d96, _, trail_d96, hwm_d96 = seq[96]
    assert underwater_d96 == 0, (
        f"Day 96 (new ATH): underwater_days should be 0, got {underwater_d96}"
    )
    assert dd_d96 == 0.0
    assert trail_d96 == 105_000.0
    assert hwm_d96 == 105_000.0   # all-time HWM updates

    # Sanity: max drawdown observed = 5%
    max_dd = max(dd for _, dd, *_ in seq)
    assert 0.04 <= max_dd <= 0.06


# ------------------------------------------------------------
# Test 4 — monotonically rising equity (counter + HWM tracking)
# ------------------------------------------------------------

def test_monotonic_rise_underwater_zero_hwm_tracks_equity():
    """Strictly increasing equity — underwater_days stays 0, current_dd_frac
    stays 0, all-time HWM tracks equity exactly at every step.

    Catches: (a) underwater_days incorrectly incrementing when equity is at
    a new high, (b) current_dd_frac incorrectly going negative or non-zero,
    (c) high_water failing to update on each new high.
    """
    eq = [100_000.0 + 100 * i for i in range(120)]  # +$100/day for 120 days

    seq = _simulate_trailing_window(eq, lookback=90)

    for i, (underwater, dd, halted, trail_high, hwm) in enumerate(seq):
        assert underwater == 0, (
            f"Day {i}: monotonically rising equity should never accumulate "
            f"underwater_days; got {underwater}"
        )
        assert dd == 0.0, (
            f"Day {i}: drawdown should be 0 on monotonic rise; got {dd}"
        )
        assert not halted, (
            f"Day {i}: monotonic rise should never trigger Layer 4"
        )
        # All-time HWM tracks equity exactly
        assert hwm == eq[i], (
            f"Day {i}: all-time HWM should equal equity ({eq[i]}); got {hwm}"
        )
        # trailing_high == today's equity (since today is the new max in the deque)
        assert trail_high == eq[i], (
            f"Day {i}: trailing_high should equal today's equity; got {trail_high}"
        )
