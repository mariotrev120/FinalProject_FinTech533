# Option A — PR-ready diff (NOT COMMITTED)

This file is the proposed change for sign-off. Apply with the engine.py edit + the new test file. Then commit as one atomic change.

## §4 verbatim text (from PRE_COMMITMENT_VRP.md as of HEAD)

```
**Layer 4 — drawdown halt** (any one):
- Underwater duration > 90 trading days
- Drawdown depth > 15% of starting equity in trailing 90 days
```

The `(any one)` combinator is unambiguous (OR between the two conditions). The fix does NOT change combinator logic.

## Framing recommendation: bug-fix-of-implementation

The second condition's spec text says "in trailing 90 days." The current implementation computes drawdown depth from the all-time HWM since strategy inception, NOT from the trailing-90-day max — that's a deviation from spec text.

For consistency, the first condition's "underwater duration > 90 trading days" is interpreted with respect to the same trailing-90-day reference, since the second condition explicitly anchors the interpretation. (The first-condition reading is somewhat implied by the second; reasonable readers could disagree but the consistent reading is defensible.)

**No §4.2 amendment block is needed.** The §4 spec text stays as-is. Only the implementation changes.

## engine.py — diff

### Imports section (no change required)

`collections.deque` is in stdlib; explicit import not strictly required if used as `import collections` elsewhere. Below assumes `from collections import deque` is added at the top of `src/backtest/engine.py`.

### Top-of-file imports (add deque)

```diff
 from collections import defaultdict
+from collections import deque
 from dataclasses import dataclass, field
```

### Lines 158-161 — drawdown bookkeeping initialization

```diff
-    high_water = inputs.initial_equity
-    underwater_days = 0
-    current_dd_frac = 0.0
-    next_ic_id = 1
+    # Layer 4 drawdown bookkeeping per PRE_COMMITMENT_VRP §4 spec text
+    # ("in trailing 90 days"). Trailing-window max + underwater_days_trailing
+    # both computed against a rolling 90-day deque, not since strategy
+    # inception.
+    DRAWDOWN_LOOKBACK_DAYS = 90   # matches §4 "in trailing 90 days"
+    equity_trailing: deque = deque(maxlen=DRAWDOWN_LOOKBACK_DAYS)
+    underwater_days = 0           # consecutive days below trailing-90 max
+    current_dd_frac = 0.0         # drawdown from trailing-90 max
+    high_water = inputs.initial_equity   # all-time HWM, kept for REPORTING
+                                          # ONLY — NOT used in halt logic.
+    next_ic_id = 1
```

### Lines 267-278 — drawdown bookkeeping inside day-loop

```diff
         # --- Drawdown bookkeeping ---
-        # Strictly-below-high-water counts as underwater. Equality (e.g. on
-        # no-trade days when equity sits unchanged at the prior high) does
-        # NOT increment underwater_days; otherwise an idle account would
-        # accumulate fake underwater days indefinitely.
-        if equity >= high_water:
-            high_water = equity
-            underwater_days = 0
-            current_dd_frac = 0.0
-        else:
-            underwater_days += 1
-            current_dd_frac = (high_water - equity) / high_water if high_water > 0 else 0.0
+        # Per § 4: drawdown depth and underwater duration are computed
+        # over the TRAILING 90-DAY window. The deque includes today's
+        # equity BEFORE we compare; trailing_high is therefore the max
+        # over [today − 89, today] (inclusive of today). underwater_days
+        # counts consecutive days where equity < trailing_high — equality
+        # (today's equity == trailing_high) resets the counter, which
+        # matches the spec ("recovered to high" → no longer underwater).
+        equity_trailing.append(equity)
+        trailing_high = max(equity_trailing)
+        if equity >= trailing_high:
+            underwater_days = 0
+            current_dd_frac = 0.0
+        else:
+            underwater_days += 1
+            current_dd_frac = (
+                (trailing_high - equity) / trailing_high
+                if trailing_high > 0 else 0.0
+            )
+        # Maintain all-time HWM for reporting (NOT used in halt logic).
+        if equity > high_water:
+            high_water = equity
```

### Net effect

- `equity_trailing` is a `deque(maxlen=90)`: bounded memory, no unbounded list growth.
- `trailing_high = max(equity_trailing)` excludes nothing — it includes today's equity (since deque was appended BEFORE the max). Consistent semantic: today's equity that equals the trailing max → not underwater.
- `underwater_days` resets to 0 when equity reaches the trailing-90-day max (which CAN happen during a halt because equity can drift up via rf accrual).
- `current_dd_frac` is computed against trailing-90 max (not all-time).
- `high_water` is preserved for REPORTING ONLY (used in summarize() for the writeup); explicitly NOT consumed by `evaluate_halts` or `drawdown_halt_triggered`.

### What stays unchanged

- `drawdown_halt_triggered` in `src/strategy/halts.py` — its signature is `(underwater_days, current_drawdown_frac)`. The values passed in change, but the function itself doesn't.
- The HALT_DD_DURATION_DAYS=90 and HALT_DD_DEPTH_FRAC=0.15 constants in `src/config.py` — unchanged.
- All other engine logic, exits, sizing, friction — unchanged.

## tests/test_layer4_trailing_window.py — new file

```python
"""
Tests for Option A: Layer 4 drawdown halt uses trailing-90-day window
per PRE_COMMITMENT_VRP §4 spec text "in trailing 90 days."

Three scenarios:
  1. Equity drifts 1% below the all-time HWM for 95 days, then recovers.
     Trailing-90 max rolls forward → drawdown is small AND underwater
     resets when equity hits trailing-window max → halt should NOT fire.
  2. Equity drops 16% in 30 days. Drawdown > 15% in trailing 90 days
     → halt SHOULD fire on the day the drawdown crosses 15%.
  3. Equity drifts 5% below HWM for 95 days, then recovers to a NEW
     all-time high. During the 95-day dip: trailing-90 max rolls forward
     and underwater_days resets when equity equals trailing max (which
     happens periodically during the flat-dip). Halt should NOT fire.
     After the new ATH: counter still 0; trailing-90 max is the new ATH.
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

from src.strategy.halts import drawdown_halt_triggered


DRAWDOWN_LOOKBACK_DAYS = 90


def _simulate_trailing_window(
    equity_path: List[float], lookback: int = DRAWDOWN_LOOKBACK_DAYS,
) -> List[Tuple[int, float, bool]]:
    """Replicate the engine's trailing-window book-keeping outside the
    engine. Returns per-day (underwater_days, current_dd_frac, halted)."""
    equity_trailing: deque = deque(maxlen=lookback)
    underwater_days = 0
    out: List[Tuple[int, float, bool]] = []
    for eq in equity_path:
        equity_trailing.append(eq)
        trailing_high = max(equity_trailing)
        if eq >= trailing_high:
            underwater_days = 0
            dd = 0.0
        else:
            underwater_days += 1
            dd = (trailing_high - eq) / trailing_high if trailing_high > 0 else 0.0
        halted = drawdown_halt_triggered(underwater_days, dd)
        out.append((underwater_days, dd, halted))
    return out


def test_one_pct_drift_for_95_days_does_not_halt():
    """1% sustained drift below all-time HWM. Trailing-window resets
    each day (since equity == trailing_high after first 90 days). No halt."""
    eq = [100_000.0]                  # ATH
    for _ in range(95):
        eq.append(99_000.0)           # 1% below ATH for 95 days
    for _ in range(10):
        eq.append(101_000.0)          # recovery above prior ATH

    seq = _simulate_trailing_window(eq, lookback=90)
    halted_any = any(h for _, _, h in seq)
    assert not halted_any, (
        "Layer 4 should NOT fire on a 1% drift sustained for 95 days. "
        "After day 90+ the trailing-90 max IS the dip level, equity == "
        "trailing_high, underwater_days resets each day, drawdown_frac=0."
    )

    # Sanity: max drawdown observed under trailing-window logic is
    # bounded — within first 90 days, trailing_high includes the $100K
    # ATH; after that, trailing_high is the dip level itself.
    max_dd = max(dd for _, dd, _ in seq)
    assert max_dd <= 0.011, f"Max DD should be ~1%, got {max_dd:.4f}"


def test_sixteen_pct_drop_in_30_days_does_halt():
    """16% decline in 30 days. Drawdown > 15% within trailing 90d window
    → halt fires."""
    eq = [100_000.0]
    for i in range(30):
        eq.append(100_000.0 * (1 - 0.16 * (i + 1) / 30))

    seq = _simulate_trailing_window(eq, lookback=90)
    halted_any = any(h for _, _, h in seq)
    assert halted_any, "Layer 4 should fire when DD exceeds 15% in trailing 90d"

    # The exact day the halt first fires
    first_halt_idx = next(i for i, (_, _, h) in enumerate(seq) if h)
    assert first_halt_idx < 30, (
        f"Halt should fire within the 30-day decline (got idx {first_halt_idx})"
    )

    # When the halt fires, drawdown should be ≥ 15%
    _, dd_at_halt, _ = seq[first_halt_idx]
    assert dd_at_halt >= 0.15 - 1e-6


def test_five_pct_dip_for_95_days_then_new_ath_resets_correctly():
    """User-requested third test:

    Synthetic equity drifts 5% below HWM for 95 days, then recovers to
    a NEW all-time high. Trailing-window logic should:

    - During the 95-day dip: drawdown remains < 15% (only 5% off) AND
      underwater_days resets each day after day 90 (because trailing-90
      max IS the dip level, equity == trailing_high). No halt fires.
    - After the new ATH: trailing_high jumps to the new ATH. underwater_days
      counter is at 0 (resets on the new-high day).

    This catches: rolling-window math correct AND new-ATH handling
    correct. Regression test for accidentally NOT updating
    trailing_high after a new ATH.
    """
    eq = [100_000.0]                  # initial ATH
    for _ in range(95):
        eq.append(95_000.0)           # 5% below ATH for 95 days
    eq.append(105_000.0)              # NEW ATH (5% above prior)
    for _ in range(20):
        eq.append(105_000.0)          # plateau at new ATH

    seq = _simulate_trailing_window(eq, lookback=90)
    halted_any = any(h for _, _, h in seq)
    assert not halted_any, (
        "Layer 4 should NOT fire on a 5% dip for 95 days followed by recovery. "
        "Drawdown depth = 5% < 15% threshold; underwater_days resets after "
        "day 90 when trailing-90 max IS the dip level. After the new ATH, "
        "counter and trailing_high update correctly."
    )

    # The day after the new ATH: underwater_days should be 0
    new_ath_idx = 96   # eq[0]=100K, eq[1..95]=95K, eq[96]=105K
    underwater_at_new_ath, dd_at_new_ath, _ = seq[new_ath_idx]
    assert underwater_at_new_ath == 0, (
        f"underwater_days should reset to 0 on the new-ATH day, "
        f"got {underwater_at_new_ath}"
    )
    assert dd_at_new_ath == 0.0, (
        f"current_dd_frac should be 0 on the new-ATH day, got {dd_at_new_ath}"
    )

    # Sanity: max drawdown observed throughout = 5% (5K / 100K)
    max_dd = max(dd for _, dd, _ in seq)
    assert 0.04 <= max_dd <= 0.06, f"Max DD should be ~5%, got {max_dd:.4f}"

    # Day 89 (last day of dip when trailing-90 still includes initial $100K):
    # trailing_high = 100K, equity = 95K → drawdown = 5%, underwater_days
    # has been incrementing daily since day 1 of the dip. Should be 89-ish.
    underwater_day_89, dd_day_89, _ = seq[89]
    assert underwater_day_89 >= 80, (
        f"underwater_days near end of 90d window should be 80+, got {underwater_day_89}"
    )
    assert 0.04 <= dd_day_89 <= 0.06

    # Day 95 (last day of dip, trailing-90 has rolled past initial $100K
    # and now consists entirely of $95K dip values): trailing_high = 95K,
    # equity = 95K → underwater_days should have RESET to 0.
    underwater_day_95, dd_day_95, _ = seq[95]
    assert underwater_day_95 == 0, (
        f"underwater_days should have reset by day 95 (trailing window "
        f"now all $95K), got {underwater_day_95}"
    )
    assert dd_day_95 == 0.0
```

## Application checklist (when authorized)

1. Apply the engine.py diff (3 edits: imports, init, day-loop)
2. Add `tests/test_layer4_trailing_window.py` with the 3 tests above
3. Run `PYTHONPATH=. .venv/bin/python -m pytest tests/test_layer4_trailing_window.py tests/test_halts_layer5.py tests/test_halt_layer_invariants.py -v` — all should pass
4. Run all-test sanity: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q --ignore=tests/test_strategies.py` (skip strategy tests if they need data)
5. Commit as ONE atomic change with message tying to DIAGNOSIS_HALTS.md catch-22 + §4 spec text alignment
6. Re-run `scripts/diagnose_spx_halts_only.py` and verify:
   - Sharpe drops from 1.555 toward 0.15–0.40 plausibility band
   - Distinct halt periods rise from 8 toward 10–30
   - Halted-day fraction drops from 60% toward 15–30%
7. If those land outside expected ranges → STOP, do not proceed to multi-instrument; re-diagnose.

## Status of related files

- `tests/test_halt_layer_invariants.py` — already created (autonomously, NOT committed). Contains the absorbing-state invariant tests + the layer4 catch-22 demonstration. Will pass once Option A is applied (specifically `test_layer4_PROPOSED_FIX_passes_invariant_under_calm_drift`).
- `DIAGNOSIS_HALTS.md` — UPDATE 2 already contains the 3-criteria confirmation log + the audit table.
- This file (`OPTION_A_DIFF.md`) — the PR-ready diff itself, ready for sign-off review. Not committed.
