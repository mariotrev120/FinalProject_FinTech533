# DIAGNOSIS_HALTS.md

Status as of 2026-05-03 03:40 UTC.

## SPX halts_only OOS — diagnostic results

| Metric | Value |
|---|---|
| n_trades | 237 |
| Final equity | $107,856 (started $100K) |
| Total return | +7.86% over 7y |
| Annualized return | +1.09% |
| Max drawdown | **−1.45%** |
| Naive Sharpe | **1.555** |
| v1.5 anchor (halts_only put-only) | 0.286 |
| Delta vs anchor | +1.269 |

**User's plausibility band:** 0.15–0.40. **1.555 is far above the 0.5 ceiling. Per the rule, paused multi-instrument launch.**

## Halt-log breakdown (2018-2024 OOS)

| State / trigger | Count | Notes |
|---|---|---|
| state=active | 703 days (39.9%) | |
| state=halted (any) | **1057 days (60.1%)** | |
| trigger=drawdown | 760 | Layer 4 firing — primary blocker |
| trigger=awaiting_resume | 238 | Layer 5 holding |
| trigger=time_fallback_resumed | 5 | 60-day fallback fires rarely |
| trigger=auto_resumed | 2 | Market conditions clearing rarely |
| trigger=vix_spike,term_inversion_hard | 6 | Hard tail Layer 1 |
| trigger=spx_5pct_move,term_inversion_hard | 7 | |

## 8 distinct halt periods

| Start | End | Duration (trading days) | First trigger |
|---|---|---|---|
| 2018-02-05 | 2018-05-02 | 61 | vix_spike + term_inversion_hard (Volmageddon) |
| 2018-05-21 | 2018-07-19 | 42 | drawdown |
| 2018-10-10 | 2019-05-10 | 146 | vix_spike + term_inversion_hard (Q4 selloff) |
| 2019-08-05 | 2019-10-29 | 61 | term_inversion_hard |
| **2020-02-24 → 2022-10-21** | | **673** | vix_spike + term_inversion_hard (COVID + ongoing) |
| 2022-11-10 | 2022-11-25 | 11 | spx_5pct_move |
| 2024-08-05 | 2024-10-29 | 61 | vix_spike + term_inversion_hard |
| 2024-12-18 | 2024-12-31 | 9 | vix_spike + term_inversion_hard (still halted) |

Halts-active weeks per calendar year:
- 2018: 31.4 weeks
- 2019: 29.8 weeks
- 2020: 43.6 weeks
- 2021: **50.4 weeks** (basically all of 2021)
- 2022: 42.4 weeks
- 2023: 0.0 weeks
- 2024: 13.8 weeks

## Root-cause analysis

The **2020-02-24 → 2022-10-21 (673-day) halt** is the smoking gun. Tracing:

1. **2020-02-24:** COVID-driven Layer 1 hard halt fires (VIX spike + term inversion).
2. **By June 2020:** Layer 1 conditions clear (term structure normalized post-Fed intervention).
3. **By Sep 2020:** market conditions all calm by every reasonable measure.
4. But: **drawdown halt** (Layer 4) keeps firing because of how `underwater_days` is computed.

Engine drawdown logic (`src/backtest/engine.py`):
```python
if equity >= high_water:
    high_water = equity
    underwater_days = 0
    current_dd_frac = 0.0
else:
    underwater_days += 1
```

`underwater_days` only resets when equity reaches a **NEW ALL-TIME HIGH**. If equity recovers to *near* but not exactly the prior all-time high, the counter keeps incrementing forever.

`drawdown_halt_triggered`:
```python
if underwater_days >= 90:
    return True
if current_drawdown_frac >= 0.15:
    return True
```

So **even a 1% drawdown that lasts 90 days** fires the halt — same logical structure as the v1 Layer 5 catch-22 we already fixed.

### The cycle

1. Time fallback fires after 60 days halted → state="active"
2. Strategy enters a few trades
3. Some lose → equity dips below all-time-high
4. `underwater_days` starts climbing
5. After 90 more days underwater (even with tiny PnL drift), Layer 4 drawdown halt re-fires
6. Wait 60 more days for time fallback...
7. Cycle repeats

Net effect: ~60% halted. Strategy barely trades, doesn't accumulate enough P&L to reach a new high-water mark. The Sharpe of 1.555 is a Sharpe-on-near-zero-volatility-of-near-zero-return illusion.

## Why this isn't a "win" (Sharpe-magic-lifter pattern)

A Sharpe of 1.555 with **only 237 trades over 7 years** and **max DD of 1.45%** is not a credible result. The strategy is gaming the rf-rate accrual on idle cash by being halted 60% of the time. Vestal would call this out instantly:

- The v1.5 baseline (which had the same Layer 4 design) generated 210 halts_only trades and Sharpe 0.286. So why does v2 with the same Layer 4 give Sharpe 1.555?
- Iron-condor mode produces ~2× the trade count of put-only mode. With Layer 4 catch-22, IC produces small per-trade losses (call wing during 2020-2024 bull market) → strategy more often underwater → Layer 4 more often firing.
- The v1.5 baseline used put-only spreads which were profitable enough to keep equity at near-all-time-highs more often → Layer 4 fired less.

So the same Layer 4 design that was fine for v1's put-only happens to be pathological for v2's iron condors. **This is a v2-specific issue.**

## Proposed Layer 4 fix (NOT IMPLEMENTING — needs user sign-off)

Two structural options:

**Option A — Trailing-90d window (matches the spec text in v1 PRE_COMMITMENT):**
```python
# v1 PRE_COMMITMENT.md said: "Drawdown depth > 15% of starting equity in trailing 90 days"
# Implementation should track:
trailing_high = equity_curve.rolling(90).max().iloc[-1]
trailing_dd = (trailing_high - equity_now) / trailing_high
underwater_days_trailing = (equity_curve.iloc[-90:] < trailing_high).sum()
```
This matches the actual spec text "trailing 90 days," not "since inception all-time high."

**Option B — Reset underwater_days when equity is within 1% of HWM:**
```python
if equity >= high_water * 0.99:   # within 1% of high water → "recovered enough"
    underwater_days = 0
```

**Option C — Add a Layer 4 time fallback:**
Same as Layer 5: drawdown halt auto-lifts after 60 trading days regardless. (But this might just produce the same cycle.)

I recommend **Option A** because:
- It matches the v1 PRE_COMMITMENT spec text literally ("in trailing 90 days").
- The current implementation is arguably a deviation from spec already.
- The change is "fixing implementation to match spec" rather than "changing spec."

But **NOT IMPLEMENTING WITHOUT USER SIGN-OFF.** This is a methodology change to a previously-committed parameter, similar to the Layer 5 catch-22 fix.

## What I'm doing autonomously while you sleep

1. ✅ Diagnostic report (this file) — committed
2. **Run train_head1_per_instrument** — labels naked-mode trades, not affected by halt issues. Will produce per-instrument win-rate labels for Head 1 training.
3. **Run train_head3_per_instrument** — labels skew direction (which side hurt more), not affected by halts.
4. **Run multi-instrument NAKED mode** — no halts at all. Lower-bound baseline that's clean of Layer 4 issues. Tells us how aggregate VRP cluster performs without any halt machinery.
5. **Build C5 Hoeffding monitor + C7 Hodrick SE** — pure logic, no data load, accumulates while data runs execute.
6. **Build the ablation framework runner scaffolding** — even if mode results are pending, the infrastructure can be ready.

Final overnight delivery: a writeup-ready set of artifacts + clear list of methodology decisions awaiting your morning sign-off.

---

# UPDATE 2 (2026-05-03 ~10:13 UTC)

User reset overnight scope. Strict order: Layer 4 confirmation → §4 framing → Option A diff (uncommitted) → halt-layer audit → no Head 1/3, no multi-instrument. **Don't auto-commit anything.**

## Layer 4 confirmation — 3 criteria, all PASS → diagnosis LOCKED

Confirmation script: `scripts/confirm_layer4_diagnosis.py`. Re-runable.

| Criterion | Threshold | Observed | Result |
|---|---|---|---|
| (A) avg per-trade PnL | within ±0.10% of starting capital | **−0.0199%** | **PASS** |
| (B) median trade duration | < 5 trading days | **3 days** | **PASS** |
| (C) trades in halted-then-released windows | ≥ 200 of 237 | **231 of 237** | **PASS** |

Ancillary statistics:
- Mean PnL per trade: −$19.93 on $100,000 starting capital
- Median PnL per trade: −$8.50
- Mean trade duration (calendar days): 5.0
- First halt fires 2018-02-05 (Volmageddon). 6 trades occur before; 231 of remaining 237 trades occur during active-windows that follow at least one halt.

**Diagnosis is locked.** Layer 4 catch-22 confirmed. Proceeding with the framing and PR-ready diff per user's plan.

## §4 verbatim text from PRE_COMMITMENT_VRP.md

Pulled directly from the file as it currently stands (after the Layer 5 v2 amendment):

```
**Layer 4 — drawdown halt** (any one):
- Underwater duration > 90 trading days
- Drawdown depth > 15% of starting equity in trailing 90 days
```

Two conditions, OR-combinator ("any one"). Note the operative phrase **"in trailing 90 days"** in the second condition.

## Framing recommendation — bug fix, NOT methodology amendment

**Combinator logic ("any one" / OR) is unambiguous in §4 text.** The fix does NOT change combinator logic. Proceeding without §4.2 amendment disclosure.

**The fix corrects the second condition's IMPLEMENTATION to match the spec text.** Current code interprets "drawdown depth" as `(all_time_HWM − current_equity) / all_time_HWM`. Spec text says "drawdown depth > 15% of starting equity **in trailing 90 days**." The literal reading is: drawdown computed over a TRAILING-90-DAY window, not since strategy inception.

For consistency, the first condition's "underwater duration > 90 trading days" should also be interpreted with respect to a trailing-90-day reference (i.e., underwater w.r.t. the trailing-90-day max). This is a consistency-with-spec interpretation; the spec doesn't explicitly say but the second condition's "in trailing 90 days" anchors the framing. **First-condition consistency reading is implied, not strictly forced; reasonable readers could disagree.**

**Recommendation:** treat as bug-fix-of-implementation for the second condition (which is unambiguous). Apply the SAME trailing-90-day reference to the first condition for consistency. If user prefers strictly literal: only fix the second condition and leave the first condition's "all-time HWM" reading. Either choice closes the catch-22 because the trailing-window resets within 90 days.

---

## Option A diff — PR-ready, NOT committed

### Diff for `src/backtest/engine.py` (~10 lines changed)

Replace the all-time-HWM book-keeping with trailing-90-day book-keeping. Conceptually:

```python
# BEFORE (current implementation, deviates from spec text "in trailing 90 days"):
high_water = inputs.initial_equity
underwater_days = 0
current_dd_frac = 0.0
...
# Inside day loop:
if equity >= high_water:
    high_water = equity
    underwater_days = 0
    current_dd_frac = 0.0
else:
    underwater_days += 1
    current_dd_frac = (high_water - equity) / high_water if high_water > 0 else 0.0
```

```python
# AFTER (matches spec "in trailing 90 days"):
DRAWDOWN_LOOKBACK_DAYS = 90  # imported from src.config; matches §4 "in trailing 90 days"
equity_history = []   # rolling, only last DRAWDOWN_LOOKBACK_DAYS kept
underwater_days = 0
current_dd_frac = 0.0
...
# Inside day loop:
equity_history.append(equity)
if len(equity_history) > DRAWDOWN_LOOKBACK_DAYS:
    equity_history = equity_history[-DRAWDOWN_LOOKBACK_DAYS:]
trailing_high = max(equity_history)
if equity >= trailing_high:
    underwater_days = 0
    current_dd_frac = 0.0
else:
    underwater_days += 1
    current_dd_frac = (trailing_high - equity) / trailing_high if trailing_high > 0 else 0.0
```

Net effect: `underwater_days` resets when equity reaches the TRAILING-90-day max (not the all-time HWM). `current_dd_frac` is computed against the trailing-90 max. Both align with the spec text "in trailing 90 days."

### Reference unit tests (in `tests/test_layer4_trailing_window.py`, NOT committed)

```python
"""
Tests for Option A: Layer 4 drawdown halt uses trailing-90-day window
per PRE_COMMITMENT_VRP §4 spec text.

Two synthetic scenarios:
  1. Equity drifts 1% below HWM for 95 days, then recovers to a NEW
     trailing-90-day high. Halt should NOT fire (within trailing window
     the dd is small AND duration counter resets when equity ≥ trailing
     max).
  2. Equity drops 16% in 30 days. Halt SHOULD fire (drawdown >15% in
     trailing 90 days, satisfying the second condition).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.halts import drawdown_halt_triggered
from src.backtest.engine import run_backtest, BacktestInputs
# Note: the actual rolling-window logic lives in run_backtest; tests
# here exercise the helper as well as a synthetic engine-equivalent
# day-loop.


def _synthetic_run(equity_path, lookback=90):
    """Simulate the trailing-90-day book-keeping outside the engine to
    test the math directly. Returns the per-day (underwater_days,
    current_dd_frac, halted) sequence."""
    trail = []
    underwater_days = 0
    out = []
    for eq in equity_path:
        trail.append(eq)
        if len(trail) > lookback:
            trail = trail[-lookback:]
        trailing_high = max(trail)
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
    """Synthetic equity drifts to 99% of HWM for 95 days then recovers.
    Drawdown is 1% (< 15% threshold) and underwater-days resets when
    equity hits the trailing-90-day max — should NOT halt."""
    eq = [100_000.0]
    for _ in range(95):
        eq.append(99_000.0)   # 1% below the all-time start
    for _ in range(10):
        eq.append(101_000.0)  # recovery above prior HWM

    seq = _synthetic_run(eq, lookback=90)
    halted_any = any(h for _, _, h in seq)
    assert not halted_any, (
        "Layer 4 should NOT fire on a 1% drift sustained for 95 days "
        "(drawdown depth is below 15%, and trailing-window reset "
        "prevents underwater_days from accumulating beyond 90)."
    )


def test_sixteen_pct_drop_in_30_days_does_halt():
    """Synthetic equity drops 16% in 30 days. Drawdown depth > 15%
    (well in trailing 90 days) → halt SHOULD fire on the day the
    drawdown crosses 15%."""
    eq = [100_000.0]
    # 30 days of monotonic decline to 84% of starting
    for i in range(30):
        eq.append(100_000.0 * (1 - 0.16 * (i + 1) / 30))

    seq = _synthetic_run(eq, lookback=90)
    halted_any = any(h for _, _, h in seq)
    assert halted_any, "Layer 4 should fire when drawdown exceeds 15% within the trailing 90d window."

    # The exact day the halt first fires is when (trailing_high - eq)/trailing_high > 0.15
    first_halt_idx = next(i for i, (_, _, h) in enumerate(seq) if h)
    assert first_halt_idx < 30, "Halt should fire within the 30-day decline."
```

### How to apply

When user signs off:
1. Apply the engine.py diff
2. Add `tests/test_layer4_trailing_window.py` with the two tests above
3. Run all tests under `tests/`
4. Re-run `scripts/diagnose_spx_halts_only.py` — expect SPX halts_only Sharpe to land in the 0.15–0.40 plausibility range, halt-active % to drop from 60% to ~10–20%, distinct halt periods to be in the 10–30 range
5. Commit as a single atomic change

**NOTHING IN THIS UPDATE HAS BEEN COMMITTED.** This file is the DRAFT for morning review.

---

## Halt-layer absorbing-state audit — design pattern lesson

Layer 4 (current) and Layer 5 v1 BOTH had the same catch-22: a halt that latches because the lift condition references strategy state that the halt itself prevents. Pattern is:

> "If a halt's lift condition references a quantity Q that monotonically deteriorates while the halt is active (because the halt prevents Q from improving), then the halt is an absorbing state."

This is a design pattern issue, not two independent bugs. Future halt layers must satisfy an INVARIANT: **the lift condition must be reachable while the halt is active.** Equivalently: every halt layer must include either (a) a market-state-only lift condition (no dependence on strategy state) or (b) a time-based escape hatch.

### Audit table (pre-fix state)

| Layer | What triggers it | Lift condition (pre-fix) | Reachable while halted? |
|---|---|---|---|
| Layer 1 (hard) | VIX spike > 50%; VVIX > 150; HYG-LQD > 3 SD | next-day market state (Layer 1 fresh check is False) | YES — market-only, no strategy dependence |
| Layer 2 (soft, dropped post-audit) | n/a — explicitly removed in v1.5 | n/a | n/a |
| Layer 3 (slow, dropped post-audit) | n/a | n/a | n/a |
| Layer 4 (drawdown) | underwater_days ≥ 90 OR DD ≥ 15% | drawdown < 15% AND underwater_days < 90 (computed from ALL-TIME HWM in current impl) | **NO — strategy state cannot improve while halted; absorbing.** |
| Layer 5 (auto-resume v1) | implicit (lift gate after Layer 1/4) | 4 conditions including DD-recovery to within 3% of HWM | **NO — same DD dependency; absorbing.** |
| Layer 5 v2 (current) | implicit | 3 market-only conditions OR 60-day time fallback | YES — market-only + time escape hatch |

### Invariant unit test template (for any future halt layer)

```python
def test_layer_X_lift_condition_reachable_while_halted():
    """Invariant: Layer X's lift condition must be satisfiable while
    Layer X itself is halted.

    Construct an "all market conditions normalized, strategy idle"
    state and assert the lift condition fires. If it doesn't, the
    layer is an absorbing state — same design pathology as the
    v1 Layer 4 / Layer 5 catch-22.
    """
    # Build a synthetic environment: market is normal in every dim,
    # strategy is idle (no fresh PnL).
    state = build_normalized_market_state_with_strategy_idle()

    # Force the halt to be active.
    state.halted = True

    # Check whether the layer's lift condition would fire.
    can_lift = layer_X_lift_condition(state)
    assert can_lift, (
        f"Layer X is an absorbing state — lift condition cannot fire "
        f"when market is normal and strategy is idle. This is the same "
        f"design pattern as v1 Layer 4 and v1 Layer 5 catch-22. Either "
        f"add a market-only condition or a time-based escape hatch."
    )
```

This template should be added as a `tests/test_halt_layer_invariants.py` and ANY new halt layer must include a passing test of this form.

### Layer 4 fix verification (when applied)

After Option A is applied, the trailing-90-day window means: even if a halt is active and equity is roughly constant (rf accrual ≈ 1% APY), within 90 trading days the equity will have new trailing-90-day highs whenever rf accrual ≥ epsilon. So Layer 4 becomes self-resolving when strategy state is benign — satisfying the invariant.

Verifying this with a unit test that's the inverse of the synthetic test above: "build a benign equity curve with no PnL events, assert Layer 4 lift condition fires within trailing-90-day window."

---

## Morning review checklist (deliverables produced this overnight session)

1. ✅ `DIAGNOSIS_DATA.md` — per-ticker delta/IV/Greeks ranges, NaN-by-year, sample chains. **Documentation only**, no fix proposed.
2. ✅ `DIAGNOSIS_HALTS.md` (this file, updated) — Layer 4 confirmation log (3-criteria stop-test PASS), §4 verbatim quote, framing recommendation (bug-fix of implementation), **Option A diff** (PR-ready, **uncommitted**), 2 unit tests, halt-layer absorbing-state audit, invariant unit test template.
3. ✅ `scripts/confirm_layer4_diagnosis.py`, `scripts/dump_data_quality.py` — re-runable.
4. ✅ `src/metrics/hodrick_se.py` + `tests/test_hodrick_se.py` — Phase C task C7 (committed before user's "don't auto-commit" instruction).
5. ✅ `src/metrics/hoeffding.py` + `scripts/hoeffding_monitor.py` + `tests/test_hoeffding_trader_form.py` — Phase C task C5 (committed before user's "don't auto-commit" instruction).
6. ❌ Multi-instrument NAKED — **NOT RUN.** Per latest user instruction, canceled pending data quality triage.
7. ❌ Head 1 + Head 3 — Head 1 ran before user's "stop" instruction arrived; results documented for review (4 of 5 tickers showed 0% win rate, but this is downstream of the data-quality investigation in DIAGNOSIS_DATA.md). Head 3 NOT run.

**Awaiting morning sign-off on:**
- Option A application (Layer 4 fix)
- Resolution of data-quality investigation paths in DIAGNOSIS_DATA.md
- Authorization to re-run Head 1 / Head 3 / multi-instrument against the corrected baseline
