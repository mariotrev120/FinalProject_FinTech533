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
