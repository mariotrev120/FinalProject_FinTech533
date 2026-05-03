# PRE_COMMITMENT_VRP.md  *(DRAFT — pending review)*

**Strategy:** Defined-risk volatility-risk-premium harvesting via iron condors on a 5-instrument index-and-ETF cluster.

**Status:** DRAFT. Once the user signs and the corresponding git commit lands, this file is frozen. Any subsequent edit is a methodology breach and is logged as such in the writeup.

**Anchor for v2 acceptance:** v1.5 baseline `halts_only` OOS Sharpe = 0.286 over 210 trades, win rate 69.0% (`artifacts/v1_5_baseline/component_attribution.csv`). v2 SPX iron condor must beat 0.286 *before* multi-instrument expansion is considered to have added value.

---

## 1. Universe (locked)

| Ticker | Class            | Section 1256 | Tax disclosure        | Strike increment | Wing width pts |
| ------ | ---------------- | ------------ | --------------------- | ---------------- | -------------- |
| SPX    | Cash index       | Yes          | 60/40 LTCG/STCG       | 5                | 5              |
| RUT    | Cash index       | Yes          | 60/40                 | 5                | 5              |
| NDX    | Cash index       | Yes          | 60/40                 | 25               | 25             |
| TLT    | Long-bond ETF    | No           | Asymmetric ST/LT      | 1                | 1              |
| GLD    | Gold ETF         | No           | Asymmetric ST/LT      | 1                | 1              |

**Why this set, not bigger:** Carr & Wu (2003/2004) document strongly negative VRP for S&P 500 indexes, smaller-magnitude negative VRP for NDX and most individual stocks. Indexes diversify across-name idiosyncratic risk, isolating the volatility-of-volatility premium itself. TLT and GLD diversify *across asset class*, so the cluster is not just a leveraged bet on equity vol regime. The trade-off: TLT/GLD lose 1256 treatment and are smaller in liquidity. We accept this and disclose tax asymmetry explicitly.

**Why not PEP / not 6+:** PEP is currently active in the wheel basket (PRE_COMMITMENT_WHEEL.md §1, 9-name primary), not in the VRP cluster. The VRP universe is 5 instruments locked. Adding more dilutes the per-instrument sample size below the ML acceptance threshold (~60 trades/instrument/fold).

## 2. Entry construction

- **Structure:** Defined-risk iron condor (short put credit + short call credit, both wings protected).
- **Short put delta target:** −0.16 (one-sigma OTM, BS-Greek-defined via OptionMetrics IV when available; BS pricer fallback when IV missing).
- **Short call delta target:** +0.16 (symmetric).
- **Wing widths:** Per table above (5 / 5 / 25 / 1 / 1).
- **DTE at entry:** 30 to 45 calendar days. Selected expiry = the chain whose DTE is in [30, 45], preferring the closest to 35.
- **Entry frequency:** Weekly Monday open per instrument (one structure per name per week, no overlapping spreads in same instrument). Skipped if Monday is a holiday; defer to next session.
- **Strike-snap policy (TWO variants tested, headline locked):** Round to listed-strike grid using nearest-delta. Headline result reports the canonical policy; a robustness-section variant with adjacent-strike sensitivity is required.

**Rationale per parameter:**

| Parameter | Choice | Why |
|---|---|---|
| 16-delta short legs | Standard one-sigma | BS-Greek-defined, robust to IV-skew estimation noise, well-studied in retail VRP literature |
| 30–45 DTE | "Sweet spot" | Theta acceleration begins ~45 DTE; closing at 21 leaves the high-decay window inside the trade |
| 5/5/25/1/1 wings | Match strike grid | Smaller wings = less defined risk reduction; wider wings = uncovered gap on tail; we pick the listed grid's natural minimum |
| Weekly entry cadence | Limits dependency between trades | More frequent (daily) increases overlap and serial correlation; less frequent (monthly) loses sample size |
| One structure per name per week | No stacking | Removes path-dependency where multiple overlapping spreads in one name confound exit attribution |

## 3. Exit logic (per-side independent management — five fates)

A single iron condor opens two `Spread` objects with shared `iron_condor_id`. Each side is managed independently. Five fates total per `Trade`:

| Fate | Trigger | Notes |
|---|---|---|
| `profit_target` | per-side debit ≤ 50% of side's entry credit | PT exits one side without forcing the other |
| `stop_loss` | per-side debit ≥ 200% of side's entry credit | Gap-aware execution (realized fill = min(stop_level, opening_gap_price)) |
| `time_exit` | DTE ≤ 21 (per side) | Closes both sides simultaneously if both still open |
| `emergency` | \|short_delta\| > 0.50 (per side) | Triggered when underlying breaches the short strike materially |
| `eos_force` | Backtest end | Applied at end-of-OOS to force-close residual positions |

**Acceptance:** `evaluate_exit()` (`src/strategy/exits.py`) implements all five for both sides. Engine asserts that every closed `Trade` has `fate != "open"` and `pnl_per_spread is not None` (`src/backtest/engine.py:485`).

## 4. Halt framework (Layers 1, 4, 5 active; Layers 2 & 3 dropped)

**Layer 1 — hard tail-event halt** (any one):
- VIX intraday spike > 50% from prior close
- VVIX > 150
- HYG-LQD spread > 3 SD from 252-day rolling mean

**Layer 4 — drawdown halt** (any one):
- Underwater duration > 90 trading days
- Drawdown depth > 15% of starting equity in trailing 90 days

**Layer 5 — auto-resume (ALL four required to lift halt)**:
1. VIX3M − VIX > 2 points for 5 consecutive closes
2. Realized 5-day vol below 80th percentile of trailing 252 days
3. HYG-LQD spread within 1 SD of long-run mean
4. Drawdown recovered to within 3% of high-water mark

**Layers 2 & 3 dropped** (audit-validated): winrate-below-baseline, hyg-lqd-soft, term-inversion-soft, model-prob-low. v1 audit found each was an anti-signal (-9 to -10 pp lift, blocked profitable trades). Functions retained in `halts.py` as documentation of audited-and-discarded triggers; not invoked.

**Wiring requirement:** Layer 5 helpers (`auto_resume_ready` and 4 components) currently exist but are not called from `evaluate_halts()`. Per spec, this must be wired before backtests fire. Task #2.

## 5. ML — three calibrated heads (per Niculescu-Mizil & Caruana 2005 prescription)

All heads use XGBoost with **isotonic calibration on out-of-fold predictions** via TimeSeriesSplit. Calibration set independence required (Section 2 of the paper) — the model training set is NOT reused for calibration.

**⚠ Calibration method note:** Per-instrument heads have ~60 trades/fold. Niculescu-Mizil & Caruana show isotonic regression overfits below ~200–1000 calibration points and Platt scaling dominates in that regime. **Pre-commitment:** if per-instrument calibration set < 200, switch to Platt scaling (sklearn LogisticRegression on raw scores). For the shared cross-asset Head 2 (300+ trades pooled), isotonic remains. This is a literature-driven adjustment and is reported in the writeup.

### Head 1 — trade quality (per-instrument, continuous sizing)

- **Target:** binary outcome `pnl_per_spread > 0` over the 21-DTE-or-PT-or-SL holding window.
- **Features:** vol-regime quadrant (categorical — see §7), VIX/VIX3M term, HYG-LQD, RV/IV ratio at entry, ML feature suite from v1 (`src/features/`).
- **Output:** calibrated p_quality ∈ [0,1].
- **Sizing curve (continuous, replaces v1's binary 0.55 cutoff):**
  - p < 0.50 → skip trade entirely
  - p ∈ [0.50, 0.70] → linearly scale from 0.6× → 1.0× of base size
  - p ≥ 0.70 → 1.2× (capped)
- **Acceptance criterion (pre-committed):** Head 1 must achieve OOS log-loss strictly below the within-fold majority-class baseline. If it fails on ≥ 3 of 5 instruments, the entire Head 1 sizing curve is dropped and base sizing is used. Negative finding is reported as such.

### Head 2 — regime stress (shared cross-asset)

- **Target:** binary "high-stress event in the next 30–90 trading days" (custom flag combining VIX > 30 sustained, drawdown event in equity benchmarks, credit-spread blowout). Definition locked in code before training.
- **Features:** macro-cross-asset feature set (HMM regime probabilities, term, credit, vol-of-vol).
- **Output:** calibrated p_stress ∈ [0,1].
- **Book scaler:** aggregate book size = base × (1 − p_stress). One scaler shared across all 5 instruments.
- **Acceptance criterion:** Head 2 must beat naive "use trailing 1-month average stress rate" baseline OOS by ≥ 5% Brier-score reduction. If it fails, Head 2 is dropped and book scaler defaults to 1.0. Negative finding is reported.

### Head 3 — skew direction (per-instrument, dynamic strike selection)

- **Target:** which side (put or call) realizes higher loss probability over the next 30 days, given current skew, term, and momentum.
- **Output:** dynamic delta selection from {0.10, 0.16, 0.25} per side, replacing the static 0.16/0.16.
- **Acceptance criterion (pre-committed):** Head 3 OOS log-loss must beat majority-class baseline on ≥ 4 of 5 instruments. If fails, **fall back to static 0.16/0.16** for that instrument; static fallback is the headline result for failing instruments and is reported.

## 6. Capital allocation across instruments

- **Per-instrument weight:** w_i = p_quality_i / Σ_j p_quality_j across the 5 instruments at each entry session.
- **Aggregate book scaler:** book = base_book × (1 − p_stress) per Head 2.
- **Per-trade hard cap:** 1.0% of account equity per spread (defined-risk = max-loss is bounded).
- **Combined sizing formula at trade open:**
  ```
  size_i = base_size × w_i × (1 − p_stress) × continuous_quality(p_quality_i)
  ```

**Why this rule and not equal-weight:** equal-weight ignores cross-sectional differences in trade-quality signal. Quality-proportional weighting puts capital where the model most confidently expects edge.

## 7. Vol-regime quadrant classifier (dual role — input AND ablation baseline)

- **Per-instrument calibration:** standardize VIX/RVX/VXN/MOVE/GVZ to z-units using trailing 252-day mean and std; classify into 4 quadrants by the signs of (level z, derivative z).
- **Role 1:** categorical input feature to Head 1.
- **Role 2:** standalone gate ablation (vol-regime-only mode in the ablation matrix), comparator against the full ML stack.

## 8. Friction model

| Item | Spec |
|---|---|
| Slippage | 30 / 50 / 75 / 100 % of bid-ask half-spread for VIX buckets <20 / 20–30 / 30–40 / >40 |
| Commissions | $0.65 per option contract per side |
| Gap-aware fill | Stop-loss fill = min(stop_level, opening_gap_price) |
| Monday gap rule | Gap ≥ 1.5% adds 50% slippage; gap ≥ 3% skips entry that week |
| Tax overlay (1256) | SPX/RUT/NDX: 60% LTCG / 40% STCG mark-to-market annually |
| Tax overlay (non-1256) | TLT/GLD: full ST treatment of options gains; held shares get standard ST/LT split if assigned |

## 9. Sample windows

- **IS (training):** 2012-03-26 → 2017-12-31  *(per `src/config.py` IS_START / IS_END)*
- **OOS (reporting):** 2018-01-01 → 2024-12-31  *(per `src/config.py` OOS_START / OOS_END)*
- **Walk-forward refit cadence:** annually on the IS partition; OOS predictions made with prior-trading-day feature shift (Bug #1 fix preserved from v1).

## 10. Reproducibility

- `SEED = 42` (`src/config.py`) consumed by XGBoost `random_state`, isotonic CV split, block-bootstrap, all sklearn `random_state`.
- Reproducibility test: same commit, same data, same seed → identical equity curve to floating-point precision. Required to pass before each gate.

## 11. Sensitivity grid for DSR / PBO

The sensitivity grid for Deflated Sharpe Ratio (Bailey & López de Prado 2014) and PBO via CSCV (Bailey/Borwein/López de Prado/Zhu 2015) must be **declared in advance**, not chosen post hoc:

| Dimension | Grid values | Count |
|---|---|---|
| Short delta target | 0.10, 0.16, 0.25 | 3 |
| Spread width (×base) | 0.5×, 1.0×, 2.0× | 3 |
| DTE entry window | (25,40), (30,45), (35,50) | 3 |
| Profit target | 25%, 50%, 75% | 3 |
| Stop loss multiple | 1.5×, 2.0×, 3.0× | 3 |
| Time exit DTE | 14, 21, 28 | 3 |

**Total trials N = 3⁶ = 729.** Per the DSR paper, average pairwise correlation between trials must be estimated for the implied-independent-trials count N̂ via Eq. 9 (`N̂ = ρ̄ + (1−ρ̄)·M`). The headline DSR uses N̂, not raw N.

**Strike-snap policy is handled separately** (NOT as a grid dimension). The grid above is purely strategy-parameter sensitivity. Strike-snap is a *data-handling* policy choice (which listed strike to round to when the delta-target falls between two listed strikes — nearest-delta vs nearest-strike). Adding it to the grid would conflate strategy-parameter sensitivity with execution-policy sensitivity. Instead: the headline result is locked under the canonical policy (nearest-delta rounding); the alternative policy (nearest-strike rounding) is reported as a single-number robustness check in the writeup, not multiplied through the grid. Two policies × 729 grid points = 1458 trials would also unnecessarily inflate N for DSR.

## 12. Acceptance gates (pre-committed, in order)

1. SPX iron condor OOS Sharpe > 0.286 (v1.5 anchor) — *required to proceed to multi-instrument*.
2. Aggregate 5-instrument iron condor OOS Sharpe ≥ SPX-only Sharpe — *required to keep multi-instrument architecture*.
3. Each ML head meets its own §5 acceptance criterion or is dropped per fallback rule.
4. DSR (PSR threshold) ≥ 0.95 against the N̂-trial benchmark — *required for "statistically distinguishable from selection-bias winner"*.
5. PBO via CSCV ≤ 0.30 on the §11 sensitivity grid — *if violated, headline is reported with explicit "high overfit probability" caveat*.
6. Reproducibility test passes (seed = 42, same equity curve to FP precision).

## 13. Sign-off

- **Authored:** Mario Trevino
- **Reviewed:** [pending user signature]
- **Timestamp:** [git commit datetime — populated at commit]
- **Git commit hash:** [populated at commit]
- **IS-data commit hash:** [populated at commit]
- **Universe locked:** SPX, RUT, NDX, TLT, GLD. No additions, no PEP.
