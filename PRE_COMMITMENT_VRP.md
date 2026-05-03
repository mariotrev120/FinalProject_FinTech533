# PRE_COMMITMENT_VRP.md  *(DRAFT — pending review)*

**Strategy:** Defined-risk volatility-risk-premium harvesting via iron condors on a 5-instrument index-and-ETF cluster.

**Status:** DRAFT. Once the user signs and the corresponding git commit lands, this file is frozen. Any subsequent edit is a methodology breach and is logged as such in the writeup.

**Anchor for v2 acceptance:** v1.5 baseline `halts_only` OOS Sharpe = 0.286 over 210 trades, win rate 69.0% (`artifacts/v1_5_baseline/component_attribution.csv`). v2 SPX iron condor must beat 0.286 *before* multi-instrument expansion is considered to have added value.

---

## 1. Universe (locked)

| Ticker | Class            | Section 1256 | Tax disclosure        | Strike increment | Wing width pts | VRP evidence base |
| ------ | ---------------- | ------------ | --------------------- | ---------------- | -------------- | ----------------- |
| SPX    | Cash index       | Yes          | 60/40 LTCG/STCG       | 5                | 5              | **Strongly supported** |
| RUT    | Cash index       | Yes          | 60/40                 | 5                | 5              | Inferred (similar to SPX) |
| NDX    | Cash index       | Yes          | 60/40                 | 25               | 25             | Weaker support |
| TLT    | Long-bond ETF    | No           | Asymmetric ST/LT      | 1                | 1              | Novel application |
| GLD    | Gold ETF         | No           | Asymmetric ST/LT      | 1                | 1              | Novel application |

**VRP evidence base column — sourced per Carr & Wu (2003/2004) Section 6 Table 5:**

- **SPX (and OEX, DJX in the original paper) — strongly supported.** Carr/Wu §6.1: "the largest t-statistics come from the S&P 500 and S&P 100 indexes and the Dow Jones Industrial Average, which are strongly significant for both variance risk premia and log variance risk premia." Mean log VRP magnitude exceeds −50% per month for S&P and Dow. RP form: t-statistics highly significant. LRP form: t-statistics highly significant.
- **RUT — inferred.** Not in Carr/Wu's original 5-index sample (which was SPX, OEX, DJX, NDX, and the Nasdaq-100 tracking stock QQQ). RUT is a large-cap-equivalent US index; we treat it as similar in VRP structure to SPX, but this is an extrapolation, not direct empirical evidence.
- **NDX — weaker support.** Carr/Wu §6.1: "The Nasdaq-100 index and its tracking stock generate t-statistics that are much lower. The t-statistics on the two Nasdaq indexes are not statistically significant for the variance risk premia RP, albeit significant for the log variance risk premia LRP." Smaller-magnitude negative VRP than SPX.
- **TLT, GLD — novel application of the framework.** Carr/Wu studied 5 stock indexes and 35 individual stocks. Bond ETFs (TLT) and commodity ETFs (GLD) were not in their sample. Theoretical VRP framework (variance swap rate as risk-neutral expected variance, realized variance as ex-post counterpart) carries over to any underlying with a liquid options market. We apply it to TLT and GLD as a novel extension; empirical magnitude is to be measured by the OOS results, with no prior literature anchor.

**Why this set, not bigger:** Indexes diversify across-name idiosyncratic risk, isolating the volatility-of-volatility premium itself. TLT and GLD diversify *across asset class*, so the cluster is not just a leveraged bet on equity vol regime. The trade-off: TLT/GLD lose 1256 treatment, smaller liquidity, and lack direct Carr/Wu empirical support. We accept these and disclose explicitly. The writeup limitations section will state that any positive VRP harvest on TLT and GLD is a novel finding without prior-literature replication.

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

### §5.1 Head 2 stress definition (locked, sourced)

The §5 spec text "VIX > 30 sustained, drawdown event in equity benchmarks, credit-spread blowout" was originally drafted as a programmatic threshold definition. After verifying the literature against this approach, the threshold values are NOT supported by the 8 reference papers — Bollerslev/Tauchen/Zhou treat VRP magnitude as a continuous predictor and use NBER recession bars as a visual overlay only; Carr/Wu study VRP across instruments without dichotomizing into stress/non-stress regimes. Inventing programmatic thresholds (`VIX > 30 sustained 2 days`, `SPX DD > 5% from 60d max`, `credit > 2 SD`, etc.) would require defending each value without literature support.

**Approach (locked): labeled-event source.** Stress labels come from the 5 vetted historical stress events already locked in `scripts/stress_events.py` `EVENTS` constant from v1 audit work:

| Event | Peak date |
|---|---|
| Aug2015 China devaluation | 2015-08-24 |
| Feb2018 Volmageddon | 2018-02-05 |
| Q4-2018 selloff | 2018-12-24 |
| Mar2020 COVID crash | 2020-03-23 |
| Mar2023 banking crisis | 2023-03-13 |

Source: vetted in v1 audit (commit `abdd644` and earlier). No threshold values invented; every label is traceable to a named historical episode.

**Forward-window horizon: 63 trading days (one quarter).**
Source: Bollerslev/Tauchen/Zhou 2009 (`ExpectedStockReturns_and_VarianceRiskPremia.pdf`), Section 3 "Forecasting Stock Market Returns" Table 2 and Figure 3. Specifically:

- **Monthly (h=1) horizon**: VRP slope coefficient 0.39, robust t-stat 1.76, adjusted R² = 1.07%.
- **Quarterly (h=3) horizon**: VRP slope coefficient 0.47, robust t-stat 2.86, adjusted R² = 6.82%. **This is the maximum.**
- 6-month horizon: 0.30, t=2.15, R²=5.42%.
- 9-month horizon: 0.17, t=1.36, R²=2.30% (no longer significant at 5%).
- 12-month: t=1.00, R²=1.23%.
- 24-month: t=0.11, R²=−0.50%.

BTZ Section 3.1: "The quarterly return regression results in a much more impressive t-statistic of 2.86 and a corresponding R² of 6.82%. The t-statistic remains significant at the six-month horizon, but the numerical values and significance then gradually taper off for longer return horizons." This anchors Head 2 to the forecast horizon at which VRP-based predictability is empirically strongest, with explicit numerical evidence.

**With control variables added** (Table 4, multiple regression), quarterly R² rises to 16.76% (VRP+P/E), 17.42% (+CAY), 19.74% (+TMSP+RREL). VRP coefficient stays significant in every joint regression.

**Labeling logic (locked):**
For each trading day D in the trading calendar (taken from `features.parquet.index`):

- Forward window = trading-day positions `(pos(D), pos(D)+63]` — inclusive of D+63, **exclusive of D itself**.
- Label is positive if any of the 5 peak dates has its trading-day position in this window.
- Equivalently, for each peak P at trading-day position `pp`, mark trading-day positions `[pp-63, pp-1]` as positive.
- Peak day itself: position `pp` is at the *end* of the slice (excluded) → negative label. Warning is for the future, not the present.
- Days after peak: negative (no peak in their forward window unless another event is ahead).

Total positive label count is bounded above by 5 × 63 = 315 trading days; actual count is lower if any pair of windows overlaps (which they do not for these 5 dates, separated by 18+ months minimum).

**Naive baseline forecast: trailing 22-day mean of `stress_today`.**
~1 trading month. No literature anchor; flagged as a methodology choice. Brier-score acceptance gate compares model OOS Brier to baseline OOS Brier on the same `forward_stress_label` target.

**Feature set (locked):**
- All columns of `data/processed/features.parquet`: vix, vix3m, vix3m_minus_vix, vvix, vrp_30d, vrp_60d, yc_c0, yc_c1, plus the additional 7 features in that file.
- **VRP itself** (`vrp_30d` and `vrp_60d`) is the strongest single predictor at the quarterly horizon per BTZ 2009 Section 3.1 Table 2 and Carr/Wu 2003 Section 6.1 Tables 5–6. Both papers establish VRP magnitude as a continuous predictor of next-quarter realized variance and excess returns. Including VRP as a Head 2 feature directly leverages this empirical finding.
- **Vol-regime quadrant** (`vol_regime_quadrant` from `src/strategy/vol_regime.py`) — categorical 0..3 based on VIX (level z, derivative z) signs. Per the dual-role design (PRE_COMMITMENT_VRP §7), this is BOTH a Head 1 input AND a standalone gate ablation.
- **Optional**: HMM regime probabilities (`hmm_regimes.parquet`) — 2-state Gaussian HMM. Already in v1 codebase.

**Standard errors for evaluation:**
**Hodrick (1992) standard errors**, NOT Newey-West, for overlapping multi-period forecast significance testing. Source: BTZ 2009 footnote 21 — "Ang and Bekaert (2007) have forcefully shown that in the context of predictive regressions with overlapping observations, the standard errors obtained by summing the regressors in the past, as advocated by Hodrick (1992), are generally more reliable than the more traditional standard errors based on the summation of the residuals into the future as in, for example, Newey and West (1987)." Our 63-day forward window with daily-frequency observations creates exactly this overlap pattern.

**R² interpretation caveat:**
Per Boudoukh, Richardson & Whitelaw (2008) as cited in BTZ Section 3 footnote 22: "even in the absence of any increase in the true predictability, the values of the R²s with highly persistent predictor variables and overlapping returns will by construction increase roughly proportional with the return horizon and the length of the overlap." We will report Hodrick-adjusted t-stats as the PRIMARY significance measure, not raw R². R² is reported descriptively only.

**Acceptance gate (unchanged from §5):** Head 2 OOS Brier must be ≥ 5% lower than the naive baseline. Failure → drop Head 2; book scaler defaults to 1.0; report negative finding.

### §5.2 Head 2 limitations and disclosures (writeup-required)

Head 2 is trained on 5 archetypes of regime break, with multiple methodological gaps relative to the literature's preferred specifications. The following must be disclosed in the writeup limitations section:

1. **Pattern coverage.** The model can only flag regime breaks that *resemble* the 5 historical archetypes — vol-spike-driven sharp peaks. Slow grinding bears (e.g., 2022-style) and structural-shift regimes (e.g., 2008 GFC, which is outside our 2012-2024 sample) are NOT in the training distribution and will not be flagged with high confidence.

2. **Generalization.** Performance OOS depends on future stress regimes resembling the 5 historical archetypes. Disclose explicitly that Head 2 is a "pattern-match against past archetypes" head, not a general-purpose stress detector.

3. **Sample-size disclosure.** Five labeled stress peaks × 63-day forward window ≤ 315 positive-class days against ~3,500 total trading days = ~9% positive base rate. XGBoost can learn signal at this rate but the effective sample size for the rare-class is small. The Brier-score gate is calibrated to be informative at this sample size.

4. **Effective independent observations.** Our 63-day forward window with daily observations creates a high-overlap predictive setup. Effective independent windows ≈ 12 years × 252 trading days / 63 = ~48 non-overlapping windows. BTZ 2009 Section 3 footnote 23 explicitly cautions: "Our limited post-1990 sample prevents us from effectively studying issues having to do with longer return horizons spanning multiple years." Our sample is shorter than BTZ's (12 vs 17 years).

5. **R² inflation per Boudoukh-Richardson-Whitelaw 2008.** Even in the absence of any increase in true predictability, R²s with highly persistent predictors and overlapping returns inflate proportionally with horizon × overlap. We report Hodrick-adjusted t-stats as primary significance measure, R² descriptively only.

6. **Calibration sample-size note.** Niculescu-Mizil & Caruana 2005 Section 5 (learning-curve analysis) shows isotonic calibration overfits when calibration set < ~200 cases. Head 2 with ~300 pooled cross-asset events is borderline. Production training will switch to Platt scaling if any per-fold calibration set falls below 200 cases.

These six disclosures are required in the limitations / going-forward section of the writeup. Failure to disclose them is a methodology breach.

### §5.3 Methodological gaps relative to BTZ's preferred specification

These are documented in the writeup as known approximations, NOT as bugs:

1. **Daily-frequency RV components, not 5-minute intraday.** BTZ Section 3.2.1: "Estimation of the same predictive regressions based on the traditional Black–Scholes implied variances and/or realized variances constructed from lower frequency daily data does not give rise to nearly as significant results." Our HAR-RV inputs use daily-frequency RV. Effect: lower expected R² and weaker t-stats than BTZ's intraday baseline. Magnitude unknown; documented honestly.

2. **Sample length.** BTZ 1990–2007 = 17 years. Our 2012–2024 = 12 years. Effect: smaller effective sample size; wider Hodrick-adjusted CIs.

3. **No stochastic-volatility model layer.** BTZ's underlying theoretical model has explicit volatility-of-volatility process (qt). We treat VRP as a black-box predictor without modeling the underlying vol-of-vol structure. Effect: Head 2 captures empirical regularities but not the structural mechanism. This is consistent with the project's empirical-not-theoretical orientation.

---

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
