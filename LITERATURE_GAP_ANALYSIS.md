# LITERATURE_GAP_ANALYSIS.md

For each of the 8 papers in `/home/mht120/projects/FinTech533/Literature/`, what the paper prescribes, where the codebase falls short, and the severity. Categorized: **ship-blocking** (must fix or the deliverable is methodologically unsound), **ship-improving** (the result is defensible without it but stronger with it), **post-ship** (interesting follow-up).

---

## 1. Bailey & López de Prado (2014) — Deflated Sharpe Ratio

**Prescribes:**
- Eq. 1: SR̂_0 = √V[{SR̂_n}] · ((1−γ)·Z⁻¹[1−1/N] + γ·Z⁻¹[1−1/(N·e)]) — the noise-floor Sharpe under H0 of zero true skill across N independent trials.
- Eq. 2: DSR = Z[((SR̂ − SR̂_0)·√(T−1)) / √(1 − γ̂_3·SR̂ + ((γ̂_4−1)/4)·SR̂²)] — probability that true Sharpe exceeds SR̂_0 given observed SR̂, sample length T, skew, kurtosis.
- Eq. 9: N̂ = ρ̄ + (1−ρ̄)·M — number of *implied independent* trials given M dependent trials with average pairwise correlation ρ̄.
- All quantities computed at per-period scale (not annualized).

**Codebase state (`src/metrics/deflated_sharpe.py`):**
- ✅ Eq. 1 and Eq. 2 implemented (per audit report).
- ✅ `deflated_sharpe_ratio()` takes returns, n_trials, sharpe_var_across_trials_per_period, periods_per_year.
- ✅ Optional-wired into `PortfolioReport` via `dsr_n_trials` and `dsr_sharpe_var_per_period` kwargs.
- ❌ **GAP — Eq. 9 (N̂ from ρ̄) is NOT implemented.** Current code requires the user to pass raw N (number of grid trials, e.g. 729). The DSR paper warns explicitly: "using M instead of N will overstate E[max{y_n}]". For correlated grid trials (which a sensitivity grid always produces — adjacent delta values produce highly correlated equity curves), raw N inflates SR̂_0, making DSR more conservative than warranted. Need a helper that takes the (T × N) returns matrix M, computes ρ̄, returns N̂.
- ❌ **GAP — DSR is currently NOT WIRED into any headline result.** The optional kwargs exist but no caller passes them. Per audit, `build_portfolio_report` is invoked without DSR kwargs in current scripts.

**Severity:** **ship-blocking** for DSR being credible (Eq. 9 fix + wiring); **ship-improving** if user accepts raw-N as a deliberately conservative estimate.

**Fix (in order of priority):**
1. Add `n_eff_from_matrix(M: pd.DataFrame) -> tuple[int, float]` to `deflated_sharpe.py` returning (N̂, ρ̄).
2. Wire the DSR sensitivity-grid runner (Phase C task C1) to pass N̂ and `sharpe_var_per_period`.
3. Re-run all headline reports with DSR populated.

---

## 2. Bailey/Borwein/López de Prado/Zhu (2015) — Probability of Backtest Overfitting via CSCV

**Prescribes:**
- Algorithm 2.3: form (T × N) performance matrix M; partition rows into S=16 disjoint submatrices; form C(S, S/2) = 12,780 combinations; for each combination compute IS-best strategy n* and its OOS relative rank ω̄_c; compute logit λ_c = ln(ω̄_c / (1−ω̄_c)); PBO = fraction of c with λ_c < 0.
- Reproducible, model-free, non-parametric.
- Also enables: performance-degradation slope (R̄ vs R regression), probability of OOS loss, stochastic dominance check.

**Codebase state:**
- ❌ **GAP — Not implemented.** No `src/metrics/pbo.py`. No CSCV machinery anywhere.
- ✅ Foundation exists: the sensitivity grid (Phase C task C1) will produce the (T × N) matrix; `src/metrics/bootstrap.py` already does block resampling that shares some shape.

**Severity:** **ship-blocking.** PBO is the rigorous answer to "is this backtest result a fluke from the sensitivity grid?" Without it, the writeup cannot answer the multiple-testing problem that DSR alone only partially addresses.

**Fix:**
- New module `src/metrics/pbo.py` implementing CSCV with S=16. ~250 LOC including the four reportable analyses (PBO, performance-degradation, prob-of-loss, stochastic-dominance check). Single test on the sensitivity-grid output.

---

## 3. Niculescu-Mizil & Caruana (2005) — Predicting Good Probabilities with Supervised Learning

**Prescribes:**
- XGBoost / boosted trees push probability mass *away from 0 and 1* (sigmoid-shaped distortion). Without calibration, raw scores are NOT true posterior probabilities.
- Calibration MUST be done on an independent set, not the model training set.
- **Isotonic Regression dominates Platt** when calibration set ≥ ~1000 cases. **Platt dominates** below ~200 cases. In between, results are mixed but Platt is more robust.

**Codebase state (verified per audit):**
- ✅ `src/models/calibration.py:22-37` — `fit_isotonic_on_oof()` uses sklearn `IsotonicRegression` on OOF predictions.
- ✅ `src/models/xgboost_primary.py:49-72` — `fit_xgb_with_isotonic()` uses TimeSeriesSplit OOF (independent calibration set requirement satisfied).
- ✅ `walk_forward_predict()` calls `fit_xgb_with_isotonic` per fold.
- ⚠ **GAP — sample-size threshold not enforced.** Per-instrument heads will have ~60 trades/fold. Niculescu-Mizil & Caruana Section 5 (learning curve analysis): "Platt Scaling outperforms Isotonic Regression with all nine learning methods" when cal set < ~200–1000. Per-instrument isotonic at n=60 is in the regime where the paper warns isotonic overfits.
- The pooled cross-asset Head 2 (~300 trades pooled) is also below the 1000 threshold; isotonic is borderline.

**Severity:** **ship-improving.** The calibration is correct in form, but the method choice is sub-optimal for the per-instrument heads. Switching to Platt for small-n heads would tighten log-loss and avoid step-function artifacts.

**Fix:**
- Augment `fit_xgb_with_isotonic` with a `calibration_method: Literal["isotonic","platt","auto"]` kwarg. `auto` picks Platt below n=200, isotonic above.
- Pre-commit it in PRE_COMMITMENT_VRP §5 (already drafted).

---

## 4. Egger & Vestal — Hoeffding's Inequality for Early-Warning Regime Change

**Prescribes (trader's application, distinct from ML statistical-learning application):**
- Commit to a baseline μ from prior backtest / theoretical edge (NOT inferred from observed X̄).
- Compute P[X̄ − μ ≥ t | H0] ≤ e^(−2t²N) on the observed underperformance t = μ − X̄.
- For RVs not bounded to (0,1), use general form: P[X̄ − μ ≥ t] ≤ e^(−2t²N/(b−a)). Recommendation: keep U and D bounds close together to retain signal strength.
- Threshold interpretation:
  - **< 50%** → trader's beliefs about regime probably no longer fully right; consider stake reduction.
  - **< 25%** → significant regime risk; substantial concern.
  - **< 10%** → almost certain regime change; halt and rethink.

**Codebase state (`src/metrics/hoeffding.py` per audit):**
- ✅ `hoeffding_lower_bound()` — one-sided confidence bound on win rate.
- ✅ `rolling_winrate()` — time-series rolling mean.
- ✅ `hoeffding_breach_dates()` — detect breaches.
- ⚠ **PARTIAL GAP — current implementation is the ML statistical-learning form.** The audit indicates these are bounds on win rate, but the user's spec wants the Egger/Vestal *trader-application form* with the 50%/25%/10% thresholds against a *pre-committed* μ from PRE_COMMITMENT_VRP/WHEEL. Need to confirm by reading the actual file, but the audit phrasing ("bound on win rate") suggests it's not yet committed-μ-based.
- ❌ **GAP — no live-monitoring runner.** A `scripts/hoeffding_monitor.py` that reads the pre-committed μ and emits dated 50%/25%/10% threshold status across rolling 60-trade window is missing (Phase C task C5).

**Severity:** **ship-improving.** The theoretical machinery is in place. The *application* (regime-change trader-style monitoring with explicit threshold semantics) needs a thin wrapper.

**Fix:**
- Verify `hoeffding.py` matches Egger/Vestal Eq. 1.4 — `P[X̄ − μ ≥ t] ≤ e^(−2t²N)`. Add helper `regime_change_signal(observed_winrate, committed_mu, n_trades) -> Literal["green","yellow","red","critical"]`.
- Implement `scripts/hoeffding_monitor.py` per Phase C task C5.

**Methodology bonus:** Egger is the user's professor (per the paper byline). Vestal is also Duke FinTech faculty. Using their *exact* application of Hoeffding's inequality is the highest-leverage methodological grounding for the live-monitoring section. The writeup should be precise about "P[X̄ − μ ≥ t | H0] ≤ e^(−2t²N) for the trader's-application form" (no citation per user spec, but methodologically explicit).

---

## 5. Carr & Wu (2003/2004) — Variance Risk Premia

**Prescribes:**
- Variance swap rate ≈ portfolio of options (theoretical replication).
- VRP = realized variance − variance swap rate.
- Empirical: strongly negative VRP for SPX/SPX100/DJIA; smaller-magnitude negative for NDX and most individual stocks; CAPM/Fama-French cannot explain the magnitude.

**Codebase state:**
- ✅ Strategy harvests VRP via short-vol options structures (iron condors). The economic premise IS the Carr/Wu finding.
- ⚠ **METHODOLOGY HOLE — universe not stratified by Carr/Wu's evidence on VRP magnitude.** TLT and GLD are in the VRP cluster but Carr/Wu didn't study them. The writeup should disclose that the evidence base for SPX is strongest, NDX moderate, RUT untested in the original paper, TLT/GLD novel application of the same theoretical framework.

**Severity:** **ship-improving** (writeup-only). The strategy is methodologically sound — it just needs honest disclosure of which instruments have the strongest theoretical prior.

**Fix:**
- VRP thesis page (Phase F task F2) discloses the heterogeneity in theoretical prior across the 5 instruments.

---

## 6. Bollerslev/Tauchen/Zhou — Expected Stock Returns and Variance Risk Premia

**Prescribes:**
- VRP forecasts excess stock returns at short horizons (~3 months).
- Mechanism: VRP is a risk-aversion proxy.

**Codebase state:**
- ⚠ **GAP — feature not used.** VRP itself (the IV² − RV² residual) is not in the ML feature suite. If BTZ is right, including VRP as a feature for Head 2 (regime stress, cross-asset, 30–90 day horizon) would directly leverage their finding.

**Severity:** **ship-improving.** Adding VRP as a Head 2 feature is one extra column in the feature matrix; cheap, high-conceptual-leverage.

**Fix:**
- Add `vrp_30d` feature to Head 2 training pipeline (Phase D task D3).
- Mention in writeup that this is the BTZ feature, applied within the project's own framework.

---

## 7. Black & Scholes (1973) — The Pricing of Options

**Prescribes:**
- Closed-form European option pricing under GBM.
- Greeks: delta, gamma, theta, vega, rho via partial derivatives.
- Implied volatility: invert pricing formula on observed market quote.

**Codebase state:**
- ✅ `src/strategy/black_scholes.py` — full pricer with put/call, delta/gamma/theta/vega.
- ✅ Used as fallback when OptionMetrics IV missing (per audit).
- ✅ 16-delta strike selection uses BS Greeks (`select_short_strike` in `spread_construction.py`).
- ✅ Friction model uses BS Greeks for slippage scaling.

**Severity:** **no gap.** Foundation is solid.

---

## 8. Merton (1973) — Theory of Rational Option Pricing

**Prescribes:**
- Continuous-time framework extending BS.
- American options: early-exercise bounds.
- Dividend handling.
- Rational pricing bounds (no-arbitrage).

**Codebase state:**
- ✅ Wheel mechanics implicitly use early-exercise bounds for assigned shares (CSP can be assigned American-style).
- ⚠ **GAP — explicit dividend handling for GLD.** GLD pays no dividends, so European/American pricing equivalence holds; SPX index options are European-style cash-settled. This is correct in code but should be stated.
- ⚠ **GAP — wheel basket has dividend-paying names.** AAPL, MSFT, JNJ, KO, PG, WMT, JPM, PEP all pay dividends. American-style equity options have early-exercise risk for in-the-money puts on ex-dividend dates. The wheel pricing currently treats them as European-equivalent. Material in stress scenarios.

**Severity:** **ship-improving.** For the headline period (2018–2024), early-exercise effects on retail-sized positions are small. Disclosure in the writeup is sufficient.

**Fix:**
- Wheel mechanics page (Phase F task F3) discloses American-style early-exercise risk and that the model approximates with European pricing as a documented simplification.

---

## Summary table

| Paper | Severity | Fix LOC est | Fix time |
|---|---|---|---|
| 1. DSR — Eq. 9 + wiring | **ship-blocking** | 50 | 2h |
| 2. PBO via CSCV | **ship-blocking** | 250 | 6h |
| 3. Niculescu-Mizil — calibration method auto-select | ship-improving | 30 | 1h |
| 4. Egger/Vestal — trader-application monitoring | ship-improving | 100 | 3h |
| 5. Carr/Wu — universe stratification disclosure | ship-improving (writeup) | — | 1h |
| 6. Bollerslev/Tauchen/Zhou — VRP feature | ship-improving | 20 | 1h |
| 7. Black-Scholes | no gap | — | — |
| 8. Merton — early-exercise disclosure | ship-improving (writeup) | — | 0.5h |
| **Total** | | ~450 LOC | **~14.5h** |

**Critical-path ship-blockers:** 1 + 2 = 8h on top of Phase A–B work.
