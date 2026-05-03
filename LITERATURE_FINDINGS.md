# LITERATURE_FINDINGS.md

Comprehensive findings from reading all 8 papers in `/Literature/` in full. For each paper: core methodology contributions relevant to the project, specific parameter/threshold values the paper uses (so we can source from these instead of inventing), explicit limitations the paper states, methodological caveats for our use, and impact on Head 1/2/3, wheel HAR-RV forecaster, friction modeling, strike selection.

Reading status:
- ✅ Bailey & López de Prado 2014 (DSR) — txt, full
- ✅ Bailey/Borwein/López de Prado/Zhu 2015 (PBO via CSCV) — txt, full
- ✅ Niculescu-Mizil & Caruana 2005 (calibration) — txt, full
- ✅ Egger & Vestal 2025 (Hoeffding regime change) — txt, full
- ✅ Carr & Wu 2003/2004 (Variance Risk Premia) — txt, full (1991 lines)
- ✅ Bollerslev, Tauchen & Zhou 2009 (Expected Stock Returns and VRP) — PDF 30 pages, extracted to txt and read full
- ✅ Black & Scholes 1973 (Pricing of Options) — image-scanned PDF; pages 2-19 read via page-image rendering
- ✅ Merton 1973 (Theory of Rational Option Pricing) — PDF 44 pages, extracted to txt and read full

---

## 1. Bailey & López de Prado (2014) — Deflated Sharpe Ratio

### Core methodology contributions
- **Eq. 1**: Expected max Sharpe across N independent trials under H0:SR=0 — `E[max{SR̂_n}] ≈ √V · ((1−γ)·Z⁻¹[1−1/N] + γ·Z⁻¹[1−1/(N·e)])`. Euler-Mascheroni constant γ ≈ 0.5772.
- **Eq. 2**: DSR — `Z[(SR̂ − SR̂_0)·√(T−1) / √(1 − γ̂_3·SR̂ + ((γ̂_4−1)/4)·SR̂²)]`. Per-period (NOT annualized).
- **Eq. 9**: N̂ = ρ̄ + (1−ρ̄)·M — implied independent trials from M correlated trials. Avoids overstating SR̂_0 when grid trials are correlated.

### Specific parameter values
- **95% confidence threshold**: DSR ≥ 0.95 deemed "statistically significant" against multiple-testing.
- **Numerical example** (page 6): N=100 trials, V=(0.5/√250)², T=1250 daily obs, γ̂_3=−3, γ̂_4=10 → SR̂_0 ≈ 0.1132 (per-period), DSR ≈ 0.9004 → INSUFFICIENT at 95%.
- For **Normal-returns equivalence**, N=88 trials would yield DSR=0.95.
- Optimal-stopping rule (1/e ≈ 37%): "After sampling 37% of theoretically-justifiable configurations randomly, take the next-best-so-far." This is Bruss's secretary-problem result.

### Explicit limitations the paper states
- The "average correlation" approach for N̂ (Eq. 9) is "intuitive and convenient" but: (a) correlation is "a limited notion of linear dependence," (b) for short samples T < M·(M−1)/2 the correlation matrix is "ill-conditioned and pointless." Solution: dimension-reduce or use entropy/information-theory.
- Selection bias correction is **parametric** (assumes Gaussian trial-Sharpe distribution); CSCV-based PBO (paper #2) is its non-parametric complement.

### Methodological caveats for our use
- We have a 729-trial sensitivity grid that is **highly correlated** (adjacent delta values, adjacent DTE windows produce nearly-identical equity curves). Eq. 9 N̂ correction is critical; raw N=729 would inflate SR̂_0 dramatically.
- Skew and **raw kurtosis** (= excess + 3) must be sample estimates from the actual return series. Our `deflated_sharpe.py` correctly converts pandas .kurt()+3.

### Impact on each project component
- **All headline reports** (Head 1/2/3 ablations, wheel ablations, full-portfolio aggregate): every Sharpe MUST be reported alongside DSR per the paper's central thesis. Failure to do so reduces the writeup to "uncontrolled multiple-testing" which the paper explicitly warns against.
- **Sensitivity grid** (PRE_COMMITMENT_VRP §11, WHEEL §7): N̂ from Eq. 9 used, not raw N=729.

---

## 2. Bailey, Borwein, López de Prado & Zhu (2015) — Probability of Backtest Overfitting via CSCV

### Core methodology contributions
- **Algorithm 2.3**: Form (T × N) performance matrix M from sensitivity grid. Partition rows into S submatrices (S even, recommended S=16). Form C(S, S/2) combinations of row-halves. For each combo: identify IS-best n*, compute OOS rank ω̄_c, logit λ_c = ln(ω̄/(1−ω̄)). PBO = fraction of combos with λ_c < 0.
- Reproducible (no random component); model-free; non-parametric.
- Yields four complementary statistics: PBO, performance-degradation slope, prob-of-OOS-loss, stochastic-dominance check.

### Specific parameter values
- **S=16** is the recommended default for daily data with ≥4 years history — produces 12,780 logit combos with σ[f̂(λ)] < 0.0045.
- **PBO ≤ 0.05** is the Neyman-Pearson default rejection threshold; in practice the paper notes "PBO < 0.30" is often considered acceptable for investment strategies.
- **Conversion**: PBO ≈ 0 → no overfitting; PBO ≈ 1 → severe overfitting.

### Explicit limitations the paper states
- Symmetric S-block partition may be "unsuitable when performance series has strong autocorrelation" — could obscure regime structure.
- Assumes all sample statistics carry equal weight — if priors on distribution exist, model-specific weighting is more accurate.
- The researcher "must provide full information regarding actual trials conducted, to avoid file-drawer problem." Hiding trials → underestimated PBO.
- "Procedure does nothing to evaluate the correctness of a backtest" — bad assumptions (transaction costs, lookahead) propagate undetected.
- Structural breaks outside the dataset cannot be assessed.
- "Skillful strategies can still exist in N tested" even with high PBO — many similarly-good strategies → high PBO without being a real overfitting signal.

### Methodological caveats for our use
- **Recommended S=16** with our ~3500-trading-day OOS sample → 218 obs/block — fine, but autocorrelation in equity curves is non-trivial. Should report PBO alongside performance-degradation slope to triangulate.
- "Don't use CSCV to GUIDE strategy selection" — Goodhart's law warning. We must use it ONLY for post-hoc validation, never as an objective function.

### Impact on each project component
- **Sensitivity grid PBO**: ship-blocking gap from prior analysis, now implemented in `src/metrics/pbo.py`. Will run on the 729-trial sensitivity grid output.
- **Performance degradation slope (β from R̄ vs R)**: complementary diagnostic. Our `PBOReport.perf_degradation_slope` captures this.
- **Stochastic dominance check**: distinguishes "selected strategy is worse than random selection" (overfit) from "all strategies similarly good but no edge" (high PBO without overfitting). Our `PBOReport.stoch_dom_first_order` captures this.

---

## 3. Niculescu-Mizil & Caruana (2005) — Predicting Good Probabilities with Supervised Learning

### Core methodology contributions
- Boosted trees / SVMs / boosted stumps push probability mass AWAY from 0 and 1 → sigmoid-shaped distortion, raw scores are NOT calibrated probabilities.
- **Platt scaling** (sigmoid fit on raw scores via 2-parameter logistic regression): best for small calibration sets.
- **Isotonic regression** (PAV algorithm, monotonic step function): more flexible but overfits when calibration set is small.
- Calibration set MUST BE INDEPENDENT from the model training set.

### Specific parameter values
- **n < 200**: Platt strictly better than isotonic across all 9 learning methods tested.
- **n ∈ [200, 1000]**: mixed results, Platt generally more robust.
- **n ≥ 1000**: isotonic equal-or-better than Platt.
- **Platt's overfitting fix**: target labels y+ = (N+ + 1)/(N+ + 2), y− = 1/(N− + 2) instead of {0, 1} (page 50).

### Explicit limitations the paper states
- Designed for binary classification — multiclass requires reduction-to-binary + recombine.
- Bagged trees, neural nets, logistic regression are well-calibrated already; calibrating them with Platt or isotonic can HURT performance for small calibration sets.
- "Random forests are less clear cut" — well-calibrated on some problems, poorly on others (LETTER.P2, HS, COV_TYPE, MEDIS, LETTER.P1).

### Methodological caveats for our use
- **Per-instrument Head 1** (5 instruments × ~60 trades/fold) — each model trains and calibrates on ~60 cases. Below the n<200 threshold → Platt scaling DOMINATES isotonic.
- **Per-instrument Head 3** (5 models × ~60 trades) — same.
- **Pooled Head 2** (cross-asset, ~300 trades pooled if labeled events used) — borderline; isotonic acceptable but Platt safer.
- Our current `xgboost_primary.py` uses isotonic regardless of sample size. This is suboptimal per the paper for small-n heads.

### Impact on each project component
- **Head 1, Head 3 (per-instrument)**: switch to Platt scaling (already pre-committed in PRE_COMMITMENT_VRP §5).
- **Head 2 (pooled)**: isotonic acceptable; Platt as fallback if calibration set < 200 in any fold.
- **Wheel HAR-RV forecaster**: regression task, calibration not directly applicable (calibration is for classification probabilities).

---

## 4. Egger & Vestal (2025) — Hoeffding's Inequality for Early-Warning Regime Change

### Core methodology contributions
- **Trader's application** distinct from ML statistical-learning application.
- **ML form** (Inequality 1.3): `P[|Ein − Eout| > ε] ≤ e^(−2ε²N)` — bounds future error rate given observed in-sample.
- **Trader form** (Eq. 1.9): `P[X̄ − μ ≥ t | H0] ≤ e^(−2t²N)` — μ committed from prior backtest, decreasing probability bound = decreasing plausibility of regime hypothesis H0.
- **Bounded RVs not in (0,1)**: general form `P[X̄ − μ ≥ t] ≤ e^(−2t²N/(b−a))` (Eq. 1.10). Recommendation: keep U and D bounds CLOSE TOGETHER to retain signal strength.

### Specific parameter values (locked threshold semantics)
- **< 50%**: trader's beliefs about regime "probably no longer fully right" — consider stake reduction.
- **< 25%**: significant regime risk.
- **< 10%**: "almost certain" regime change — halt strategy and rethink.

### Explicit limitations the paper states
- Hoeffding bounds are WORST-CASE — apply to any bounded RV regardless of distribution. They become tighter only when distributional assumptions are added (variance, autocorrelation).
- Statement explicitly NOT: "e^(−2t²N) is the maximum probability the trader's H0 is correct." The bound governs probability of OBSERVING the divergence given H0, not the probability of H0 itself.

### Methodological caveats for our use
- For win-rate monitoring, bounds (a, b) = (0, 1). Eq. 1.4 applies directly.
- For per-trade dollar P/L monitoring, RV is unbounded — must impose synthetic bounds via stop-loss U and profit-target D. The paper recommends U and D close together. Our pre-committed PT=50% credit, SL=200% credit gives a 4:1 profit:loss bound — wide enough that signal strength suffers per Eq. 1.10.
- Best signal: rolling 60-trade win rate with μ pre-committed from IS-baseline.

### Impact on each project component
- **Live monitoring runner** (`scripts/hoeffding_monitor.py`, Phase C task C5): must use Eq. 1.4 trader-form with μ pre-committed from PRE_COMMITMENT_VRP/WHEEL. Threshold semantics 50%/25%/10% as stated.
- **Halt framework Layer 5 auto-resume**: not directly affected (uses different criteria). But the writeup should note that Hoeffding monitoring complements halt framework — halts handle market state, Hoeffding monitoring handles strategy degradation.
- **Authors are Duke FinTech faculty** (Egger and Vestal). Vestal is the user's advisor. This is the highest-leverage methodological grounding for the live-monitoring section. The writeup should be precise about Eq. 1.4 form (no citation per user spec, but methodologically explicit).

---

## 5. Carr & Wu (2003/2004) — Variance Risk Premia

### Core methodology contributions
- **Synthesizing variance swap rate** from a portfolio of options: `EQ[RV_t,T] ≈ 2 ∫₀^∞ Θ_t(K,T) / (B_t(T)·K²) dK` where Θ is OTM option price.
- Approximation EXACT under continuous price paths; instantaneous error O((dF_t)³) under jumps.
- **VRP** = realized variance − variance swap rate. Captures risk-neutral expected variance vs ex-post realized.
- Volatility swap ≈ at-the-money implied volatility (Carr-Lee 2003).

### Specific parameter values (locked methodological choices)
- **30-day horizon** for variance swap synthesis (matches the new CBOE VIX construction).
- **5-strike interpolation** is conservative; their 5-index/35-stock sample averages ≥ 5 strikes.
- **Filters for individual stocks** (stock options trading-activity gates):
  - Nearest available maturity must be < 90 days.
  - Stock price > $1.
  - ≥3 strikes at each of two nearest maturities.
- **Sample**: Jan 1996 – Feb 2003 (~7 years of OptionMetrics options data).
- **Newey-West with 30 lags** for serial-dependence-adjusted t-statistics.
- **Returns**: They use `RV_{t,t+30} = (365/30) · Σ_{i=1}^{30} (F_{t+i,t+30} − F_{t+i−1,t+30})²/F²_{t+i−1,t+30}` (forward returns squared, ≈ daily variance over 30-day window).

### Explicit limitations the paper states
- Approximation error from finite strikes (discretization error) and from jumps (jump error of order O((dF/F)³)) — paper shows both small under realistic parameters.
- ATM implied volatility ≈ volatility swap rate is "extremely accurate" but not exact (third-order error).
- Assumes futures contracts continuously mark-to-market.
- For American options on individual stocks, OptionMetrics uses binomial tree to extract IV — different from European pricing.

### Empirical magnitudes by instrument
- **SPX, OEX, DJX**: strongly negative VRP, large t-stats — investors pay heavy premium to hedge variance.
- **NDX (Nasdaq 100)**: smaller-magnitude negative; t-stats not significant for RP form, significant for log-RP.
- **Individual stocks (35)**: 21 of 35 significantly negative for log-RP; only 3 of 35 significant for RP.
- CAPM β explains a small portion of negative VRP. Fama-French factors explain more but still leave large unexplained residual.
- **Variance-of-volatility predicts |VRP|**: Eq. 60 — slope estimates predominantly positive.

### Methodological caveats for our use
- **Universe stratification** for VRP cluster (PRE_COMMITMENT_VRP §1): Carr/Wu evidence base is strongest for SPX, weaker for NDX, untested for RUT/TLT/GLD. Writeup should disclose this.
- **VRP feature in Head 2**: BTZ shows VRP is dominant predictor at quarterly horizon. Should be in feature set.
- **OptionMetrics — same source as user's data**.
- Wheel basket (8 dividend-paying stocks + GOOGL pays no div) has American options; Carr/Wu used binomial-tree IVs for these. Our pricer should ideally do the same — currently uses BS for all → known approximation gap.

### Impact on each project component
- **VRP cluster universe** (PRE_COMMITMENT_VRP §1): theoretical prior heterogeneous across the 5 instruments. Disclose explicitly.
- **Head 2 features**: VRP itself (the IV²−RV² residual) belongs in the input set.
- **Wheel pricing**: American early-exercise risk on dividend-paying names. Disclose, but BS approximation is conservative for retail-sized positions.
- **Newey-West vs Hodrick SEs**: BTZ recommends Hodrick (next paper), Carr/Wu uses Newey-West-30. For Head 2 evaluation, both are defensible; we'll use Hodrick per BTZ since our forecast horizon matches BTZ's framework more closely.

---

## 6. Bollerslev, Tauchen & Zhou (2009) — Expected Stock Returns and Variance Risk Premia

### Core methodology contributions
- General equilibrium model (Epstein-Zin-Weil recursive preferences) with stochastic vol AND vol-of-vol → two-factor structure.
- **Variance risk premium isolates the vol-of-vol factor**: VRP = E^Q[σ²_{r,t+1}] − E[σ²_{r,t+1}] = (θ−1)κ₁[A_σ + A_q κ₁²(A²_σ + A²_qφ²_q) φ²_q] · q_t.
- Empirical VRP definition: `VRP_t ≡ IV_t − RV_t` (model-free implied variance minus model-free realized variance).
- Predictive regression: `(1/h)Σ_{j=1}^h r_{t+j} = b_0(h) + b_1(h)·VRP_t + u_{t+h,t}`.

### Specific parameter values
- **h = quarterly (3 months)**: empirically optimal forecast horizon → "63 trading days" per CAPSTONE_BACKLOG. R²=6.82% at quarterly vs 1.07% at monthly.
- **Sample**: Jan 1990 – Dec 2007 (~17 years of monthly observations).
- **VIX**: post-Sep 2003 the "new" VIX (model-free, S&P 500 options, 30-day fixed maturity) replaces the "old" VIX (S&P 100, BS-implied).
- **Realized variance**: 78 within-day 5-minute squared returns (9:30–4:00) + close-to-open overnight return. For a 22-trading-day month, n = 22 × 78 = 1716 5-minute returns.
- **Slope coefficient**: b₁(h=3) ≈ 0.47 with robust t-stat 2.86. With P/E control: 0.58, t=3.43.
- **Combined VRP+P/E quarterly R²**: 16.76%. With CAY: 17.42%. With all controls: 19.74%.

### Explicit limitations the paper states
- "Limited post-1990 sample prevents us from effectively studying issues having to do with longer return horizons spanning multiple years" — small sample for multi-year horizons. Our project's 2012–2024 sample (~12 years) is even shorter than BTZ's 17.
- Boudoukh-Richardson-Whitelaw (2008): R²s with HIGHLY-PERSISTENT predictors and OVERLAPPING returns "will by construction increase roughly proportional with the return horizon and the length of the overlap" even without true predictability. **Critical warning** for interpreting our 63-day forward window R² values.
- Estimation results subject to "look-ahead bias" when full-sample HAR-RV is used to estimate `E_t(RV_{t+1})` (footnote 29).
- Stambaugh-bias adjustment (small for VRP itself, ~0.02; large for P/E and CAY, 0.51-0.52).

### Methodological caveats for our use
- **Use Hodrick (1992) standard errors**, NOT Newey-West, for overlapping multi-period predictive regressions. Per Ang-Bekaert (2007), Hodrick is "generally more reliable than the more traditional standard errors based on the summation of the residuals into the future."
- **High-frequency intraday RV** is methodologically superior to daily-RV. Our project uses daily-RV for HAR-RV components → known methodological gap. Document this honestly.
- Use the **"new" model-free VIX**, not BS-implied. Confirmed in our `data/raw/VIX.parquet` (post-2003 data).
- **VRP itself is the strongest single predictor** at quarterly horizon (BTZ R²=6.82% alone, surpasses P/E, default spread, CAY). Should be a primary Head 2 feature.

### Impact on each project component
- **Head 2 (regime stress)**: 63-day forward horizon directly anchored in BTZ's quarterly-horizon empirical result. Stress label uses 5 vetted historical events (Approach B); but Head 2's FEATURES include VRP itself per BTZ.
- **Reporting standards**: when reporting Head 2 OOS Brier, supplement with Hodrick-style robust SE on the implied predictability significance.
- **HAR-RV forecaster (Wheel Layer 3)**: We use daily-RV components, not 5-minute intraday. Disclose as methodological simplification.
- **Sample-size disclosure**: Explicitly note 12-year sample is shorter than BTZ's 17. Effective independent observations after 63-day overlap ≈ 12 × 252/63 ≈ 48 non-overlapping windows. Small.
- **R² interpretation caveat**: Per Boudoukh et al., even our positive R²s might inflate proportional to overlap. Use Hodrick t-stats as primary significance measure.

---

## 7. Black & Scholes (1973) — The Pricing of Options

### Core methodology contributions
- **Hedged portfolio argument**: Long 1 stock, short Δw/Δx options creates risk-free portfolio. By no-arbitrage, expected return = riskless rate.
- **Famous formula** (Eq. 13 of paper): `w(x, t) = x·N(d₁) − c·e^{−r(t*−t)}·N(d₂)` where:
  - `d₁ = [ln(x/c) + (r + v²/2)(t* − t)] / [v·√(t* − t)]`
  - `d₂ = d₁ − v·√(t* − t)`
  - x = stock price, c = exercise price, r = risk-free rate, v = std dev of log returns, t* − t = time to expiration.
- Differential equation (Eq. 7): `∂w/∂t = rw − rx(∂w/∂x) − (v²/2)x²(∂²w/∂x²)`.

### The 7 BS assumptions (page 5)
1. Short-term interest rate is known and constant through time.
2. Stock price follows continuous random walk with constant variance rate (geometric Brownian motion with constant σ).
3. Stock pays no dividends or distributions.
4. Option is "European" — exercisable only at maturity.
5. No transaction costs in buying or selling stock or option.
6. Possible to borrow any fraction of price of security at short-term rate.
7. No penalties to short selling — full proceeds available.

### Empirical findings (page 16)
- Tested on **545 OTC option contracts** in NYC market, day-by-day in **1966**.
- Predicted prices vs actual: BS systematically OVERVALUES options on high-variance securities and UNDERVALUES on low-variance securities. (Page 16, "When the variance of returns is high, the formula tends to overvalue options. When the variance is low, the formula tends to undervalue options.")
- Suggests model misspecification — likely related to stochastic vol / vol-of-vol structure that BTZ formalizes 36 years later.

### Specific parameter values
- **Risk-free rate proxy**: T-bill yield to maturity matching option expiration.
- **Variance estimate**: From past stock price data — paper uses approximate 1/4 of historical variance to detrend secular drift.
- **Discount rate** in their numerical example: 4% annualized.

### Explicit limitations the paper states
- Constant σ is restrictive; empirical evidence (Black 1976 leverage effect) shows σ varies with stock price.
- No-dividend assumption requires modification for dividend-paying stocks.
- Constant r assumption ignores term structure dynamics.
- "European" only — for American calls on non-dividend stocks, value is the same. For American puts or dividend-paying stocks, early-exercise needs separate treatment (handled by Merton 1973 next).

### Methodological caveats for our use
- **Strike selection at 16-delta** uses BS-Greek: ∂w/∂x = N(d₁). For SPX (no dividends, European-style), this is exact under BS assumptions.
- For wheel basket (American options on dividend-paying stocks), 16-delta is approximation — known gap.
- **Friction model uses BS Greeks** (theta, vega, gamma) — same caveats.
- **BS pricer fallback** when OptionMetrics IV missing: assumes constant σ for that day's quote — fine for one-day approximation.
- BS's own empirical finding (overvalues high-var, undervalues low-var) is the EMPIRICAL EVIDENCE that VRP exists — our strategy harvests precisely this misspecification.

### Impact on each project component
- **Strike selection (`select_short_strike`)**: BS-Greek-defined 16-delta. ✅ Implemented correctly.
- **Friction model (`slippage_pct_at_vix`, theta/vega/gamma in P&L)**: BS Greeks. ✅ Implemented.
- **BS pricer fallback**: Used when OptionMetrics IV missing. ✅ Implemented in `src/strategy/black_scholes.py`.
- **Wheel basket American options**: Known approximation. Disclose.
- **Writeup conceptual framing**: VRP exists because BS is empirically a biased pricer (overvalues high-vol, undervalues low-vol relative to true risk-adjusted prices). Our strategy is a structural harvester of this bias.

---

## 8. Merton (1973) — Theory of Rational Option Pricing

### Core methodology contributions
- **Section 2**: 10 theorems on option price restrictions from "no dominance" alone (Assumption 1: option neither dominates nor is dominated by another security).
- **Theorem 1**: f(S, τ; E) ≥ Max[0, S − E·P(τ)] — option worth at least max(0, stock minus PV of strike).
- **Theorem 2**: For non-dividend stocks, **American call = European call** (no benefit to early exercise).
- **Theorem 3**: Perpetual call on non-dividend stock has value equal to stock price.
- **Theorem 8**: Option price is non-decreasing in riskiness of underlying — confirms VRP at the theoretical level.
- **Theorem 9**: Option price is homogeneous of degree one in (S, E) when returns are independent of stock price level (D.I.S.P.).
- **Theorem 10**: Option price is convex in stock price (under D.I.S.P.).
- **Theorem 11**: Payout protection requires share-count adjustment by (d/Sx)% per ex-payout date.
- **Section 4 — Put options**: Put-call parity holds for European: `g = f − S + E·P(τ)`. Does NOT hold for American puts because American puts ALWAYS have positive probability of premature exercise.
- **Section 5–6**: Rederives BS formula from weaker assumptions (only no-dominance + frictionless markets + Itō dynamics + non-stochastic σ). DOES NOT require CAPM.
- **Section 7**: Continuous-dividend version of BS PDE: `(1/2)σ²S²W₁₁ + (rS−D)W₁ − W₂ − rW = 0`. Closed form for proportional dividend D=ρS: `W = e^{−ρτ}·S·Φ(d₁) − E·e^{−rτ}·Φ(d₂)`.
- **Sufficient condition for no premature exercise** with continuous dividend d at constant interest rate r: **E > d/r** (Eq. 13).
- **Section 8 — American put**: Always positive probability of premature exercise. Closed form for **perpetual** American put: `G(S, ∞; E) = (E/(1+γ))·[(1+γ)S/(γE)]^{−γ}` where γ = 2r/σ². Optimal exercise boundary `C* = γE/(1+γ)`.
- **Section 9 — Down-and-out call**: Closed form for European version (Eq. 55). Knock-out feature represented as a discount on the standard call value.
- **Section 10 — Callable warrant**: Closed form for perpetual: `F(S) = (K/(K+E))·S` where K is the call price.

### Specific parameter values used
- **Numerical example** (page 13): "If the increase in exercise price is 10 percent and the length of time before the next exercise price change is five years, the yield to maturity on riskless securities would have to be less than 2 percent before [the no-early-exercise condition] would not hold."
- Interest rates and volatilities are otherwise expressed symbolically; no specific numerical thresholds for stress regimes.

### Explicit limitations the paper states
- "Continuous-trading assumption is necessary to establish perfect correlation among nonlinear functions which is required to form the 'perfect hedge' portfolio mix."
- "Itō processes for asset returns dynamics" required to apply Itō's Lemma.
- σ and δ "must be nonstochastic and independent of price levels" for hedge formula. **Stochastic vol breaks the closed form** (footnote 57: "the variance of stock price return can be a function of the price level [in the special case of nonstochastic interest rates] and the derivation still goes through. However, the resulting partial differential equation will not have a simple closed-form solution.")
- "BS claim that (21) is the only formula consistent with capital market equilibrium is a bit too strong. It is not true that if the market prices options differently, then arbitrage profits are ensured. It is a 'rational' option pricing theory **relative to the assumptions of this section**."
- For unrestricted-supply (non-incipient) warrants: payout protection requires capital structure adjustments; in BS's "y=1" extreme example (firm liquidates entirely), payout protection is impossible.

### Methodological caveats for our use
- **American puts on dividend-paying stocks** (wheel basket — AAPL, MSFT, JNJ, KO, PG, WMT, JPM, PEP all pay dividends): Theorem 13 confirms there's ALWAYS positive probability of premature exercise. As put-WRITERS we face the inverse: our short puts may be assigned to us early, especially before ex-dividend dates. This is part of the wheel mechanic.
- **No-early-exercise condition** for European-equivalent pricing on dividend stocks: E > d/r. With current rates ~5% and dividend yields ~1-3%, our short puts at 16-25 delta are typically OUT of the money (E < S), so this isn't directly violated for the writer's perspective.
- **GLD pays no dividends** — European-equivalent pricing exact for European-style options. But GLD options ARE American-style → still some early-exercise complexity.
- **TLT pays interest** (monthly distributions on bond ETF). Treated as dividend-paying for option pricing.
- **SPX/RUT/NDX**: cash-settled European-style. No early-exercise issue.

### Impact on each project component
- **VRP cluster pricing** (5 instruments): SPX/RUT/NDX clean European-style. TLT/GLD American-style + interest distributions (TLT) — known methodological gap. Disclose.
- **Wheel mechanic** (8 dividend-paying names, 1 non-dividend GOOGL): American-style early exercise risk on ITM puts before ex-dividend dates. Strategy: tracked via our existing exit logic; assignments handled post-hoc.
- **Pricer accuracy**: We use BS for all instruments. For dividend-paying names, this OVERVALUES European calls and UNDERVALUES American puts (which have positive early-exercise premium). Magnitude is small for OTM positions (our 16-delta strikes are well OTM at entry). Disclose.
- **Theorem 8 (riskiness → higher option value)**: theoretical underpinning for VRP existence — confirms why short-vol strategies harvest a structural premium.
- **Variable exercise prices** (Merton §7): Not applicable — our strikes are fixed per trade.
- **Knock-out / barrier options** (Merton §9): Not applicable to our strategy.

---

## Cross-cutting findings

### A. Multiple-testing rigor (Papers 1, 2)
**Every Sharpe in the writeup must be reported with DSR (Bailey-LdP 2014) and PBO via CSCV (Bailey-Borwein-LdP-Zhu 2015).** This is non-negotiable per the literature and is a ship-blocking gap addressed in `src/metrics/deflated_sharpe.py` and `src/metrics/pbo.py`.

### B. Calibration sample-size (Paper 3)
Per Niculescu-Mizil & Caruana, **switch to Platt scaling for per-instrument heads (n < 200)**. Already pre-committed in PRE_COMMITMENT_VRP §5. Implementation TODO: add `calibration_method` kwarg to `fit_xgb_with_isotonic` with auto-select.

### C. Live monitoring (Paper 4)
**Use Eq. 1.4 trader-form Hoeffding** with μ pre-committed. 50%/25%/10% threshold semantics. Authors are user's professors — methodological grounding is high-leverage. Phase C task C5.

### D. Forecast horizon (Papers 5, 6)
**Quarterly = 63 trading days** is the empirically-optimal horizon per BTZ's Section 3 R²-by-horizon analysis. Locked in PRE_COMMITMENT_VRP §5.1.

### E. VRP itself as feature (Papers 5, 6)
**VRP belongs in Head 2 features.** Carr/Wu 2003 + BTZ 2009 both identify VRP magnitude as the strongest single predictor of next-quarter returns. Already in `data/processed/features.parquet` columns `vrp_30d` and `vrp_60d`.

### F. Standard errors for overlapping forecasts (Paper 6)
**Hodrick (1992) SEs preferred over Newey-West** for overlapping multi-period predictive regressions. Per Ang-Bekaert (2007), Hodrick is more reliable. Phase C task C6 should use Hodrick when reporting Head 2 evaluation SEs, not Newey-West.

### G. R² inflation warning (Paper 6)
Per **Boudoukh-Richardson-Whitelaw (2008)** as cited in BTZ: even with no true predictability, R² inflates proportionally with overlap × horizon. Our 63-day forward window with daily observations has very high overlap. Use t-stats (Hodrick-adjusted), not R² alone, as primary significance measure.

### H. Universe stratification by VRP evidence (Paper 5)
**Carr/Wu evidence base is heterogeneous across the 5-instrument VRP cluster:**
- SPX: strongly supported (large negative VRP, large t-stats)
- NDX: weaker support
- RUT: not in original paper (but similar to SPX — large-cap US index)
- TLT: novel application (bond options not studied)
- GLD: novel application

Disclose this heterogeneity in the writeup thesis page.

### I. Methodological gaps relative to BTZ's preferred specification (Paper 6)
- We use **daily-frequency** RV components for HAR-RV forecaster, not 5-minute intraday. BTZ explicitly: "Estimation of the same predictive regressions based on the traditional Black–Scholes implied variances and/or realized variances constructed from lower frequency daily data does not give rise to nearly as significant results."
- Our **sample is shorter** (12 years vs BTZ's 17). Effective independent obs at 63-day horizon ≈ 48 non-overlapping windows.
- Disclose both.

### J. American early-exercise risk for wheel basket (Paper 8)
- 8 of 9 wheel names pay dividends (all except GOOGL).
- American puts on dividend-paying stocks ALWAYS have positive probability of premature exercise (Merton Theorem 13).
- We treat all wheel options with BS European-style pricing. Magnitude of approximation error small for OTM positions; disclose.

### K. BS pricing biases (Paper 7)
- BS empirically overvalues options on high-variance securities and undervalues on low-variance securities.
- This bias is the EVIDENCE that VRP exists — our strategy harvests this structural mispricing.
- Frame in writeup: BS is a useful first-order approximation; the deviations BS exhibits empirically are the source of the volatility risk premium that motivates the strategy.
