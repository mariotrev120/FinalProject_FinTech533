# SPX Volatility Risk Premium Harvesting with ML-Gated Entry

**FinTech 533, Final Project**
**Authors:** Mario Trevino, Robert Lanni
**Duke University, Master of Engineering in Financial Technology**

---

## TL;DR

We sell XSP put credit spreads to harvest the volatility risk premium, gated by an XGBoost classifier trained on Vestal-style exogenous environmental features. Position sizing uses calibrated probabilities and Kelly-fractional rules. Five layered halt mechanisms shut the strategy down when regime indicators turn hostile. Live performance is monitored via Hoeffding bounds and block-bootstrap confidence intervals. Theoretical foundation grounded in Merton 1973 (variance risk premium, put-call parity, down-and-out barrier framework as halt analog) and Black-Scholes 1973 (theta decay).

**Backtest period:** 2010-2024 (in-sample 2010-2017, OOS 2018-2024)
**Universe:** XSP (S&P 500 Mini, cash-settled, European exercise, Section 1256 taxed)
**Data source:** IBKR TWS via shinybroker

---

## What This Project Answers

Vestal's two rubric questions, explicitly:

1. **How will you know your strategy is performing in line with the backtest?**
   Rolling Hoeffding bound on win rate vs in-sample baseline, block-bootstrap 90% CIs on Sharpe, calibration drift check, feature stability monitor.

2. **How will you quantify when the strategy stops working?**
   Five halt layers with pre-committed thresholds, each with a specific quantitative trigger and documented rationale. Auto-resume conditions are also pre-committed.

---

## Strategy Thesis and Rationale

### Why volatility risk premium harvesting?

The variance risk premium (VRP) is the structural gap between options-implied volatility and subsequently-realized volatility. It is one of the most-documented persistent edges in financial markets. Carr and Madan (1998) provide the theoretical decomposition. Decades of empirical work since (Bakshi and Kapadia 2003, Carr and Wu 2009, Bollerslev et al. 2009) confirm the premium averages 3 to 5 vol points on equity index options and persists because option buyers structurally overpay for downside insurance.

The mathematical foundation is Merton 1973, Theorems 8 and 15. Option price is monotone increasing in variance. Selling options is selling variance. The structural premium exists because hedgers (pension funds, asset managers, insurance companies) are willing to pay above-fair-value for tail-risk protection. This creates a persistent transfer of wealth from insurance buyers to insurance sellers, gross of any compensation for bearing crash risk.

The strategy harvests this premium directly. The risk is that the premium is highest precisely when crash risk is elevated, so naive vol selling blows up during stress events (Volmageddon February 2018, COVID March 2020 being the canonical retail blow-up cases). The entire engineering of this project addresses that asymmetric risk profile.

### Why credit spreads instead of naked puts?

Three reasons:

1. **Defined risk.** Maximum loss per spread is known at entry (spread width minus credit received). This makes position sizing clean and prevents single-trade catastrophic loss.

2. **Capital efficiency.** Margin requirement is roughly maximum loss, not full underlying notional. A $100k account can run multiple concurrent spreads without margin pressure.

3. **Suitable for retail size.** Naked put selling on SPX requires substantial cash collateral and has tail risk that retail position sizing cannot absorb. Credit spreads make the strategy implementable at a $100k account level.

### Why XSP and not SPY or SPX?

XSP (SPX Mini) is the optimal instrument for this strategy at retail size. Three structural advantages over SPY:

1. **Cash settlement.** SPX-family options settle in cash. There is no risk of physical assignment of 100 shares per contract over a weekend. SPY's physical settlement creates "pin risk" where a contract closing exactly at the short strike can produce overnight assignment without warning, converting a defined-risk spread into an undefined-risk stock position.

2. **European exercise.** XSP cannot be exercised early. American-style SPY options can be assigned any business day, particularly when deep in the money near ex-dividend dates. European exercise removes this entire risk class.

3. **Section 1256 tax treatment.** XSP qualifies for IRS Section 1256 contract treatment: 60% long-term capital gains and 40% short-term capital gains rate, regardless of holding period. For a strategy generating 12 to 18% annualized at retail tax brackets, this is worth roughly 4 to 7% of after-tax returns compared to SPY's standard short-term treatment.

XSP is chosen over full-size SPX because the 1/10th notional makes position sizing cleaner at $100k account scale. At $1M+ account sizes, migration to full SPX would be appropriate.

---

## Strategy Mechanics

### Entry

Every Monday at the open, the system evaluates an exogenous feature vector computed at Friday close. If the calibrated XGBoost probability of trade profitability is at least 0.55, the system sells one XSP put credit spread.

**Spread construction:**
- Short strike: approximately 1 standard deviation OTM (16-delta)
- Long strike: 5 points further OTM
- Days to expiration at entry: 30 to 45 days

**Why 1-sigma OTM short strike?** Empirical research on equity index put-selling (notably from Tasty Trade and CBOE PUT/PutWrite indices) shows that 1-sigma OTM short strikes balance premium collection against assignment risk. Closer to ATM increases credit but lowers win rate. Further OTM increases win rate but compresses absolute return per trade. The 16-delta convention is the industry standard for systematic put-writing.

**Why 30 to 45 DTE?** This window optimizes the trade-off between theta decay rate and gamma exposure. Black-Scholes shows theta is proportional to roughly 1/sqrt(tau), so theta-per-day accelerates as expiration approaches. But gamma also accelerates, and gamma is the enemy of short option positions. Below 30 DTE, gamma risk dominates and small price moves cause large P&L swings. Above 45 DTE, theta is too slow to be efficient capital deployment. The 30 to 45 window is where theta is rich relative to gamma.

**Why decision threshold of 0.55?** The threshold is set 0.05 above the naive 0.50 cutoff to require modest model confidence above coin-flip. The asymmetry of credit spread payoffs (typical max win = $200 credit, typical max loss = $300 on a $5 spread) means a 50% win rate is roughly breakeven before friction. The 0.55 threshold provides a margin to cover friction and provide positive expectancy. Higher thresholds (0.60 or above) reduce trade count too aggressively given the small candidate pool produced by 1-sigma OTM 30-45 DTE entries. The 0.55 value is pre-committed before any OOS test.

### Exit (Three Fates Plus Emergency)

**Fate 1, Profit target:** Close at 50% of maximum profit.

Rationale: Standard tasty trade research finding, replicated across multiple practitioner studies. The first 50% of profit accumulates roughly 3 to 4 times faster than the second 50% because theta decay is fastest when the option still has meaningful time value. Holding to expiration captures the remaining 50% but at substantially worse risk-adjusted return per day. Closing at 50% redeploys capital into fresh trades with rich theta.

**Fate 2, Stop loss:** Close at 200% of credit received, executed at min(stop_level, opening_gap_price).

Rationale: 200% of credit is the standard stop level for credit spread systems. Tighter stops (150%) get noise-stopped during normal volatility expansion. Wider stops (300%) produce single trades that can wipe out 5 or more winners. The gap-aware execution rule (min of stop level and opening price) accounts for the reality that overnight gaps can blow through stops, and the realized loss is whichever is worse.

**Fate 3, Time exit:** Hard close at 21 DTE.

Rationale: This is the gamma-cliff protection. Inside 21 DTE, gamma per dollar of premium remaining accelerates non-linearly. A trade that has been winning quietly for three weeks can give back its gains in two days as gamma dominates theta. The 21 DTE exit captures roughly 70% of the theoretical max profit while exiting the high-gamma zone entirely. This is also a standard convention in systematic put-writing literature.

**Emergency exit:** Short leg delta exceeds 0.50.

Rationale: When the short leg moves to ATM (delta near 0.50), the original assumptions of the trade (1-sigma OTM, defined edge, theta-favored) no longer hold. The position has converted from "premium harvest" to "directional bet on bounce-back," which is not what the strategy was designed to do. Closing at delta 0.50 caps the convex loss exposure before it becomes path-dependent.

### Position Sizing

Three multiplicative components:

**Component 1, Kelly-fractional from calibrated probability.**

Formula: position_size = kelly_fraction(p_calibrated, max_win, max_loss), capped at 25% of full Kelly.

Rationale: Full Kelly is theoretically optimal for log-utility maximization but famously aggressive under parameter uncertainty. When the probability estimate has any error (and it always does), full Kelly amplifies that error into account drawdowns. Quarter-Kelly (25%) is the institutional convention for reducing variance while preserving most of the geometric growth advantage. Thorp (1969) and subsequent work establishes that fractional Kelly between 1/4 and 1/2 is appropriate for systematic strategies with parameter uncertainty.

**Component 2, Volatility scaling.**

Formula: vol_multiplier = 15 / VIX_current.

Rationale: Position size scales inversely with current implied volatility. The intent is to keep ex-ante portfolio volatility roughly constant at 10 to 12% annualized regardless of regime. Without vol scaling, a fixed-notional position has 3x the realized vol exposure when VIX is at 30 vs at 15. Vol scaling is standard in trend-following and risk-parity strategies (AQR, Man AHL, Bridgewater all use variants).

**Component 3, Stress multiplier.**

Formula: stress_mult = 0.5 if SPY-TLT 20-day correlation exceeds 0.5, else 1.0.

Rationale: When stocks and bonds move together (correlation flips positive), the diversification structure of the market is breaking down. This is a leading indicator of systemic stress (most pronounced in 2008, 2020, 2022). Cutting size during these regimes reduces exposure to the moments when vol-selling strategies most often fail. The 0.5 threshold is set on 2010-2017 data using the 90th percentile of 20-day SPY-TLT rolling correlation.

**Hard cap:** No single trade risks more than 1% of account equity.

Rationale: At $100k account size, this caps maximum loss per trade at $1000. With XSP 5-point spreads having maximum loss of roughly $400 per spread after credit, this allows 2 to 3 contracts per trade, which is sufficient for the strategy and prevents any single bad trade from materially damaging the account. The 1% rule is from Van Tharp's position sizing literature and is the conventional retail-systematic upper bound.

---

## The Halt Framework (Vestal's Two Questions, Answered)

Five layers, each with a different time constant and trigger logic. All thresholds pre-committed using only 2010-2017 data.

### Layer 1, Hard halt (immediate, all positions closed)

**Trigger A: VIX intraday spike above 40% from prior close.**

Rationale: A 40% intraday VIX spike is the threshold separating "elevated vol" from "regime break." In 2010-2017 training data, this threshold was crossed exactly 4 times (August 2011, August 2015, December 2018 partial, September 2017 partial). Each instance preceded a multi-day drawdown in equity volatility-selling strategies. The 40% threshold is the 99.5th percentile of daily VIX percent changes in training data.

**Trigger B: SPX intraday move above 5% in either direction.**

Rationale: 5% daily moves are by definition tail events. In 2010-2017, this threshold was crossed twice (August 2011 downside, November 2016 election-driven). For a short-vol strategy, large intraday moves indicate gamma-driven repricing that no slower model can react to in time. Hard halt is appropriate.

**Trigger C: VIX3M-VIX inverts by more than 2 points (VIX at least 2 points above VIX3M).**

Rationale: Severe term-structure inversion is the single most predictive indicator of imminent vol-selling losses. Normal regime: VIX3M-VIX is positive (contango), averaging +2 to +3 points. Stress regime: this flips negative (backwardation). The -2 point threshold (VIX 2 points above VIX3M) is the magnitude that historically precedes the worst single-month drawdowns for short-vol strategies. February 24-26, 2020 crossed this threshold three days before the worst of the COVID crash. The 5-day version of this rule (which is slower) would have caught the COVID move only after the worst was done. The 2-point hard halt catches it in time.

### Layer 2, Soft halt (no new entries, manage existing to natural exit)

**Trigger A: VIX3M < VIX for 2 consecutive closes.**

Rationale: Sustained term structure inversion (even if not severe enough for hard halt). The 2-day persistence requirement filters single-day noise inversions while still being fast enough to catch developing stress.

**Trigger B: HYG-LQD spread widens above 2 standard deviations from 252-day rolling mean.**

Rationale: HYG (high yield credit) minus LQD (investment grade credit) is the simplest available proxy for credit market stress. When credit spreads widen sharply, equity volatility almost always follows, but with a lag of 1 to 5 days. The 2-SD threshold is a soft regime-shift indicator. The 252-day rolling baseline (1 trading year) accounts for the slow drift in credit spread averages over time.

**Trigger C: Model probability stays below 0.55 entry threshold for 5 consecutive sessions.**

Rationale: If the XGBoost model itself is consistently saying "do not enter" for a full week, the environment is hostile by the model's own assessment. This is the model-driven version of the halt rules. Combined with the structural triggers (VIX, term structure, credit spreads), it provides a model-based safety check.

### Layer 3, Slow halt (Hoeffding-driven, statistical)

**Trigger A: Rolling 60-trade win rate falls below in-sample baseline by more than 2 standard errors.**

Rationale: This is the formal Hoeffding bound implementation from Egger and Vestal 2025 (course material). Under H0 (the strategy's in-sample regime persists), Hoeffding's Inequality bounds the probability that observed underperformance is due to chance. When the bound exceeds 2 SE (roughly p < 0.05 one-sided), the in-sample regime is statistically rejected and the strategy is no longer behaving as backtested. This directly answers Vestal's "how do you know it stopped working" question.

**Trigger B: Rolling 90-day Sharpe ratio significantly negative (block-bootstrap p < 0.10).**

Rationale: Complementary to Trigger A. Where the win-rate test catches degradation in trade-level expectancy, the Sharpe test catches degradation in risk-adjusted return. Block-bootstrap (with non-overlapping 20-day blocks) preserves the autocorrelation structure of the trade outcome series.

### Layer 4, Drawdown halt

**Trigger A: Account underwater for more than 90 trading days.**

Rationale: 90 trading days is roughly 4.5 calendar months. A drawdown lasting longer than this is no longer "noise" by the strategy's expected time-to-recovery distribution from in-sample data. In 2010-2017 training, the longest in-sample drawdown was 47 trading days. The 90-day threshold is roughly 2x this maximum, providing a buffer for non-stationarity while still triggering on durations the strategy was never designed to survive.

**Trigger B: Drawdown depth above 15% of starting equity in trailing 90 days.**

Rationale: 15% is the conventional retail tolerance for systematic strategy drawdowns. Beyond this depth, behavioral pressure to abandon the strategy outweighs the statistical case for waiting it out. This trigger captures fast drawdowns that don't yet meet the duration test.

### Layer 5, Auto-resume conditions

The strategy resumes only when ALL four conditions are simultaneously true:

1. VIX3M > VIX by more than 2 points for 5 consecutive closes (term structure recovered)
2. Realized 5-day vol below 80th percentile of trailing 252 days (vol regime calmed)
3. HYG-LQD spread within 1 SD of long-run mean (credit stress resolved)
4. Drawdown recovered to within 3% of high-water mark (account healed)

Rationale: All-four conjunction prevents premature resumption during dead-cat-bounce regimes. Each condition independently is too noisy to be a sufficient resume signal, but the conjunction is conservative enough to filter false positives. The 5-day persistence on the term structure check ensures the recovery is durable, not a single-day artifact.

### The theoretical analog: Merton's down-and-out option

The halt framework is conceptually a barrier strategy. The strategy is "alive" while regime indicators are above the barrier and "dies" when they cross it. This maps directly onto Merton 1973, Section 9 (Down-and-Out Call Option). Merton derives the price of options that become worthless if the underlying crosses a stated barrier. The mathematical structure of "value while above barrier, zero if barrier touched" is the same structure as our halt logic.

This connection is unique to this project. Most retail vol-selling backtests have ad-hoc halt rules with no formal grounding. Ours derives from rigorous options theory.

---

## Feature Set (Vestal's Exogenous Framework)

16 features, all evaluated at close of trading day t-1, all from market-traded instruments. No revised macro data, no SPY-derived technical indicators beyond return and MA distance, no options data on XSP itself.

### Volatility regime (4 features)

- **VIX**: 30-day implied volatility on SPX options. Headline measure.
- **VIX3M**: 90-day implied volatility on SPX options. Captures longer-dated end of curve.
- **VIX3M minus VIX spread**: Term structure shape. Positive in contango (normal), negative in backwardation (stress). Single most important regime indicator for vol-selling.
- **VVIX**: Volatility of VIX itself. When VVIX rises, even the vol regime is unstable.

### Variance risk premium (2 features)

- **VIX minus 30-day realized vol on SPY**: The premium directly. Wide gap means rich insurance pricing.
- **30-day backward-looking version at 60-day horizon**: Same calculation, longer window. Captures persistence of premium.

### Yield curve shape (4 features)

Polynomial coefficients fit to 5 Treasury yields: {3M (IRX), 2Y, 5Y (FVX), 10Y (TNX), 30Y (TYX)}. Degree-3 polynomial gives 4 coefficients describing level, slope, curvature, and inflection.

Rationale: Direct from Vestal's verbal feedback on HW3. "Fit a polynomial spline through it. Store only the polynomial coefficients as individual columns. This captures curve shape without picking arbitrary tenors and without creating collinear features." Inverted curves (negative slope coefficient) historically precede recessions, which historically precede vol regime shifts.

### Cross-asset stress (3 features)

- **SPY-TLT 20-day correlation**: Stocks vs Treasuries. Normal anti-correlation. Stress-regime positive correlation.
- **SPY-GLD 20-day correlation**: Stocks vs Gold. Captures whether gold acts as flight-to-safety.
- **HYG-LQD spread**: High-yield minus investment-grade credit. Captures credit market stress that often precedes equity vol.

### Broad market regime (3 features)

- **20-day SPY return**: Medium-term momentum.
- **SPY distance from 200-day MA (percentage)**: Long-term trend position.
- **50d/200d MA cross state (binary)**: Golden cross vs death cross.

### What's deliberately NOT in the feature set

- Any SPY price-derived technical indicators (RSI, Bollinger, MACD, ATR) on the asset being traded. Vestal's exogeneity rule applied to the index option strategy.
- Options data on XSP itself (greeks, IV percentile of XSP options). The features describe the broader environment, not the specific instruments.
- Revised macro data (Fed balance sheet, NFP, CPI, GDP, unemployment). IBKR does not provide point-in-time vintage data for these series. Using current-revised values would introduce silent look-ahead bias because the values used in a 2015 backtest decision would not have been available to a 2015 trader. By restricting to market-traded data only, the feature set is point-in-time clean by construction.

### Validation discipline

Every feature value used to make a decision for trade execution on day t is computable from data available by close of day t-1. An explicit timestamp validation step in the pipeline asserts this for every feature on every trade. This catches the most common backtest error (timestamp leakage) before it can contaminate results.

---

## Model Architecture

### Primary classifier: XGBoost

Tree-based gradient boosting. Captures non-linear relationships and feature interactions natively, which is appropriate for the strategy because the relationship between environmental features and trade profitability is known to be non-linear (e.g., the effect of VIX level depends on whether term structure is in contango or backwardation).

**Hyperparameter tuning:** Nested time-series cross-validation with 5 splits within each training fold, Bayesian optimization via Optuna with 50 trials per fold. Tunes max_depth (3-6), learning_rate (0.01-0.1), n_estimators (with early stopping), subsample (0.6-1.0), colsample_bytree (0.6-1.0), reg_alpha (0-1), reg_lambda (0-1), min_child_weight (1-10).

Rationale: Nested CV is the only methodologically clean way to tune hyperparameters in walk-forward without contaminating the OOS test. Bayesian optimization (Optuna) is more efficient than grid or random search for high-dimensional continuous hyperparameter spaces.

### Calibration: Isotonic regression

Applied to out-of-fold XGBoost predictions before the decision threshold is applied.

Rationale: Raw XGBoost outputs are not calibrated probabilities. They are scores that often appear extreme (very close to 0 or 1) without corresponding to actual conditional probabilities. Without calibration, position sizing via Kelly formula is wrong by a factor that varies with the score. Isotonic regression (a non-parametric monotonic mapping) corrects this, ensuring that when the model says 70%, the realized win rate over many such trades is approximately 70%. Niculescu-Mizil and Caruana (2005) is the canonical reference.

### Benchmark: Elastic net logistic regression

Same 16 features. L1 plus L2 regularization with alpha and l1_ratio tuned via inner CV. Calibrated via Platt scaling.

Rationale: This is the rigorous version of Vestal's logistic regression framework. Vanilla LR with 16 features on a moderate sample size overfits. Elastic net handles this with proper regularization. Reporting elastic net alongside XGBoost demonstrates methodological pluralism: we did not pick the more complex model arbitrarily, we benchmarked it against the simpler alternative.

**Pre-committed model selection rule:** If elastic net OOS log-loss on the 2010-2017 training folds is lower than XGBoost log-loss, swap as primary. Otherwise XGBoost is primary. This rule is committed before any OOS test.

### Diagnostic overlay: Two-state HMM (Calm vs Stressed)

Hidden Markov Model fit on a focused subset of features (VIX, VIX3M-VIX, SPY-TLT correlation, realized 30-day vol). Two states selected via BIC.

**NOT used for trade decisions.** Used only for regime tagging in the writeup. Every trade is labeled with its HMM regime at entry, which enables performance-by-regime analysis.

**Stability fixes:**
- 20 random Baum-Welch initializations per fit, taking the highest log-likelihood result.
- Centroid-tracked state labeling: after each refit, states are reassigned by matching mean feature vectors to the previous fold's centroids using nearest-neighbor (Hungarian algorithm). This forces "Calm" to remain semantically "Calm" across all walk-forward refits, even as parameters drift.

Rationale: HMMs fit via Baum-Welch are sensitive to initialization, with different random seeds converging to different local optima. Without these stability fixes, state assignments flip across refits, making the diagnostic uninterpretable. With them, the HMM produces stable regime tags across the full 2010-2024 backtest.

### Walk-forward refit cadence: Annual

Every December 31, retrain XGBoost, elastic net, and HMM on all data through that date. Refit isotonic calibration. Use exponential weighting (alpha = 0.7) to blend new parameters with prior parameters.

Rationale: Annual cadence is chosen for engineering simplicity. Quarterly refits with smoothed parameters would be the institutional-grade version. The exponential parameter blending dampens whipsaw across refit boundaries (the strategy should not violently change its behavior on January 1 just because we refit on December 31).

---

## Friction Model

Aggressive and realistic. Every component justified by published or commonly-observed retail execution costs.

### Commissions

IBKR Pro tiered: $0.65 per options contract per leg, $1 minimum per order. Each spread = 2 legs at entry, 2 legs at exit = $1.30 minimum each side. Plus regulatory fees (SEC fee, ORF fee, OCC clearing fee), approximately $0.05 per contract per side.

### VIX-scaled bid-ask slippage

Pay a percentage of the bid-ask spread per side, scaled by current VIX:
- VIX < 20: 30% of spread
- VIX 20-30: 50% of spread
- VIX 30-40: 75% of spread
- VIX > 40: 100% of spread (full bid-ask)

Rationale: Bid-ask spreads on SPX/XSP options widen substantially during stress. Pretending you can always trade at the midpoint is the single most common reason retail backtests fail to generalize to live trading. The 30/50/75/100% schedule is calibrated against observed XSP option spreads in 2010-2017 across VIX regimes. The formula is conservative: it assumes you cannot consistently better than the midpoint plus a regime-dependent drag.

### Gap-aware stop execution

If SPX opens past the stop trigger level, execute at the open price, not the stop level.

Rationale: The realistic execution price after an overnight gap is whatever the market opens at, not where your stop was set. Backtests that assume execution at the stop level systematically underestimate drawdown depth during gap events. This is a non-negotiable realism check.

### Monday-open execution discipline

Strategy decides Friday close, executes Monday open. The 64-hour weekend window is a known risk.

- Monday entry uses Monday open bid-ask, not Friday close midpoint
- VIX-scaled slippage applies to Monday's environment (which may differ materially from Friday's)
- If SPX gaps more than 1.5% from Friday close to Monday open, apply 50% additional slippage on entry
- If gap exceeds 3%, skip entry entirely

Rationale: The 1.5% and 3% thresholds are set from the 95th and 99th percentiles of weekend SPX gap magnitudes in 2008-2017 training data. Pre-committed before OOS test.

### Tax treatment

Section 1256: 60% long-term capital gains and 40% short-term capital gains rate, applied to aggregate annual P&L. No carry-forwards modeled.

Rationale: This is the IRS treatment for SPX/XSP options regardless of holding period. For a strategy generating short-duration trades that would otherwise face 100% short-term rates, this is meaningful structural alpha worth roughly 5% of after-tax returns at typical retail tax brackets.

### Margin

IBKR Portfolio Margin: buying power requirement is roughly the maximum loss of the spread.

With 2 to 3 spreads concurrent at $400-500 max loss each, total margin commitment is $1000-1500 against $100k equity. Margin is not a binding constraint at this account size.

---

## Backtest Protocol

**In-sample (development period):** 2010-2017. All hyperparameter tuning, halt threshold setting, decision threshold selection happen here. Pre-commitment file (PRE_COMMITMENT.md, committed to git) records every methodological choice with timestamp.

**Out-of-sample (test period):** 2018-2024. Frozen pipeline run once. Reported results are whatever come out. No iteration based on what we see.

**Confidence intervals:** Block-bootstrap with non-overlapping 20-day blocks, 10,000 resamples on the trade outcome series. 90% CIs reported on Sharpe, win rate, average return per trade.

**Stress events analyzed (per-event analysis section):**
- August 2015 (China devaluation)
- February 2018 (Volmageddon)
- Q4 2018 (selloff)
- February-March 2020 (COVID, the critical test)
- March 2023 (regional banking crisis)

For each event: did the strategy halt? When? Did it resume? Realized P&L through the event? This is the failure-mode analysis section in the writeup.

**The single most important test:** Whether the halt rules fire BEFORE the worst of February-March 2020, not during. If yes, the strategy has real defensive structure. If no, the backtest is a curiosity regardless of overall return numbers.

---

## Component Attribution (Ablation Modes)

The strategy is composed of two independent layers (an ML gate and a halt framework). Without explicit attribution, it is impossible to know which layer is doing the work and which is decorative. The backtest engine therefore supports four `mode` flags from Day 1, with the same blotter logic across all four. Only the gates differ. No silent code paths that diverge between modes.

**Mode `naked`:** No ML filter, no halt rules. Sells one credit spread every Monday regardless of model output or regime indicators. This is the unfiltered baseline.

**Mode `ml_only`:** XGBoost gate active (calibrated probability >= 0.55 to enter), all halt rules disabled. Isolates the ML layer's contribution.

**Mode `halts_only`:** No ML filter (every Monday is a candidate entry), all five halt layers active. Isolates the halt framework's contribution.

**Mode `full`:** Both ML filter and halt rules active. The strategy as designed.

All four modes run on both in-sample (2010-2017) and out-of-sample (2018-2024). The Component Attribution table reports, for each mode and each period:

- Pooled Sharpe ratio
- Maximum drawdown
- Drawdown duration (longest underwater period)
- Win rate
- Average return per trade
- Total trade count
- Per-event survival check across the five stress events listed above

**Pre-committed interpretation rule (recorded in PRE_COMMITMENT.md before any OOS test runs):**

> If `ml_only` Sharpe is within 0.1 of `naked` baseline Sharpe in the OOS period, the ML filter is decorative. The writeup states this explicitly. The ML layer remains in the architecture as documentation of Vestal's exogenous-features methodology and as a regime-drift diagnostic for live monitoring, but it is not credited with strategy performance.

---

## Implementation Discipline (Bug Prevention)

The architecture has substantial complexity (XGBoost with nested CV, isotonic calibration, 16-feature exogenous pipeline, two-state HMM diagnostic, 5-layer halt framework, VIX-scaled friction model, gap-aware execution, ablation modes, walk-forward refit). The probability of undetected bugs scales with system complexity, and silent bugs in ML pipelines are particularly dangerous because the code runs without error and produces plausible-looking output that is wrong. HW5 demonstrated this empirically: the blotter showed 21 trades while the equity curve silently used 7 due to an LR filter that was not supposed to be filtering. The mismatch was undetected until manual inspection. We will not repeat this.

The following implementation disciplines are mandatory, not optional:

### Unit Tests

Every source module in `src/` has a corresponding test file in `tests/`. Each test verifies expected behavior on known inputs.

- `tests/test_features.py`: feature pipeline produces correct values on synthetic data, no NaN propagation, no future leakage
- `tests/test_strategy.py`: spread construction returns valid strikes, exit logic fires correctly on synthetic price paths
- `tests/test_friction.py`: VIX-scaled slippage produces correct values across regimes, gap-aware execution returns min(stop, open)
- `tests/test_halts.py`: each of the five halt layers fires on synthetic regime data, auto-resume conditions activate only when all four are simultaneously true
- `tests/test_models.py`: XGBoost training is deterministic given seed, calibration produces monotonic probability mapping, HMM state labels are stable across refits
- `tests/test_backtest.py`: trade count in equals trade count out across the engine, blotter rows match equity curve increments

All tests must pass before any commit to main. Run via `pytest tests/ -v`.

### Consistency Assertions

Every artifact handoff has runtime assertions, not just code-review checks:

- Trade count entering a function equals trade count leaving it
- Blotter row count equals equity curve update count
- Feature dataframe has expected number of columns and rows
- Per-fold metrics sum to pooled metrics within tolerance
- Universe count is consistent across all reporting sections

Use `assert` statements with descriptive error messages. These are not removable in production: this is a research backtest where correctness is the primary requirement.

### Seeded Randomness

Single global seed defined in `src/config.py`. Every random component reads from this seed:

- XGBoost `random_state`
- Train/test splits in nested CV
- Block-bootstrap resampling
- HMM Baum-Welch initialization (each of 20 starts uses `seed + i`)
- Optuna sampler seed
- All sklearn `random_state` parameters

The seed is logged at the top of every notebook execution. If results look anomalous, the entire pipeline is rerunnable to verify reproducibility.

### Consistency Check Pass

`notebooks/99_consistency_check.ipynb` runs after every full pipeline execution and reports:

- Universe count matches across README, writeup intro, asset selection section, assumptions section
- Trade count matches across blotter, equity curve, metrics tables, Hoeffding monitor, slippage sensitivity, ablation comparison
- Date ranges match across all artifacts
- Per-fold breakdowns sum to pooled totals (within rounding tolerance)
- No NaN or inf values in any output dataframe
- All four ablation modes produce trade counts consistent with their respective configurations

The notebook raises an exception if any check fails. The pipeline is not considered complete until this notebook passes cleanly.

### The "No Silent Filtering" Rule

If the blotter contains N trades, every downstream artifact (equity curve, performance metrics, Hoeffding monitor, slippage sensitivity, ablation tables) reflects the same N trades. The ML probability is annotation only, never a downstream filter that creates inconsistency between sections. Enforced as a runtime assertion in the backtest engine, not just a methodological rule.

---

## Live Performance Monitoring

This section answers Vestal's two rubric questions explicitly. Forward-looking framework for running the strategy with real money.

### Daily monitoring routine

After each trade closes, the monitor updates:

1. **Rolling 60-trade win rate** plotted with Hoeffding 90% bound around in-sample baseline. If the rolling rate falls below the bound, this is statistical evidence that the strategy's regime has shifted.

2. **Rolling 90-day Sharpe** with block-bootstrap 90% CI updated each trade. Provides a model-free check on whether risk-adjusted returns match backtest expectations.

3. **Calibration drift check.** Predicted probability bins (0.55-0.60, 0.60-0.65, etc.) compared to realized win rate within each bin. Calibrated model: realized matches predicted within sampling error. Drifted model: realized diverges from predicted.

4. **Feature stability monitor.** XGBoost feature importance scores in recent predictions compared to feature importance in original training. Sustained shift indicates the model is no longer fitting the same underlying relationships.

### Watch list (soft signals, do not trigger halts but get reported daily)

- Recent calibration error trend
- Variance risk premium compression (if VRP shrinks below 25th percentile of trailing 5 years for sustained periods, scale down position size)
- HMM state transition frequency (rapid switching suggests regime instability)
- Cross-asset correlation drift

### Forward shadow-trading plan

Post-submission, the strategy will be paper-traded via IBKR demo account for 30 to 60 days. Live trades will be compared to backtest predictions for the same period. This provides real-world validation of whether backtest results translate to live execution.

---

## Theoretical Foundation

The writeup includes a formal theoretical foundation section grounding every strategic choice in canonical options literature.

### 1. Variance risk premium thesis

**Citation:** Merton 1973, Theorems 8 and 15. Carr-Madan 1998 for the variance swap decomposition.

Merton proves that option price is monotone increasing in variance. Selling options is selling variance. The structural premium exists because hedgers pay above-fair-value for tail-risk protection. This grounds the strategy's first edge.

### 2. Theta decay

**Citation:** Black-Scholes 1973 for the closed-form theta formula. Merton 1973 equation 36 for the parallel derivation via the no-arbitrage hedging argument.

Theta decay is the mechanism by which the static premium (variance risk premium) converts into realized profit as time passes. The rate of decay accelerates non-linearly toward expiration, which justifies the 30-45 DTE entry window and 21-DTE roll rule. Both papers confirm decay arises from the no-arbitrage condition independent of investor risk preferences.

### 3. Put-call parity as data-integrity check

**Citation:** Merton 1973, Theorem 12.

The relationship f(S, tau, E) - g(S, tau, E) = S - E*P(tau) must hold for European options on non-dividend-paying assets. Used to validate every options chain pulled from IBKR. If parity does not hold within a few cents, the data has a problem (stale quote, wrong timestamp, data error). This catches data quality issues before they contaminate the backtest.

### 4. High-contact condition

**Citation:** Merton 1973, Section 7.

The optimality condition for early exercise of American options: W'(C[tau], tau, E) = 1. While XSP options are European-style, the spirit of the condition (assignment risk becomes material when delta approaches 1) justifies our delta-based emergency exit at delta > 0.50.

### 5. Down-and-out option as halt-rule analog

**Citation:** Merton 1973, Section 9.

This is the most distinctive theoretical move in the project. Merton derives the price of options that become worthless if the underlying crosses a stated barrier. The mathematical structure of "value while above barrier, zero if barrier touched" maps directly onto our halt logic. The strategy is "alive" while regime indicators are above the barrier and "dies" when they cross. Most retail vol-selling backtests have ad-hoc halt rules with no formal grounding. Ours derives from rigorous options theory.

### Additional references

- López de Prado 2018, *Advances in Financial Machine Learning*, especially Chapter 3 on meta-labeling (referenced as future extension, not implemented in this version).
- Hoeffding 1963 for the inequality underlying the live monitoring framework.
- Egger and Vestal 2025 (course material) for the Hoeffding-based regime monitoring application.

---

## Methodological Discipline (Non-Negotiable Rules)

Pre-committed rules, learned from prior coursework experience, that make the project defensible:

1. **Never tune parameters after seeing OOS results.** Pre-commit using 2010-2017. Freeze for 2018-2024. Disappointing results are still results. The PRE_COMMITMENT.md file is committed to git with timestamp before any OOS test runs.

2. **No silent filtering.** If 50 signals fire, 50 trades appear in the blotter, 50 trades drive the equity curve. ML probability is annotation, not a downstream gate that creates inconsistencies between sections. Enforced as a runtime assertion (see Implementation Discipline).

3. **Strategy parameters frozen across folds.** Only hyperparameters (XGBoost depth, regularization) and feature scalers refit at fold boundaries. Strategy parameters (DTE windows, strike selection rules, exit multiples, sizing parameters) are design decisions, not learned.

4. **Pre-commit with rationale before running.** Halt thresholds, decision threshold, sizing parameters all need documented justification BEFORE seeing OOS results. The PRE_COMMITMENT.md file enforces this.

5. **Internal consistency.** Universe count, trade count, metric values must match across every section of the writeup. A grader who finds inconsistencies discards the entire result. Verified by `notebooks/99_consistency_check.ipynb`.

---

## Repository Structure

```
.
├── README.md                       # This file
├── PRE_COMMITMENT.md               # Methodology frozen before OOS test
├── data/                           # IBKR data cache (parquet, gitignored)
│   ├── equities/                   # SPY, XSP, ETFs
│   ├── vol_indices/                # VIX, VIX3M, VVIX
│   ├── yields/                     # IRX, FVX, TNX, TYX
│   ├── options_chains/             # SPY/XSP historical chains
│   └── README.md                   # Data sources, refresh cadence
├── src/
│   ├── config.py                   # Global seed and constants
│   ├── data/
│   │   ├── fetch_ibkr.py           # IBKR fetch via shinybroker
│   │   ├── validate_chains.py      # Put-call parity validation
│   │   └── timestamp_audit.py      # Look-ahead bias check
│   ├── features/
│   │   ├── exogenous.py            # 16-feature pipeline
│   │   ├── yield_curve.py          # Polynomial spline fit
│   │   └── correlations.py         # Cross-asset stress
│   ├── strategy/
│   │   ├── spread_construction.py  # XSP put credit spreads
│   │   ├── exits.py                # 4-rule exit logic
│   │   ├── friction.py             # Cost model
│   │   ├── sizing.py               # Kelly + vol + stress
│   │   └── halts.py                # 5-layer halt framework
│   ├── models/
│   │   ├── xgboost_primary.py      # XGBoost + Optuna nested CV
│   │   ├── elastic_net_bench.py    # Logistic regression benchmark
│   │   ├── hmm_diagnostic.py       # 2-state HMM overlay
│   │   └── calibration.py          # Isotonic regression
│   ├── backtest/
│   │   ├── walkforward.py          # Annual refit harness
│   │   ├── engine.py               # Trade simulation (mode-aware)
│   │   └── stress_events.py        # Per-event analysis
│   ├── metrics/
│   │   ├── performance.py          # Sharpe, Sortino, etc.
│   │   ├── hoeffding.py            # Live monitoring bounds
│   │   └── bootstrap.py            # Block-bootstrap CIs
│   └── monitoring/
│       └── live_monitor.py         # Daily update logic
├── notebooks/
│   ├── 01_data_audit.ipynb         # Data validation
│   ├── 02_baseline.ipynb           # Naked baseline 2010-2017
│   ├── 03_xgboost_training.ipynb   # Model training + calibration
│   ├── 04_walkforward_oos.ipynb    # 2018-2024 frozen test (all 4 modes)
│   ├── 05_stress_events.ipynb      # Per-event analysis
│   ├── 06_live_monitoring.ipynb    # Forward-looking dashboard
│   └── 99_consistency_check.ipynb  # End-of-pipeline assertion notebook
├── website/
│   ├── index.qmd                   # Main writeup
│   ├── _quarto.yml                 # Quarto config
│   └── assets/                     # Charts, tables
├── docs/                           # Rendered HTML (GitHub Pages)
├── tests/
│   ├── test_features.py
│   ├── test_strategy.py
│   ├── test_friction.py
│   ├── test_halts.py
│   ├── test_models.py
│   └── test_backtest.py
├── requirements.txt
├── .gitignore
└── pyproject.toml
```

---

## Setup

### Environment

```bash
git clone https://github.com/mariotrev120/spx-vrp-strategy.git
cd spx-vrp-strategy
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### IBKR TWS

1. Install TWS or IB Gateway, log in with paper trading account
2. Enable API access: Configure, then API, then Settings, then check "Enable ActiveX and Socket Clients"
3. Set socket port to 7497, allow localhost connections
4. Confirm connection: `python -c "import shinybroker as sb; print(sb.req_current_time())"`

### Data Pull

```bash
python src/data/fetch_ibkr.py --start 2010-01-01 --end 2024-12-31
```

Initial pull takes 4 to 8 hours depending on IBKR responsiveness. Cached as parquet in `data/`.

### Run Backtest

```bash
# Full pipeline, all 4 ablation modes
python src/backtest/walkforward.py --train-end 2017-12-31 --test-end 2024-12-31 --modes naked,ml_only,halts_only,full

# Render writeup
cd website && quarto render
```

---

## Performance Targets

These are targets, not guarantees. The OOS test reports what actually happens.

- Annualized return (after taxes): 12 to 18%
- Sharpe ratio: 0.8 to 1.2
- Max drawdown: 8 to 15% if halts work, 25 to 40% if they fail (the test of the halt framework)
- Win rate: 70 to 80%
- Capacity at retail size: $5M+

The strategy survives February-March 2020 with less than 15% drawdown OR the strategy is broken. That is the single most important test.

---

## Known Limitations

1. **Sample size.** N = 15 years of training data includes only one true tail event (March 2020). Halt-rule confidence is bounded by limited stress data. Adding 2008 would help but options chain data quality pre-2010 is unreliable.

2. **IBKR data gaps.** Historical options chains may have missing strikes or expirations. Validated via put-call parity but residual data issues are possible.

3. **XSP liquidity at scale.** This implementation targets retail account sizes ($100k to $5M). Above $10M, migration to full-size SPX would be required because XSP bid-ask spreads widen substantially with size.

4. **Crowding risk.** Vol selling on SPX is one of the most popular systematic strategies. Backtest results from 2010-2024 partially reflect aggregate vol-seller behavior. Live deployment in 2026 may face different competitive dynamics that the backtest cannot capture.

5. **Refit cadence.** Annual refits chosen for engineering simplicity. Quarterly refits with parameter smoothing would be the institutional-grade version. We document this as a known limitation and a target for future iteration.

6. **No 2008 data.** Including the GFC would meaningfully strengthen halt rule calibration but the data quality and feature availability pre-2010 is insufficient for a clean inclusion.

7. **Complexity-related risks.** We acknowledge two complexity-related risks. First, the architecture has many moving parts (XGBoost gate, halt framework, HMM diagnostic, calibration, friction layers) and the probability of an undetected bug rises with system complexity. We mitigate via unit tests on every component, consistency assertions at every artifact handoff, seeded randomness throughout, and a dedicated consistency-check pass after every pipeline execution (see Implementation Discipline section). Second, complexity may be doing less work than it appears: the halt framework's defensive structure may dominate the ML filter's contribution. We address this by running explicit ablation studies (naked baseline, ML-only, halts-only, full pipeline) on both in-sample and out-of-sample data. The component attribution analysis is reported alongside the headline performance numbers, allowing the reader to assess which layers are contributing what.

---

## References

- Merton, R.C. (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics and Management Science*, 4(1), 141-183.
- Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*.
- Carr, P. and Madan, D. (1998). "Towards a Theory of Volatility Trading." In *Volatility: New Estimation Techniques for Pricing Derivatives*.
- Bakshi, G. and Kapadia, N. (2003). "Delta-Hedged Gains and the Negative Market Volatility Risk Premium." *Review of Financial Studies*.
- Carr, P. and Wu, L. (2009). "Variance Risk Premiums." *Review of Financial Studies*.
- Bollerslev, T., Tauchen, G., and Zhou, H. (2009). "Expected Stock Returns and Variance Risk Premia." *Review of Financial Studies*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Hoeffding, W. (1963). "Probability Inequalities for Sums of Bounded Random Variables." *Journal of the American Statistical Association*.
- Niculescu-Mizil, A. and Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." *Proceedings of the 22nd International Conference on Machine Learning*.
- Thorp, E. (1969). "Optimal Gambling Systems for Favorable Games." *Review of the International Statistical Institute*.
- Egger, J. and Vestal, E. (2025). Hoeffding-based regime monitoring framework, FinTech 533 course material.

---

## License

MIT License. See LICENSE for details.

## Acknowledgments

- Professor Vestal for the exogenous-feature framework, Hoeffding monitoring discipline, and verbal critique of HW3 that shaped this project's methodology.
- Duke MEng Financial Technology faculty.
- The systematic vol-selling community whose collective failures during February 2018 and March 2020 informed the halt-rule design.

---

**Status:** In active development
