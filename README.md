# SPX Volatility Risk Premium Harvesting with ML-Gated Entry

**FinTech 533, Final Project**
**Authors:** Mario Trevino, Robert Lanni
**Duke University, Master of Engineering in Financial Technology**

---

## TL;DR

We sell SPX put credit spreads to harvest the variance risk premium, gated by an XGBoost classifier trained on Vestal-style exogenous environmental features. Position sizing uses calibrated probabilities and Kelly-fractional rules. Five layered halt mechanisms shut the strategy down when regime indicators turn hostile. Performance is evaluated via Hoeffding bounds and block-bootstrap confidence intervals. Theoretical foundation grounded in Merton 1973 (variance risk premium, put-call parity, with the down-and-out barrier as a conceptual analog for the halt framework) and Black-Scholes 1973 (theta decay).

**Backtest period:** 2012-03-26 to 2024-12-31 (in-sample 2012-03-26 to 2017-12-31, OOS 2018-01-01 to 2024-12-31)
**Universe:** SPX (S&P 500 Index Options, cash-settled, European exercise, Section 1256 taxed)
**Data sources:** OptionMetrics IvyDB US for SPX option chains (via WRDS); IBKR TWS via shinybroker for index, yield, and ETF history.

**Headline OOS results (2018-2024, real OPRA quotes, post-bugfix):**

| Mode | Trades | Win Rate | Annualized Return | Sharpe (trade) | Max DD |
|---|---|---|---|---|---|
| naked | 272 | 64.0% | +16.69% | 0.447 [-0.15, 0.97] | 2.73% |
| ml_only | 272 | 64.0% | +16.69% | 0.447 [-0.15, 0.97] | 2.73% |
| halts_only | 7 | 57.1% | -0.02% | -1.81 [-7.24, -0.74] | 0.21% |
| full | 7 | 57.1% | -0.02% | -1.81 [-7.24, -0.74] | 0.21% |

(Square brackets are 90% block-bootstrap CIs. CIs that cross zero indicate the trade-level Sharpe is not strictly positive at 90% confidence.)

**Pre-committed interpretation rule fires:** the OOS Sharpe gap between `ml_only` and `naked` is **+0.000** — the two modes produce *identical* trade-by-trade output. After fixing a 1-trading-day feature look-ahead leak (see Bug Audit section), the ML model's predictions on Friday-close features no longer have an artificial edge over the unfiltered baseline. Every OOS Monday passes the 0.55 calibrated-probability threshold, so the gate is fully inert in OOS. The XGBoost layer adds zero economic value beyond annotation. The XGBoost gate does NOT add material value over the unfiltered baseline. The structural variance risk premium edge (and the friction model that captures it) is the entire source of OOS profitability. The halt framework, calibrated on 2012-2017 IS data, is too aggressive on OOS — it kept the strategy out of nearly every trade and missed the bull-market premium. The honest result is that the simple, ungated baseline is the most profitable mode in the OOS period.

We report this finding directly. The pre-commitment to disclose negative ML attribution was the right discipline; it is now the honest centerpiece of this writeup.

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

The strategy harvests this premium directly. The risk is that the premium is highest precisely when crash risk is elevated, so naive vol selling blows up during stress events (Volmageddon February 2018, COVID March 2020 being the canonical retail blow-up cases). The entire engineering of this project addresses that asymmetric risk profile through the halt framework.

### Why credit spreads instead of naked puts?

Three reasons:

1. **Defined risk.** Maximum loss per spread is known at entry (spread width minus credit received). This makes position sizing clean and prevents single-trade catastrophic loss.

2. **Capital efficiency.** Margin requirement is roughly maximum loss, not full underlying notional. An account can run multiple concurrent spreads without margin pressure.

3. **Risk-bounded.** Naked put selling on SPX has tail risk that no responsible position-sizing rule can absorb. Credit spreads make the strategy deployable in a portfolio context.

### Why SPX

SPX is the canonical cash-settled European-style index option contract. Three structural advantages over equity ETF options like SPY:

1. **Cash settlement.** SPX options settle in cash. There is no risk of physical assignment of 100 shares per contract over a weekend. SPY's physical settlement creates "pin risk" where a contract closing exactly at the short strike can produce overnight assignment without warning, converting a defined-risk spread into an undefined-risk stock position.

2. **European exercise.** SPX cannot be exercised early. American-style SPY options can be assigned any business day, particularly when deep in the money near ex-dividend dates. European exercise removes this entire risk class.

3. **Section 1256 tax treatment.** SPX qualifies for IRS Section 1256 contract treatment: 60% long-term capital gains and 40% short-term capital gains rate, regardless of holding period. For a strategy generating short-duration trades that would otherwise face 100% short-term rates, this is meaningful structural alpha worth roughly 5 to 7% of after-tax returns.

In addition, SPX has the deepest, most-liquid index option chain available — continuous OPRA-grade coverage since the 1990s with tight bid-ask spreads even at strikes 1 standard deviation OTM. This data quality is what makes a clean systematic backtest possible.

---

## Strategy Mechanics

### Entry

Every Monday at the open, the system evaluates an exogenous feature vector computed at Friday close. If the calibrated XGBoost probability of trade profitability is at least 0.55, the system sells one SPX put credit spread.

**Spread construction:**
- Short strike: approximately 1 standard deviation OTM (16-delta)
- Long strike: 5 points further OTM
- Days to expiration at entry: 30 to 45 days

**Why 1-sigma OTM short strike?** Empirical research on equity index put-selling (notably from Tasty Trade and CBOE PUT/PutWrite indices) shows that 1-sigma OTM short strikes balance premium collection against assignment risk. Closer to ATM increases credit but lowers win rate. Further OTM increases win rate but compresses absolute return per trade. The 16-delta convention is the industry standard for systematic put-writing.

**Why 30 to 45 DTE?** This window optimizes the trade-off between theta decay rate and gamma exposure. Black-Scholes shows theta is proportional to roughly 1/sqrt(tau), so theta-per-day accelerates as expiration approaches. But gamma also accelerates, and gamma is the enemy of short option positions. Below 30 DTE, gamma risk dominates and small price moves cause large P&L swings. Above 45 DTE, theta is too slow to be efficient capital deployment. The 30 to 45 window is where theta is rich relative to gamma.

**Why decision threshold of 0.55?** The threshold is set 0.05 above the naive 0.50 cutoff to require modest model confidence above coin-flip. The asymmetry of credit spread payoffs (typical max win of credit collected vs. max loss of width-minus-credit) means a 50% win rate is roughly breakeven before friction. The 0.55 threshold provides a margin to cover friction and provide positive expectancy. The 0.55 value was pre-committed before any OOS test.

### Exit (Three Fates Plus Emergency)

**Fate 1, Profit target:** Close at 50% of maximum profit.

Rationale: Standard Tasty Trade research finding, replicated across multiple practitioner studies. The first 50% of profit accumulates roughly 3 to 4 times faster than the second 50% because theta decay is fastest when the option still has meaningful time value. Closing at 50% redeploys capital into fresh trades with rich theta.

**Fate 2, Stop loss:** Close at 200% of credit received, executed at min(stop_level, opening_gap_price).

Rationale: 200% of credit is the standard stop level for credit spread systems. Tighter stops (150%) get noise-stopped during normal volatility expansion. Wider stops (300%) produce single trades that can wipe out 5 or more winners. The gap-aware execution rule (min of stop level and opening price) accounts for the reality that overnight gaps can blow through stops.

**Fate 3, Time exit:** Hard close at 21 DTE.

Rationale: This is the gamma-cliff protection. Inside 21 DTE, gamma per dollar of premium remaining accelerates non-linearly. The 21 DTE exit captures roughly 70% of the theoretical max profit while exiting the high-gamma zone entirely. This is also a standard convention in systematic put-writing literature.

**Emergency exit:** Short leg delta exceeds 0.50.

Rationale: When the short leg moves to ATM (delta near 0.50), the original assumptions of the trade (1-sigma OTM, defined edge, theta-favored) no longer hold. The position has converted from "premium harvest" to "directional bet on bounce-back." Closing at delta 0.50 caps the convex loss exposure before it becomes path-dependent.

### Position Sizing

Three multiplicative components:

**Component 1, Kelly-fractional from calibrated probability.** Capped at 25% of full Kelly. (Note: for the loss/win ratio of typical credit spreads, Kelly evaluates to zero or negative below ~85% win rate. The strategy uses Kelly as a documented diagnostic, not as the binding sizing constraint — see below.)

**Component 2, Volatility scaling.** `vol_multiplier = 15 / VIX_current`. Position size scales inversely with current implied volatility, keeping ex-ante portfolio vol roughly constant across regimes.

**Component 3, Stress multiplier.** `stress_mult = 0.5` if SPY-TNX 20-day correlation exceeds 0.5, else 1.0. (Original spec was SPY-TLT correlation; TWS paper account history on TLT begins 2016, while TNX 10Y yield extends back to 2011. Same regime signal.)

**Hard cap:** No single trade risks more than 1% of account equity. With SPX 5-point spreads having maximum loss of roughly $400 per spread after credit, this caps maximum loss per trade and is the *binding* constraint in practice. Kelly enters as a multiplier on top of the cap when ML is active, which (per the empirical observation above) typically multiplies by 0 to 1 depending on calibrated probability.

---

## The Halt Framework

Five layers, each with a different time constant and trigger logic. All thresholds were pre-committed using only 2012-2017 IS data.

### Layer 1, Hard halt (immediate, all positions closed)

**Trigger A: VIX intraday spike above 40% from prior close.**

Rationale: A 40% intraday VIX spike is the threshold separating "elevated vol" from "regime break." The 40% threshold is the 99.5th percentile of daily VIX percent changes in IS training data.

**Trigger B: SPX intraday move above 5% in either direction.**

Rationale: 5% daily moves are tail events. For a short-vol strategy, large intraday moves indicate gamma-driven repricing that no slower model can react to in time. Hard halt is appropriate.

**Trigger C: VIX3M-VIX inverts by more than 2 points (VIX at least 2 points above VIX3M).**

Rationale: Severe term-structure inversion is the single most predictive indicator of imminent vol-selling losses. Normal regime: VIX3M-VIX positive (contango), averaging +2 to +3 points. Stress regime: this flips negative (backwardation). The -2 point threshold is the magnitude that historically precedes the worst single-month drawdowns for short-vol strategies.

### Layer 2, Soft halt (no new entries, manage existing to natural exit)

- VIX3M < VIX for 2 consecutive closes
- HYG-LQD spread widens above 2 standard deviations from 252-day rolling mean
- Model probability stays below 0.55 entry threshold for 5 consecutive sessions

### Layer 3, Slow halt (Hoeffding-driven, statistical)

- Rolling 60-trade win rate falls below in-sample baseline by more than 2 standard errors (Hoeffding bound from Egger-Vestal 2025)
- Rolling 90-day Sharpe ratio significantly negative (block-bootstrap p < 0.10)

### Layer 4, Drawdown halt

- Account underwater for more than 90 trading days, OR
- Drawdown depth above 15% of starting equity in trailing 90 days

### Layer 5, Auto-resume conditions

The strategy resumes only when ALL four conditions are simultaneously true:

1. VIX3M > VIX by more than 2 points for 5 consecutive closes
2. Realized 5-day vol below 80th percentile of trailing 252 days
3. HYG-LQD spread within 1 SD of long-run mean
4. Drawdown recovered to within 3% of high-water mark

### The theoretical analog: Merton's down-and-out option

The halt framework is conceptually a *barrier strategy*: the strategy is "alive" while regime indicators sit above their barriers and stops out when one is crossed. This is structurally analogous to Merton 1973, Section 9 (Down-and-Out Call Option), where an option's value is positive while the underlying remains above a barrier and becomes zero if the barrier is touched.

We claim only the analogy, not the math. Merton derives a closed-form price for a financial instrument with a fixed barrier on the underlying under GBM; our halts are stop conditions on a live trade with empirically-calibrated barriers on multiple regime indicators (VIX, term structure, credit spreads, realized drawdown). The thresholds are 99.5th-percentile-style cutoffs from 2012-2017 training data, not optimized inside a barrier-pricing framework. The Merton parallel is the conceptual frame we used to organize the halt design; it is not a derivation of the thresholds.

What we believe is genuinely uncommon in retail vol-selling backtests is the explicit barrier framing combined with pre-committed thresholds and ablation attribution, not the use of options theory itself.

---

## Feature Set (Vestal's Exogenous Framework)

15 features, all evaluated at close of trading day t-1, all from market-traded instruments. No revised macro data, no SPY-derived technical indicators beyond return and MA distance, no options data on SPX itself.

### Volatility regime (4 features)

- **VIX**: 30-day implied volatility on SPX options.
- **VIX3M**: 90-day implied volatility on SPX options.
- **VIX3M minus VIX spread**: Term structure shape.
- **VVIX**: Volatility of VIX itself.

### Variance risk premium (2 features)

- **VIX minus 30-day realized vol on SPY**
- **VIX minus 60-day realized vol on SPY** (longer-window persistence check)

### Yield curve shape (3 features)

Polynomial coefficients fit to 4 Treasury yields: {3M (IRX), 5Y (FVX), 10Y (TNX), 30Y (TYX)}. Degree-2 polynomial gives 3 coefficients describing level, slope, and curvature.

(Original spec was degree-3 over a 5-tenor curve including 2Y. TWS does not expose a 2Y CBOE yield index, so the implementation uses degree-2 over the 4 tenors available.)

### Cross-asset stress (3 features)

- **SPY-TNX 20-day correlation**: Stocks vs Treasury yields. (Original spec was SPY-TLT; TLT history on TWS paper begins 2016 while TNX extends to 2011. Same signal.)
- **SPY-GLD 20-day correlation**: Stocks vs Gold.
- **HYG-LQD spread**: High-yield minus investment-grade credit price spread.

### Broad market regime (3 features)

- **20-day SPY return**
- **SPY distance from 200-day MA (percentage)**
- **50d/200d MA cross state (binary)**

### What's deliberately NOT in the feature set

- Any SPY price-derived technical indicators (RSI, Bollinger, MACD, ATR) on the asset family being traded. Vestal's exogeneity rule applied to the index option strategy.
- Options data on SPX itself (greeks, IV percentile of SPX options).
- Revised macro data (Fed balance sheet, NFP, CPI, GDP, unemployment). Using current-revised values would introduce silent look-ahead bias because the values used in a 2015 backtest decision would not have been available to a 2015 trader. By restricting to market-traded data only, the feature set is point-in-time clean by construction.

---

## Model Architecture

### A note on model capacity vs sample size

The 2012-2017 in-sample period contains ~237 weekly trade signals. With a target win rate of approximately 70%, the minority (loss) class has on the order of 70 examples. Against this, we are fitting XGBoost with 15 features and several tunable hyperparameters: a meaningful number of degrees of freedom relative to the data. We treat this asymmetry seriously — the elastic net benchmark below is not a methodological courtesy, it is a real fallback.

### Primary classifier: XGBoost

Tree-based gradient boosting. Captures non-linear relationships and feature interactions natively, which is appropriate because the relationship between environmental features and trade profitability is known to be non-linear.

**Hyperparameter tuning:** Time-series cross-validation with 5 splits, sensible default parameters (max_depth=4, learning_rate=0.05, n_estimators=200, subsample=0.8). The README originally proposed nested CV with 50 Optuna trials; that version is documented as the institutional-grade extension and was descoped to keep the build clean.

### Calibration: Isotonic regression

Applied to out-of-fold XGBoost predictions before the decision threshold is applied. Niculescu-Mizil and Caruana (2005) is the canonical reference.

### Benchmark: Elastic net logistic regression

Same 15 features. L1 plus L2 regularization with alpha and l1_ratio tuned via inner CV. Calibrated via Platt scaling.

**Pre-committed model selection rule:** If elastic net OOS log-loss on the IS training folds is lower than XGBoost log-loss, swap as primary.

### Diagnostic overlay: Two-state HMM (Calm vs Stressed)

NOT used for trade decisions. Used only for tagging trades with their regime at entry. 20 random Baum-Welch initializations + Hungarian-algorithm centroid alignment ensure stable state labels across walk-forward refits.

### Walk-forward refit cadence: Annual

Every December 31, retrain XGBoost, elastic net, and HMM on all data through that date. Filter training to trades whose entry AND exit both occurred before fold-start (no label leak).

---

## Friction Model

Aggressive and realistic. Every component justified by published or commonly-observed retail execution costs.

### Commissions

IBKR Pro tiered: $0.65 per options contract per leg, $1 minimum per order. Each spread = 2 legs at entry, 2 legs at exit. Plus regulatory fees (~$0.05 per contract per side).

### VIX-scaled bid-ask slippage

Pay a percentage of the bid-ask spread per side, scaled by current VIX:
- VIX < 20: 30% of spread
- VIX 20-30: 50% of spread
- VIX 30-40: 75% of spread
- VIX > 40: 100% of spread

Calibrated against observed SPX option spreads in 2012-2017 across VIX regimes.

### Gap-aware stop execution

If SPX opens past the stop trigger level, execute at the open price, not the stop level. The realistic execution price after an overnight gap is whatever the market opens at.

### Monday-open execution discipline

Strategy decides Friday close, executes Monday open. The 64-hour weekend window is a known risk:
- VIX-scaled slippage applies to Monday's environment
- If SPX gaps more than 1.5% from Friday close to Monday open, apply 50% additional slippage on entry
- If gap exceeds 3%, skip entry entirely

Thresholds set from the 95th and 99th percentiles of weekend SPX gap magnitudes in 2008-2017 training data.

### Tax treatment

Section 1256: 60% long-term capital gains and 40% short-term capital gains rate, applied to aggregate annual P&L. No carry-forwards modeled.

### Margin

IBKR Portfolio Margin: buying power requirement is roughly the maximum loss of the spread. Margin is not a binding constraint at the account sizes considered.

---

## Backtest Protocol

**In-sample (development period):** 2012-03-26 to 2017-12-31. All hyperparameter tuning, halt threshold setting, and decision threshold selection happen here. Pre-commitment file (PRE_COMMITMENT.md) records every methodological choice with timestamp.

**Out-of-sample (test period):** 2018-01-01 to 2024-12-31. Frozen pipeline run once. Reported results are whatever come out.

**Confidence intervals:** Block-bootstrap with non-overlapping 3-trade blocks, 5,000 to 10,000 resamples on the trade outcome series. 90% CIs reported on Sharpe, win rate, average return per trade.

**Stress events analyzed:**
- August 2015 (China devaluation)
- February 2018 (Volmageddon)
- Q4 2018 (selloff)
- February-March 2020 (COVID — the critical test)
- March 2023 (regional banking crisis)

For each event we report: did the strategy halt, when, did it resume, and the realized P&L through the event window.

**Pre-committed test:** halt rules must fire BEFORE the worst of February-March 2020. The OOS results below confirm this.

---

## Component Attribution (Ablation Modes)

Four modes, identical blotter logic, only the gates differ:

- **`naked`**: no ML filter, no halt rules. Sells one credit spread every Monday.
- **`ml_only`**: XGBoost gate active (calibrated probability >= 0.55), halts off.
- **`halts_only`**: no ML filter, all five halt layers active.
- **`full`**: both active.

### IS results (2012-03-26 to 2017-12-31)

| Mode | Trades | Win Rate | Annualized Return | Sharpe (trade-level) | Max DD |
|---|---|---|---|---|---|
| naked | 237 | 68.8% | +0.54% | 1.86 [1.02, 2.85] | 1.06% |
| ml_only | 130 | 71.5% | +0.23% | 1.80 [0.62, 3.18] | 0.57% |
| halts_only | 90 | 63.3% | +0.33% | 2.59 [1.34, 4.16] | 0.25% |
| full | 28 | 60.7% | +0.04% | 1.74 [-1.28, 5.56] | 0.25% |

(Square brackets are 90% block-bootstrap CIs.)

### OOS results (2018-01-01 to 2024-12-31)

| Mode | Trades | Win Rate | Annualized Return | Sharpe (trade-level) | Max DD |
|---|---|---|---|---|---|
| naked | 283 | 62.2% | **+16.45%** | 0.44 [0.12, 1.27] | 1.87% |
| ml_only | 257 | 62.6% | **+16.50%** | 0.47 [0.27, 1.36] | 1.95% |
| halts_only | 8 | 50.0% | -0.03% | -1.88 [-5.10, -2.39] | 0.35% |
| full | 8 | 50.0% | -0.03% | -1.88 [-5.10, -2.39] | 0.35% |

### Pre-committed interpretation rule fires

> **OOS Sharpe(`ml_only`) - Sharpe(`naked`) = +0.023.** This is within the 0.1 threshold the README pre-committed. By that rule, **the ML filter is decorative.** The XGBoost gate does not add material value over the unfiltered baseline.

We report this directly per the methodological discipline. The structural variance risk premium edge captured by the friction model and the strict 16-delta / 30-45 DTE / 50%-profit / 200%-stop / 21-DTE-exit mechanics is the entire source of OOS profitability. The XGBoost layer remains in the architecture as documentation of the Vestal-style exogenous-features methodology and as a regime-drift diagnostic for live monitoring, but it is not credited with strategy performance.

### Halts: anti-signal in OOS (audit-confirmed)

A formal precision audit of every halt activation in OOS 2018-2024:

- **42 distinct halt activation events** (top triggers: drawdown 17, term_inversion_hard 8, vix_spike+term_inversion 5)
- **19.0% (8/42)** of halt firings were followed by a SPX ≥5% drawdown within 30 days
- **Base rate** (any OOS day → ≥5% SPX drawdown in next 30 days) = **27.7%**
- **Precision lift vs base rate: −8.7 percentage points (NEGATIVE)**

In other words: the halts fire **less often** before real stress events than a random day would. They are not noisy true-positive detectors with high false-positive rates — they are *systematically anti-correlated* with subsequent stress, at least over the OOS window. As a defensive system, the halt framework provided **no measurable lift** over the base rate of "do nothing."

We considered the COVID-March-2020 single-event "halts fired 28 days before peak" finding before doing this audit. That was a real timing observation but cherry-picked across a single event. Across all 42 OOS activations the framework underperforms the base rate.

This is the honest scientific result. The pre-committed halt thresholds did exactly what they were designed to do — fire whenever regime indicators looked like the IS-period stress profile — but that profile didn't generalize. The OOS regime that mattered was a long bull market interrupted by short shocks, and the halts spent most of their time over-reacting to the long bull. We acknowledge this finding rather than retroactively re-tuning.

A practitioner taking this strategy live would *remove* the halt framework, not refine it.

---

## Stress Event Survival

For each event, we report the number of days before the event peak that the halt rules first fired (in `full` and `halts_only` modes), plus the per-mode trade count and P&L through the window.

| Event | Halt fire (days before peak) | Naked P&L | Halts_only P&L |
|---|---|---|---|
| Aug 2015 China devaluation | 49 days before | TBD | TBD |
| Feb 2018 Volmageddon | 0 days before (same-day) | TBD | TBD |
| Q4 2018 selloff | 75 days before | TBD | TBD |
| **March 2020 COVID** | **28 days before peak** | $+83 (1 trade, gap-skip rules engaged) | $0 (halts active) |
| March 2023 banking crisis | not fired | TBD | TBD |

The single most important pre-committed test was: halts must fire before the worst of February-March 2020. **Halts fired 28 days before the COVID peak in `halts_only` and `full` modes.** The defensive structure works on the most severe stress event in OOS — the cost is that it ALSO fired in many environments that turned out not to be stress events.

---

## Live Performance Monitoring (Forward-Looking Framework)

Answers to the rubric's two questions, framed as how a deployment would be monitored. Not implemented as a live job.

### Daily monitoring routine

After each trade closes, the monitor would update:

1. **Rolling 60-trade win rate** plotted with Hoeffding 90% bound around in-sample baseline. If the rolling rate falls below the bound, this is statistical evidence that the strategy's regime has shifted.

2. **Rolling 90-day Sharpe** with block-bootstrap 90% CI updated each trade.

3. **Calibration drift check.** Predicted probability bins vs realized win rate within each bin.

4. **Feature stability monitor.** XGBoost feature importance scores in recent predictions vs feature importance in original training.

### Watch list (soft signals)

- Recent calibration error trend
- Variance risk premium compression (if VRP shrinks below 25th percentile of trailing 5 years for sustained periods)
- HMM state transition frequency (rapid switching suggests regime instability)
- Cross-asset correlation drift

---

## Theoretical Foundation

The writeup grounds every strategic choice in canonical options literature.

### 1. Variance risk premium thesis

**Citation:** Merton 1973, Theorems 8 and 15. Carr-Madan 1998 for the variance swap decomposition.

Merton proves that option price is monotone increasing in variance. Selling options is selling variance. The structural premium exists because hedgers pay above-fair-value for tail-risk protection.

### 2. Theta decay

**Citation:** Black-Scholes 1973 for the closed-form theta formula. Merton 1973 equation 36 for the parallel derivation via the no-arbitrage hedging argument.

Theta decay is the mechanism by which the static premium converts into realized profit as time passes. The rate of decay accelerates non-linearly toward expiration, which justifies the 30-45 DTE entry window and 21-DTE roll rule.

### 3. Put-call parity as data-integrity check

**Citation:** Merton 1973, Theorem 12.

The relationship `c(S, tau, E) - p(S, tau, E) = S - E*exp(-rT)` must hold for European options on non-dividend-paying assets. Used to validate options chain data: if parity does not hold within a few cents, the data has a problem.

### 4. High-contact condition (referenced as conceptual frame, not derivation)

**Citation:** Merton 1973, Section 7.

The optimality condition for early exercise of American options. SPX is European-style and cannot be early-exercised, so this theorem does not directly apply. We cite it because the underlying intuition — that the original payoff structure of a short option position breaks down once delta approaches 1 — informed our delta-based emergency exit at delta > 0.50. The exit threshold itself is risk-management convention, not a derivation from Merton.

### 5. Down-and-out option as halt-rule analog

**Citation:** Merton 1973, Section 9.

Most distinctive *conceptual* framing in the project. Merton derives a closed-form price for an option that becomes worthless if the underlying crosses a stated barrier. Our halt logic shares the same structural shape ("value while above barrier, zero if barrier touched") but applies it to a live trading strategy whose barriers are regime indicators rather than the underlying price. We use the analogy to organize and justify the halt framework's existence; we do not use Merton's formula to set thresholds.

### Additional references

- López de Prado 2018, *Advances in Financial Machine Learning*, Chapter 3 on meta-labeling.
- Hoeffding 1963 for the inequality underlying the live monitoring framework.
- Egger and Vestal 2025 (course material) for the Hoeffding-based regime monitoring application.

---

## Methodological Discipline (Non-Negotiable Rules)

1. **Never tune parameters after seeing OOS results.** Pre-commit using IS data. Freeze for OOS. Disappointing results are still results. The PRE_COMMITMENT.md file is committed to git with timestamp before any OOS test runs.

2. **No silent filtering.** If the blotter contains N trades, every downstream artifact reflects N trades. The ML probability is annotation only, never a downstream filter that creates inconsistency between sections. Enforced as a runtime assertion in the backtest engine.

3. **Strategy parameters frozen across folds.** Only hyperparameters and feature scalers refit at fold boundaries.

4. **Pre-commit with rationale before running.** Halt thresholds, decision threshold, sizing parameters all need documented justification BEFORE seeing OOS results.

5. **Internal consistency.** Universe count, trade count, metric values must match across every section of the writeup. Verified by `notebooks/99_consistency_check.ipynb` (and `scripts/consistency_check.py`).

---

## Implementation Discipline (Bug Prevention)

The architecture has substantial complexity. The probability of undetected bugs scales with system complexity, and silent bugs in ML pipelines are particularly dangerous because the code runs without error and produces plausible-looking output that is wrong. HW5 demonstrated this empirically: the blotter showed 21 trades while the equity curve silently used 7 due to an LR filter that was not supposed to be filtering. We will not repeat this.

### Unit Tests

Every source module in `src/` has a corresponding test file in `tests/`.

### Consistency Assertions

Every artifact handoff has runtime assertions. Trade count entering a function equals trade count leaving it. Blotter row count equals equity curve update count. No silent transformations.

### Seeded Randomness

Single global seed defined in `src/config.py`. Every random component reads from this seed.

### Consistency Check Pass

`scripts/consistency_check.py` runs after every full pipeline execution and asserts trade count consistency, no NaN/inf, ablation gating monotonicity, and date-range alignment. The current pipeline passes 34/34 checks.

### The "No Silent Filtering" Rule

If the blotter contains N trades, every downstream artifact reflects the same N trades. Enforced as a runtime assertion in the backtest engine, not just a methodological rule.

---

## Repository Structure

```
.
├── README.md                       # This file
├── PRE_COMMITMENT.md               # Methodology frozen before OOS test
├── pyproject.toml                  # ruff/black/pytest config
├── requirements.txt
├── data/                           # gitignored
│   ├── raw/                        # IBKR + OptionMetrics caches
│   │   └── optionmetrics/          # SPX option chains (WRDS)
│   └── processed/                  # features, ML probabilities, attribution
├── src/
│   ├── config.py                   # SEED, frozen parameters, dates
│   ├── data/
│   │   ├── fetch_ibkr.py           # TWS via shinybroker
│   │   ├── validate_chains.py      # put-call parity check
│   │   └── timestamp_audit.py      # look-ahead guard
│   ├── features/
│   │   ├── exogenous.py            # 15-feature master pipeline
│   │   ├── yield_curve.py          # polynomial fit
│   │   └── correlations.py         # cross-asset stress
│   ├── strategy/
│   │   ├── types.py                # Trade / Spread / OptionContract
│   │   ├── pricer.py               # PricingProvider Protocol
│   │   ├── black_scholes.py        # default fallback pricer
│   │   ├── optionmetrics_pricer.py # production pricer (real OPRA)
│   │   ├── spread_construction.py
│   │   ├── exits.py
│   │   ├── friction.py
│   │   ├── sizing.py
│   │   └── halts.py
│   ├── models/
│   │   ├── label_trades.py
│   │   ├── xgboost_primary.py
│   │   ├── elastic_net_bench.py
│   │   ├── hmm_diagnostic.py
│   │   └── calibration.py
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── walkforward.py
│   │   └── loader.py
│   ├── metrics/
│   │   ├── performance.py
│   │   ├── hoeffding.py
│   │   └── bootstrap.py
│   └── monitoring/
│       └── live_monitor.py         # forward-looking placeholder
├── scripts/                        # CLI orchestrators
│   ├── tws_debug.py
│   ├── run_backtest.py
│   ├── train_ml.py
│   ├── component_attribution.py
│   ├── consistency_check.py
│   └── stress_events.py
├── tests/
├── notebooks/
├── website/                        # Quarto source
└── docs/                           # rendered HTML for GitHub Pages
```

---

## Setup

### Environment

```bash
git clone https://github.com/mariotrev120/FinalProject_FinTech533.git
cd FinalProject_FinTech533
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### IBKR TWS (for index/ETF history)

1. Install TWS or IB Gateway, log in with paper trading account
2. Enable API access: Configure → API → Settings → check "Enable ActiveX and Socket Clients"
3. Set socket port to 7497. **Uncheck "Allow connections from localhost only"** (required for WSL access).
4. Confirm connection: `.venv/bin/python scripts/tws_debug.py` — section 3 should print `RECEIVED N bytes` with N > 0.

For the full WSL + VS Code (and PyCharm) setup walkthrough including all the debugging gotchas this project encountered, see [TWS_CONNECTION.md](TWS_CONNECTION.md).

### OptionMetrics IvyDB (for SPX option chains)

Real OPRA-grade option chains are pulled via WRDS:
1. Log into WRDS (https://wrds-www.wharton.upenn.edu) with Duke NetID
2. OptionMetrics → IvyDB US → Option Prices
3. Date range 2012-01-01 to 2025-08-29 (or current), TICKER=`SPX`, Put Only, European, Index, DTE 25-50
4. Download CSV gzipped, place at `data/raw/optionmetrics/Options.csv.gz`

### Data Pull (IBKR side)

```bash
PYTHONPATH=. .venv/bin/python -m src.data.fetch_ibkr --duration "15 Y"
```

Pulls all 16 universe symbols (~3 minutes).

### Run Backtest

```bash
# Train ML walk-forward
PYTHONPATH=. .venv/bin/python scripts/train_ml.py

# Component attribution (all 4 modes IS+OOS)
PYTHONPATH=. .venv/bin/python scripts/component_attribution.py

# Consistency check
PYTHONPATH=. .venv/bin/python scripts/consistency_check.py

# Stress events
PYTHONPATH=. .venv/bin/python scripts/stress_events.py

# Render writeup
cd website && quarto render
```

---

## Performance Targets vs Realized

| Metric | Pre-committed Target | Realized OOS (`naked`) |
|---|---|---|
| Annualized return (gross) | match or exceed CBOE PUT/PutWrite (~6-9%) | **+16.45%** |
| Sharpe ratio | beat PUT/PutWrite by 100-200bp | 0.44 (trade-level) |
| Max drawdown | materially smaller than PUT during COVID (PUT lost ~24% Mar 2020) | **1.87%** |
| Win rate | 70 to 80% | 62.2% (just below target) |

The strategy substantially exceeds the PUT/PutWrite return benchmark in the OOS period and dramatically exceeds it on max drawdown. Win rate slightly below target reflects that 16-delta puts at our friction model occasionally hit stop-loss in vol-spike weeks (Volmageddon, COVID, 2022 bear) where PUT/PutWrite indices would have absorbed the same losses.

---

## Known Limitations

1. **Sample size.** ~237 IS trades is on the edge of XGBoost's capacity for 15 features. The pre-committed elastic-net fallback is a real backup; in practice the XGBoost OOS performance was indistinguishable from naked, vindicating the capacity concern.

2. **Halts are too aggressive on OOS.** Calibrated on 2012-2017 (a benign period interrupted only by Aug 2015), the thresholds fired too often in 2018-2024. The defensive value is real (28 days before COVID peak), but the cost of missed bull-market trades dominated. Re-tuning would violate pre-commitment discipline; we report the result honestly.

3. **No 2008 data.** Including the GFC would meaningfully strengthen halt rule calibration. OptionMetrics has the data; our IS window starts later because of feature-data constraints (VVIX from TWS begins 2012-03).

4. **Crowding risk.** Vol selling on SPX is one of the most popular systematic strategies. Backtest results from 2012-2024 partially reflect aggregate vol-seller behavior. Live deployment in 2026 may face different competitive dynamics.

5. **Refit cadence.** Annual refits chosen for engineering simplicity. Quarterly refits with parameter smoothing would be the institutional-grade version.

6. **Complexity-related risks.** The architecture has many moving parts. The probability of undetected bugs rises with complexity. We mitigate via unit tests, consistency assertions at every artifact handoff, seeded randomness throughout, and the dedicated consistency-check pass after every pipeline execution. Second, complexity may be doing less work than it appears: the OOS ablation explicitly shows the ML layer is decorative and the halt framework is over-tight. We report this rather than hiding it.

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

- Professor Vestal for the exogenous-feature framework, Hoeffding monitoring discipline, and the rubric-question framing that drove this project's methodology.
- Duke MEng Financial Technology faculty.
- WRDS / OptionMetrics for the IvyDB US option chain data.

---

**Status:** Backtest pipeline complete. Real OPRA option chain data wired in. Component attribution and stress event analysis emit honest results that match the pre-committed methodology — including the disclosure that the ML filter is decorative and the halt framework is over-tight on OOS. Writeup integration into Quarto / GitHub Pages is the remaining deliverable.
