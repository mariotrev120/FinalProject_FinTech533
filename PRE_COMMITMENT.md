# PRE_COMMITMENT.md

**Purpose:** This file records every methodological choice for the strategy *before* the out-of-sample test runs. It is committed to git with a timestamp so the grader can verify that no parameter, threshold, or rule was tuned after seeing OOS results.

**Status:** TEMPLATE. To be filled out and committed BEFORE the first invocation of `walkforward.py` against the OOS window (2018-2024).

**Once committed, this file is frozen.** Subsequent edits are forbidden until the OOS run is complete and the writeup is final.

---

## 1. Decision Threshold

- **Calibrated XGBoost probability cutoff:** 0.55
- **Rationale:** [...]

## 2. Strategy Parameters (frozen across all walk-forward folds)

- **Short strike target delta:** 0.16 (1-sigma OTM)
- **Spread width:** 5 points
- **DTE at entry:** 30 to 45 days
- **Profit target:** 50% of max profit
- **Stop loss:** 200% of credit received
- **Time exit:** hard close at 21 DTE
- **Emergency exit delta:** 0.50

## 3. Position Sizing Parameters

- **Kelly fraction cap:** 0.25 (quarter Kelly)
- **Vol scaling formula:** `vol_mult = 15 / VIX_current`
- **Stress multiplier threshold:** 0.5 (SPY-TLT 20-day correlation)
- **Stress multiplier value:** 0.5x size when triggered
- **Hard cap per trade:** 1% of account equity

## 4. Halt Framework Thresholds (calibrated on 2010-2017 only)

### Layer 1 (hard halt)
- VIX intraday spike: > 40% from prior close
- SPX intraday move: > 5% in either direction
- VIX3M-VIX inversion: VIX > VIX3M by more than 2 points

### Layer 2 (soft halt)
- VIX3M < VIX for 2 consecutive closes
- HYG-LQD spread: > 2 SD from 252-day rolling mean
- Model probability < 0.55 for 5 consecutive sessions

### Layer 3 (slow halt)
- Rolling 60-trade win rate falls > 2 SE below in-sample baseline
- Rolling 90-day Sharpe significantly negative (block-bootstrap p < 0.10)

### Layer 4 (drawdown halt)
- Underwater duration: > 90 trading days
- Drawdown depth: > 15% of starting equity in trailing 90 days

### Layer 5 (auto-resume, all four required)
- VIX3M > VIX by more than 2 points for 5 consecutive closes
- Realized 5-day vol below 80th percentile of trailing 252 days
- HYG-LQD spread within 1 SD of long-run mean
- Drawdown recovered to within 3% of high-water mark

## 5. Friction Parameters

- VIX-scaled bid-ask slippage: 30/50/75/100% by VIX bucket (<20, 20-30, 30-40, >40)
- Gap-aware stop execution: realized loss = min(stop_level, opening_gap_price)
- Monday-open gap thresholds: 1.5% adds 50% slippage; 3% skips entry
- Section 1256 tax overlay: 60% LTCG / 40% STCG

## 6. Model Selection Rule (pre-committed)

If elastic net OOS log-loss on the 2010-2017 training folds is lower than XGBoost log-loss, swap as primary. Otherwise XGBoost is primary. Decision recorded here once training is complete: [TBD before OOS]

## 7. Ablation Interpretation Rule (pre-committed)

If `ml_only` Sharpe is within 0.1 of `naked` baseline Sharpe in the OOS period, the ML filter is decorative. The writeup states this explicitly. The ML layer remains in the architecture as documentation of Vestal's exogenous-features methodology and as a regime-drift diagnostic for live monitoring, but it is not credited with strategy performance.

## 8. Global Random Seed

- `SEED = 42` (defined in `src/config.py`)
- Consumed by: XGBoost `random_state`, train/test splits, block-bootstrap, HMM Baum-Welch (each of 20 starts uses `seed + i`), Optuna sampler, all sklearn `random_state` parameters.

## 9. Data Snapshot

- IBKR fetch start date: [...]
- IBKR fetch end date: [...]
- Number of trading days in IS (2010-2017): [...]
- Number of trading days in OOS (2018-2024): [...]
- Universe: XSP (or [pivot if applicable])
- Primary commit hash of fetched-and-validated data cache: [...]

## 10. Sign-off

- **Pre-commitment authored by:** Mario Trevino, Robert Lanni
- **Timestamp:** [git commit datetime]
- **Git commit hash:** [filled by post-commit hook or manual entry]
- **Confirmation:** No edits to this file after this commit. Any future change is null and noted in the writeup as a methodology breach.
