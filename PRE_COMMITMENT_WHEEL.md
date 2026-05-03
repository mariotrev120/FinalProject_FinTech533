# PRE_COMMITMENT_WHEEL.md  *(DRAFT — pending review)*

**Strategy:** Adaptive 5-layer wheel on a 9-name basket of liquid US equities.

**Status:** DRAFT. Once user signs and the corresponding git commit lands, this file is frozen. Any subsequent edit is a methodology breach and is logged as such in the writeup.

---

## 1. Basket selection (locked)

**Primary basket — 9 names (activated):**
AAPL, MSFT, GOOGL, JNJ, KO, PG, WMT, JPM, PEP

**Robustness basket A — 7-name strict (drops JPM and PEP):**
AAPL, MSFT, GOOGL, JNJ, KO, PG, WMT

**Robustness basket B — 8-name (adds JPM only):**
AAPL, MSFT, GOOGL, JNJ, KO, PG, WMT, JPM

The headline result runs on the 9-name primary. Both robustness baskets are reported alongside the headline so the basket-selection sensitivity is visible.

### Selection criteria (applied to the strict 7)

| Criterion | Threshold | Rationale |
|---|---|---|
| Market cap | ≥ $100B as of IS_END | Liquidity and option chain depth |
| Option chain depth | ≥ 6 weekly expirations, weekly volume ≥ 5k contracts | Sufficient liquidity for 30-45 DTE trades |
| Max drawdown 2012-2017 | ≤ 50% | Avoid names with structural single-event collapse risk |
| Earnings cadence | Quarterly, predictable | Required for Layer 4 earnings-aware management |
| Sector | Diversified across at least 5 GICS sectors | Reduce within-basket correlation in stress |

**Why JPM is in robustness, not strict:** JPM fails the 50% max-drawdown criterion *only* due to GFC sector-wide stress. Activating it tests sensitivity to the strict criterion's potential over-rejection of cyclicals. Disclosed: GFC drawdown not a name-specific failure.

**Why PEP is activated despite sector overlap with KO:** PEP enters the activated basket for the **within-sector correlation probe**. The KO/PEP pair is methodologically deliberate — same sector (Consumer Staples), different brands. Reporting per-name attribution lets us measure whether within-sector pairs add diversification or just average-out their idiosyncrasies. This is a probe, not free diversification. The writeup discloses the within-sector overlap explicitly.

## 2. Five-layer wheel architecture

### Layer 1 — regime-conditional delta selection

VIX percentile (trailing 252-day) drives the cash-secured put short delta:

| VIX percentile bucket | Short put delta target |
|---|---|
| Bottom quartile (calm) | 0.15 |
| Middle quartiles | 0.20 |
| Top quartile (elevated) | 0.25 |

**Direction (calm → elevated = lower → higher delta):** Higher absolute delta = strike closer to spot = larger premium captured per CSP, AND higher assignment probability. The table moves UP in delta as VIX percentile rises.

**Rationale:** In elevated VIX the IV–RV wedge (the VRP itself) is widest in absolute terms, so harvesting is most lucrative on a per-trade basis; the trade-off is paid in higher assignment risk, which on a wheel is partially-acceptable because assignment converts the position into stock + covered calls (the wheel's other leg). In calm VIX the absolute premium is thin, so going further OTM (lower delta) keeps assignment frequency low and avoids unnecessary basis risk on shares.

**Code-direction note:** Wheel code is not yet implemented (Phase B task B4). The acceptance criterion at implementation time is that the table direction in `src/strategy/wheel/regime_delta.py` (when written) matches the rationale here — i.e., monotonically increasing delta target as VIX percentile rises. If the implemented direction is opposite, the rationale text wins and the implementation is corrected to match.

### Layer 2 — per-name vol-adjusted strikes

Per-name 252-day realized volatility (RV) modifies Layer 1 delta:

| Per-name RV bucket | Delta adjustment |
|---|---|
| RV > 35% | Delta target +0.05 (widen — more conservative) |
| RV ∈ [20%, 35%] | Layer 1 delta unchanged |
| RV < 20% | Layer 1 delta unchanged |

**Why widen, not narrow on RV<20%:** A high-RV name has fatter realized tails than its option chain may price; widening the strike gives more cushion. A low-RV name doesn't get tighter strikes because that would inflate assignment-rate-per-trade vs. the per-name liquidity of the underlying.

### Layer 3 — per-name 30-day RV forecaster

XGBoost regression on:
- HAR-RV components (1d, 5d, 22d realized variance)
- Exogenous: VIX/VIX3M term, sector-ETF RV, market beta-shock, earnings-window indicator

**Output:** point forecast `rv_hat_30d` per name per session.

**Sizing modifier:** `size_multiplier = clip(target_rv / rv_hat_30d, 0.4, 1.5)` where `target_rv = name's trailing 252-day mean RV`.

**Skip rule:** if `rv_hat_30d > 95th percentile of trailing 252-day rv_hat_30d distribution for that name`, skip the trade entirely (regime too uncertain).

**Pre-committed acceptance (per-name, individual comparison — NOT aggregate):** For each of the 9 names independently, the per-name XGBoost RV forecaster's OOS RMSE must be strictly lower than that name's own pure-HAR baseline RMSE on the same OOS sample. The accept/reject decision is made *per name*, not on a pooled or averaged metric.

- **Per-name pass:** XGBoost RMSE < HAR RMSE on that name → use XGBoost forecaster + size_multiplier + skip rule for that name.
- **Per-name fail:** XGBoost RMSE ≥ HAR RMSE on that name → fall back to `size_multiplier = 1.0` and drop the skip rule for that name.
- **Headline rule:** If at least 7 of 9 names pass, the basket-level result reported as "Layer 3 active." If fewer than 7 pass, the basket-level result is reported as "Layer 3 inconclusive — failing names use static fallback."

The aggregate count (≥ 7 of 9) is for narrative classification of the result, not for an aggregate-metric pass/fail. Each name's fallback decision is independent.

#### §2.1 Layer 3 methodological gap — daily-RV vs 5-minute intraday RV

**Source:** Bollerslev/Tauchen/Zhou 2009 Section 3.2.1 ("Old" variance measures).

The HAR-RV forecaster's input components (1d, 5d, 22d realized variance) are computed from **daily** close-to-close returns of each underlying. BTZ §3.2.1 explicitly cautions that this is a weaker realization measure than 5-minute intraday-summed squared returns:

> "Estimation of the same predictive regressions based on the traditional Black–Scholes implied variances and/or realized variances constructed from lower frequency daily data does not give rise to nearly as significant results."

BTZ's preferred specification uses 78 within-day 5-minute squared returns (9:30am–4:00pm) plus close-to-open overnight return, totaling n = 22 × 78 = 1716 5-minute returns per typical trading month. Their R² gain from intraday vs daily is meaningful: the high-frequency-RV variant produces stronger empirical predictability than the daily-RV variant in their Tables.

**Disclosure:** Layer 3's HAR-RV forecaster may underperform what the literature could achieve with intraday data. Acquiring 5-minute intraday data for the 9-name basket over 2012–2024 is out of scope for the capstone (would require a separate data pipeline and ~9 × 12 years × 252 days × ~78 bars = ~2M rows per name). We document this gap honestly per BTZ's own caveat. The acceptance criterion (Layer 3 forecaster must beat HAR baseline OOS RMSE per name) remains; we expect lower absolute RMSE-improvement magnitudes than BTZ's intraday baseline.

**Impact on the writeup:** Limitations section explicitly cites BTZ §3.2.1 and states that Layer 3 results are upper-bounded by the daily-frequency input granularity. A future-work item is to re-run Layer 3 with intraday data when accessible.

### Layer 4 — earnings-aware management

| Window | Behavior |
|---|---|
| 7-day blackout pre-earnings | No new wheel entries |
| Holding shares pre-earnings | Roll the covered call to the nearest higher strike (narrow CC) to reduce upside cap |
| 1-day post-earnings | Opportunistic re-entry if IV crush has produced rich premium |

**Why blackout, not lean-into:** Earnings is a single-event vol regime where realized often exceeds implied due to news-driven gap, not vol-mean-reversion. The VRP edge is structurally weaker in the immediate event window.

### Layer 5 — fundamental-only defensive rolls

If holding shares assigned from a put leg AND any of the following entry premises is broken, defensive-roll out:
- Quarterly EPS guidance cut > 20%
- Dividend cut or suspension
- Rating agency downgrade > 2 notches in 12 months
- Sector ETF underperformance > 25% vs. SPY trailing 60d AND name's beta-adjusted residual return < −10%

**Why fundamental, not generic delta-threshold:** Generic delta-rolls produce the well-documented "wheel turning into stock-trading" failure mode where shares are sold below cost during temporary drawdowns. Fundamental rolls trigger only when the *original entry premise* is broken — i.e., the volatility-risk-premium edge is no longer applicable to this name.

## 3. Capital allocation across the basket

- **Equal initial weight:** each name gets 1/N of the wheel's allocated capital at portfolio inception.
- **Layer 3 size_multiplier** modulates per-name allocation per session.
- **Per-trade hard cap:** 5% of basket equity per single put/call (CSP and CC both).
- **Within-sector pair handling:** KO and PEP share the Consumer Staples allocation envelope (combined cap 10% of basket equity to avoid double-loading).

## 4. Friction model (same as VRP unless noted)

- Slippage: 30 / 50 / 75 / 100 % of bid-ask half-spread for VIX buckets <20 / 20–30 / 30–40 / >40
- Commissions: $0.65 per option contract
- Equity assignment commissions: $0.005 per share
- Tax overlay: standard ST/LT (no 1256). Held shares get standard ST/LT split based on holding period from assignment date.

## 5. Sample windows

- **IS:** 2012-03-26 → 2017-12-31
- **OOS:** 2018-01-01 → 2024-12-31
- **Walk-forward refit cadence:** Layer 3 RV forecaster refit annually; Layer 1/2 thresholds locked from IS calibration.

## 6. Reproducibility

- `SEED = 42` (`src/config.py`)
- Reproducibility test: same commit + data + seed → identical equity curve to FP precision. Required to pass before each gate.

## 7. Sensitivity grid for DSR / PBO

| Dimension | Grid values | Count |
|---|---|---|
| Layer 1 calm-bucket delta | 0.10, 0.15, 0.20 | 3 |
| Layer 1 mid-bucket delta | 0.15, 0.20, 0.25 | 3 |
| Layer 1 elevated-bucket delta | 0.20, 0.25, 0.30 | 3 |
| Layer 2 RV-widen threshold | 30%, 35%, 40% | 3 |
| Layer 3 skip percentile | 90th, 95th, 99th | 3 |
| Layer 4 blackout window | 5d, 7d, 10d | 3 |

**Total trials N = 3⁶ = 729.** Same DSR / PBO machinery as VRP; N̂ via Eq. 9 of Bailey & López de Prado 2014.

## 8. Acceptance gates (pre-committed, in order)

1. 9-name primary basket OOS Sharpe ≥ 7-name strict basket OOS Sharpe within 0.05 — *required to keep the activated basket as primary*. If gap > 0.05 in favor of the strict basket, the strict basket becomes headline and activation is reported as net-negative.
2. Each ML/forecaster head meets §2 acceptance criterion or falls back per stated rule.
3. DSR (PSR threshold) ≥ 0.95 against the N̂-trial benchmark.
4. PBO via CSCV ≤ 0.30.
5. Reproducibility test passes.
6. Within-sector probe (KO/PEP pair correlation) is reported regardless of pass/fail — it is a methodological probe, not a gate.

## 9. Sign-off

- **Authored:** Mario Trevino
- **Reviewed:** [pending user signature]
- **Timestamp:** [git commit datetime — populated at commit]
- **Git commit hash:** [populated at commit]
- **IS-data commit hash:** [populated at commit]
- **Basket locked:** AAPL, MSFT, GOOGL, JNJ, KO, PG, WMT, JPM, PEP. No additions.
