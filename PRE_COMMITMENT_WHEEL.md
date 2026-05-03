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

**Rationale:** Higher delta in elevated VIX = higher premium captured to compensate for higher assignment risk. Lower delta in calm VIX = lower assignment probability when the volatility-risk-premium edge is thinner.

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

**Pre-committed acceptance:** Head 3 RV forecaster must beat HAR baseline OOS RMSE on ≥ 7 of 9 names. If it fails, **fall back to static `size_multiplier = 1.0` and the skip rule is dropped** for the failing names. The static-fallback equity curve is the headline result for those names, and the failure is reported.

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
