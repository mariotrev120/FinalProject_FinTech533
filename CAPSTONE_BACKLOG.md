# CAPSTONE_BACKLOG.md

Single source of truth for FinTech 533 capstone v2 work. Updated as tasks complete.

**Branch:** `mario/data-pipeline` (no merge to main until final).
**Anchor:** v1.5 baseline `halts_only` OOS Sharpe = 0.286 (`artifacts/v1_5_baseline/component_attribution.csv`).
**Universe:** Locked. 5 VRP instruments + 9 wheel names. PEP activated. No additions.

---

## Status legend

- `[ ]` pending
- `[~]` in progress
- `[x]` complete
- `[!]` blocked / waiting on input

---

## PHASE A — Pre-flight verification (gate to backtest runs)

| ID | Task | Est | Dep | Acceptance | Status |
|---|---|---|---|---|---|
| A1 | Iron condor smoke test on SPX OOS | 1h | — | Both wings produce trades, ≥ 1 IC pair has different fates between sides, all 5 fates appear, eos_force fires at OOS end, IC count > 200 | `[~]` |
| A2 | Re-wire Layer 5 auto-resume into `evaluate_halts` | 2h | — | When `state` enters halt and `auto_resume_ready()` returns True next session, state transitions to `active`. Unit test confirms across all 4 conditions. | `[ ]` |
| A3 | Pre-commit doc sign-off (VRP + WHEEL) | user | — | User reads both DRAFTs, edits or signs, replaces `[pending user signature]` placeholders | `[!]` |
| A4 | Reproducibility test (seed=42, 2 runs identical) | 30m | A1, A2 | Two runs of same backtest produce identical equity curves to FP precision | `[ ]` |

## PHASE B — Multi-instrument engine plumbing

| ID | Task | Est | Dep | Acceptance | Status |
|---|---|---|---|---|---|
| B1 | Multi-instrument run driver (5 VRP) | 4h | A4 | `scripts/run_multi_instrument.py` loops over 5 tickers using `make_pricers_for_universe()`. Per-instrument equity curves + aggregate book-level NAV with capital allocation. | `[ ]` |
| B2 | Capital-allocation engine (quality-weighted) | 3h | B1 | Per-session re-weight w_i = p_quality_i / Σ p_quality_j; aggregate book scaler = (1 − p_stress). Falls back to equal-weight when ML heads not present. | `[ ]` |
| B3 | Per-instrument PortfolioReport rollup | 2h | B1 | Each instrument and the aggregate book emit a full `PortfolioReport` with DSR, bootstrap CI, Hoeffding bound | `[ ]` |
| B4 | Multi-instrument run driver (9 wheel names) | 4h | B1 | Same loop framework; different entry mechanic (CSP/CC instead of IC) | `[ ]` |

## PHASE C — Statistical infrastructure (gates v2 reporting)

| ID | Task | Est | Dep | Acceptance | Status |
|---|---|---|---|---|---|
| C1 | DSR sensitivity-grid runner | 3h | B1 | `scripts/run_sensitivity_grid.py` runs N=729 grid for VRP, records per-trial daily returns matrix M (T × N). Outputs `dsr_n_trials`, `dsr_sharpe_var_per_period`, average pairwise correlation ρ̄, and N̂ via Eq. 9 of Bailey/López de Prado 2014. | `[ ]` |
| C2 | Wire DSR into headline reports | 1h | C1, B3 | Every `PortfolioReport` for headline modes is built with `dsr_n_trials=N̂` and `dsr_sharpe_var_per_period=V̂` from C1 | `[ ]` |
| C3 | CSCV-based PBO module (`src/metrics/pbo.py`) | 6h | C1 | Implements Bailey/Borwein/López de Prado/Zhu 2015 Algorithm 2.3 with S=16. Outputs PBO, performance-degradation slope, prob-of-loss, stochastic-dominance check. Reproducible (no random component). | `[ ]` |
| C4 | Wire PBO into headline reports | 1h | C3 | PBO computed once on the §11 sensitivity grid (VRP) and §7 grid (wheel); reported in writeup | `[ ]` |
| C5 | Hoeffding live-monitoring runner per Egger/Vestal trader-application | 3h | B3 | `scripts/hoeffding_monitor.py` reads pre-committed μ from PRE_COMMITMENT_VRP/WHEEL and emits 50%/25%/10% threshold status across rolling 60-trade window. Output: dated alert log. | `[ ]` |
| C6 | Bootstrap CI on Sharpe + ann return for headline | 2h | B3 | Wraps existing `src/metrics/bootstrap.py` into the report rollup. Block-bootstrap, B=10000, block size by Newey-West heuristic. | `[ ]` |
| C7 | Hodrick (1992) SE estimator for overlapping forecasts | 1h | — | `src/metrics/hodrick_se.py` implements Hodrick (1992) overlapping-regression standard errors per BTZ 2009 footnote 21 / Ang-Bekaert 2007. Used in Phase F results-page reporting for Head 2 evaluation predictive significance. ~50 LOC. | `[ ]` |

## PHASE D — ML training (gated by Phase C statistical infrastructure)

| ID | Task | Est | Dep | Acceptance | Status |
|---|---|---|---|---|---|
| D1 | Vol-regime quadrant classifier | 4h | A4 | Per-instrument z-unit standardization on (level, derivative) of VIX/RVX/VXN/MOVE/GVZ. 4-quadrant categorical output. Dual role: standalone gate ablation AND categorical Head 1 feature. | `[ ]` |
| D2 | Head 1 trade-quality (per-instrument, 5 models) | 8h | D1, B1 | XGBoost + isotonic-OR-Platt depending on cal-set size (per Niculescu-Mizil & Caruana 2005). Continuous sizing curve: skip <0.50, scale 0.6×→1.0× over [0.50,0.70], 1.2× at ≥0.70. Pre-commit fallback rule: if log-loss ≥ majority-class on ≥ 3 of 5 instruments, drop entire Head 1. Negative finding reported either way. | `[ ]` |
| D3 | Head 2 regime-stress (shared cross-asset, isotonic) | 4h | D1 | Single calibrated model on pooled cross-asset features. Brier-score acceptance vs trailing-1mo baseline. Fallback: book scaler = 1.0. Negative finding reported. | `[ ]` |
| D4 | Head 3 skew direction (per-instrument, 5 models) | 6h | D1 | Dynamic delta selection from {0.10, 0.16, 0.25}. Per-instrument log-loss vs majority-class baseline; fall back to static 0.16/0.16 per failing instrument. | `[ ]` |
| D5 | HAR-RV forecaster for wheel Layer 3 (9 names) | 6h | B4 | XGBoost regression, OOS RMSE vs HAR baseline; per-name fallback to static size_multiplier=1.0 if it fails. | `[ ]` |

## PHASE E — Ablation framework

| ID | Task | Est | Dep | Acceptance | Status |
|---|---|---|---|---|---|
| E1 | Ablation matrix runner | 4h | D2, D3, D4 | Modes: naked / halts_only / vol-regime-only / ml_only / regime+halts / full. Each mode generates full `PortfolioReport` with DSR, bootstrap CI, Hoeffding monitoring trace. | `[ ]` |
| E2 | Paired bootstrap CI between modes | 2h | E1, C6 | Each mode pair (e.g., full vs naked) gets a paired-bootstrap CI on Sharpe difference. Same block structure as C6. | `[ ]` |
| E3 | Per-instrument attribution table | 2h | E1 | Per-mode, per-instrument Sharpe / win rate / trade count / contribution to aggregate PnL | `[ ]` |
| E4 | Per-regime attribution (HMM-tagged) | 3h | E1 | Tag each trading day with HMM regime probability; per-regime Sharpe and win rate per mode | `[ ]` |

## PHASE F — Quarto site + writeup

| ID | Task | Est | Dep | Acceptance | Status |
|---|---|---|---|---|---|
| F1 | Quarto scaffold (`_quarto.yml`, navbar, theme) | 2h | — | `cd website && quarto render` produces a static site | `[ ]` |
| F2 | Thesis page (VRP grounding) | 3h | A3 | Plain-language explanation of VRP, internalized from Carr/Wu and Bollerslev/Tauchen/Zhou. No bibliography. | `[ ]` |
| F3 | Strategy mechanics (VRP + wheel) | 4h | A3, B1, B4 | BS/Merton-grounded explanation of strikes, Greeks, exit logic. Code refs to actual file paths. | `[ ]` |
| F4 | Pre-commit pages (VRP + wheel) | 1h | A3 | Render both PRE_COMMITMENT files with the signed commit hash | `[ ]` |
| F5 | Results page | 4h | E1, C2, C4 | Headline numbers per mode with raw Sharpe + DSR + bootstrap CI; per-instrument and per-regime tables; PBO block | `[ ]` |
| F6 | **Live-monitoring framework page (HEADLINE — answers BOTH course-required questions)** | 3h | C5 | Egger/Vestal Hoeffding-style monitoring methodology (no citations) — directly answers the assignment's two required questions: (1) "how will you know strategy performance is in line with backtest?" → pre-commit μ from OOS, rolling 60-trade X̄, P[X̄−μ≥t\|H0]≤e^(−2t²N), green/yellow/red while bound ≥ 50%; (2) "how will you know when it stops working?" → same framework tripwires at 50/25/10%. Show: pre-committed μ values from headline backtest, threshold semantics 50/25/10%, **worked example** on OOS sample flagging what would have triggered in 2018/2020/2022, 60-trade rolling window choice rationale. **Don't bury — this is the single best move on the website.** Vestal co-wrote the framework; he will recognize Eq. 1.4 on sight. | `[ ]` |
| F7 | Limitations + going-forward | 2h | F5 | Multiple-testing acknowledgment with DSR/PBO numbers; what would invalidate the result; live-monitoring tripwires | `[ ]` |
| F8 | Reproducibility instructions | 1h | A4 | `git clone`, env setup, single command to reproduce headline | `[ ]` |
| F9 | Final review pass + final commit + tag | 2h | F1–F8 | All claims in writeup match code; release tag `v2-final` | `[ ]` |

## PHASE G — Stretch / nice-to-have

| ID | Task | Est | Dep | Acceptance | Status |
|---|---|---|---|---|---|
| G1 | After-tax Sharpe (1256 vs ST/LT) | 2h | E1 | Per-mode after-tax PortfolioReport using existing 1256 helpers; reported alongside pre-tax | `[ ]` |
| G2 | Strike-snap policy alternative robustness | 2h | E1 | Both strike-snap policies tested; headline locked under canonical, alternative reported as robustness | `[ ]` |

---

## Critical-path summary

**Blocks everything:** A1 (smoke test), A2 (Layer 5 wiring), A3 (pre-commit sign-off), A4 (reproducibility).
**Blocks ML phase:** D1 (vol-regime quadrant) → D2/D3/D4.
**Blocks reporting:** E1 (ablation matrix) → F5 (results page) → F9 (final).
**Truly parallel:** F1 (Quarto scaffold), C5 (Hoeffding monitor) — can run while ML trains.

**Realistic timeline (single-developer, focused work):**
- Phase A: 0.5 day
- Phase B: 1.5 days
- Phase C: 1.5 days
- Phase D: 3 days
- Phase E: 1 day
- Phase F: 2.5 days
- **Total: ~10 working days from pre-commit sign-off to v2-final tag.**
