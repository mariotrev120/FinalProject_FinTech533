# R & M Trade Desk

**FinTech 533 Final Project · Duke MEng Financial Technology**
**Authors:** Robert Lanni and Mario Treviño

A defined-risk options strategy that harvests the volatility risk premium across a four-instrument cross-asset basket, with a three-layer halt framework and a calibrated XGBoost stress overlay that contracts book exposure during predicted regime breaks.

**Live site:** https://mariotrev120.github.io/FinalProject_FinTech533/

## Headline result

| Metric | Value |
|---|---:|
| Excess Sharpe (with Head 2 ML overlay) | **+0.371** |
| Excess Sharpe (halts_only base, no ML) | +0.359 |
| Anchor (v1.5 SPX put-only halts_only) | +0.286 |
| Δ vs anchor | +0.085 |
| Risk-free rate (avg IRX 2018-2024) | 2.33% |
| Annualized return | +2.41% |
| Max drawdown over 7 years OOS | -0.13% |
| Trades total (4 instruments) | 437 |
| Deflated Sharpe (PSR) ≥ 0.95 gate | 1.0000 ✓ |
| PBO via CSCV ≤ 0.30 gate | 0.0402 ✓ |

OOS window 2018-01-01 to 2024-12-31. Headline basket: AAPL, MSFT, WMT, GLD. Selected from a 12-instrument tested universe; selection bias corrected via DSR with implied-independent-trials adjustment (N̂ = 9 from average pairwise correlation 0.261) and PBO via CSCV (S = 16, 12,870 logits).

## Strategy in one paragraph

Sell weekly 16-delta put credit spreads with 5-point wing protection on each of the four instruments. Hold for 30 to 45 days. Exit on profit target (50% of credit), stop loss (200% of credit, gap-aware), time decay (DTE ≤ 21), or delta blow-out (|Δ| > 0.50). Halt entries when Layer 1 (extreme tail-event), Layer 4 (trailing 90-day drawdown), or Layer 5 (vol-regime auto-resume) conditions fire. Scale book exposure each day by `(1 - p_stress)` where `p_stress` is a calibrated XGBoost probability over 15 macro features. Risk-free cash on the unutilized portion of the book accrues at the 13-week T-bill rate (CBOE IRX).

## Two writeups in one repo

This project ships both an academic writeup and a live operations dashboard.

- **Quarto static site** under `website/_quarto.yml` and `website/*.qmd`. Nine pages covering thesis, mechanics, exogenous-factor inputs, halt framework, ML stack, ablation matrix, live-monitoring framework, limitations, and reproducibility. Renders with `cd website && quarto render` to `website/_site/`. Published on GitHub Pages.
- **Flask dashboard** under `website/app.py` with templates in `website/templates/`. KPIs per mode, equity curve with stress-event annotations, blotter, ablation comparison, test status. Reads `website/data/{metrics,blotter,test_results}.json`. Deployment via Render.com using `render.yaml`. Local: `flask --app website/app.py run`.

## Live monitoring framework

The strategy ships with a Hoeffding-inequality monitor in trader-application form to answer the two questions every trading-system writeup is supposed to address.

> *How will you know your strategy is performing as expected?* The pre-committed basket win rate over the OOS sample is μ = 0.730. A rolling 60-trade window of realized win rate X̄ produces a Hoeffding bound on the probability of regime shift: P[X̄ − μ ≥ t | H₀] ≤ exp(−2t²N) for N = 60. Threshold semantics 50% / 25% / 10% on the bound trigger reduce-size, freeze, and shut-down actions respectively.
>
> *How will you quantify when it stops working?* The framework logs a daily signal. Backtested on the headline basket over 1,760 OOS trading days, the framework produced 88% green / 7.7% yellow / 4.2% red / 0% critical signals. No regime-shutdown event triggered in 7 years.

See `website/monitoring.qmd` for the worked example and `src/metrics/hoeffding.py` for the implementation.

## Multiple-testing correction

The headline was selected from a tested universe of 12 instruments and 5 alternative basket configurations. Selection bias is corrected with two pre-committed tests:

1. **Deflated Sharpe Ratio with Eq. 9 implied-independent-trials adjustment** (Bailey and López de Prado 2014). The 12 raw trials have average pairwise return correlation 0.261, giving N̂ = 9 implied independent trials. The headline basket's PSR is 1.0000 against the noise floor for 9 trials.
2. **Probability of Backtest Overfitting via combinatorially symmetric cross-validation** (Bailey, Borwein, López de Prado, Zhu 2015) with S = 16 partitions and C(16, 8) = 12,870 logit combinations. PBO = 0.0402.

Both gates pass.

## Data sources

| Source | What it provides |
|---|---|
| OptionMetrics IvyDB US (via WRDS) | Daily option chains 2012-2025 with bid/ask/IV/Greeks. Stored as per-ticker parquets at `data/processed/options_by_ticker/{TICKER}.parquet`. |
| OptionMetrics IvyDB Securities (via WRDS) | Daily underlying OHLC for the 4 headline-basket tickers and 6 ablation tickers, 2012 to 2025. Same secid system as the chains, ensuring clean joins on date and security. Stored under `data/raw/{TICKER}.parquet`. |
| Interactive Brokers TWS feed snapshot | VIX, VIX3M, VVIX, SKEW, IRX, TNX, HYG, LQD, SPY, SPX index OHLC. Frozen to local parquet at commit time. No live network connection during a backtest. |

All data files are gitignored. The repository contains code, methodology documents, and the rendered Quarto site.

## Author contributions

- **Robert Lanni** designed the Flask dashboard architecture (`website/app.py`, dark finance theme, Plotly charts, KPI cards, ablation comparison views, test status accordion), wrote the test suite (`tests/test_*.py` covering strategy, halts, friction, models, features, backtest), set up Render.com deployment (`render.yaml`), built the shared `conftest.py` fixtures, and contributed to the joint pre-commitment baseline.
- **Mario Treviño** wrote the backtest engine (`src/backtest/`), the strategy modules (spread construction, exits, halts, sizing, friction, OptionMetrics pricer), the three-head ML stack (per-instrument quality, pooled regime stress, per-instrument skew), the multiple-testing-correction metrics (Deflated Sharpe Ratio with Eq. 9 correction, PBO via CSCV, Hoeffding trader-form bound, Hodrick standard errors), the per-ticker pre-commitment documents, and the Quarto site.

## Reproducibility

```bash
git clone https://github.com/mariotrev120/FinalProject_FinTech533.git
cd FinalProject_FinTech533
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# One-time data conversion (DuckDB streaming, ~80s)
PYTHONPATH=. python scripts/convert_optionmetrics_to_parquet.py

# Reproduce the headline result
PYTHONPATH=. python scripts/run_multi_instrument_vrp.py \
    --tickers AAPL MSFT WMT GLD \
    --mode halts_only \
    --no-ic \
    --total-capital 200000
```

`SEED = 42` is consumed by every stochastic component. Two runs of the same commit on the same data produce byte-identical equity curves.

## References

- Bailey, D. H., and López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. Journal of Portfolio Management.
- Bailey, D. H., Borwein, J. M., López de Prado, M., and Zhu, Q. J. (2015). The Probability of Backtest Overfitting. Journal of Computational Finance.
- Bollerslev, T., Tauchen, G., and Zhou, H. (2009). Expected Stock Returns and Variance Risk Premia. Review of Financial Studies.
- Carr, P., and Wu, L. (2003, 2004). Variance Risk Premia. Review of Financial Studies (working paper sequence).
- Black, F., and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy.
- Merton, R. C. (1973). Theory of Rational Option Pricing. Bell Journal of Economics and Management Science.
- Hoeffding, W. (1963). Probability Inequalities for Sums of Bounded Random Variables. Journal of the American Statistical Association.
- Niculescu-Mizil, A., and Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. Proceedings of the 22nd International Conference on Machine Learning.
- Hodrick, R. J. (1992). Dividend Yields and Expected Stock Returns. Review of Financial Studies.

## License

MIT. See `LICENSE`.
