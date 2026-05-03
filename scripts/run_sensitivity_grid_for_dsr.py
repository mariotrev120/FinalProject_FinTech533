"""
Sensitivity grid runner that persists per-trial returns matrices.

The existing scripts/audit_sensitivity_grid.py only persists summary stats
per trial (Sharpe, max DD, win rate). Computing DSR with the implied-
independent-trials correction (Eq. 9) and PBO via CSCV (BBLP 2015) both
require the (T × N_trials) returns matrix — the average pairwise correlation
across trial-return columns is what deflates the raw trial count.

This runner produces:
  data/processed/sensitivity_grid_returns.parquet  — (T x N_trials) returns matrix
  data/processed/sensitivity_grid_summary.csv      — per-trial summary stats
  data/processed/sensitivity_grid_dsr.txt          — DSR + PBO report

Grid (default — moderate cost):
  - PROFIT_TARGET_FRAC: {0.30, 0.40, 0.50, 0.60}        (4)
  - STOP_LOSS_MULT:     {1.5, 2.0, 2.5, 3.0}            (4)
  - DTE window:         {(25,40), (30,45), (35,50)}     (3)
  - ENTRY_DELTA_TARGET: {0.12, 0.16, 0.20}              (3)
  Total: 4*4*3*3 = 144 trials. SPX-only, naked, OOS.

Optional --full enables the larger 729-trial §11 grid (3 short-deltas × 3
spread-widths × 3 DTE-windows × 3 PT × 3 SL × 3 time-exits) — requires
spread-width plumbing in src/strategy/spread_construction.py which is
not currently a config knob. Defer until that plumbing lands.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/run_sensitivity_grid_for_dsr.py
  PYTHONPATH=. .venv/bin/python scripts/run_sensitivity_grid_for_dsr.py --tickers SPX TLT GLD
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_inputs
from src.config import DATA_PROCESSED_DIR, OOS_END, OOS_START
from src.metrics.deflated_sharpe import (
    deflated_sharpe_ratio,
    n_eff_from_matrix,
)
from src.metrics.pbo import cscv_pbo
from src.metrics.performance import combined_metrics
from src.strategy.optionmetrics_pricer import _TICKER_DF_CACHE, make_pricer_v2


log = logging.getLogger(__name__)


GRID_PT = [0.30, 0.40, 0.50, 0.60]
GRID_SL = [1.5, 2.0, 2.5, 3.0]
GRID_DTE = [(25, 40), (30, 45), (35, 50)]
GRID_DLT = [0.12, 0.16, 0.20]


def daily_returns_from_equity(equity: pd.Series) -> pd.Series:
    """Compute daily simple returns from an equity series indexed by date."""
    eq = equity.dropna().sort_index()
    return eq.pct_change().iloc[1:]


def run_one_trial(inputs, pricer, pt: float, sl: float,
                  dte_min: int, dte_max: int, dlt: float) -> dict:
    """Run a single backtest trial and return summary + daily returns."""
    import src.config as cfg
    cfg.PROFIT_TARGET_FRAC = pt
    cfg.STOP_LOSS_MULT = sl
    cfg.DTE_MIN = dte_min
    cfg.DTE_MAX = dte_max
    cfg.ENTRY_DELTA_TARGET = dlt

    result = run_backtest(
        inputs, pricer, mode="naked",
        start=OOS_START, end=OOS_END, use_iron_condor=True,
    )
    metrics = combined_metrics(
        result.trades, result.equity_curve,
        risk_free_curve=inputs.risk_free_curve,
    )
    eq = pd.Series({d: e for d, e in result.equity_curve}).sort_index()
    eq.index = pd.to_datetime(eq.index)
    daily_ret = daily_returns_from_equity(eq)

    return {
        "params": {"pt": pt, "sl": sl, "dte_min": dte_min,
                   "dte_max": dte_max, "delta_target": dlt},
        "metrics": metrics,
        "daily_returns": daily_ret,
        "n_trades": len(result.trades),
    }


def trial_label(pt, sl, dte_min, dte_max, dlt) -> str:
    return f"pt{pt:.2f}_sl{sl:.1f}_dte{dte_min}-{dte_max}_dlt{dlt:.2f}"


def run_grid_for_ticker(ticker: str, inputs, out_dir: Path) -> dict:
    """Run the full grid for one ticker, save returns matrix + summary."""
    log.info("=" * 70)
    log.info("Sensitivity grid for %s — %d trials", ticker,
             len(GRID_PT) * len(GRID_SL) * len(GRID_DTE) * len(GRID_DLT))
    log.info("=" * 70)

    t0 = time.time()
    pricer = make_pricer_v2(ticker)
    log.info("  pricer loaded in %.1fs", time.time() - t0)

    summary_rows: list[dict] = []
    returns_cols: dict[str, pd.Series] = {}

    import src.config as cfg
    orig = (cfg.PROFIT_TARGET_FRAC, cfg.STOP_LOSS_MULT,
            cfg.DTE_MIN, cfg.DTE_MAX, cfg.ENTRY_DELTA_TARGET)
    try:
        n_combinations = len(GRID_PT) * len(GRID_SL) * len(GRID_DTE) * len(GRID_DLT)
        for i, (pt, sl, (dte_min, dte_max), dlt) in enumerate(
            product(GRID_PT, GRID_SL, GRID_DTE, GRID_DLT)
        ):
            label = trial_label(pt, sl, dte_min, dte_max, dlt)
            try:
                t = time.time()
                result = run_one_trial(inputs, pricer, pt, sl, dte_min, dte_max, dlt)
                dt = time.time() - t
                log.info(
                    "  [%3d/%d] %s: Sharpe=%.3f, n_trades=%d, %.1fs",
                    i + 1, n_combinations, label,
                    result["metrics"]["sharpe_annualized"],
                    result["n_trades"], dt,
                )
                summary_rows.append({
                    "ticker": ticker,
                    "trial_label": label,
                    "pt": pt, "sl": sl,
                    "dte_min": dte_min, "dte_max": dte_max,
                    "delta_target": dlt,
                    "n_trades": result["n_trades"],
                    "win_rate": result["metrics"]["win_rate"],
                    "ann_ret_pct": result["metrics"]["annualized_return_pct"],
                    "ann_vol_pct": result["metrics"]["annualized_vol_pct"],
                    "sharpe_ann": result["metrics"]["sharpe_annualized"],
                    "max_dd_pct": result["metrics"]["max_drawdown_pct"],
                })
                returns_cols[label] = result["daily_returns"]
            except Exception as e:
                log.error("  [%3d/%d] %s FAILED: %s",
                          i + 1, n_combinations, label, e)
    finally:
        (cfg.PROFIT_TARGET_FRAC, cfg.STOP_LOSS_MULT,
         cfg.DTE_MIN, cfg.DTE_MAX, cfg.ENTRY_DELTA_TARGET) = orig

    # Free per-ticker pricer cache
    if ticker in _TICKER_DF_CACHE:
        del _TICKER_DF_CACHE[ticker]
    del pricer
    import gc
    gc.collect()

    if not returns_cols:
        log.error("All trials failed for %s; skipping aggregation", ticker)
        return {}

    # Align all return series on a common index (outer join, fill 0 on missing
    # days = no trade activity). Then save matrix.
    R = pd.DataFrame(returns_cols).fillna(0.0).sort_index()
    summary_df = pd.DataFrame(summary_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    R.to_parquet(out_dir / f"{ticker}_grid_returns.parquet")
    summary_df.to_csv(out_dir / f"{ticker}_grid_summary.csv", index=False)
    log.info("Saved %s_grid_returns.parquet (%d rows × %d trials)",
             ticker, len(R), R.shape[1])

    # DSR + PBO computation
    n_hat, rho_bar = n_eff_from_matrix(R)
    log.info("Eq. 9: rho_bar=%.4f, N_hat=%d (raw N=%d)",
             rho_bar, n_hat, R.shape[1])

    # Best trial by raw Sharpe
    best_idx = int(np.argmax(summary_df["sharpe_ann"].to_numpy()))
    best_trial = summary_df.iloc[best_idx]
    best_returns = R.iloc[:, best_idx]

    # Per-trial Sharpe variance (for Eq. 1 V parameter)
    trial_sharpes_ann = summary_df["sharpe_ann"].to_numpy()
    sharpe_var_pp = float(np.var(trial_sharpes_ann, ddof=1)) / 252.0
    dsr_report = deflated_sharpe_ratio(
        best_returns, n_trials=n_hat,
        sharpe_var_across_trials_per_period=sharpe_var_pp,
    )

    # PBO via CSCV (S=16 default per BBLP)
    try:
        pbo_report = cscv_pbo(R, n_partitions=16)
    except ValueError as e:
        log.warning("PBO failed: %s — using S=8", e)
        pbo_report = cscv_pbo(R, n_partitions=8)

    headline_lines = [
        "=" * 70,
        f"SENSITIVITY-GRID DSR + PBO REPORT — {ticker}",
        "=" * 70,
        f"Trials run:                  {R.shape[1]}",
        f"OOS observations (T):        {R.shape[0]}",
        "",
        "Best trial by raw Sharpe:",
        f"  label:                     {best_trial['trial_label']}",
        f"  raw Sharpe (annualized):   {best_trial['sharpe_ann']:+.4f}",
        f"  ann return:                {best_trial['ann_ret_pct']:+.3f}%",
        f"  max DD:                    {best_trial['max_dd_pct']:+.3f}%",
        f"  n_trades:                  {int(best_trial['n_trades'])}",
        "",
        "Selection-bias deflation (Bailey & López de Prado, 2014):",
        f"  raw N (trials):            {R.shape[1]}",
        f"  rho-bar (avg pairwise):    {rho_bar:+.4f}",
        f"  N_hat (Eq. 9):             {n_hat}",
        "",
        dsr_report.headline(),
        "",
        pbo_report.headline(),
        "=" * 70,
        "ACCEPTANCE GATES:",
        f"  DSR (PSR) >= 0.95?         {dsr_report.is_significant_at_95}",
        f"  PBO <= 0.30?               {pbo_report.pbo <= 0.30}",
        "=" * 70,
    ]
    headline = "\n".join(headline_lines)
    print(headline)
    log.info(headline)
    (out_dir / f"{ticker}_dsr_pbo_report.txt").write_text(headline)

    return {
        "ticker": ticker,
        "n_trials": R.shape[1],
        "n_hat": n_hat,
        "rho_bar": rho_bar,
        "best_sharpe_ann": float(best_trial["sharpe_ann"]),
        "dsr_psr": dsr_report.psr,
        "dsr_passes": dsr_report.is_significant_at_95,
        "pbo": pbo_report.pbo,
        "pbo_passes": bool(pbo_report.pbo <= 0.30),
        "best_trial_label": best_trial["trial_label"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["SPX"])
    parser.add_argument("--full", action="store_true",
                        help="(NOT YET) larger 729-trial grid (defer until "
                             "spread-width plumbing lands)")
    args = parser.parse_args(argv)

    if args.full:
        log.error("--full grid is deferred until spread-width is a config knob")
        return 1

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/sensitivity_grid_dsr.log", mode="w"),
        ],
    )

    out_dir = Path(DATA_PROCESSED_DIR) / "sensitivity_grid"
    inputs = load_inputs()

    summaries: list[dict] = []
    for ticker in args.tickers:
        try:
            row = run_grid_for_ticker(ticker, inputs, out_dir)
            if row:
                summaries.append(row)
        except Exception as e:
            log.exception("Ticker %s grid failed: %s", ticker, e)
            summaries.append({"ticker": ticker, "error": str(e)})

    if summaries:
        master = pd.DataFrame(summaries)
        master.to_csv(out_dir / "master_summary.csv", index=False)
        log.info("Master summary saved to %s", out_dir / "master_summary.csv")
        log.info("\n%s", master.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
