"""
Shared backtest helpers: load raw + processed data into a `BacktestInputs`
struct, and summarize a `BacktestResult` into a one-row dict.

These live in `src/` (importable) rather than `scripts/` (CLI entry points)
because multiple scripts and notebooks consume them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestInputs, BacktestResult
from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.strategy.optionmetrics_pricer import make_optionmetrics_pricer


def load_inputs() -> BacktestInputs:
    """Load all raw bars + (optional) ML probability series into a
    BacktestInputs ready to feed run_backtest()."""
    spx = pd.read_parquet(DATA_RAW_DIR / "SPX.parquet")
    vix = pd.read_parquet(DATA_RAW_DIR / "VIX.parquet")["close"]
    vix3m = pd.read_parquet(DATA_RAW_DIR / "VIX3M.parquet")["close"]
    hyg = pd.read_parquet(DATA_RAW_DIR / "HYG.parquet")["close"]
    lqd = pd.read_parquet(DATA_RAW_DIR / "LQD.parquet")["close"]
    spy = pd.read_parquet(DATA_RAW_DIR / "SPY.parquet")["close"]
    tnx = pd.read_parquet(DATA_RAW_DIR / "TNX.parquet")["close"]

    spy_ret = spy.pct_change()
    tnx_chg = tnx.diff()
    spy_treasury_corr = spy_ret.rolling(20).corr(tnx_chg)

    # 3M T-bill yield index for daily risk-free accrual
    irx = pd.read_parquet(DATA_RAW_DIR / "IRX.parquet")["close"]

    ml_path = DATA_PROCESSED_DIR / "ml_probabilities.parquet"
    if ml_path.exists():
        ml_prob = pd.read_parquet(ml_path)["p_calibrated"]
    else:
        ml_prob = None

    return BacktestInputs(
        bars_spx=spx[["open", "high", "low", "close"]].copy(),
        vix=vix,
        vix3m=vix3m,
        hyg_minus_lqd=(hyg - lqd),
        spy_treasury_corr=spy_treasury_corr,
        ml_probability=ml_prob,
        is_winrate_baseline=0.75,
        risk_free_curve=irx,
    )


def load_default_pricer(underlying: str = "SPX"):
    """Default pricer for the project: real OptionMetrics IvyDB quotes.

    Pre-loads ~6M rows into an in-memory index. Subsequent backtest runs
    against this pricer reuse the same instance for performance.
    """
    return make_optionmetrics_pricer(underlying=underlying)


def summarize(result: BacktestResult, mode: str) -> dict:
    """One-row text summary of a BacktestResult — used by the run script
    and a few other diagnostic callers."""
    n = len(result.trades)
    wins = sum(1 for t in result.trades if t.pnl_per_spread is not None and t.pnl_per_spread > 0)
    losses = sum(1 for t in result.trades if t.pnl_per_spread is not None and t.pnl_per_spread <= 0)
    pnls = np.array([t.total_pnl for t in result.trades if t.total_pnl is not None])
    fates = pd.Series([t.fate for t in result.trades]).value_counts().to_dict() if n else {}
    eq = result.equity_curve
    starting = float(eq.iloc[0]) if len(eq) > 0 else 100_000.0
    ending = float(eq.iloc[-1]) if len(eq) > 0 else starting
    ret = (ending - starting) / starting if starting > 0 else 0.0

    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else float("nan")

    drawdown = (eq.cummax() - eq) / eq.cummax()
    max_dd = float(drawdown.max()) if len(drawdown) else 0.0

    return {
        "mode": mode,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / n if n else 0.0,
        "total_pnl": float(pnls.sum()) if len(pnls) else 0.0,
        "avg_pnl_per_trade": float(pnls.mean()) if len(pnls) else 0.0,
        "ret_pct": ret * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "fates": fates,
        "skipped": len(result.skipped_entries),
    }
