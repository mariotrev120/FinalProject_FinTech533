"""
End-to-end runner. Loads raw data, builds engine inputs, runs all four
ablation modes, prints a summary.

Usage:
    PYTHONPATH=. .venv/bin/python -m src.backtest.run
    PYTHONPATH=. .venv/bin/python -m src.backtest.run --mode naked --start 2018-01-01
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestInputs, BacktestResult, run_backtest
from src.config import DATA_RAW_DIR, IS_END, IS_START, OOS_END, OOS_START
from src.strategy.black_scholes import make_default_pricer


log = logging.getLogger(__name__)


def load_inputs() -> BacktestInputs:
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

    # Load ML probabilities if present (set by src.models.train)
    from src.config import DATA_PROCESSED_DIR
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
    )


def summarize(result: BacktestResult, mode: str) -> dict:
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["naked", "ml_only", "halts_only", "full", "all"], default="all")
    p.add_argument("--start", default=IS_START)
    p.add_argument("--end", default=OOS_END)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    inputs = load_inputs()
    pricer = make_default_pricer()

    modes = ["naked", "ml_only", "halts_only", "full"] if args.mode == "all" else [args.mode]
    summaries = []
    for m in modes:
        if m in ("ml_only", "full") and inputs.ml_probability is None:
            print(f"[skip {m}] no ML probability series wired in yet")
            continue
        log.info("running mode=%s on %s -> %s", m, args.start, args.end)
        result = run_backtest(inputs, pricer, mode=m, start=args.start, end=args.end)
        s = summarize(result, m)
        summaries.append(s)
        print(f"\n=== mode={m} ===")
        for k, v in s.items():
            if k == "fates":
                print(f"  {k}: {v}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
