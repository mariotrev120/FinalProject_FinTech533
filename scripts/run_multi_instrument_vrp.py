"""
Multi-instrument VRP runner: 5-ticker iron condor cluster (Phase B-1).

Loops [SPX, RUT, NDX, TLT, GLD] through the engine, runs iron-condor
backtest per ticker over OOS, and aggregates per-instrument equity
curves into a book-level NAV.

Capital allocation:
  - This is the BASELINE multi-instrument run. Equal-weight: each ticker
    runs as a standalone $100K-capital book ($500K total / 5 = $100K).
    The per-instrument equity curves are reported on a $100K basis.
    Aggregate book NAV = Σ instrument_i_equity_t (paper-portfolio sum).
  - $100K per ticker is the floor for iron condor entries: the engine's
    1% per-trade hard cap = $1000, which must clear the IC max-loss
    per spread (~$400 SPX, ~$2000 NDX). At $20K/ticker the cap was $200,
    blocking every entry → 0 trades. v1.5 baseline used $100K SPX-only,
    so $100K/ticker is consistent with that anchor's sizing scale.
  - Quality-weighted allocation (w_i = p_quality_i / Σ p_quality_j) is a
    Phase B-2 task — requires Head 1 trained first, so deferred.

Memory discipline:
  - Loads one ticker's pricer at a time, runs the backtest, frees the
    pricer's DataFrame from cache, then loads the next ticker. Peak
    memory ≈ size of the largest single ticker (SPX ~3 GB in pandas).
  - Without this, holding all 5 simultaneously is ~8-10 GB → OOM on WSL.

Engine note (known limitation, documented in writeup):
  - The day-loop uses inputs.bars_spx as the trading calendar AND for
    gap-detection across ALL instruments. This is an approximation
    consistent with US options markets all sharing the NYSE calendar.
    Per-instrument bars (for tighter gap detection per ticker) is a
    refactor candidate for after-class iteration.

Output:
  - data/processed/per_instrument_results/{TICKER}_equity.parquet
  - data/processed/per_instrument_results/{TICKER}_trades.parquet
  - data/processed/per_instrument_results/{TICKER}_halt_log.parquet
  - data/processed/aggregate_book_equity.parquet
  - logs/multi_instrument_vrp.log

Console output:
  - Per-instrument PortfolioReport headline
  - Aggregate book-level PortfolioReport headline
  - Anchor comparison vs v1.5 baseline halts_only OOS Sharpe = 0.286

Usage:
  PYTHONPATH=. .venv/bin/python scripts/run_multi_instrument_vrp.py [--mode naked|halts_only|...]
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_inputs
from src.config import DATA_PROCESSED_DIR, OOS_END, OOS_START
from src.strategy.optionmetrics_pricer import _TICKER_DF_CACHE, make_pricer_v2


log = logging.getLogger(__name__)


VRP_UNIVERSE: list[str] = ["SPX", "RUT", "NDX", "TLT", "GLD"]
V1_5_ANCHOR_SHARPE: float = 0.286   # halts_only OOS, single-instrument SPX


def run_one_ticker(
    ticker: str,
    inputs,
    mode: str,
    start: str,
    end: str,
    use_iron_condor: bool,
    capital: float,
):
    """Run engine for a single ticker, then free its DataFrame from cache."""
    log.info("=" * 70)
    log.info("Running %s (capital=$%s, mode=%s, IC=%s)",
             ticker, f"{capital:,.0f}", mode, use_iron_condor)
    log.info("=" * 70)
    t0 = time.time()

    pricer = make_pricer_v2(ticker)
    log.info("  %s pricer loaded in %.1fs (put dict %d, call dict %d)",
             ticker, time.time() - t0,
             len(pricer._by_date_exp_put), len(pricer._by_date_exp_call))

    # Per-ticker inputs: just override initial_equity to per-ticker capital
    per_inputs = replace(inputs, initial_equity=capital)

    t = time.time()
    result = run_backtest(
        per_inputs, pricer,
        mode=mode, start=start, end=end,
        use_iron_condor=use_iron_condor,
    )
    log.info("  %s backtest done in %.1fs (n_trades=%d, final eq=$%s)",
             ticker, time.time() - t,
             len(result.trades), f"{result.equity_curve.iloc[-1]:,.0f}")

    # Free pricer's DataFrame so next ticker can load without OOM
    if ticker in _TICKER_DF_CACHE:
        del _TICKER_DF_CACHE[ticker]
    del pricer
    gc.collect()

    return result


def aggregate_book_equity(
    per_ticker_equity: dict[str, pd.Series],
) -> pd.Series:
    """Aggregate book-level NAV = Σ per-instrument equity curves at each
    date. Each instrument starts with capital/N and they're summed."""
    df = pd.DataFrame(per_ticker_equity)
    # Forward-fill per-ticker (different inception dates / calendar gaps)
    # then sum. Where a ticker has no data, its contribution is 0.
    df = df.ffill().fillna(0)
    return df.sum(axis=1).rename("book_equity")


def quick_metrics(equity: pd.Series, n_trades: int, label: str) -> dict:
    """Bare-bones metrics computed without external deps. Real metrics
    via PortfolioReport in the full ablation runner; here we just want
    the headline numbers."""
    ret = equity.pct_change().dropna()
    if len(ret) < 2 or ret.std() == 0:
        return {"label": label, "sharpe": float("nan"), "ann_return_pct": float("nan"),
                "max_dd_pct": float("nan"), "final_eq": float(equity.iloc[-1])}
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252))
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    drawdown = (equity / equity.cummax() - 1).min()
    return {
        "label": label,
        "n_trades": n_trades,
        "sharpe": sharpe,
        "ann_return_pct": float(ann_return * 100),
        "max_dd_pct": float(drawdown * 100),
        "final_eq": float(equity.iloc[-1]),
        "total_return_pct": float(total_return * 100),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="naked",
                        choices=["naked", "halts_only", "ml_only", "full"])
    parser.add_argument("--start", default=OOS_START)
    parser.add_argument("--end", default=OOS_END)
    parser.add_argument("--total-capital", type=float, default=500_000.0,
                        help="Total book capital. Default $500K = $100K per ticker "
                             "across 5 VRP instruments. Iron condor max-loss-per-spread "
                             "(~$400 SPX, $2000 NDX) requires $100K+ per ticker for the "
                             "1%% per-trade hard cap to allow entries.")
    parser.add_argument("--no-ic", action="store_true",
                        help="Disable iron condor (use put-only spreads).")
    parser.add_argument("--tickers", nargs="+", default=VRP_UNIVERSE,
                        help="Subset of tickers to run (default: all 5 VRP)")
    args = parser.parse_args(argv)

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/multi_instrument_vrp.log", mode="w"),
        ],
    )

    out_dir = Path(DATA_PROCESSED_DIR) / "per_instrument_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Multi-Instrument VRP Runner — %s mode, IC=%s",
             args.mode, not args.no_ic)
    log.info("=" * 70)
    log.info("Universe: %s", args.tickers)
    log.info("Total capital: $%s", f"{args.total_capital:,.0f}")
    log.info("Per-instrument capital (equal-weight): $%s",
             f"{args.total_capital / len(args.tickers):,.0f}")
    log.info("OOS window: [%s, %s]", args.start, args.end)

    log.info("Loading inputs (small parquets)...")
    t0 = time.time()
    inputs = load_inputs()
    log.info("Inputs loaded in %.1fs", time.time() - t0)

    per_capital = args.total_capital / len(args.tickers)
    per_ticker_equity: dict[str, pd.Series] = {}
    per_ticker_metrics: list[dict] = []
    use_iron_condor = not args.no_ic

    overall_t0 = time.time()
    for ticker in args.tickers:
        try:
            result = run_one_ticker(
                ticker, inputs,
                mode=args.mode, start=args.start, end=args.end,
                use_iron_condor=use_iron_condor, capital=per_capital,
            )
        except Exception as e:
            log.exception("Ticker %s FAILED: %s", ticker, e)
            continue

        # Save artifacts
        eq_path = out_dir / f"{ticker}_equity.parquet"
        result.equity_curve.to_frame("equity").to_parquet(eq_path)
        log.info("  saved %s", eq_path)

        if result.trades:
            trades_df = pd.DataFrame([
                dict(
                    entry_date=t.entry_date, exit_date=t.exit_date,
                    fate=t.fate, mode=t.mode,
                    pnl_per_spread=t.pnl_per_spread,
                    iron_condor_id=t.iron_condor_id,
                    right=t.spread.right,
                    halt_state=t.halt_state_at_entry,
                ) for t in result.trades
            ])
            trades_path = out_dir / f"{ticker}_trades.parquet"
            trades_df.to_parquet(trades_path)
            log.info("  saved %s (%d trades)", trades_path, len(trades_df))

        if not result.halt_log.empty:
            halt_path = out_dir / f"{ticker}_halt_log.parquet"
            result.halt_log.to_parquet(halt_path)

        per_ticker_equity[ticker] = result.equity_curve
        m = quick_metrics(result.equity_curve, len(result.trades), ticker)
        per_ticker_metrics.append(m)
        log.info(
            "  %s metrics: Sharpe=%.3f, ann_return=%+.2f%%, max_dd=%+.2f%%, "
            "n_trades=%d, final_eq=$%s",
            ticker, m["sharpe"], m["ann_return_pct"], m["max_dd_pct"],
            m["n_trades"], f"{m['final_eq']:,.0f}",
        )

    if not per_ticker_equity:
        log.error("No instruments produced results. Aborting.")
        return 1

    # --- Aggregate book ---
    log.info("=" * 70)
    log.info("Aggregating book-level NAV (equal-weight, %d instruments)",
             len(per_ticker_equity))
    log.info("=" * 70)

    book_equity = aggregate_book_equity(per_ticker_equity)
    book_path = Path(DATA_PROCESSED_DIR) / "aggregate_book_equity.parquet"
    book_equity.to_frame().to_parquet(book_path)
    log.info("Saved book equity to %s", book_path)

    # Aggregate trades
    total_trades = sum(m["n_trades"] for m in per_ticker_metrics)
    book_metrics = quick_metrics(book_equity, total_trades, label="BOOK")

    log.info("=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)
    log.info("Per-instrument:")
    log.info("  %-7s | %8s | %10s | %10s | %8s | %12s",
             "Ticker", "Sharpe", "AnnRet%", "MaxDD%", "Trades", "FinalEq")
    log.info("  " + "-" * 78)
    for m in per_ticker_metrics:
        log.info("  %-7s | %+8.3f | %+10.2f | %+10.2f | %8d | $%11s",
                 m["label"], m["sharpe"], m["ann_return_pct"],
                 m["max_dd_pct"], m["n_trades"], f"{m['final_eq']:,.0f}")
    log.info("  " + "-" * 78)
    log.info("  %-7s | %+8.3f | %+10.2f | %+10.2f | %8d | $%11s",
             "BOOK", book_metrics["sharpe"], book_metrics["ann_return_pct"],
             book_metrics["max_dd_pct"], book_metrics["n_trades"],
             f"{book_metrics['final_eq']:,.0f}")

    log.info("=" * 70)
    log.info("ANCHOR COMPARISON")
    log.info("=" * 70)
    log.info("  v1.5 baseline halts_only OOS Sharpe (single-inst SPX):  %.3f",
             V1_5_ANCHOR_SHARPE)
    log.info("  v2 multi-instrument BOOK Sharpe (%s mode, %d inst):     %.3f",
             args.mode, len(per_ticker_equity), book_metrics["sharpe"])
    delta = book_metrics["sharpe"] - V1_5_ANCHOR_SHARPE
    log.info("  Delta vs anchor:                                        %+.3f",
             delta)
    if delta > 0:
        log.info("  Multi-instrument BOOK BEATS v1.5 anchor by %.3f Sharpe.",
                 delta)
    else:
        log.info("  Multi-instrument BOOK does NOT beat v1.5 anchor (delta %+.3f).",
                 delta)

    log.info("Total runner wall-clock: %.1fs", time.time() - overall_t0)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
