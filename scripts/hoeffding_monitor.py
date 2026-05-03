"""
Hoeffding live-monitoring runner per Egger & Vestal (2025) trader application.

For each ticker's trade history (from per-instrument backtest output),
emits a dated alert log showing when the rolling-window win rate's
Hoeffding probability bound crosses the 50% / 25% / 10% thresholds.

Used in writeup live-monitoring section: the formal answer to "how do
you know when the strategy stopped working?"

Inputs (per ticker):
  - data/processed/per_instrument_results/{TICKER}_trades.parquet
    (from scripts/run_multi_instrument_vrp.py output) OR
  - data/processed/head1_per_instrument/{TICKER}_labels.parquet
    (from scripts/train_head1_per_instrument.py output — naked-mode
    labelled trades).

Pre-committed μ baseline:
  - For VRP cluster: μ ≈ IS-window mean of `win` per instrument from
    naked-mode IC backtest. Computed from the labels parquet's
    pre-2018 portion.

Output:
  - logs/hoeffding_monitor.log
  - data/processed/hoeffding_signals/{TICKER}_signals.parquet
    Columns: rolling_winrate, hoeffding_bound, signal.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/hoeffding_monitor.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED_DIR, IS_END
from src.metrics.hoeffding import (
    regime_change_probability_bound,
    regime_change_signal,
    rolling_regime_signal,
)


log = logging.getLogger(__name__)


VRP_UNIVERSE = ["SPX", "RUT", "NDX", "TLT", "GLD"]


def find_labels_parquet(ticker: str) -> Path | None:
    """Find the labels parquet for a ticker, preferring head1 output
    (cleanest naked-IC labels) but falling back to multi-instrument
    runner output."""
    candidates = [
        Path(DATA_PROCESSED_DIR) / "head1_per_instrument" / f"{ticker}_labels.parquet",
        Path(DATA_PROCESSED_DIR) / "per_instrument_results" / f"{ticker}_trades.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def labels_to_trade_outcomes(labels: pd.DataFrame) -> pd.Series:
    """Coerce a labels parquet into a per-trade win/loss Series indexed
    by entry date."""
    if "win" in labels.columns:
        s = labels["win"].astype(int)
    elif "pnl_per_spread" in labels.columns:
        s = (labels["pnl_per_spread"] > 0).astype(int)
    else:
        raise ValueError(
            f"labels DataFrame has no 'win' or 'pnl_per_spread' column; "
            f"got {list(labels.columns)}"
        )
    if not isinstance(s.index, pd.DatetimeIndex):
        if "entry_date" in labels.columns:
            s.index = pd.DatetimeIndex(labels["entry_date"])
    return s.sort_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=VRP_UNIVERSE)
    parser.add_argument("--window", type=int, default=60,
                        help="Rolling window in trades (default 60)")
    parser.add_argument("--mu", type=float, default=None,
                        help="Override committed μ baseline. If None, "
                             "compute per-ticker IS win rate.")
    args = parser.parse_args(argv)

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/hoeffding_monitor.log", mode="w"),
        ],
    )

    out_dir = Path(DATA_PROCESSED_DIR) / "hoeffding_signals"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Hoeffding live-monitoring runner (Egger & Vestal 2025 trader form)")
    log.info("=" * 70)
    log.info("Universe: %s", args.tickers)
    log.info("Rolling window: %d trades", args.window)

    is_end = pd.Timestamp(IS_END)

    summary_rows = []
    for ticker in args.tickers:
        labels_path = find_labels_parquet(ticker)
        if labels_path is None:
            log.warning("[%s] no labels parquet found, skipping", ticker)
            continue
        log.info("[%s] loading %s", ticker, labels_path)
        labels = pd.read_parquet(labels_path)
        outcomes = labels_to_trade_outcomes(labels)

        # Determine μ baseline
        if args.mu is not None:
            mu = float(args.mu)
        else:
            is_outcomes = outcomes.loc[outcomes.index <= is_end]
            if len(is_outcomes) < 30:
                log.warning(
                    "[%s] only %d IS trades (< 30) — insufficient to compute μ; "
                    "skipping", ticker, len(is_outcomes),
                )
                continue
            mu = float(is_outcomes.mean())
        log.info(
            "[%s] committed μ = %.3f (computed on %d IS trades through %s)",
            ticker, mu,
            int((outcomes.index <= is_end).sum()),
            is_end.date(),
        )

        # Rolling signal computation
        sig_df = rolling_regime_signal(
            outcomes, mu_committed=mu, window=args.window,
        )
        sig_path = out_dir / f"{ticker}_signals.parquet"
        sig_df.to_parquet(sig_path)
        log.info("[%s] saved %s", ticker, sig_path)

        # Summary breakdown
        oos_sigs = sig_df.loc[sig_df.index > is_end]
        n_oos = len(oos_sigs.dropna(subset=["signal"]))
        sig_counts = oos_sigs["signal"].dropna().value_counts().to_dict()
        for s in ["green", "yellow", "red", "critical"]:
            sig_counts.setdefault(s, 0)
        log.info(
            "[%s] OOS signal breakdown (n=%d): green=%d, yellow=%d, red=%d, "
            "critical=%d",
            ticker, n_oos,
            sig_counts["green"], sig_counts["yellow"],
            sig_counts["red"], sig_counts["critical"],
        )

        # Earliest critical/red dates for the writeup
        critical = oos_sigs[oos_sigs["signal"] == "critical"]
        red = oos_sigs[oos_sigs["signal"] == "red"]
        first_critical = critical.index.min() if len(critical) else None
        first_red = red.index.min() if len(red) else None
        log.info(
            "[%s] first RED:      %s   first CRITICAL: %s",
            ticker,
            first_red.date() if first_red is not None else "(none)",
            first_critical.date() if first_critical is not None else "(none)",
        )

        summary_rows.append({
            "ticker": ticker,
            "mu_committed": mu,
            "n_oos_signals": n_oos,
            "n_green": sig_counts["green"],
            "n_yellow": sig_counts["yellow"],
            "n_red": sig_counts["red"],
            "n_critical": sig_counts["critical"],
            "first_red_date": first_red,
            "first_critical_date": first_critical,
        })

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(out_dir / "summary.csv", index=False)
        log.info("=" * 70)
        log.info("Summary saved to %s/summary.csv", out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
