"""
Label trades from a naked-mode backtest as wins/losses.

Output: a DataFrame indexed by entry_date with columns [win, pnl_per_spread,
fate]. The win column is the binary classification target the ML gate is
trained to predict.

Walk-forward fold methodology: trades from before fold-boundary date go into
training; trades within fold form the test/predict set.
"""
from __future__ import annotations

import pandas as pd

from src.backtest.engine import BacktestResult


def label_trades_dataframe(result: BacktestResult) -> pd.DataFrame:
    """Per-side trade labels (one row per Trade — for an iron condor that
    means two rows per entry_date: put + call). Index = entry_date."""
    rows = []
    for t in result.trades:
        if t.pnl_per_spread is None:
            continue
        side = "P" if (t.spread is not None and t.spread.right == "P") else "C"
        rows.append({
            "entry_date": pd.Timestamp(t.entry_date),
            "exit_date": pd.Timestamp(t.exit_date) if t.exit_date is not None else pd.NaT,
            "side": side,
            "iron_condor_id": t.iron_condor_id,
            "win": int(t.pnl_per_spread > 0),
            "pnl_per_spread": float(t.pnl_per_spread),
            "fate": t.fate,
            "entry_dte": t.entry_dte,
        })
    df = pd.DataFrame(rows).set_index("entry_date").sort_index()
    return df


def label_trades_ic_level(result: BacktestResult) -> pd.DataFrame:
    """Iron-condor-level labels — one row per IC pair, with win defined as
    the sum of put-side and call-side pnl_per_spread being positive
    (i.e., the iron condor as a whole was profitable). Used by Head 1
    for per-instrument quality classification, where the per-IC outcome
    is what we want to predict at entry-date granularity. Index =
    entry_date (unique).

    Trades without iron_condor_id (standalone) get one row each."""
    by_ic: dict[int, list] = {}
    standalone = []
    for t in result.trades:
        if t.pnl_per_spread is None:
            continue
        if t.iron_condor_id is None:
            standalone.append(t)
        else:
            by_ic.setdefault(t.iron_condor_id, []).append(t)

    rows = []
    for ic_id, sides in by_ic.items():
        sides_sorted = sorted(sides, key=lambda x: pd.Timestamp(x.entry_date))
        first = sides_sorted[0]
        total_pnl = sum(float(t.pnl_per_spread) for t in sides)
        rows.append({
            "entry_date": pd.Timestamp(first.entry_date),
            "iron_condor_id": ic_id,
            "n_sides": len(sides),
            "win": int(total_pnl > 0),
            "pnl_per_spread_total": total_pnl,
        })
    for t in standalone:
        rows.append({
            "entry_date": pd.Timestamp(t.entry_date),
            "iron_condor_id": None,
            "n_sides": 1,
            "win": int(float(t.pnl_per_spread) > 0),
            "pnl_per_spread_total": float(t.pnl_per_spread),
        })
    df = pd.DataFrame(rows).set_index("entry_date").sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]
    return df
