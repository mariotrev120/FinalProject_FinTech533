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
    rows = []
    for t in result.trades:
        if t.pnl_per_spread is None:
            continue
        rows.append({
            "entry_date": pd.Timestamp(t.entry_date),
            "exit_date": pd.Timestamp(t.exit_date) if t.exit_date is not None else pd.NaT,
            "win": int(t.pnl_per_spread > 0),
            "pnl_per_spread": float(t.pnl_per_spread),
            "fate": t.fate,
            "entry_dte": t.entry_dte,
        })
    df = pd.DataFrame(rows).set_index("entry_date").sort_index()
    return df
