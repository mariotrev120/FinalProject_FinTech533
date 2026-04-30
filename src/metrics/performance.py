"""
Performance metrics on a list of closed trades.

Returns a single dict per backtest run, suitable for dropping into the
Component Attribution table directly.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.strategy.types import Trade


def trade_returns(trades: Iterable[Trade], starting_equity: float) -> pd.Series:
    """Per-trade return as a fraction of starting equity, indexed by entry_date."""
    rows = []
    for t in trades:
        if t.total_pnl is None:
            continue
        rows.append((pd.Timestamp(t.entry_date), t.total_pnl / starting_equity))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(dict(rows)).sort_index()
    s.name = "trade_return"
    return s


def summarize_blotter(trades: list[Trade], starting_equity: float = 100_000.0) -> dict:
    pnls = np.array([t.total_pnl for t in trades if t.total_pnl is not None])
    if len(pnls) == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "avg_pnl": 0.0, "median_pnl": 0.0,
            "total_pnl": 0.0, "expected_return_per_trade_pct": 0.0,
            "avg_trade_days": 0.0, "fate_distribution": {},
            "sharpe_trade_level": float("nan"),
        }

    rets = pnls / starting_equity
    wins = (pnls > 0).sum()
    fates = pd.Series([t.fate for t in trades]).value_counts().to_dict()
    days = []
    for t in trades:
        if t.exit_date is None: continue
        d = (pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days
        days.append(d)
    avg_days = float(np.mean(days)) if days else 0.0

    sharpe_trade = (rets.mean() / rets.std() * np.sqrt(52)) if rets.std() > 0 else float("nan")

    return {
        "n_trades": int(len(pnls)),
        "win_rate": float(wins / len(pnls)),
        "avg_pnl": float(pnls.mean()),
        "median_pnl": float(np.median(pnls)),
        "total_pnl": float(pnls.sum()),
        "expected_return_per_trade_pct": float(rets.mean() * 100),
        "avg_trade_days": avg_days,
        "fate_distribution": fates,
        "sharpe_trade_level": float(sharpe_trade),
    }


def equity_curve_metrics(
    equity: pd.Series, risk_free_curve: pd.Series | None = None,
) -> dict:
    """Annualized portfolio Sharpe with explicit formula:
        SR = (annualized_return - annualized_rf) / annualized_vol
    where:
        annualized_return = (equity[-1] / equity[0]) ** (1/years) - 1
        annualized_vol    = std(daily_return) * sqrt(252)
        annualized_rf     = mean of daily 3M T-bill rate over the window

    The "trade-level Sharpe" reported elsewhere is a separate concept
    (mean-trade-return / std-trade-return × sqrt(52)) and is NOT the
    portfolio Sharpe used here. Both are reported for transparency.
    """
    if len(equity) < 2:
        return {"sharpe_annualized": float("nan"), "max_drawdown_pct": 0.0,
                "max_dd_duration_days": 0, "annualized_return_pct": 0.0,
                "annualized_vol_pct": 0.0, "annualized_rf_pct": 0.0}
    daily_ret = equity.pct_change().dropna()
    cummax = equity.cummax()
    drawdown = (cummax - equity) / cummax
    max_dd = float(drawdown.max())

    underwater = drawdown > 0
    if underwater.any():
        groups = (underwater != underwater.shift()).cumsum()
        durations = underwater.groupby(groups).sum()
        max_dur = int(durations.max())
    else:
        max_dur = 0

    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    if n_years > 0 and equity.iloc[0] > 0:
        ann_ret = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1.0
    else:
        ann_ret = 0.0
    ann_vol = float(daily_ret.std() * np.sqrt(252))

    # Risk-free rate over the window
    if risk_free_curve is not None:
        rfs = risk_free_curve.reindex(daily_ret.index, method="ffill").dropna() / 10.0 / 100.0
        ann_rf = float(rfs.mean()) if len(rfs) > 0 else 0.04
    else:
        ann_rf = 0.04

    sharpe = (ann_ret - ann_rf) / ann_vol if ann_vol > 0 else float("nan")

    return {
        "sharpe_annualized": float(sharpe),
        "max_drawdown_pct": max_dd * 100,
        "max_dd_duration_days": max_dur,
        "annualized_return_pct": ann_ret * 100,
        "annualized_vol_pct": ann_vol * 100,
        "annualized_rf_pct": ann_rf * 100,
    }


def combined_metrics(
    trades: list[Trade], equity: pd.Series, starting_equity: float = 100_000.0,
    risk_free_curve: pd.Series | None = None,
) -> dict:
    """One-row dict suitable for dropping into the Component Attribution table."""
    out = {}
    out.update(summarize_blotter(trades, starting_equity))
    out.update(equity_curve_metrics(equity, risk_free_curve=risk_free_curve))
    return out
