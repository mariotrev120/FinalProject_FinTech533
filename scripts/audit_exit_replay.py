"""
Replay 5 stop-loss-with-gap trades + 5 time-exit-at-21-DTE trades to verify
gap-aware execution and the 21-DTE time exit fire correctly.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_default_pricer, load_inputs
from src.config import OOS_END, OOS_START


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    inputs = load_inputs()
    pricer = load_default_pricer()
    naked = run_backtest(inputs, pricer, mode="naked", start=OOS_START, end=OOS_END)

    rows = []
    for t in naked.trades:
        if t.exit_date is None or t.fate is None:
            continue
        rows.append({
            "entry_date": pd.Timestamp(t.entry_date),
            "exit_date": pd.Timestamp(t.exit_date),
            "fate": t.fate,
            "short_strike": t.spread.short_leg.strike,
            "expiry": pd.Timestamp(t.spread.short_leg.expiry),
            "entry_credit_per_spread": t.entry_credit_per_spread,
            "exit_debit_per_spread": t.exit_debit_per_spread,
            "pnl_per_spread": t.pnl_per_spread,
            "total_pnl": t.total_pnl,
            "contracts": t.contracts,
            "entry_spx": t.entry_spx,
            "exit_spx": t.exit_spx,
            "exit_short_delta": t.exit_short_delta,
        })
    df = pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)
    print(f"\nTotal trades: {len(df)}")
    print("\nFate distribution:")
    print(df["fate"].value_counts().to_string())

    spx = inputs.bars_spx.copy()
    spx.index = pd.to_datetime(spx.index)

    print("\n" + "=" * 78)
    print("STOP-LOSS replay: do exit prices respect the gap-aware MAX rule?")
    print("=" * 78)
    sl = df[df["fate"] == "stop_loss"].head(5)
    for _, r in sl.iterrows():
        exit_dt = r["exit_date"]
        if exit_dt not in spx.index:
            print(f"  {exit_dt.date()}: exit date not in SPX index, skipping")
            continue
        idx = spx.index.get_loc(exit_dt)
        prior = spx.index[idx - 1] if idx > 0 else None
        prior_close = spx.iloc[idx - 1]["close"] if idx > 0 else None
        gap_pct = (spx.loc[exit_dt, "open"] / prior_close - 1.0) * 100 if prior_close else 0.0
        intraday_low = spx.loc[exit_dt, "low"]
        print(f"  {exit_dt.date()} (entry {r['entry_date'].date()}, K={r['short_strike']}):")
        print(f"     prior close ={prior_close:>8.2f}    open ={spx.loc[exit_dt,'open']:>8.2f}    "
              f"low ={intraday_low:>8.2f}    close ={spx.loc[exit_dt,'close']:>8.2f}")
        print(f"     gap_pct      ={gap_pct:>+6.2f}%    "
              f"entry_credit ={r['entry_credit_per_spread']:>8.2f}    "
              f"exit_debit ={r['exit_debit_per_spread']:>8.2f}    "
              f"pnl_per_spread ={r['pnl_per_spread']:>+9.2f}")
        if r["exit_debit_per_spread"] > 5 * 100:
            print(f"     !! exit debit exceeds spread width $500 — anomaly")

    print("\n" + "=" * 78)
    print("TIME EXIT replay: 21-DTE exit fires correctly on calendar days?")
    print("=" * 78)
    te = df[df["fate"] == "time_exit"].head(5)
    for _, r in te.iterrows():
        dte_at_exit = (r["expiry"] - r["exit_date"]).days
        dte_day_before = dte_at_exit + 1
        print(f"  entry {r['entry_date'].date()}  exit {r['exit_date'].date()}  "
              f"expiry {r['expiry'].date()}  "
              f"DTE_at_exit={dte_at_exit:>2d} (day before would have been {dte_day_before})")
        if dte_at_exit > 21:
            print(f"  !! exit fired at DTE > 21 — time exit triggered too early")
        elif dte_at_exit < 18:
            print(f"  !! exit fired at DTE < 18 — possible gap in time check")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
