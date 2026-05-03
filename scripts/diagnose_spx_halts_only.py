"""
Diagnostic: SPX halts_only OOS run with full halt-log analysis.

Reports:
  - SPX halts_only OOS Sharpe vs v1.5 anchor (0.286)
  - Halt activation periods: dates of fresh halts and dates of resumes
  - Halt-active weeks per year
  - Time-fallback resume vs market-conditions resume breakdown

Usage:
  PYTHONPATH=. .venv/bin/python scripts/diagnose_spx_halts_only.py
"""
from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_inputs
from src.config import OOS_END, OOS_START
from src.strategy.optionmetrics_pricer import make_pricer_v2


log = logging.getLogger(__name__)


def find_halt_periods(halt_log: pd.DataFrame) -> pd.DataFrame:
    """Detect contiguous halt episodes and their resume mechanism."""
    log_df = halt_log.copy()
    log_df["is_halted"] = log_df["state"] != "active"

    # Mark transitions: 1 where state changes from active → halt; -1 from halt → active
    transition = log_df["is_halted"].astype(int).diff().fillna(0)
    starts = log_df.index[transition == 1].tolist()
    ends = log_df.index[transition == -1].tolist()

    # Edge: still halted at series end
    if len(log_df) and log_df["is_halted"].iloc[-1]:
        ends.append(log_df.index[-1])

    if len(starts) == 0:
        return pd.DataFrame(columns=[
            "start", "end", "duration_days", "first_trigger", "resume_trigger",
        ])

    rows = []
    for st, en in zip(starts, ends[: len(starts)]):
        period = log_df.loc[st:en]
        duration = len(period)
        first_trigger = period["triggers"].iloc[0]
        # Resume trigger is what was logged on the LAST active day after this halt
        try:
            after = log_df.loc[en:].iloc[1] if log_df.index.get_loc(en) + 1 < len(log_df) else None
            resume_trigger = after["triggers"] if after is not None else "still_halted"
        except (IndexError, KeyError):
            resume_trigger = "still_halted"
        rows.append({
            "start": st, "end": en, "duration_days": duration,
            "first_trigger": first_trigger,
            "resume_trigger": resume_trigger,
        })
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    t0 = time.time()
    inputs = load_inputs()
    pricer = make_pricer_v2("SPX")
    log.info("SPX pricer loaded in %.1fs", time.time() - t0)

    log.info("Running SPX halts_only IC over OOS [%s, %s]...", OOS_START, OOS_END)
    t = time.time()
    result = run_backtest(
        inputs, pricer, mode="halts_only",
        start=OOS_START, end=OOS_END, use_iron_condor=True,
    )
    log.info("Done in %.1fs. n_trades=%d", time.time() - t, len(result.trades))

    eq = result.equity_curve
    ret = eq.pct_change().dropna()
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else float("nan")
    final_eq = float(eq.iloc[-1])
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    max_dd = float((eq / eq.cummax() - 1).min())

    print()
    print("=" * 70)
    print("SPX halts_only OOS [%s, %s]" % (OOS_START, OOS_END))
    print("=" * 70)
    print(f"  n_trades:       {len(result.trades)}")
    print(f"  Final equity:   ${final_eq:,.0f}  (started $100,000)")
    print(f"  Total return:   {total_return*100:+.2f}%")
    print(f"  Ann. return:    {ann_return*100:+.2f}%")
    print(f"  Max drawdown:   {max_dd*100:+.2f}%")
    print(f"  Naive Sharpe:   {sharpe:.3f}   (anchor v1.5 halts_only = 0.286)")
    delta = sharpe - 0.286
    sign = "BEATS" if delta > 0 else "below"
    print(f"  Delta vs anchor: {delta:+.3f}  ({sign})")
    print()

    # Halt log analysis
    log_df = result.halt_log
    print("=" * 70)
    print("Halt log breakdown")
    print("=" * 70)
    print(f"  total log rows:          {len(log_df)}")
    print(f"  active days:             {(log_df['state'] == 'active').sum()}")
    print(f"  halted days:             {(log_df['state'] != 'active').sum()}")
    pct_halted = 100 * (log_df['state'] != 'active').mean()
    print(f"  halted %:                {pct_halted:.1f}%")
    print()
    print("  state value counts:")
    for state, n in log_df["state"].value_counts().items():
        print(f"    {state:20s}: {n}")
    print()
    print("  trigger value counts (top 10):")
    for trig, n in log_df["triggers"].value_counts().head(10).items():
        print(f"    {trig!r:40s}: {n}")
    print()

    halt_periods = find_halt_periods(log_df)
    print(f"  number of distinct halt periods: {len(halt_periods)}")
    if len(halt_periods) > 0:
        print(f"  median duration (trading days):  {int(halt_periods['duration_days'].median())}")
        print(f"  max duration:                    {int(halt_periods['duration_days'].max())}")
        print(f"  min duration:                    {int(halt_periods['duration_days'].min())}")
        print()
        print("  PERIOD               DURATION  FIRST TRIGGER                    RESUME TRIGGER")
        print("  " + "-" * 90)
        for _, row in halt_periods.iterrows():
            print(f"  {row['start'].date()} → {row['end'].date()}   "
                  f"{row['duration_days']:6d}d  "
                  f"{str(row['first_trigger'])[:30]:30s}   "
                  f"{str(row['resume_trigger'])[:30]:30s}")

    # Halt-active weeks per year
    print()
    print("  Halts-active weeks per calendar year:")
    log_df["year"] = log_df.index.year
    log_df["halted"] = (log_df["state"] != "active").astype(int)
    by_year = log_df.groupby("year")["halted"].sum() / 5  # ~5 trading days per week
    for year, weeks in by_year.items():
        bar = "█" * int(weeks)
        print(f"    {year}: {weeks:5.1f} weeks halted  {bar}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
