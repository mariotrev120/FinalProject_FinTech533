"""
Per-stress-event analysis.

For each of five named stress events, run all 4 modes through a window
straddling the event, and ask: did the mode SURVIVE? Specifically:
  - Did halts fire BEFORE the worst of the event? (halts_only / full)
  - What was the realized P&L through the event window?
  - Drawdown depth and recovery duration?

Events (per README):
  Aug 2015: China devaluation (2015-08-15 -> 2015-09-30)
  Feb 2018: Volmageddon (2018-01-15 -> 2018-03-15)
  Q4 2018: equity selloff (2018-10-01 -> 2018-12-31)
  Feb-Mar 2020: COVID — the critical test (2020-02-15 -> 2020-04-30)
  Mar 2023: regional banking crisis (2023-03-01 -> 2023-04-15)

Note: the IS window ends 2017-12, so the Aug 2015 event is INSIDE the IS
window — its "survival" results are an in-sample diagnostic, not an OOS
test of the halt rules.

Usage:
    PYTHONPATH=. .venv/bin/python -m src.backtest.stress_events
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.loader import load_inputs
from src.config import DATA_PROCESSED_DIR
from src.strategy.black_scholes import make_default_pricer
from src.strategy.types import Mode


log = logging.getLogger(__name__)


EVENTS: list[tuple[str, str, str, str]] = [
    # (label, peak_date_for_alignment, window_start, window_end)
    ("Aug2015_china",     "2015-08-24", "2015-07-01", "2015-09-30"),
    ("Feb2018_volmagedon","2018-02-05", "2018-01-15", "2018-03-15"),
    ("Q4_2018_selloff",   "2018-12-24", "2018-10-01", "2018-12-31"),
    ("Mar2020_covid",     "2020-03-23", "2020-02-15", "2020-04-30"),
    ("Mar2023_banking",   "2023-03-13", "2023-03-01", "2023-04-15"),
]


def first_halt_fire_date(result: BacktestResult, before: pd.Timestamp) -> pd.Timestamp | None:
    """Return the first date where the halt log shows a non-active state at
    or before the peak date. None if halts never fired in this window."""
    if result.halt_log.empty:
        return None
    pre_peak = result.halt_log.loc[:before]
    fired = pre_peak[pre_peak["state"] != "active"]
    if fired.empty:
        return None
    return fired.index[0]


def event_summary(result: BacktestResult, peak: pd.Timestamp) -> dict:
    eq = result.equity_curve
    if len(eq) == 0:
        return {"trades": 0, "pnl": 0.0, "max_dd_pct": 0.0,
                "halt_fire_date": None, "halt_pre_peak_days": None,
                "ending_equity": float("nan")}
    pnls = [t.total_pnl for t in result.trades if t.total_pnl is not None]
    drawdown = (eq.cummax() - eq) / eq.cummax()
    halt_fire = first_halt_fire_date(result, peak) if not result.halt_log.empty else None
    days_before = (peak - halt_fire).days if halt_fire is not None else None
    return {
        "trades": len(pnls),
        "pnl": float(np.sum(pnls)) if pnls else 0.0,
        "max_dd_pct": float(drawdown.max() * 100) if len(drawdown) > 0 else 0.0,
        "halt_fire_date": halt_fire.date() if halt_fire is not None else None,
        "halt_pre_peak_days": days_before,
        "ending_equity": float(eq.iloc[-1]),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    inputs = load_inputs()
    pricer = make_default_pricer()

    rows = []
    modes: list[Mode] = ["naked", "ml_only", "halts_only", "full"]
    for label, peak_str, start, end in EVENTS:
        peak = pd.Timestamp(peak_str)
        for mode in modes:
            log.info("[%s] mode=%s window=%s..%s", label, mode, start, end)
            try:
                r = run_backtest(inputs, pricer, mode=mode, start=start, end=end)
            except Exception as e:
                log.error("  failed: %s", e)
                continue
            s = event_summary(r, peak)
            s["event"] = label
            s["mode"] = mode
            rows.append(s)

    df = pd.DataFrame(rows)
    out_path = DATA_PROCESSED_DIR / "stress_events.parquet"
    df.to_parquet(out_path)
    csv_path = DATA_PROCESSED_DIR / "stress_events.csv"
    df.to_csv(csv_path, index=False)
    log.info("wrote %s and %s", out_path, csv_path)

    print("\n" + "=" * 90)
    print("STRESS EVENT SURVIVAL TABLE")
    print("=" * 90)
    pivot = df.pivot_table(
        index="event", columns="mode",
        values=["trades", "pnl", "max_dd_pct", "halt_pre_peak_days"],
        aggfunc="first",
    )
    print(pivot)

    # COVID is the critical test
    print("\n" + "=" * 90)
    print("COVID Mar 2020 — the single most important test")
    print("=" * 90)
    covid = df[df.event == "Mar2020_covid"]
    for _, row in covid.iterrows():
        halt_msg = (f"halt fired {row['halt_pre_peak_days']} days before peak"
                    if row['halt_pre_peak_days'] is not None
                    else "no halt fired")
        print(f"  {row['mode']:11s}: {int(row['trades'])} trades, "
              f"pnl=${row['pnl']:+.0f}, max_dd={row['max_dd_pct']:.2f}%  ({halt_msg})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
