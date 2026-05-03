"""Per-ticker bar loading and engine spot-resolution tests.

Spec test: load_underlying_bars(ticker) returns OHLC bars indexed by date
with the correct close on a known reference date.

Engine spec test: when run_backtest is called with inputs.underlying_bars
set to a per-ticker bar series, the trades' entry_spx field reflects the
PER-TICKER spot, not SPX.
"""
from __future__ import annotations

import pandas as pd
import pytest


def test_load_underlying_bars_returns_correct_spot():
    """SPX/TLT/GLD on 3 reference dates each, asserted against published values."""
    from src.backtest.loader import load_underlying_bars

    expected = {
        "SPX": [("2018-04-16", 2670, 2685),    # actual 2677.84
                ("2020-03-23", 2200, 2270),    # actual 2237.40 (COVID low)
                ("2024-12-31", 5870, 5890)],   # actual 5881.63
        "TLT": [("2018-04-16", 119, 122),      # actual 120.94
                ("2020-03-23", 165, 167),      # actual 166.00
                ("2024-12-31", 87, 88)],       # actual 87.33
        "GLD": [("2018-04-16", 127, 128),      # actual 127.63
                ("2020-03-23", 145, 147),      # actual 146.30
                ("2024-12-31", 241, 243)],     # actual 242.13
    }

    for ticker, cases in expected.items():
        bars = load_underlying_bars(ticker)
        assert "close" in bars.columns, f"{ticker} bars missing close column"
        assert isinstance(bars.index, pd.DatetimeIndex), f"{ticker} bars not DatetimeIndex"
        for date_str, lo, hi in cases:
            ts = pd.Timestamp(date_str)
            assert ts in bars.index, f"{ticker} missing {date_str}"
            close = float(bars.loc[ts, "close"])
            assert lo <= close <= hi, (
                f"{ticker} {date_str}: close=${close:.2f} not in [${lo}, ${hi}]"
            )


def test_load_underlying_bars_unknown_ticker_raises():
    from src.backtest.loader import load_underlying_bars
    with pytest.raises((FileNotFoundError, ValueError)):
        load_underlying_bars("NOPE")


def test_engine_uses_underlying_bars_for_spot_when_set():
    """When inputs.underlying_bars is set to TLT bars, the engine's recorded
    entry_spx for trades should match TLT spot (not SPX)."""
    from src.backtest.engine import run_backtest
    from src.backtest.loader import load_inputs, load_underlying_bars
    from src.strategy.optionmetrics_pricer import make_pricer_v2

    inputs = load_inputs()
    inputs.underlying_bars = load_underlying_bars("TLT")
    pricer = make_pricer_v2("TLT")

    # Run a 3-month slice (smoke test)
    result = run_backtest(
        inputs, pricer, mode="naked",
        start="2020-01-01", end="2020-04-01", use_iron_condor=True,
    )
    if not result.trades:
        pytest.skip("No trades in smoke window — engine wiring covered by other tests")

    # Sanity: every trade's entry_spx should be in TLT range, not SPX range
    for t in result.trades[:5]:
        assert 100.0 <= t.entry_spx <= 200.0, (
            f"trade entry_spx={t.entry_spx} not in TLT range; engine probably "
            f"still hardcoded to SPX bars"
        )
