"""
Parity + behavior tests for the per-ticker parquet loader.

Verifies that scripts/convert_optionmetrics_to_parquet.py produced parquets
that match the original CSV's row counts per ticker, that the schema has
the columns downstream code depends on, and that load_ticker_dataframe +
make_pricer_v2 + make_pricers_for_universe work end-to-end without
loading the 2GB CSV.

These tests skip cleanly if the parquets aren't on disk (CI / fresh
checkouts where the conversion hasn't been run yet).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.strategy.optionmetrics_pricer import (
    OPTIONS_BY_TICKER_DIR,
    OptionMetricsPricer,
    load_ticker_dataframe,
    make_pricer_v2,
    make_pricers_for_universe,
)


VRP_UNIVERSE = ["SPX", "RUT", "NDX", "TLT", "GLD"]
WHEEL_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "JNJ", "KO", "PG", "WMT", "JPM", "PEP"]
ALL_14 = VRP_UNIVERSE + WHEEL_UNIVERSE

# Expected row counts from the conversion script's parity check
EXPECTED_ROWS = {
    "AAPL": 2_491_630, "GLD": 3_545_675, "GOOGL": 4_624_094, "JNJ": 1_387_153,
    "JPM": 1_587_732, "KO": 1_212_308, "MSFT": 1_882_157, "NDX": 15_451_063,
    "PEP": 1_327_336, "PG": 1_452_732, "RUT": 8_841_879, "SPX": 20_657_223,
    "TLT": 2_419_203, "WMT": 1_474_655,
}

REQUIRED_COLUMNS = {
    "ticker", "date", "exdate", "cp_flag", "strike_price", "strike",
    "best_bid", "best_offer", "impl_volatility", "delta",
}


def _parquets_exist() -> bool:
    return all(
        (Path(OPTIONS_BY_TICKER_DIR) / f"{t}.parquet").exists()
        for t in ALL_14
    )


@pytest.fixture(scope="module")
def require_parquets():
    if not _parquets_exist():
        pytest.skip(
            "Per-ticker parquets not present. Run "
            "`scripts/convert_optionmetrics_to_parquet.py` first."
        )


@pytest.fixture(autouse=True)
def _clear_ticker_cache():
    """Clear the module-level ticker cache before AND after each test so
    memory doesn't accumulate across the parametrized matrix (each ticker
    is 50-750 MB in pandas; SPX+RUT+NDX combined > 7 GB).
    Tests should not depend on cross-test cache state."""
    from src.strategy.optionmetrics_pricer import _TICKER_DF_CACHE
    _TICKER_DF_CACHE.clear()
    yield
    _TICKER_DF_CACHE.clear()
    import gc
    gc.collect()


# --- Parity: row counts ---------------------------------------------------

@pytest.mark.parametrize("ticker", ALL_14)
def test_per_ticker_row_count_matches_csv(require_parquets, ticker):
    df = load_ticker_dataframe(ticker)
    assert len(df) == EXPECTED_ROWS[ticker], (
        f"{ticker}: parquet has {len(df)} rows, expected {EXPECTED_ROWS[ticker]} "
        f"(parity vs original CSV)"
    )


def test_total_rows_match_csv_grand_total(require_parquets):
    """Sum across all 14 tickers must equal the CSV's 68,354,840.

    Uses DuckDB to count rows directly from disk instead of materializing
    into pandas — keeps memory bounded (loading all 14 into pandas at once
    is ~8-10 GB and OOMs WSL)."""
    import duckdb
    con = duckdb.connect()
    total = 0
    for ticker in ALL_14:
        n = con.execute(
            f"SELECT COUNT(*) FROM read_parquet("
            f"'{OPTIONS_BY_TICKER_DIR}/{ticker}.parquet')"
        ).fetchone()[0]
        total += n
    assert total == sum(EXPECTED_ROWS.values()) == 68_354_840


# --- Schema ---------------------------------------------------------------

@pytest.mark.parametrize("ticker", ALL_14)
def test_schema_has_required_columns(require_parquets, ticker):
    df = load_ticker_dataframe(ticker)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"{ticker} parquet missing required columns: {missing}"


@pytest.mark.parametrize("ticker", ["SPX", "AAPL"])  # spot-check 2 tickers
def test_strike_column_is_strike_price_div_1000(require_parquets, ticker):
    """`strike` was added during conversion as strike_price / 1000."""
    df = load_ticker_dataframe(ticker).head(10000)
    diff = (df["strike"] - df["strike_price"].astype(float) / 1000.0).abs()
    assert diff.max() < 1e-9, (
        f"{ticker}: strike != strike_price/1000 in first 10k rows"
    )


@pytest.mark.parametrize("ticker", ALL_14)
def test_date_columns_are_datetime(require_parquets, ticker):
    df = load_ticker_dataframe(ticker)
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), (
        f"{ticker}: 'date' column is not datetime64 — got {df['date'].dtype}"
    )
    assert pd.api.types.is_datetime64_any_dtype(df["exdate"]), (
        f"{ticker}: 'exdate' column is not datetime64 — got {df['exdate'].dtype}"
    )


# --- Date range coverage --------------------------------------------------

@pytest.mark.parametrize("ticker", ["SPX", "AAPL", "GLD"])
def test_date_range_spans_oos(require_parquets, ticker):
    """All tickers must cover at least 2018-2024 (the OOS period)."""
    df = load_ticker_dataframe(ticker)
    assert df["date"].min() <= pd.Timestamp("2018-01-01")
    assert df["date"].max() >= pd.Timestamp("2024-12-31")


# --- Loader behavior ------------------------------------------------------

def test_make_pricer_v2_returns_optionmetrics_pricer(require_parquets):
    pricer = make_pricer_v2("SPX")
    assert isinstance(pricer, OptionMetricsPricer)
    # Pricer must have both put and call lookup dicts populated
    assert len(pricer._by_date_exp_put) > 0
    assert len(pricer._by_date_exp_call) > 0


def test_make_pricers_for_universe_smoke(require_parquets):
    """Use the 2 smallest tickers (TLT, GLD) to verify the loop without
    OOMing the test runner. Loading all 5 VRP simultaneously is ~8 GB
    in pandas — exercised by the actual multi-instrument runner, not by
    this unit test."""
    pricers = make_pricers_for_universe(["TLT", "GLD"])
    assert set(pricers.keys()) == {"TLT", "GLD"}
    for ticker, pricer in pricers.items():
        assert isinstance(pricer, OptionMetricsPricer)


def test_make_pricers_for_universe_skips_missing_ticker(require_parquets, tmp_path):
    """If a ticker's parquet doesn't exist, make_pricers_for_universe
    logs a warning and skips it without crashing."""
    pricers = make_pricers_for_universe(["SPX", "NONEXISTENT_TICKER"])
    assert set(pricers.keys()) == {"SPX"}


def test_loader_caches_per_ticker(require_parquets):
    """Calling load_ticker_dataframe twice for the same ticker should
    return the cached DataFrame (object identity).

    Uses KO (smallest ticker, 1.2M rows ~45MB) to keep memory low."""
    df1 = load_ticker_dataframe("KO")
    df2 = load_ticker_dataframe("KO")
    assert df1 is df2
