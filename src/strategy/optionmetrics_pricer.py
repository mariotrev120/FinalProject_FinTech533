"""
OptionMetrics IvyDB pricer — real OPRA quotes, no synthetic reconstruction.

Implements the `PricingProvider` Protocol. Plugs in as a drop-in replacement
for `BlackScholesPricer` in `make_default_pricer()`.

Data source: WRDS / OptionMetrics IvyDB US, pulled via the form-based query
(see `data/raw/optionmetrics/Options.csv.gz`).

The CSV is loaded once into a pandas DataFrame, filtered to drop illiquid
rows (best_bid <= 0 or delta is NaN), then indexed by (ticker, exdate, date)
for O(1) lookups. The index stores per-row best_bid, best_offer, delta, IV
so the engine can compute realistic round-trip execution prices instead of
synthesizing slippage from a heuristic.

Returns:
  - price_put(...)     -> mid price (legacy, used where pricer-protocol mid
                          is the simplest abstraction; engine prefers
                          quote_put for realistic execution)
  - implied_delta(...) -> the delta column directly
  - quote_put(...)     -> (bid, ask, delta, iv) for the nearest-strike row,
                          letting the engine compute realistic
                          short_bid - long_ask at entry and
                          short_ask - long_bid at exit

Bug-fix history (this build):
  - filter best_bid > 0 AND delta is not NaN at ingestion (Bug #3)
  - expose real bid/ask via quote_put for engine to use realistic
    execution prices (Bug #2)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from src.config import DATA_RAW_DIR


log = logging.getLogger(__name__)


class PutQuote(NamedTuple):
    bid: float
    ask: float
    mid: float
    delta: float
    iv: float
    strike: float


@dataclass
class OptionMetricsPricer:
    """Pluggable real-quote pricer.

    df: long-format DataFrame with columns: ticker, date, exdate, strike,
        cp_flag, best_bid, best_offer, delta, impl_volatility.
        Strike is in dollars (loader handles ×1000 conversion).
    underlying_filter: e.g. 'SPX', 'RUT' — filtered once at load time.

    Internally maintains separate (date, exdate) -> chain dicts for puts and
    calls. Critical for iron-condor support: a strike below spot has both an
    OTM put (delta ~ -0.16) and an ITM call (delta ~ +0.84) — the prior
    single-dict implementation would return whichever row sorted first,
    which is a real bug for iron condor entry-strike selection.
    """
    df: pd.DataFrame
    underlying_filter: str = "SPX"
    _by_date_exp_put: dict = field(default_factory=dict, repr=False)
    _by_date_exp_call: dict = field(default_factory=dict, repr=False)
    _drop_stats: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        sub = self.df[self.df["ticker"] == self.underlying_filter].copy()
        if len(sub) == 0:
            raise ValueError(
                f"No rows for ticker={self.underlying_filter} in OptionMetrics CSV. "
                f"Found tickers: {sorted(self.df['ticker'].unique())}"
            )
        n_total = len(sub)

        # Bug #3 fix: drop rows where bid is 0 (no buyer side) OR delta is NaN
        # (no implied vol could be computed). These are illiquid contracts the
        # engine should never trade on.
        bad_bid = sub["best_bid"] <= 0
        bad_delta = sub["delta"].isna()
        bad_iv = sub["impl_volatility"].isna()
        bad_ask = sub["best_offer"] <= 0
        drop_mask = bad_bid | bad_delta | bad_iv | bad_ask
        n_dropped = int(drop_mask.sum())

        if n_dropped > 0:
            drop_dates = sub.loc[drop_mask, "date"].dt.year.value_counts().sort_index()
            self._drop_stats = {
                "total_rows": n_total,
                "dropped_rows": n_dropped,
                "dropped_pct": 100.0 * n_dropped / n_total,
                "by_year": drop_dates.to_dict(),
            }
            log.info(
                "OptionMetricsPricer ingest filter: dropped %d/%d rows (%.2f%%)",
                n_dropped, n_total, 100.0 * n_dropped / n_total,
            )
            log.info("  drops by year: %s", dict(drop_dates))

        sub = sub[~drop_mask].copy()
        sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
        sub["exdate"] = pd.to_datetime(sub["exdate"]).dt.normalize()
        sub["mid"] = (sub["best_bid"] + sub["best_offer"]) / 2.0

        # Per-side groupby. cp_flag is "P" or "C".
        for cp_flag, target in [("P", self._by_date_exp_put),
                                  ("C", self._by_date_exp_call)]:
            side_sub = sub[sub["cp_flag"] == cp_flag]
            for (d, e), grp in side_sub.groupby(["date", "exdate"]):
                target[(pd.Timestamp(d), pd.Timestamp(e))] = (
                    grp[["strike", "best_bid", "best_offer", "mid",
                         "delta", "impl_volatility"]]
                       .sort_values("strike")
                       .reset_index(drop=True)
                )

        log.info(
            "OptionMetricsPricer loaded: %d rows after filtering, "
            "%d put-(date,exdate) groups, %d call-(date,exdate) groups, ticker=%s",
            len(sub), len(self._by_date_exp_put),
            len(self._by_date_exp_call), self.underlying_filter,
        )

    def _lookup_row(
        self, as_of: pd.Timestamp, expiry: date, strike: float,
        right: str = "P",
    ) -> Optional[pd.Series]:
        d = pd.Timestamp(as_of).normalize()
        e = pd.Timestamp(expiry).normalize()
        target = self._by_date_exp_put if right == "P" else self._by_date_exp_call
        grp = target.get((d, e))
        if grp is None or len(grp) == 0:
            return None
        idx = (grp["strike"] - strike).abs().idxmin()
        return grp.loc[idx]

    # --- PricingProvider Protocol -----------------------------------------

    def price_put(
        self,
        as_of: pd.Timestamp,
        underlying: str,    # noqa: ARG002
        spot: float,        # noqa: ARG002
        strike: float,
        expiry: date,
        vix: float,         # noqa: ARG002
    ) -> float:
        row = self._lookup_row(as_of, expiry, strike, right="P")
        if row is None:
            return 0.0
        mid = float(row["mid"])
        return mid if not np.isnan(mid) else 0.0

    def price_call(
        self,
        as_of: pd.Timestamp,
        underlying: str,    # noqa: ARG002
        spot: float,        # noqa: ARG002
        strike: float,
        expiry: date,
        vix: float,         # noqa: ARG002
    ) -> float:
        row = self._lookup_row(as_of, expiry, strike, right="C")
        if row is None:
            return 0.0
        mid = float(row["mid"])
        return mid if not np.isnan(mid) else 0.0

    def implied_delta(
        self,
        as_of: pd.Timestamp,
        underlying: str,    # noqa: ARG002
        spot: float,        # noqa: ARG002
        strike: float,
        expiry: date,
        vix: float,         # noqa: ARG002
    ) -> float:
        """Returns put delta. Negative for OTM/ATM puts."""
        row = self._lookup_row(as_of, expiry, strike, right="P")
        if row is None:
            return 0.0
        d = float(row["delta"])
        return d if not np.isnan(d) else 0.0

    def implied_call_delta(
        self,
        as_of: pd.Timestamp,
        underlying: str,    # noqa: ARG002
        spot: float,        # noqa: ARG002
        strike: float,
        expiry: date,
        vix: float,         # noqa: ARG002
    ) -> float:
        """Returns call delta. Positive for OTM/ATM calls."""
        row = self._lookup_row(as_of, expiry, strike, right="C")
        if row is None:
            return 0.0
        d = float(row["delta"])
        return d if not np.isnan(d) else 0.0

    # --- Realistic-quote methods (Bug #2 fix) -----------------------------

    def quote_put(
        self,
        as_of: pd.Timestamp,
        underlying: str,    # noqa: ARG002
        strike: float,
        expiry: date,
    ) -> Optional[PutQuote]:
        row = self._lookup_row(as_of, expiry, strike, right="P")
        if row is None:
            return None
        return PutQuote(
            bid=float(row["best_bid"]),
            ask=float(row["best_offer"]),
            mid=float(row["mid"]),
            delta=float(row["delta"]),
            iv=float(row["impl_volatility"]),
            strike=float(row["strike"]),
        )

    def quote_call(
        self,
        as_of: pd.Timestamp,
        underlying: str,    # noqa: ARG002
        strike: float,
        expiry: date,
    ) -> Optional[PutQuote]:
        """Same NamedTuple shape as quote_put — bid/ask/mid/delta/iv/strike
        for a call contract. Iron-condor exit logic uses this for the call
        side."""
        row = self._lookup_row(as_of, expiry, strike, right="C")
        if row is None:
            return None
        return PutQuote(
            bid=float(row["best_bid"]),
            ask=float(row["best_offer"]),
            mid=float(row["mid"]),
            delta=float(row["delta"]),
            iv=float(row["impl_volatility"]),
            strike=float(row["strike"]),
        )


# --- Loader ---------------------------------------------------------------

def load_optionmetrics_dataframe(
    path: str = "data/raw/optionmetrics/Options.csv.gz",
) -> pd.DataFrame:
    """Read an OptionMetrics CSV (gzipped), normalize strike scale,
    return a clean DataFrame.

    OptionMetrics quirks handled:
      - strike_price stored ×1000 (5500000 -> $5500). Divide.
      - dates as 'YYYY-MM-DD' strings; convert to datetime.
    """
    df = pd.read_csv(path)
    df["strike"] = df["strike_price"].astype(float) / 1000.0
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])
    return df


def make_optionmetrics_pricer(underlying: str = "SPX") -> OptionMetricsPricer:
    """Single-ticker pricer from the legacy puts-only file (kept for back-compat
    with v1.5 baseline reproduction). For new work use
    `make_pricer_v2(underlying)` which loads from strat2.2.csv.gz."""
    df = load_optionmetrics_dataframe()
    return OptionMetricsPricer(df=df, underlying_filter=underlying)


# --- v2 multi-instrument loader (per-ticker parquet, post-DuckDB conversion) ---

# After scripts/convert_optionmetrics_to_parquet.py runs once, each ticker
# lives in its own ~50-200 MB parquet at OPTIONS_BY_TICKER_DIR/{TICKER}.parquet.
# Loading one ticker is ~50-300 MB in memory and ~1s, vs the legacy 2GB CSV
# load that crashed WSL. The legacy CSV path is kept as a fallback below.

OPTIONS_BY_TICKER_DIR = "data/processed/options_by_ticker"

# Module-level per-ticker cache. Each entry is a small (~100MB) DataFrame.
_TICKER_DF_CACHE: dict[str, pd.DataFrame] = {}
# Legacy full-universe cache (ONLY populated if a caller explicitly invokes
# load_full_universe_dataframe with the CSV path — discouraged for new code).
_FULL_DF_CACHE: Optional[pd.DataFrame] = None


def load_ticker_dataframe(
    ticker: str,
    parquet_dir: str = OPTIONS_BY_TICKER_DIR,
) -> pd.DataFrame:
    """Load one ticker's options data from a per-ticker parquet.

    Pre-condition: scripts/convert_optionmetrics_to_parquet.py has been run
    at least once to produce {parquet_dir}/{ticker}.parquet. The parquet has
    the same columns as the original CSV plus a `strike` column (= strike_price
    / 1000.0) and date/exdate cast to DATE.

    Cached at module level: subsequent calls for the same ticker return the
    cached DataFrame.
    """
    if ticker in _TICKER_DF_CACHE:
        return _TICKER_DF_CACHE[ticker]
    path = f"{parquet_dir}/{ticker}.parquet"
    log.info("Loading per-ticker parquet: %s", path)
    df = pd.read_parquet(path)
    # Ensure pandas datetime dtype for downstream lookups (parquet DATE
    # round-trips as datetime64[ns] in pandas already, but be explicit).
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    if not pd.api.types.is_datetime64_any_dtype(df["exdate"]):
        df["exdate"] = pd.to_datetime(df["exdate"])
    log.info("Loaded %s: %d rows", ticker, len(df))
    _TICKER_DF_CACHE[ticker] = df
    return df


def load_full_universe_dataframe(
    path: str = "strat2.2.csv.gz",
) -> pd.DataFrame:
    """LEGACY full-universe loader: reads the 2GB CSV.

    Discouraged. Use load_ticker_dataframe(ticker) instead — it reads from
    per-ticker parquets that are ~50-200 MB each. This function is preserved
    only for backwards compatibility with v1 audit scripts that pre-date the
    conversion.

    14-ticker universe: SPX, RUT, NDX, TLT, GLD, AAPL, MSFT, GOOGL, JNJ, KO,
    PG, WMT, JPM, PEP. Both puts and calls.

    Cached at module level; subsequent calls return the cached DataFrame.
    """
    global _FULL_DF_CACHE
    if _FULL_DF_CACHE is not None:
        return _FULL_DF_CACHE
    log.warning(
        "Loading full universe via legacy CSV path: %s. "
        "This loads the entire 2GB file into memory and crashes WSL on "
        "constrained machines. Prefer load_ticker_dataframe(ticker).",
        path,
    )
    df = pd.read_csv(path)
    df["strike"] = df["strike_price"].astype(float) / 1000.0
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])
    log.info("Loaded %d rows, %d unique tickers", len(df), df["ticker"].nunique())
    _FULL_DF_CACHE = df
    return df


def make_pricer_v2(
    underlying: str,
    parquet_dir: str = OPTIONS_BY_TICKER_DIR,
) -> OptionMetricsPricer:
    """Construct a per-ticker pricer from the per-ticker parquet.

    Reads ONLY {parquet_dir}/{underlying}.parquet — the 14-ticker
    full-universe CSV is NOT loaded. ~1s and ~50-300 MB per ticker.
    """
    df = load_ticker_dataframe(underlying, parquet_dir=parquet_dir)
    return OptionMetricsPricer(df=df, underlying_filter=underlying)


def make_pricers_for_universe(
    underlyings: list[str],
    parquet_dir: str = OPTIONS_BY_TICKER_DIR,
) -> dict[str, OptionMetricsPricer]:
    """Build per-ticker pricers for a list of underlyings.

    Each pricer is constructed from its own per-ticker parquet — the
    14-ticker full-universe CSV is NOT loaded. Total memory scales with
    sum of per-ticker sizes (~1-2 GB for full 14-ticker basket, but
    typically only the 5 VRP or 9 wheel names are loaded at once).

    Returns {ticker: OptionMetricsPricer}. Tickers whose parquet file
    is missing are logged and skipped.
    """
    from pathlib import Path

    out: dict[str, OptionMetricsPricer] = {}
    for u in underlyings:
        path = Path(parquet_dir) / f"{u}.parquet"
        if not path.exists():
            log.warning(
                "Per-ticker parquet missing for %s at %s — skipping. "
                "Run scripts/convert_optionmetrics_to_parquet.py first.",
                u, path,
            )
            continue
        out[u] = make_pricer_v2(u, parquet_dir=parquet_dir)
    return out
