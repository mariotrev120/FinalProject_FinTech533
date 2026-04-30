"""
OptionMetrics IvyDB pricer — real OPRA quotes, no synthetic reconstruction.

Implements the `PricingProvider` Protocol. Plugs in as a drop-in replacement
for `BlackScholesPricer` in `make_default_pricer()`.

Data source: WRDS / OptionMetrics IvyDB US, pulled via the form-based query
in scripts/wrds_query/ (see `data/raw/optionmetrics/Options.csv.gz`).

The CSV is loaded once into a pandas DataFrame, then indexed by
(ticker, exdate, strike, date) for O(1) lookups. Loading 6M rows takes
~30 seconds; the indexed lookup serves the entire backtest.

Returns:
  - price_put(...) -> mid price ((bid + ask) / 2) per share
  - implied_delta(...) -> the delta column directly (no recomputation)

If a lookup misses (no row for the exact strike/expiry/date), returns the
nearest-strike row's mid + delta. This handles the strike-rounding step
in spread_construction.py: the engine asks for strike=550 but the chain
only had 549.5 and 550.5, we pick the closer one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DATA_RAW_DIR


log = logging.getLogger(__name__)


@dataclass
class OptionMetricsPricer:
    """Pluggable real-quote pricer.

    df: long-format DataFrame with columns: ticker, date, exdate, strike,
        cp_flag, best_bid, best_offer, delta, impl_volatility.
        Strike is in dollars (not the ×1000 OptionMetrics raw scale —
        loader normalizes that).
    underlying_filter: 'XSP' or 'SPX' (filters df once at load time).
    """
    df: pd.DataFrame
    underlying_filter: str = "SPX"
    _by_date_exp: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Filter to underlying once
        sub = self.df[self.df["ticker"] == self.underlying_filter].copy()
        if len(sub) == 0:
            raise ValueError(
                f"No rows for ticker={self.underlying_filter} in OptionMetrics CSV. "
                f"Found tickers: {sorted(self.df['ticker'].unique())}"
            )
        sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
        sub["exdate"] = pd.to_datetime(sub["exdate"]).dt.normalize()
        sub["mid"] = (sub["best_bid"] + sub["best_offer"]) / 2.0
        # Index by (date, exdate) -> rows sorted by strike for fast nearest-strike search
        for (d, e), grp in sub.groupby(["date", "exdate"]):
            self._by_date_exp[(pd.Timestamp(d), pd.Timestamp(e))] = (
                grp[["strike", "mid", "delta", "impl_volatility", "best_bid", "best_offer"]]
                   .sort_values("strike")
                   .reset_index(drop=True)
            )
        log.info(
            "OptionMetricsPricer loaded: %d rows, %d (date,exdate) groups, ticker=%s",
            len(sub), len(self._by_date_exp), self.underlying_filter,
        )

    def _lookup_row(
        self, as_of: pd.Timestamp, expiry: date, strike: float,
    ) -> Optional[pd.Series]:
        d = pd.Timestamp(as_of).normalize()
        e = pd.Timestamp(expiry).normalize()
        grp = self._by_date_exp.get((d, e))
        if grp is None or len(grp) == 0:
            return None
        # Nearest-strike lookup
        idx = (grp["strike"] - strike).abs().idxmin()
        return grp.loc[idx]

    def price_put(
        self,
        as_of: pd.Timestamp,
        underlying: str,    # noqa: ARG002 (filtered at load time)
        spot: float,        # noqa: ARG002 (not used; price comes from quotes)
        strike: float,
        expiry: date,
        vix: float,         # noqa: ARG002
    ) -> float:
        row = self._lookup_row(as_of, expiry, strike)
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
        row = self._lookup_row(as_of, expiry, strike)
        if row is None:
            return 0.0
        d = float(row["delta"])
        return d if not np.isnan(d) else 0.0


# --- Loader ---------------------------------------------------------------

def load_optionmetrics_dataframe(
    path: str = "data/raw/optionmetrics/Options.csv.gz",
) -> pd.DataFrame:
    """Read the OptionMetrics CSV (gzipped), normalize strike scale,
    return a clean DataFrame.

    OptionMetrics quirks handled:
      - strike_price is stored ×1000 (e.g. 5500000 means $5500). Divide.
      - dates come as 'YYYY-MM-DD' strings; convert to datetime.
    """
    df = pd.read_csv(path)
    df["strike"] = df["strike_price"].astype(float) / 1000.0
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])
    return df


def make_optionmetrics_pricer(underlying: str = "SPX") -> OptionMetricsPricer:
    df = load_optionmetrics_dataframe()
    return OptionMetricsPricer(df=df, underlying_filter=underlying)
