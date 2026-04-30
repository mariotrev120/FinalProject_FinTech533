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
    underlying_filter: 'SPX' or 'XSP' — filtered once at load time.
    """
    df: pd.DataFrame
    underlying_filter: str = "SPX"
    _by_date_exp: dict = field(default_factory=dict, repr=False)
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

        # Date distribution of drops (for audit logging)
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

        for (d, e), grp in sub.groupby(["date", "exdate"]):
            self._by_date_exp[(pd.Timestamp(d), pd.Timestamp(e))] = (
                grp[["strike", "best_bid", "best_offer", "mid", "delta", "impl_volatility"]]
                   .sort_values("strike")
                   .reset_index(drop=True)
            )
        log.info(
            "OptionMetricsPricer loaded: %d rows after filtering, %d (date,exdate) groups, ticker=%s",
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

    # --- Realistic-quote method (Bug #2 fix) ------------------------------

    def quote_put(
        self,
        as_of: pd.Timestamp,
        underlying: str,    # noqa: ARG002
        strike: float,
        expiry: date,
    ) -> Optional[PutQuote]:
        """Return the full quote (bid, ask, mid, delta, iv, snapped_strike)
        for the nearest-strike row. Engine uses this to compute realistic
        execution prices instead of synthesizing slippage."""
        row = self._lookup_row(as_of, expiry, strike)
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
    """Read the OptionMetrics CSV (gzipped), normalize strike scale,
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
    df = load_optionmetrics_dataframe()
    return OptionMetricsPricer(df=df, underlying_filter=underlying)
