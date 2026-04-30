"""
Black-Scholes default option pricer.

Implements the `PricingProvider` protocol. Used as the fallback when no real
option quote source is wired up. Two layers:

1. Closed-form European put price under BS, using VIX/100 as the at-the-money
   IV proxy. (VIX is by construction the 30-day IV on SPX.)
2. Optional SKEW adjustment derived from the CBOE SKEW Index, which scales
   the IV used for OTM puts up relative to ATM in a way calibrated to the
   historical observed skew. This is critical because pure ATM-IV BS
   massively underprices OTM puts (it's the variance risk premium we are
   trying to harvest).

The skew adjustment is a heuristic — it is NOT a substitute for real OPRA
quotes. The writeup discloses this as a primary risk; the synthetic-pricing
limitation is the reason WRDS / Polygon / ORATS data should replace this
provider before final results are reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp, log, sqrt
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import DATA_RAW_DIR


def _years_to_expiry(as_of: pd.Timestamp, expiry: date) -> float:
    delta_days = (pd.Timestamp(expiry) - as_of).days
    return max(delta_days, 0) / 365.0


def _bs_put(spot: float, strike: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Return (put_price, put_delta) using closed-form BS.

    Both values per share (multiply by 100 for contract value)."""
    if T <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        intrinsic = max(strike - spot, 0.0)
        delta = -1.0 if spot < strike else 0.0
        return intrinsic, delta
    d1 = (log(spot / strike) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    put = strike * exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    delta = norm.cdf(d1) - 1.0   # put delta
    return put, delta


@dataclass
class BlackScholesPricer:
    """Default fallback pricer.

    risk_free_curve: pd.Series of TNX yield (in tenths of percent, e.g. 43.96
        for 4.396%) indexed by trading date. Used as r in BS.
    skew_index: pd.Series of CBOE SKEW Index values indexed by trading date,
        or None to disable the skew adjustment.
    skew_slope_per_sigma: how much extra IV (in vol points) to add per 1 sigma
        OTM at SKEW=130 baseline. Default 4.0 means a 1-sigma OTM put gets
        4 vol-points of additional IV at the average historical skew.
    """
    risk_free_curve: pd.Series
    skew_index: Optional[pd.Series] = None
    skew_slope_per_sigma: float = 4.0
    skew_baseline: float = 130.0   # CBOE SKEW long-run mean

    def _r_for(self, as_of: pd.Timestamp) -> float:
        # CBOE 10x indices: TNX=43.96 means 4.396% annualized
        try:
            rate_pct = float(self.risk_free_curve.asof(as_of)) / 10.0 / 100.0
        except (KeyError, ValueError):
            rate_pct = 0.04   # fallback if pre-history
        if np.isnan(rate_pct):
            rate_pct = 0.04
        return rate_pct

    def _iv_for(
        self, as_of: pd.Timestamp, spot: float, strike: float, T: float, vix: float,
    ) -> float:
        """Compute IV to use, including optional skew adjustment.

        Sigma_atm = vix / 100. If a SKEW provider is attached and the strike
        is OTM (strike < spot for puts), bump sigma by skew_slope * |sigma_offset|
        where sigma_offset = (spot - strike) / (spot * sigma_atm * sqrt(T))
        is the strike's distance from spot in standard deviations.
        """
        sigma_atm = max(vix / 100.0, 1e-4)
        if self.skew_index is None or T <= 0:
            return sigma_atm
        try:
            skew_val = float(self.skew_index.asof(as_of))
        except (KeyError, ValueError):
            return sigma_atm
        if np.isnan(skew_val):
            return sigma_atm
        # Approximate moneyness in sigmas
        sigma_offset = max((spot - strike), 0.0) / (spot * sigma_atm * sqrt(T))
        skew_scale = (skew_val - 100.0) / (self.skew_baseline - 100.0)  # 1.0 at baseline
        bump_vol_pts = self.skew_slope_per_sigma * skew_scale * sigma_offset
        return sigma_atm + bump_vol_pts / 100.0

    def price_put(
        self,
        as_of: pd.Timestamp,
        underlying: str,                      # noqa: ARG002 (kept for interface)
        spot: float,
        strike: float,
        expiry: date,
        vix: float,
    ) -> float:
        T = _years_to_expiry(as_of, expiry)
        r = self._r_for(as_of)
        sigma = self._iv_for(as_of, spot, strike, T, vix)
        price, _ = _bs_put(spot, strike, T, r, sigma)
        return price

    def implied_delta(
        self,
        as_of: pd.Timestamp,
        underlying: str,                      # noqa: ARG002
        spot: float,
        strike: float,
        expiry: date,
        vix: float,
    ) -> float:
        T = _years_to_expiry(as_of, expiry)
        r = self._r_for(as_of)
        sigma = self._iv_for(as_of, spot, strike, T, vix)
        _, delta = _bs_put(spot, strike, T, r, sigma)
        return delta


def make_default_pricer() -> BlackScholesPricer:
    """Construct a pricer with TNX as the risk-free curve and SKEW for the
    skew adjustment, both loaded from data/raw/."""
    tnx = pd.read_parquet(DATA_RAW_DIR / "TNX.parquet")["close"]
    try:
        skew = pd.read_parquet(DATA_RAW_DIR / "SKEW.parquet")["close"]
    except FileNotFoundError:
        skew = None
    return BlackScholesPricer(risk_free_curve=tnx, skew_index=skew)
