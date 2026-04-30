"""
Put-call parity validation for option chain data.

STATUS: stub. Activated when real OPRA-grade option chain data lands (WRDS /
Polygon / ORATS). With the current synthetic Black-Scholes pricer, parity
holds by construction (BS prices satisfy parity exactly), so this validator
is a no-op until real-source quotes are in scope.

When real chains land, this module will:
  - Iterate every (date, expiry, strike) row
  - Check |C(K) - P(K) - (S - K * exp(-rT))| < tolerance
  - Flag rows that fail (typically stale or wrong-timestamp ticks)
  - Save a per-date violation report to data/processed/parity_violations.parquet

The Merton 1973 Theorem 12 reference in the writeup grounds this check.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def validate_put_call_parity(
    chains: pd.DataFrame,
    spot_series: pd.Series,
    risk_free_series: pd.Series,
    tolerance: float = 0.05,
) -> Optional[pd.DataFrame]:
    """Check put-call parity on a chain DataFrame.

    Args:
        chains: DataFrame with columns ['date', 'expiry', 'strike',
                'call_mid', 'put_mid'].
        spot_series: pd.Series indexed by date, value = underlying spot.
        risk_free_series: pd.Series of TNX-style 10x yield (so 43.96 = 4.396%).
        tolerance: max allowed parity deviation in dollars.

    Returns:
        DataFrame of violations, or None if no chains data is supplied
        (current synthetic-pricing state).
    """
    if chains is None or len(chains) == 0:
        return None
    raise NotImplementedError(
        "validate_put_call_parity is a stub. Implement when real OPRA chain "
        "data is available. Synthetic BS prices satisfy parity by construction."
    )
