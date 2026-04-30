"""
Treasury yield-curve shape features.

Pulled out of `exogenous.py` per Robert's structure (one module per feature
group, easier per-function tests). The polynomial fit is over log(tenor) to
keep coefficients well-conditioned across the {3M, 5Y, 10Y, 30Y} span.

Note on README deviation: the original spec was a degree-3 polynomial over
{3M, 2Y, 5Y, 10Y, 30Y} → 4 coefficients. TWS does not expose a 2Y CBOE
yield index, so we fit degree-2 over the 4 tenors we have, producing 3
coefficients (level, slope, curvature). This is disclosed in the writeup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# CBOE 10x yield indices: TNX=43.96 means 4.396%, but here we operate on the
# raw "10x" values directly — the polynomial fit is scale-invariant and the
# coefficients are interpretable relative to the input scale.
TENORS_YEARS: dict[str, float] = {
    "IRX": 0.25,    # 13-week T-bill
    "FVX": 5.0,     # 5-year T-note
    "TNX": 10.0,    # 10-year T-note
    "TYX": 30.0,    # 30-year T-bond
}


def yield_curve_coefficients(yields: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    """Fit a polynomial in log(tenor_years) to the yield curve at each date.

    Returns a DataFrame with one column per polynomial coefficient
    (`yc_c0`, `yc_c1`, ...) sized at degree+1.

    yields: DataFrame indexed by date, columns from TENORS_YEARS keys.
    """
    cols = list(yields.columns)
    log_tenors = np.log(np.array([TENORS_YEARS[c] for c in cols]))

    coefs = np.full((len(yields), degree + 1), np.nan)
    for i, (_, row) in enumerate(yields.iterrows()):
        y = row.values.astype(float)
        if np.any(np.isnan(y)):
            continue
        coefs[i] = np.polyfit(log_tenors, y, degree)

    out = pd.DataFrame(
        coefs,
        index=yields.index,
        columns=[f"yc_c{i}" for i in range(degree + 1)],
    )
    return out
