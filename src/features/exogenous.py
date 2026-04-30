"""
Exogenous feature pipeline.

Builds the Vestal-style exogenous feature vector from raw IBKR-pulled bars.
Every feature value at row t uses only data with timestamp <= t close (the
test_leakage.py suite enforces this at runtime).

Feature groups (matches README):
1. Volatility regime: VIX, VIX3M, VIX3M-VIX, VVIX (4)
2. Variance risk premium: VIX - 30d realized vol on SPY, and 60d window (2)
3. Yield curve shape: polynomial coefficients fit to {IRX, FVX, TNX, TYX}
   (3 coefficients via degree-2 fit — TWS does not expose a 2Y index, so
   the README's degree-3 / 4-coefficient design is reduced to degree-2 / 3
   coefficients across the 4 tenors we have. This deviation is disclosed in
   the writeup.)
4. Cross-asset stress: SPY-TLT 20d corr, SPY-GLD 20d corr, HYG-LQD spread (3)
5. Broad market regime: 20d SPY return, SPY pct from 200d MA, 50/200 MA cross (3)

Total: 15 features. Loadable as a single DataFrame indexed by date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_RAW_DIR


# --- IO --------------------------------------------------------------------

def load_raw(symbol: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_RAW_DIR / f"{symbol}.parquet")


def load_close(symbol: str) -> pd.Series:
    df = load_raw(symbol)
    s = df["close"].copy()
    s.name = symbol
    return s


# --- Realized volatility ---------------------------------------------------

def realized_vol(close: pd.Series, window: int = 30) -> pd.Series:
    """Annualized realized vol from log returns over a rolling window. The
    return at row t uses returns through t-1 -> t close, so it's safe to use
    in a feature vector evaluated at t close."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252) * 100.0  # in vol points


# --- Yield curve fit -------------------------------------------------------

# CBOE yield indices are quoted in tenths of percent (e.g. TNX=43.96 means
# 4.396%). The polyfit is over the LOG of tenor in years to compress the long
# end and keep the polynomial well-conditioned.
_YIELD_TENORS_YR: dict[str, float] = {
    "IRX": 0.25,   # 13-week T-bill
    "FVX": 5.0,    # 5-year T-note
    "TNX": 10.0,   # 10-year T-note
    "TYX": 30.0,   # 30-year T-bond
}


def yield_curve_coefficients(yields: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    """Fit a polynomial in log(tenor_years) to the yield curve at each date.

    Returns a DataFrame with one column per polynomial coefficient
    (`yc_c0`, `yc_c1`, ...) sized at degree+1.

    yields: DataFrame indexed by date, columns from _YIELD_TENORS_YR keys.
    """
    cols = list(yields.columns)
    log_tenors = np.log(np.array([_YIELD_TENORS_YR[c] for c in cols]))

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


# --- Main builder ---------------------------------------------------------

def build_features() -> pd.DataFrame:
    """Build the full exogenous feature DataFrame, indexed by trading date.

    Every feature value at row t is computable from data with timestamp <= t
    close. (Test suite enforces this on synthetic data.)
    """
    # Load all closes
    closes = pd.concat(
        [load_close(s) for s in
         ["VIX", "VIX3M", "VVIX", "SPY", "GLD", "HYG", "LQD",
          "IRX", "FVX", "TNX", "TYX"]],
        axis=1, sort=False,
    )
    closes.columns = ["VIX", "VIX3M", "VVIX", "SPY", "GLD", "HYG", "LQD",
                      "IRX", "FVX", "TNX", "TYX"]

    feat = pd.DataFrame(index=closes.index)

    # Group 1: volatility regime
    feat["vix"] = closes["VIX"]
    feat["vix3m"] = closes["VIX3M"]
    feat["vix3m_minus_vix"] = closes["VIX3M"] - closes["VIX"]
    feat["vvix"] = closes["VVIX"]

    # Group 2: variance risk premium (VIX - realized vol over matched horizon)
    rv30 = realized_vol(closes["SPY"], window=30)
    rv60 = realized_vol(closes["SPY"], window=60)
    feat["vrp_30d"] = closes["VIX"] - rv30
    feat["vrp_60d"] = closes["VIX"] - rv60

    # Group 3: yield curve shape (degree-2 over 4 tenors -> 3 coefficients)
    yc = yield_curve_coefficients(
        closes[["IRX", "FVX", "TNX", "TYX"]], degree=2
    )
    feat = feat.join(yc)

    # Group 4: cross-asset stress
    # SPY-TLT corr is the canonical stocks-vs-Treasuries indicator, but TWS
    # paper caps TLT history at 2016-02-03. To preserve a 2011-05 start, we
    # use the 10Y yield (TNX) change as a Treasury proxy: when SPY and TNX
    # changes correlate positively, both stocks are selling off and yields
    # are rising, the same regime signal as positive SPY-TLT correlation.
    spy_ret = closes["SPY"].pct_change()
    tnx_chg = closes["TNX"].diff()
    gld_ret = closes["GLD"].pct_change()
    feat["spy_tnx_corr_20d"] = spy_ret.rolling(20).corr(tnx_chg)
    feat["spy_gld_corr_20d"] = spy_ret.rolling(20).corr(gld_ret)
    feat["hyg_lqd_spread"] = closes["HYG"] - closes["LQD"]

    # Group 5: broad market regime
    feat["spy_ret_20d"] = closes["SPY"].pct_change(20)
    ma200 = closes["SPY"].rolling(200).mean()
    feat["spy_dist_200ma_pct"] = (closes["SPY"] - ma200) / ma200
    ma50 = closes["SPY"].rolling(50).mean()
    feat["ma50_above_ma200"] = (ma50 > ma200).astype(int)

    # Drop the warmup region where rolling windows haven't filled
    feat = feat.dropna()

    return feat


def main() -> int:
    feat = build_features()
    out_path = DATA_RAW_DIR.parent / "processed" / "features.parquet"
    feat.to_parquet(out_path)
    print(f"features shape: {feat.shape}")
    print(f"date range: {feat.index.min().date()} -> {feat.index.max().date()}")
    print(f"columns: {list(feat.columns)}")
    print(f"saved to {out_path}")
    print("\nlast row:")
    print(feat.iloc[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
