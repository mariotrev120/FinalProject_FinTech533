"""
Exogenous feature pipeline — master assembler.

Builds the Vestal-style exogenous feature vector from raw IBKR-pulled bars
by composing the per-group helpers in:
  - src/features/yield_curve.py   (polynomial yield-curve coefficients)
  - src/features/correlations.py  (cross-asset rolling correlations + spread)

Every feature value at row t uses only data with timestamp <= t close
(test_leakage.py enforces this at runtime).

Feature groups (matches README):
1. Volatility regime: VIX, VIX3M, VIX3M-VIX, VVIX (4)
2. Variance risk premium: VIX - 30d realized vol on SPY, and 60d window (2)
3. Yield curve shape: 3 polynomial coefficients fit to {IRX, FVX, TNX, TYX}
4. Cross-asset stress: SPY-TNX 20d corr, SPY-GLD 20d corr, HYG-LQD spread (3)
5. Broad market regime: 20d SPY return, SPY pct from 200d MA, 50/200 MA cross (3)

Total: 15 features. Loadable as a single DataFrame indexed by date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_RAW_DIR
from src.features.correlations import (
    hyg_lqd_spread, spy_gld_correlation_20d, spy_tnx_correlation_20d,
)
from src.features.yield_curve import yield_curve_coefficients


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

    # Group 4: cross-asset stress (helpers in src/features/correlations.py)
    feat["spy_tnx_corr_20d"] = spy_tnx_correlation_20d(closes["SPY"], closes["TNX"])
    feat["spy_gld_corr_20d"] = spy_gld_correlation_20d(closes["SPY"], closes["GLD"])
    feat["hyg_lqd_spread"] = hyg_lqd_spread(closes["HYG"], closes["LQD"])

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
