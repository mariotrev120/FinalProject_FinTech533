"""
Vol-regime quadrant classifier (per PRE_COMMITMENT_VRP §7).

Each vol-index series (VIX for SPX, RVX for RUT, VXN for NDX, MOVE for TLT,
GVZ for GLD) is independently standardized to z-units against its own
trailing 252-day rolling mean and std, in BOTH the level and the
first-difference derivative. Each session is then placed into one of
four quadrants by the signs of (level_z, derivative_z):

  Q0  level_z < 0  AND deriv_z < 0   "calm, calming"
  Q1  level_z < 0  AND deriv_z ≥ 0   "calm, ramping"
  Q2  level_z ≥ 0  AND deriv_z ≥ 0   "stressed, escalating"
  Q3  level_z ≥ 0  AND deriv_z < 0   "stressed, mean-reverting"

Dual role:
  1. Categorical feature input to Head 1 (trade quality).
  2. Standalone gate ablation baseline: in the ablation matrix, a
     vol-regime-only mode skips entry when in Q2 (escalating stress)
     and otherwise allows entry. Comparator against the full ML stack.

Per-instrument calibration: the rolling-window standardization is
PER-INSTRUMENT — VIX gets its own 252d window, RVX gets its own, etc.
This avoids treating equity-index vol the same as bond vol or gold vol
when the absolute levels and dynamics differ.

The function is instrument-agnostic: feed it any close-price series and
it returns a 0..3 classification. Per-instrument file-loading is handled
by `quadrant_for_instrument()`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# --- Per-instrument vol index mapping (locked) ----------------------------

VOL_INDEX_FOR: dict[str, str] = {
    "SPX": "VIX",
    "RUT": "RVX",
    "NDX": "VXN",
    "TLT": "MOVE",
    "GLD": "GVZ",
}


# --- Quadrant labels (0..3 = Q0..Q3) --------------------------------------

QUADRANT_LABELS: dict[int, str] = {
    0: "calm_calming",
    1: "calm_ramping",
    2: "stressed_escalating",
    3: "stressed_mean_reverting",
}

# When the input is below the warmup threshold (level or deriv z = NaN),
# return -1 to signal "uncategorized". Downstream consumers should treat
# this as a missing feature, not as Q0.
QUADRANT_UNCATEGORIZED: int = -1


# --- Core algorithm -------------------------------------------------------

def vol_regime_quadrant(
    vol_index_close: pd.Series,
    lookback: int = 252,
    min_obs: int = 50,
) -> pd.DataFrame:
    """Classify each date in `vol_index_close` into a quadrant 0..3.

    Inputs:
      vol_index_close: pandas Series of daily close prices for the vol
        index (e.g. VIX). Must be date-indexed.
      lookback: rolling-window length for z-score standardization
        (default 252 trading days = 1 calendar year).
      min_obs: minimum observations in the rolling window before a
        non-NaN z-score is computed (default 50). Sessions with fewer
        observations get quadrant = QUADRANT_UNCATEGORIZED (-1).

    Returns:
      pandas DataFrame indexed by date with columns:
        - level_z:    z-score of the level
        - deriv_z:    z-score of the first-difference of the level
        - quadrant:   integer 0..3, or -1 for warmup / NaN sessions
        - label:      string label per QUADRANT_LABELS
    """
    s = pd.Series(vol_index_close).dropna().astype(float).copy()
    if len(s) == 0:
        return pd.DataFrame(columns=["level_z", "deriv_z", "quadrant", "label"])

    # Level z-score on rolling 252d
    roll = s.rolling(window=lookback, min_periods=min_obs)
    level_z = (s - roll.mean()) / roll.std(ddof=1)

    # First-difference of the level, then z-score on rolling 252d of
    # the difference series.
    diff = s.diff()
    diff_roll = diff.rolling(window=lookback, min_periods=min_obs)
    deriv_z = (diff - diff_roll.mean()) / diff_roll.std(ddof=1)

    out = pd.DataFrame({"level_z": level_z, "deriv_z": deriv_z})
    out["quadrant"] = QUADRANT_UNCATEGORIZED
    valid = level_z.notna() & deriv_z.notna()
    if valid.any():
        # Q0: level<0, deriv<0 — calm calming
        # Q1: level<0, deriv>=0 — calm ramping
        # Q2: level>=0, deriv>=0 — stressed escalating
        # Q3: level>=0, deriv<0 — stressed mean-reverting
        lvl_pos = (level_z >= 0).astype(int)
        drv_pos = (deriv_z >= 0).astype(int)
        # Use a 2x2 lookup: (lvl_pos, drv_pos) → quadrant
        # (0,0) → 0, (0,1) → 1, (1,1) → 2, (1,0) → 3
        # This is: lvl_pos*2 ^ drv_pos? No. Let me just do it explicitly:
        q = pd.Series(QUADRANT_UNCATEGORIZED, index=s.index, dtype=int)
        q[(level_z < 0) & (deriv_z < 0)] = 0
        q[(level_z < 0) & (deriv_z >= 0)] = 1
        q[(level_z >= 0) & (deriv_z >= 0)] = 2
        q[(level_z >= 0) & (deriv_z < 0)] = 3
        out["quadrant"] = q
        # Suppress any quadrant assignment on warmup rows where either z is NaN
        out.loc[~valid, "quadrant"] = QUADRANT_UNCATEGORIZED

    out["label"] = out["quadrant"].map(
        lambda q: QUADRANT_LABELS.get(q, "uncategorized")
    )
    return out


# --- Per-instrument convenience wrapper -----------------------------------

def quadrant_for_instrument(
    underlying: str,
    data_raw_dir: str = "data/raw",
    lookback: int = 252,
    min_obs: int = 50,
) -> Optional[pd.DataFrame]:
    """Load the per-instrument vol index parquet and compute its quadrant
    classification.

    Returns None if the corresponding vol-index parquet does not exist
    on disk (e.g. RVX/VXN/MOVE/GVZ may not be sourced yet). Caller
    decides how to handle: skip the instrument, fall back to the SPX
    vol regime, or pre-fetch the missing index.
    """
    if underlying not in VOL_INDEX_FOR:
        raise KeyError(
            f"No vol-index mapping for {underlying!r}. "
            f"Known: {sorted(VOL_INDEX_FOR.keys())}."
        )
    vol_ticker = VOL_INDEX_FOR[underlying]
    path = f"{data_raw_dir}/{vol_ticker}.parquet"
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        return None
    if "close" not in df.columns:
        raise ValueError(
            f"{path} has no 'close' column; got {list(df.columns)}"
        )
    return vol_regime_quadrant(
        df["close"], lookback=lookback, min_obs=min_obs,
    )


# --- Ablation gate (Role 2): vol-regime-only entry filter -----------------

def regime_allows_entry(quadrant: int) -> bool:
    """Standalone vol-regime-only ablation rule: skip entry when in Q2
    (stressed_escalating). Allow Q0, Q1, Q3, and uncategorized warmup.

    Q2 means level is high AND vol-of-vol is increasing — the worst-case
    moment for a short-vol VRP harvest. The other three quadrants have
    at least one mitigating component (low level, or mean-reverting).
    """
    return quadrant != 2
