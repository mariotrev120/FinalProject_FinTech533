"""
Unit tests for src/strategy/vol_regime.py.

Synthetic-pattern tests verify each of the four quadrants is correctly
identified. One small integration test reads the real VIX parquet
(~95KB) and checks that all four quadrants appear in the historical
sample with non-trivial frequency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.strategy.vol_regime import (
    QUADRANT_LABELS,
    QUADRANT_UNCATEGORIZED,
    VOL_INDEX_FOR,
    quadrant_for_instrument,
    regime_allows_entry,
    vol_regime_quadrant,
)


# --- Synthetic-pattern tests ----------------------------------------------

def _build_series(values: list[float]) -> pd.Series:
    """Daily-indexed series for clean reproducibility."""
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_quadrant_q0_calm_calming():
    """Below-mean and dropping further today → Q0.

    Construction: long flat history at level 18, then the last few sessions
    drop to level ~13, with the FINAL session showing a sharp additional
    drop of ~3 points (much larger than the typical ±0 daily diff during
    the flat history). Final level_z < 0 (below the high mean) and
    deriv_z < 0 (today's drop is unusually large)."""
    base = [18.0] * 90
    flat_low = [13.0] * 5
    final_drop = [10.0]
    s = _build_series(base + flat_low + final_drop)
    out = vol_regime_quadrant(s, lookback=80, min_obs=20)
    last = out.iloc[-1]
    assert last["level_z"] < 0
    assert last["deriv_z"] < 0
    assert last["quadrant"] == 0
    assert last["label"] == "calm_calming"


def test_quadrant_q1_calm_ramping():
    """Below-mean but level rising sharply today → Q1."""
    # Long base near 18, then a dip to 11, then a sharp jump back up to 14.
    # Final level still < mean(18) but today's diff is positive and unusually
    # large vs the recent diff distribution.
    base = [18.0] * 80
    dip = list(np.linspace(18.0, 11.0, 30))
    jump = [14.0]
    vals = base + dip + jump
    s = _build_series(vals)
    out = vol_regime_quadrant(s, lookback=80, min_obs=20)
    last = out.iloc[-1]
    assert last["level_z"] < 0
    assert last["deriv_z"] > 0
    assert last["quadrant"] == 1
    assert last["label"] == "calm_ramping"


def test_quadrant_q2_stressed_escalating():
    """Above-mean and rising further → Q2."""
    # 90 days at 15, then a ramp up to 30
    base = [15.0] * 90
    ramp = list(np.linspace(15.0, 30.0, 30))
    s = _build_series(base + ramp)
    out = vol_regime_quadrant(s, lookback=90, min_obs=20)
    last = out.iloc[-1]
    assert last["level_z"] > 0
    assert last["deriv_z"] > 0
    assert last["quadrant"] == 2
    assert last["label"] == "stressed_escalating"


def test_quadrant_q3_stressed_mean_reverting():
    """Above-mean but falling today → Q3."""
    # 90 days at 15, ramp up to 30, then today drops back to 24.
    base = [15.0] * 90
    ramp = list(np.linspace(15.0, 30.0, 30))
    drop = [24.0]
    s = _build_series(base + ramp + drop)
    out = vol_regime_quadrant(s, lookback=90, min_obs=20)
    last = out.iloc[-1]
    assert last["level_z"] > 0
    assert last["deriv_z"] < 0
    assert last["quadrant"] == 3
    assert last["label"] == "stressed_mean_reverting"


def test_warmup_returns_uncategorized():
    """First few rows have no rolling-window data → quadrant = -1."""
    vals = list(np.linspace(15.0, 18.0, 20))
    s = _build_series(vals)
    out = vol_regime_quadrant(s, lookback=60, min_obs=50)
    # min_obs=50 > len(s)=20 → all rows uncategorized
    assert (out["quadrant"] == QUADRANT_UNCATEGORIZED).all()
    assert (out["label"] == "uncategorized").all()


def test_empty_input_returns_empty_frame():
    out = vol_regime_quadrant(pd.Series([], dtype=float))
    assert out.empty
    assert list(out.columns) == ["level_z", "deriv_z", "quadrant", "label"]


def test_quadrant_label_map_complete():
    """Every value 0..3 has a label."""
    for q in range(4):
        assert q in QUADRANT_LABELS
        assert QUADRANT_LABELS[q] != ""


# --- Instrument-mapping tests ---------------------------------------------

def test_instrument_mapping_locked():
    """Universe per PRE_COMMITMENT_VRP §1 maps to its declared vol index."""
    assert VOL_INDEX_FOR == {
        "SPX": "VIX",
        "RUT": "RVX",
        "NDX": "VXN",
        "TLT": "MOVE",
        "GLD": "GVZ",
    }


def test_quadrant_for_instrument_unknown_raises():
    with pytest.raises(KeyError, match="No vol-index mapping"):
        quadrant_for_instrument("AAPL")


def test_quadrant_for_instrument_missing_file_returns_none():
    """RVX/VXN/MOVE/GVZ are not on disk yet — wrapper returns None
    cleanly so callers can fall back without crashing."""
    # Use a dir that doesn't exist; SPX/VIX file under that won't be found.
    out = quadrant_for_instrument("SPX", data_raw_dir="/tmp/__nonexistent_data_dir__")
    assert out is None


# --- Real-data integration test (VIX parquet ~95KB only) ------------------

@pytest.mark.skipif(
    not Path("data/raw/VIX.parquet").exists(),
    reason="VIX parquet not present",
)
def test_quadrant_for_spx_real_vix():
    """Integration: real VIX history should produce all 4 quadrants
    with non-trivial frequency. Validates that the algorithm doesn't
    collapse into one regime on real-world input."""
    out = quadrant_for_instrument("SPX")
    assert out is not None
    assert len(out) > 1000
    counts = out["quadrant"].value_counts()
    # Every quadrant should appear at least 100 times in a 10+ year history
    for q in (0, 1, 2, 3):
        assert counts.get(q, 0) >= 100, (
            f"Quadrant {q} only appears {counts.get(q, 0)} times — "
            f"expected ≥ 100 in real VIX history"
        )


# --- Ablation gate (Role 2) tests -----------------------------------------

def test_regime_allows_entry_blocks_only_q2():
    assert regime_allows_entry(0)        # calm_calming
    assert regime_allows_entry(1)        # calm_ramping
    assert not regime_allows_entry(2)    # stressed_escalating BLOCKED
    assert regime_allows_entry(3)        # stressed_mean_reverting
    assert regime_allows_entry(QUADRANT_UNCATEGORIZED)  # warmup OK
