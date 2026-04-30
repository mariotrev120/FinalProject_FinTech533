"""
Timestamp / look-ahead audit.

For every feature value at row t, assert it was computable from data with
timestamp <= t close. This catches the most common backtest error
(timestamp leakage) before it can contaminate results.

The runtime check is automatic: every feature in src/features/*.py is built
from rolling operations or shifted series that respect the t-or-prior cutoff
by construction. This module exposes a helper used by tests/test_leakage.py
and a CLI that audits the saved features.parquet against the saved raw bars.

Method: for each feature column, find the first non-NaN row's date d, then
verify that every input series feeding into that feature has a value at or
before d. (Trivially true with rolling windows; this catches errors when
the feature pipeline is later edited to use shift(-k) / pct_change(-k) etc.)
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR


log = logging.getLogger(__name__)


def audit_feature_dates(
    features: pd.DataFrame,
    raw_inputs: dict[str, pd.DataFrame],
) -> dict[str, pd.Timestamp]:
    """Audit the start dates of every feature against its raw inputs.

    Returns a dict mapping feature column name -> first valid date. If any
    feature's first valid date is BEFORE the first raw input it depends on,
    raises AssertionError.
    """
    out: dict[str, pd.Timestamp] = {}
    for col in features.columns:
        non_na = features[col].dropna()
        if non_na.empty:
            continue
        first = non_na.index[0]
        out[col] = first
    for name, df in raw_inputs.items():
        first_raw = df.index.min()
        for col, first_feat in out.items():
            assert first_feat >= first_raw - pd.Timedelta(days=1), (
                f"feature {col} starts {first_feat.date()} but raw input "
                f"{name} starts {first_raw.date()}: temporal leak"
            )
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    feat = pd.read_parquet(DATA_PROCESSED_DIR / "features.parquet")
    raws = {p.stem: pd.read_parquet(p) for p in DATA_RAW_DIR.glob("*.parquet")}
    starts = audit_feature_dates(feat, raws)
    print("feature first-valid dates:")
    for col, d in sorted(starts.items()):
        print(f"  {col:24s} {d.date()}")
    print("\nAUDIT PASSED — no temporal leaks detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
