"""
Two-state HMM regime diagnostic (Calm vs Stressed).

NOT used for trade decisions. Used only for tagging trades with their regime
at entry, enabling performance-by-regime breakdowns in the writeup.

Stability fixes per README:
  1. 20 random Baum-Welch initializations per fit, take highest log-likelihood
  2. Centroid-tracked state labeling: after each refit, states are reassigned
     by matching mean feature vectors to the previous fold's centroids using
     the Hungarian algorithm. This forces "Calm" to remain semantically
     "Calm" across all walk-forward refits.

Features used for the HMM (subset of the full 15, focused on the regime
indicators most discriminating between calm and stressed):
  - VIX
  - VIX3M minus VIX (term structure)
  - SPY-TNX 20d corr (cross-asset stress)
  - 30d realized vol on SPY

Output: a Series indexed by trading date with values 'calm' or 'stressed'.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED_DIR, SEED


log = logging.getLogger(__name__)


HMM_FEATURE_COLS: list[str] = [
    "vix",
    "vix3m_minus_vix",
    "spy_tnx_corr_20d",
    "vrp_30d",
]


def _fit_one(X: np.ndarray, n_states: int = 2, seed: int = 0,
             max_iter: int = 200) -> tuple[GaussianHMM, float]:
    m = GaussianHMM(
        n_components=n_states, covariance_type="full",
        n_iter=max_iter, random_state=seed, tol=1e-4,
    )
    m.fit(X)
    return m, float(m.score(X))


def fit_hmm_robust(
    features_df: pd.DataFrame, n_states: int = 2, n_inits: int = 20,
    seed: int = SEED,
) -> tuple[GaussianHMM, StandardScaler, np.ndarray]:
    """Fit GaussianHMM with n_inits random restarts, return best by log-likelihood.

    Returns (model, scaler, centroids) where centroids[k] = mean feature vector
    of state k in the SCALED feature space (used for relabeling alignment).
    """
    X = features_df[HMM_FEATURE_COLS].dropna().values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best = None
    best_ll = -np.inf
    for i in range(n_inits):
        try:
            model, ll = _fit_one(Xs, n_states=n_states, seed=seed + i)
            if ll > best_ll:
                best, best_ll = model, ll
        except Exception as e:
            log.debug("HMM init %d failed: %s", i, e)
    if best is None:
        raise RuntimeError("All HMM initializations failed")

    centroids = best.means_.copy()
    return best, scaler, centroids


def relabel_to_match(
    new_centroids: np.ndarray, prev_centroids: np.ndarray,
) -> dict[int, int]:
    """Hungarian alignment: build mapping new_state -> prev_state index that
    minimizes total Euclidean distance between centroids.

    Returns dict mapping new state -> previous state.
    """
    n = len(new_centroids)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = np.linalg.norm(new_centroids[i] - prev_centroids[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind)}


def label_states_by_vol(model: GaussianHMM, scaler: StandardScaler) -> dict[int, str]:
    """Map each state index to 'calm' or 'stressed' based on which centroid
    has the higher VIX mean (after de-scaling)."""
    raw_centroids = scaler.inverse_transform(model.means_)
    vix_idx = HMM_FEATURE_COLS.index("vix")
    vix_means = raw_centroids[:, vix_idx]
    sorted_states = np.argsort(vix_means)   # ascending by VIX
    return {int(sorted_states[0]): "calm", int(sorted_states[-1]): "stressed"}


def predict_states(
    model: GaussianHMM, scaler: StandardScaler, features_df: pd.DataFrame,
) -> pd.Series:
    """Predict state for each row of features_df. Drops rows with NaN."""
    X = features_df[HMM_FEATURE_COLS].dropna()
    Xs = scaler.transform(X.values)
    states = model.predict(Xs)
    return pd.Series(states, index=X.index, name="hmm_state")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    feat = pd.read_parquet(DATA_PROCESSED_DIR / "features.parquet")
    log.info("HMM training on %d rows, features=%s", len(feat), HMM_FEATURE_COLS)

    model, scaler, centroids = fit_hmm_robust(feat, n_states=2, n_inits=20)
    label_map = label_states_by_vol(model, scaler)
    log.info("state -> regime: %s", label_map)

    states = predict_states(model, scaler, feat)
    regimes = states.map(label_map)
    out_path = DATA_PROCESSED_DIR / "hmm_regimes.parquet"
    regimes.to_frame("regime").to_parquet(out_path)

    print(f"\n=== HMM regime diagnostic ===")
    print(f"  trained on {len(feat)} rows")
    print(f"  state -> regime mapping: {label_map}")
    print(f"  regime distribution overall: ")
    print(regimes.value_counts(normalize=True).round(3))
    print(f"  saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
