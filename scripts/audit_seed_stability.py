"""
Tier 4 audit: 5-seed XGBoost stability.

Re-trains the ML pipeline under 5 different random seeds, runs ml_only OOS
each time, and reports the variance in OOS annualized Sharpe attributable
to seed alone.

If seed variance is meaningful (e.g., Sharpe range > 0.10) then the
"ML decorative" finding is fragile to model initialization — the conclusion
strengthens because seed-luck drives the result, not signal.

Outputs:
  - data/processed/audit_seed_stability.parquet (per-seed metrics)
  - stdout summary
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_default_pricer, load_inputs
from src.backtest.walkforward import annual_walk_forward_xgb
from src.config import DATA_PROCESSED_DIR, IS_START, OOS_END, OOS_START
from src.metrics.performance import combined_metrics
from src.models.label_trades import label_trades_dataframe


log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    inputs = load_inputs()
    pricer = load_default_pricer()
    features = pd.read_parquet(DATA_PROCESSED_DIR / "features.parquet")

    # Generate naked-mode labels once (deterministic since no ML/halts in naked)
    log.info("Generating naked-mode labels...")
    naked = run_backtest(inputs, pricer, mode="naked", start=IS_START, end=OOS_END)
    labels = label_trades_dataframe(naked)
    log.info("Got %d labels with win rate %.1f%%", len(labels), 100 * labels["win"].mean())

    rows = []
    for seed in [42, 7, 100, 2024, 77]:
        log.info(f"=== Seed {seed} ===")
        probs = annual_walk_forward_xgb(features=features, labels=labels,
                                         is_start=IS_START, oos_end=OOS_END,
                                         seed=seed)
        # Inject probs into engine inputs and run ml_only
        inputs_seed = inputs
        inputs_seed.ml_probability = probs
        oos = run_backtest(inputs_seed, pricer, mode="ml_only",
                            start=OOS_START, end=OOS_END)
        m = combined_metrics(oos.trades, oos.equity_curve,
                              risk_free_curve=inputs.risk_free_curve)
        log.info(f"  seed={seed}: trades={m['n_trades']}, "
                 f"win={m['win_rate']:.3f}, "
                 f"ann_ret={m['annualized_return_pct']:.2f}%, "
                 f"sharpe_ann={m['sharpe_annualized']:.3f}")
        rows.append({
            "seed": seed,
            "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
            "annualized_return_pct": m["annualized_return_pct"],
            "sharpe_annualized": m["sharpe_annualized"],
            "max_drawdown_pct": m["max_drawdown_pct"],
        })

    df = pd.DataFrame(rows)
    df.to_parquet("data/processed/audit_seed_stability.parquet")

    print()
    print("=" * 70)
    print("XGBoost SEED STABILITY (ml_only mode, OOS)")
    print("=" * 70)
    print(df.round(3).to_string(index=False))
    print()
    print(f"Sharpe spread across seeds: {df['sharpe_annualized'].max() - df['sharpe_annualized'].min():.3f}")
    print(f"Trade count spread: {df['n_trades'].max() - df['n_trades'].min()}")
    if df['sharpe_annualized'].max() - df['sharpe_annualized'].min() > 0.10:
        print(">>> Seed-driven Sharpe variance > 0.10 — model is unstable.")
        print(">>> The 'ML decorative' finding is robust to seed only if the variance")
        print(">>> brackets zero, not if it brackets wide positive.")
    else:
        print(">>> Seed-driven Sharpe variance < 0.10 — model is stable across seeds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
