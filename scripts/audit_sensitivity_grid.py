"""
Tier 4 audit: parameter sensitivity grid.

Re-runs the naked backtest under variations of the four "obvious" parameters:
  - profit_target_frac:  {0.30, 0.40, 0.50, 0.60}
  - stop_loss_mult:      {1.5, 2.0, 2.5, 3.0}
  - dte window:          {(25,40), (30,45), (35,50)}
  - entry_delta_target:  {0.12, 0.16, 0.20}

Reports each combo's OOS annualized Sharpe and annualized return. If results
flip dramatically across the grid, the strategy is parameter-sensitive
(potentially cherry-picked). If results are stable, the strategy is robust
and the chosen parameters are defensible.

Outputs a long-format DataFrame and a summary heatmap-friendly table.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_default_pricer, load_inputs
from src.config import OOS_END, OOS_START
from src.metrics.performance import combined_metrics


log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    inputs = load_inputs()
    pricer = load_default_pricer()

    grid_pt = [0.30, 0.40, 0.50, 0.60]
    grid_sl = [1.5, 2.0, 2.5, 3.0]
    grid_dte = [(25, 40), (30, 45), (35, 50)]
    grid_dlt = [0.12, 0.16, 0.20]

    rows = []
    # Modify config in place. We could plumb these through, but for sweep
    # we monkey-patch src.config which the engine reads at call time.
    import src.config as cfg
    orig = (cfg.PROFIT_TARGET_FRAC, cfg.STOP_LOSS_MULT,
            cfg.DTE_MIN, cfg.DTE_MAX, cfg.ENTRY_DELTA_TARGET)
    try:
        for pt in grid_pt:
            for sl in grid_sl:
                for dte_min, dte_max in grid_dte:
                    for dlt in grid_dlt:
                        cfg.PROFIT_TARGET_FRAC = pt
                        cfg.STOP_LOSS_MULT = sl
                        cfg.DTE_MIN = dte_min
                        cfg.DTE_MAX = dte_max
                        cfg.ENTRY_DELTA_TARGET = dlt
                        log.info("pt=%.2f sl=%.1f dte=%s delta=%.2f",
                                 pt, sl, (dte_min, dte_max), dlt)
                        try:
                            r = run_backtest(inputs, pricer, mode="naked",
                                              start=OOS_START, end=OOS_END)
                            m = combined_metrics(r.trades, r.equity_curve,
                                                  risk_free_curve=inputs.risk_free_curve)
                            rows.append({
                                "pt": pt, "sl": sl, "dte_min": dte_min, "dte_max": dte_max,
                                "delta_target": dlt,
                                "n_trades": m["n_trades"],
                                "win_rate": m["win_rate"],
                                "ann_ret_pct": m["annualized_return_pct"],
                                "ann_vol_pct": m["annualized_vol_pct"],
                                "sharpe_ann": m["sharpe_annualized"],
                                "max_dd_pct": m["max_drawdown_pct"],
                            })
                        except Exception as e:
                            log.error("  failed: %s", e)
    finally:
        # Restore originals
        cfg.PROFIT_TARGET_FRAC, cfg.STOP_LOSS_MULT, cfg.DTE_MIN, cfg.DTE_MAX, cfg.ENTRY_DELTA_TARGET = orig

    df = pd.DataFrame(rows)
    df.to_parquet("data/processed/audit_sensitivity_grid.parquet")
    csv_path = "data/processed/audit_sensitivity_grid.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print(f"PARAMETER SENSITIVITY GRID — naked OOS, {len(df)} configurations")
    print("=" * 80)
    print()
    print(f"Sharpe (annualized) summary across grid:")
    print(f"  min:    {df['sharpe_ann'].min():.3f}")
    print(f"  median: {df['sharpe_ann'].median():.3f}")
    print(f"  max:    {df['sharpe_ann'].max():.3f}")
    print(f"  range:  {df['sharpe_ann'].max() - df['sharpe_ann'].min():.3f}")
    print()
    print(f"Annualized return summary:")
    print(f"  min:    {df['ann_ret_pct'].min():.2f}%")
    print(f"  median: {df['ann_ret_pct'].median():.2f}%")
    print(f"  max:    {df['ann_ret_pct'].max():.2f}%")
    print()
    # Best and worst combos
    print(f"Top 5 by Sharpe:")
    print(df.nlargest(5, "sharpe_ann")[["pt", "sl", "dte_min", "dte_max", "delta_target",
                                          "n_trades", "win_rate", "ann_ret_pct", "sharpe_ann",
                                          "max_dd_pct"]].round(3).to_string(index=False))
    print()
    print(f"Bottom 5 by Sharpe:")
    print(df.nsmallest(5, "sharpe_ann")[["pt", "sl", "dte_min", "dte_max", "delta_target",
                                           "n_trades", "win_rate", "ann_ret_pct", "sharpe_ann",
                                           "max_dd_pct"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
