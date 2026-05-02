"""
Halt anti-signal diagnosis.

A halt is "correct" on a given Monday entry day if the NAKED trade entered
that day went on to lose money over its life. False positive = halt blocked
a day where the naked trade was profitable.

Three views:
  1. Per state (hard / soft / slow / drawdown) — was that *layer* useful?
  2. Per individual trigger — granular: which signals predict losers?
  3. Regime characterization — VIX, VIX3M-VIX, fwd 30d SPX return on halt
     vs non-halt days. If halt periods were actually MORE profitable for the
     naked trade than non-halt periods, we'd expect halts to be anti-signal.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_default_pricer, load_inputs
from src.config import OOS_END, OOS_START


log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    inputs = load_inputs()
    pricer = load_default_pricer()

    log.info("Running naked OOS for outcome reference...")
    naked = run_backtest(inputs, pricer, mode="naked", start=OOS_START, end=OOS_END)
    log.info("Running halts_only OOS for halt log...")
    halts = run_backtest(inputs, pricer, mode="halts_only", start=OOS_START, end=OOS_END)

    naked_pnl_by_date = pd.Series({
        pd.Timestamp(t.entry_date): t.total_pnl
        for t in naked.trades if t.total_pnl is not None
    }).sort_index()

    halt_log = halts.halt_log.copy()
    halt_log.index = pd.to_datetime(halt_log.index)
    halt_log["triggers"] = halt_log["triggers"].fillna("")

    naked_dates = set(naked_pnl_by_date.index)
    blocked = halt_log[
        (halt_log["state"] != "active") & halt_log.index.isin(naked_dates)
    ].copy()
    blocked["naked_pnl"] = naked_pnl_by_date.reindex(blocked.index)
    blocked["was_loser"] = blocked["naked_pnl"] < 0

    base_loser_rate = float((naked_pnl_by_date < 0).mean())

    print("\n" + "=" * 78)
    print("HALT DIAGNOSIS — naked-trade-outcome precision (OOS 2018-2024)")
    print("=" * 78)
    print(f"\nNaked OOS:        {len(naked_pnl_by_date)} trades, "
          f"loser rate {base_loser_rate*100:.1f}%")
    print(f"Halt-blocked entries that naked would have taken: {len(blocked)}")
    if len(blocked) > 0:
        prec = float(blocked["was_loser"].mean())
        print(f"Halt precision overall:  {prec*100:.1f}% "
              f"(lift vs base {(prec - base_loser_rate)*100:+.1f} pp)")
        print(f"Naked PnL the halts blocked (sum): "
              f"${blocked['naked_pnl'].sum():+,.2f}")
        print(f"   (positive = halts cost us money; negative = halts saved us)")

    print("\n--- Per state (halt LAYER) ---")
    for state, sub in blocked.groupby("state"):
        if len(sub) == 0:
            continue
        precision = float(sub["was_loser"].mean())
        mean_pnl = float(sub["naked_pnl"].mean())
        sum_pnl = float(sub["naked_pnl"].sum())
        print(f"  {state:<15s} n={len(sub):>3d}  "
              f"precision={precision*100:>5.1f}%  "
              f"mean PnL={mean_pnl:>+8.2f}  "
              f"sum PnL={sum_pnl:>+10.2f}  "
              f"lift={(precision-base_loser_rate)*100:>+5.1f} pp")

    print("\n--- Per individual trigger (a blocked day may match multiple) ---")
    all_trigs: set[str] = set()
    for trig_str in blocked["triggers"]:
        for t in trig_str.split(","):
            t = t.strip()
            if t:
                all_trigs.add(t)
    trig_rows = []
    for trig in sorted(all_trigs):
        sub = blocked[blocked["triggers"].str.contains(trig, regex=False)]
        if len(sub) == 0:
            continue
        precision = float(sub["was_loser"].mean())
        trig_rows.append({
            "trigger": trig,
            "n": len(sub),
            "precision_pct": round(precision * 100, 1),
            "mean_naked_pnl": round(float(sub["naked_pnl"].mean()), 2),
            "sum_naked_pnl": round(float(sub["naked_pnl"].sum()), 2),
            "lift_vs_base_pp": round((precision - base_loser_rate) * 100, 1),
        })
    trig_df = pd.DataFrame(trig_rows).sort_values("lift_vs_base_pp", ascending=False)
    print(trig_df.to_string(index=False))

    print("\n--- Regime characterization (entire OOS daily) ---")
    spx_close = inputs.bars_spx["close"].copy()
    spx_close.index = pd.to_datetime(spx_close.index)
    fwd_21 = (spx_close.shift(-21) / spx_close - 1.0)

    h = halt_log.copy()
    h["vix"] = inputs.vix.reindex(h.index)
    h["vix3m"] = inputs.vix3m.reindex(h.index)
    h["vrp_proxy"] = h["vix3m"] - h["vix"]
    h["fwd_21d_ret"] = fwd_21.reindex(h.index)
    h["is_halt"] = h["state"] != "active"

    print(f"\nOf {len(h)} OOS days, halt-active: {int(h['is_halt'].sum())} "
          f"({h['is_halt'].mean()*100:.1f}%)")
    print()
    for label, mask in [("halt-period", h["is_halt"]),
                         ("non-halt period", ~h["is_halt"])]:
        sub = h[mask]
        if len(sub) == 0:
            continue
        print(f"  {label:<18s} (n={len(sub)}):")
        print(f"    VIX                  mean={sub['vix'].mean():>6.2f}  median={sub['vix'].median():>6.2f}")
        print(f"    VIX3M-VIX (VRP+)     mean={sub['vrp_proxy'].mean():>+6.2f}  median={sub['vrp_proxy'].median():>+6.2f}")
        print(f"    fwd 21d SPX ret      mean={sub['fwd_21d_ret'].mean()*100:>+6.2f}%  median={sub['fwd_21d_ret'].median()*100:>+6.2f}%")
        print()

    print("--- Trigger-combo (full triggers string) ranked by sum naked PnL blocked ---")
    combo_rows = []
    for combo, sub in blocked.groupby("triggers"):
        if combo == "":
            continue
        combo_rows.append({
            "triggers": combo,
            "n": len(sub),
            "precision_pct": round(float(sub["was_loser"].mean()) * 100, 1),
            "mean_naked_pnl": round(float(sub["naked_pnl"].mean()), 2),
            "sum_naked_pnl": round(float(sub["naked_pnl"].sum()), 2),
        })
    combo_df = pd.DataFrame(combo_rows).sort_values("sum_naked_pnl")
    print(combo_df.head(15).to_string(index=False))

    blocked.to_csv("data/processed/audit_halt_diagnosis_blocked.csv")
    h.to_csv("data/processed/audit_halt_diagnosis_regime.csv")
    if len(trig_df) > 0:
        trig_df.to_csv("data/processed/audit_halt_diagnosis_per_trigger.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
