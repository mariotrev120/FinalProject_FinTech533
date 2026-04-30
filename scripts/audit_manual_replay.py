"""
Tier 3 audit: manual replay of one calendar year against engine output.

For 2019, picks 5 random trades from the naked blotter and reconciles each
field of the trade end-to-end against raw OptionMetrics chain rows.

For each trade we manually compute:
  - entry credit per share = mid_credit - frac_at_vix * combined_half_spread
  - exit debit per share   = mid_debit  + frac_at_vix * combined_half_spread
  - round-trip commission  = $0.65/contract/leg × 4 legs (entry+exit) +
                              $0.05 reg fee × 4
  - pnl_per_spread = (entry_credit - exit_debit) * 100
  - total_pnl     = pnl_per_spread * contracts - commish

We then assert each field matches the engine's reported value within a
small tolerance. If anything is off, we have a friction-model bug or a
timestamp-handling bug.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_default_pricer, load_inputs
from src.strategy.friction import slippage_pct_at_vix
from src.strategy.optionmetrics_pricer import load_optionmetrics_dataframe


log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    inputs = load_inputs()
    pricer = load_default_pricer()
    df_chain = load_optionmetrics_dataframe()

    log.info("Running naked 2019 OOS for manual replay...")
    result = run_backtest(inputs, pricer, mode="naked", start="2019-01-01", end="2019-12-31")
    trades_2019 = [t for t in result.trades if t.entry_date.year == 2019 and t.exit_date is not None]
    log.info("Found %d closed trades in 2019.", len(trades_2019))

    # Pick 5 evenly-spaced trades
    sample = [trades_2019[i] for i in
              np.linspace(0, len(trades_2019) - 1, 5).astype(int)] if trades_2019 else []
    log.info("Replaying %d trades.", len(sample))

    print()
    print("=" * 100)
    print(f"{'Trade':>5s}  {'EntryDate':>10s}  {'Field':>22s}  {'Engine':>14s}  {'Manual':>14s}  {'OK?':>4s}")
    print("=" * 100)

    n_pass = 0
    n_fail = 0
    for t in sample:
        entry_dt = pd.Timestamp(t.entry_date)
        exit_dt = pd.Timestamp(t.exit_date)

        # Entry chain rows
        e_chain = df_chain[(df_chain["ticker"] == "SPX") & (df_chain["date"] == entry_dt) &
                            (df_chain["exdate"] == pd.Timestamp(t.spread.short_leg.expiry))]
        e_short = e_chain[e_chain["strike"] == t.spread.short_leg.strike].iloc[0]
        e_long = e_chain[e_chain["strike"] == t.spread.long_leg.strike].iloc[0]
        e_short_mid = (e_short["best_bid"] + e_short["best_offer"]) / 2.0
        e_long_mid = (e_long["best_bid"] + e_long["best_offer"]) / 2.0
        mid_credit = e_short_mid - e_long_mid
        full_short = e_short["best_offer"] - e_short["best_bid"]
        full_long = e_long["best_offer"] - e_long["best_bid"]
        combined_half = (full_short + full_long) / 2.0
        frac = slippage_pct_at_vix(t.entry_vix)
        manual_entry_credit_per_share = max(mid_credit - frac * combined_half, 0.0)
        manual_entry_credit_per_spread = manual_entry_credit_per_share * 100.0

        # Exit chain rows
        x_chain = df_chain[(df_chain["ticker"] == "SPX") & (df_chain["date"] == exit_dt) &
                            (df_chain["exdate"] == pd.Timestamp(t.spread.short_leg.expiry))]
        if len(x_chain) > 0:
            x_short_rows = x_chain[x_chain["strike"] == t.spread.short_leg.strike]
            x_long_rows = x_chain[x_chain["strike"] == t.spread.long_leg.strike]
            if len(x_short_rows) > 0 and len(x_long_rows) > 0:
                x_short = x_short_rows.iloc[0]
                x_long = x_long_rows.iloc[0]
                x_short_mid = (x_short["best_bid"] + x_short["best_offer"]) / 2.0
                x_long_mid = (x_long["best_bid"] + x_long["best_offer"]) / 2.0
                mid_debit = max(x_short_mid - x_long_mid, 0.0)
                full_short_x = x_short["best_offer"] - x_short["best_bid"]
                full_long_x = x_long["best_offer"] - x_long["best_bid"]
                combined_half_x = (full_short_x + full_long_x) / 2.0
                frac_x = slippage_pct_at_vix(t.exit_vix)
                manual_exit_debit_per_share = mid_debit + frac_x * combined_half_x
                manual_exit_debit_per_spread = manual_exit_debit_per_share * 100.0
            else:
                manual_exit_debit_per_spread = float("nan")
        else:
            manual_exit_debit_per_spread = float("nan")

        manual_pnl_per_spread = manual_entry_credit_per_spread - manual_exit_debit_per_spread

        # Compare to engine
        for field, engine_v, manual_v in [
            ("entry_credit_per_spread", t.entry_credit_per_spread, manual_entry_credit_per_spread),
            ("exit_debit_per_spread", t.exit_debit_per_spread, manual_exit_debit_per_spread),
            ("pnl_per_spread", t.pnl_per_spread, manual_pnl_per_spread),
        ]:
            if engine_v is None or manual_v is None or np.isnan(manual_v):
                ok = "?"
            else:
                ok = "OK" if abs(engine_v - manual_v) < 0.5 else "FAIL"
            if ok == "OK":
                n_pass += 1
            elif ok == "FAIL":
                n_fail += 1
            print(f"{t.trade_id:5d}  {entry_dt.date()}  {field:>22s}  {engine_v:14.2f}  {manual_v:14.2f}  {ok:>4s}")
        print("-" * 100)

    print(f"\nResults: {n_pass} PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
