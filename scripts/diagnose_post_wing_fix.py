"""
Consolidated post-wing-fix diagnostic — runs all 5 tickers naked-mode, reports:

  1. Full skip-reason histogram per ticker
  2. Entry-credit distribution per ticker (mean, median, % > 0, sample of 5)
  3. Wing-width values: spot, computed pct, $ wing, grid increment, snapped
  4. Sample 5 actual successful trades per ticker

Single consolidated output. No website updates. No commits. Just diagnose.
"""
from __future__ import annotations

import gc
import logging
import sys
from datetime import date

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_inputs
from src.config import OOS_END, OOS_START, SEED
from src.strategy.optionmetrics_pricer import _TICKER_DF_CACHE, make_pricer_v2
from src.strategy.spread_construction import (
    STRIKE_INCREMENT_BY_UNDERLYING, WING_PCT_OF_SPOT, compute_wing_width,
)


VRP_UNIVERSE: list[str] = ["SPX", "RUT", "NDX", "TLT", "GLD"]


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)
    inputs = load_inputs()

    # Capture spot snapshots from SPX-feed (multi-instrument runner uses
    # ticker-specific spot from option chain mid; we approximate here via
    # the actual trades' entry_spx)
    print("=" * 90)
    print("POST-WING-FIX CONSOLIDATED DIAGNOSTIC — all 5 tickers, naked OOS")
    print(f"Wing policy: max(strike_increment, {WING_PCT_OF_SPOT*100:.2f}% × spot), snapped to grid")
    print(f"Strike increments: {STRIKE_INCREMENT_BY_UNDERLYING}")
    print("=" * 90)

    rows_summary = []
    for ticker in VRP_UNIVERSE:
        print(f"\n{'#' * 90}")
        print(f"# {ticker}")
        print(f"{'#' * 90}")
        try:
            pricer = make_pricer_v2(ticker)
        except Exception as e:
            print(f"  pricer-load FAILED: {e}")
            continue
        try:
            result = run_backtest(
                inputs, pricer, mode="naked",
                start=OOS_START, end=OOS_END, use_iron_condor=True,
            )
        except Exception as e:
            print(f"  backtest FAILED: {e}")
            continue

        n_trades = len(result.trades)
        n_skipped = len(result.skipped_entries)
        print(f"\n  trades opened: {n_trades}")
        print(f"  entries skipped: {n_skipped}")
        print(f"  total entry attempts: {n_trades + n_skipped}")

        # --- Output 1: Full skip-reason histogram ---
        print(f"\n--- (1) {ticker} skip-reason histogram ---")
        if result.skipped_entries:
            skip_df = pd.DataFrame(result.skipped_entries)
            counts = skip_df["reason"].value_counts()
            for reason, cnt in counts.items():
                print(f"    {reason:30s}  {cnt}")
        else:
            print("    (no skipped entries)")

        # --- Output 2: Entry-credit distribution ---
        print(f"\n--- (2) {ticker} entry-credit distribution (post-fix) ---")
        if n_trades > 0:
            credits = [t.entry_credit_per_spread for t in result.trades]
            credits_arr = np.asarray(credits)
            print(f"    n_trades:                      {n_trades}")
            print(f"    mean entry_credit_per_spread:  ${credits_arr.mean():>10.2f}")
            print(f"    median entry_credit_per_spread:${float(np.median(credits_arr)):>10.2f}")
            print(f"    min  entry_credit_per_spread:  ${credits_arr.min():>10.2f}")
            print(f"    max  entry_credit_per_spread:  ${credits_arr.max():>10.2f}")
            n_pos = int((credits_arr > 0).sum())
            n_zero = int((credits_arr == 0).sum())
            print(f"    > $0:    {n_pos:5d} ({100*n_pos/n_trades:5.1f}%)")
            print(f"    = $0:    {n_zero:5d}")
            # Sample 5 entry credits
            print(f"    sample of 5 entry credits: {[f'${c:.2f}' for c in credits[:5]]}")
        else:
            print("    (no trades opened)")

        # --- Output 3: Wing-width values being computed (5 sample dates) ---
        print(f"\n--- (3) {ticker} wing-width values (5 dates from successful trades or skips) ---")
        # Try to derive 5 sample (date, spot) pairs
        sample_pairs: list[tuple] = []
        if n_trades > 0:
            for t in result.trades[:5]:
                sample_pairs.append((t.entry_date, t.entry_spx, t.spread.short_leg.strike,
                                     t.spread.long_leg.strike, abs(t.spread.short_leg.strike - t.spread.long_leg.strike)))
        # Fallback: derive spot from ticker-specific OHLC ladder (use SPX feed for SPX,
        # otherwise approximate by searching the chain for a near-the-money strike)
        if not sample_pairs:
            print("    (no successful trades to sample; computing wing for synthetic spot range)")
            for spot_test in [1500, 2000, 2500, 3500, 5000]:
                w = compute_wing_width(ticker, spot_test)
                inc = STRIKE_INCREMENT_BY_UNDERLYING.get(ticker, 5)
                target = WING_PCT_OF_SPOT * spot_test
                print(f"    spot=${spot_test:>6.0f}  inc=${inc:<3d}  target=${target:>6.2f}  snapped wing=${w:<3d}  ({100*w/spot_test:.3f}%)")
        else:
            inc = STRIKE_INCREMENT_BY_UNDERLYING.get(ticker, 5)
            print(f"    [strike grid step: ${inc}]")
            print(f"    {'date':10s}  {'spot':>8s}  {'short_K':>8s}  {'long_K':>8s}  {'wing':>5s}  {'pct_of_spot':>11s}")
            for d, spot, short_k, long_k, w in sample_pairs:
                print(f"    {str(d):10s}  ${spot:>7.2f}  ${short_k:>7.0f}  ${long_k:>7.0f}  ${w:>4.0f}  {100*w/spot:>10.3f}%")

        # --- Output 4: Sample 5 actual successful trades ---
        print(f"\n--- (4) {ticker} sample 5 successful trades ---")
        if n_trades > 0:
            # Random sample reproducible
            rng = np.random.default_rng(SEED)
            idx = rng.choice(n_trades, size=min(5, n_trades), replace=False)
            print(f"    {'date':10s}  {'side':4s}  {'spot':>8s}  {'shortK':>7s}  {'longK':>7s}  {'wing':>4s}  {'credit':>7s}  {'debit':>7s}  {'fate':14s}  {'pnl':>9s}")
            for j in sorted(idx):
                t = result.trades[j]
                side = t.spread.right
                short_k = t.spread.short_leg.strike
                long_k = t.spread.long_leg.strike
                wing = abs(short_k - long_k)
                credit = t.entry_credit_per_spread
                debit = t.exit_debit_per_spread or 0.0
                pnl = t.pnl_per_spread or 0.0
                print(f"    {str(t.entry_date):10s}  {side:4s}  ${t.entry_spx:>7.2f}  ${short_k:>6.0f}  ${long_k:>6.0f}  ${wing:>3.0f}  ${credit:>6.2f}  ${debit:>6.2f}  {t.fate:14s}  ${pnl:>+8.2f}")
        else:
            print("    (no trades opened)")

        # Track summary row
        rows_summary.append({
            "ticker": ticker,
            "n_trades": n_trades,
            "n_skipped": n_skipped,
            "size_zero_skips": int(pd.DataFrame(result.skipped_entries)["reason"].str.startswith("size_zero").sum())
                if result.skipped_entries else 0,
        })

        # Free per-ticker
        if ticker in _TICKER_DF_CACHE:
            del _TICKER_DF_CACHE[ticker]
        del pricer, result
        gc.collect()

    print(f"\n{'=' * 90}")
    print("CONSOLIDATED SUMMARY")
    print(f"{'=' * 90}")
    print(pd.DataFrame(rows_summary).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
