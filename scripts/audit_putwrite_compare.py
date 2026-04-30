"""
Tier 3 audit: PutWrite reference comparison on the same OOS window.

Computes annualized return, annualized vol, Sharpe (vs same risk-free),
and max drawdown for the CBOE PutWrite index (^PUT) over 2018-2024 using
yfinance, on the SAME methodology as our strategy. Reports side-by-side.

If yfinance doesn't return PutWrite (sometimes only ^PUT.Q is available),
falls back to the published CBOE numbers as a reference range.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.config import DATA_RAW_DIR, OOS_END, OOS_START
from src.metrics.performance import equity_curve_metrics


warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# CBOE published reference (for fallback): PUT index 2018-2024
PUBLISHED_PUTWRITE_REF = {
    "annualized_return_pct": 6.5,    # rough midpoint of published ranges
    "annualized_vol_pct": 11.0,
    "sharpe_annualized": 0.40,
    "max_drawdown_pct": 24.0,        # March 2020 crash
}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rfr = pd.read_parquet(DATA_RAW_DIR / "IRX.parquet")["close"]

    try:
        import yfinance as yf
        log.info("Pulling ^PUT (CBOE PutWrite Index) from yfinance...")
        # Try several tickers PutWrite has been listed under
        for tic in ["^PUT", "PUT", "^PUTW", "PUTW"]:
            d = yf.download(tic, start=OOS_START, end=OOS_END,
                             auto_adjust=False, progress=False)
            if not d.empty:
                log.info("  yfinance returned PutWrite under ticker '%s' (%d days)", tic, len(d))
                break
        else:
            d = pd.DataFrame()

        if d.empty:
            print("yfinance has no PutWrite data; using published CBOE numbers:")
            print(PUBLISHED_PUTWRITE_REF)
            return 0

        if isinstance(d.columns, pd.MultiIndex):
            close = d[("Close", tic)]
        else:
            close = d["Close"]
        close.index = pd.to_datetime(close.index)
        close = close.loc[OOS_START:OOS_END].dropna()

        # Use CBOE's index value as the equity series.
        m = equity_curve_metrics(close, risk_free_curve=rfr)

        print("\n" + "=" * 60)
        print(f"PutWrite ({tic}) reference, OOS {OOS_START} -> {OOS_END}")
        print("=" * 60)
        print(f"  trading days observed:    {len(close)}")
        print(f"  annualized return:        {m['annualized_return_pct']:+.2f}%")
        print(f"  annualized vol:           {m['annualized_vol_pct']:.2f}%")
        print(f"  annualized risk-free:     {m['annualized_rf_pct']:.2f}%")
        print(f"  annualized Sharpe:        {m['sharpe_annualized']:+.3f}")
        print(f"  max drawdown:             {m['max_drawdown_pct']:.2f}%")
        print()

        # Compare to current strategy attribution if available
        try:
            attr = pd.read_parquet("data/processed/component_attribution.parquet")
            naked_oos = attr[(attr["mode"] == "naked") & (attr["label"] == "OOS")].iloc[0]
            print("Strategy (naked, post-bugfix):")
            print(f"  annualized return:        {naked_oos['annualized_return_pct']:+.2f}%")
            print(f"  annualized vol:           {naked_oos['annualized_vol_pct']:.2f}%")
            print(f"  annualized Sharpe:        {naked_oos['sharpe_annualized']:+.3f}")
            print(f"  max drawdown:             {naked_oos['max_drawdown_pct']:.2f}%")
            print()
            print("DELTA (strategy - PutWrite):")
            print(f"  annualized return:        {naked_oos['annualized_return_pct'] - m['annualized_return_pct']:+.2f} pp")
            print(f"  annualized Sharpe:        {naked_oos['sharpe_annualized'] - m['sharpe_annualized']:+.3f}")
            print(f"  max drawdown:             {naked_oos['max_drawdown_pct'] - m['max_drawdown_pct']:+.2f} pp")
        except Exception as e:
            print(f"(could not load component attribution for comparison: {e})")

        return 0
    except ImportError:
        print("yfinance not installed; using published CBOE numbers:")
        print(PUBLISHED_PUTWRITE_REF)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
