"""
IBKR / TWS data fetcher.

Pulls daily TRADES bars for the universe of indices and ETFs the strategy
needs from the Windows host running TWS, via ib_async. Saves each series as a
parquet file under data/raw/.

Confirmed working on TWS paper account, April 2026 probe:
- IND/CBOE: SPX, VIX, VIX3M, VVIX, SKEW, XSP, IRX, FVX, TNX, TYX (back to 2011-05+)
- STK/SMART: SPY, GLD, HYG, LQD, TLT, IEF (back to 2011-05+, TLT/IEF later)

Not implemented here:
- Historical option chain bars. Paper accounts cannot fetch them at any
  duration / whatToShow / exchange. See TWS_CONNECTION.md.

Usage:
    python -m src.data.fetch_ibkr [--duration 15Y]
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ib_async import IB, Index, Stock, util

from src.config import (
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID_BASE, DATA_RAW_DIR,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Symbol:
    name: str
    sec_type: str            # "IND" or "STK"
    exchange: str            # "CBOE" for IND, "SMART" for STK
    currency: str = "USD"


UNIVERSE: list[Symbol] = [
    # Volatility regime
    Symbol("SPX",   "IND", "CBOE"),
    Symbol("VIX",   "IND", "CBOE"),
    Symbol("VIX3M", "IND", "CBOE"),
    Symbol("VVIX",  "IND", "CBOE"),
    Symbol("SKEW",  "IND", "CBOE"),
    # Mini contract index (for size-aligned reporting)
    Symbol("XSP",   "IND", "CBOE"),
    # Yield curve (CBOE 10x yield indices)
    Symbol("IRX",   "IND", "CBOE"),  # 13-week T-bill
    Symbol("FVX",   "IND", "CBOE"),  # 5-year
    Symbol("TNX",   "IND", "CBOE"),  # 10-year
    Symbol("TYX",   "IND", "CBOE"),  # 30-year
    # Cross-asset stress
    Symbol("SPY",   "STK", "SMART"),
    Symbol("TLT",   "STK", "SMART"),
    Symbol("IEF",   "STK", "SMART"),
    Symbol("GLD",   "STK", "SMART"),
    Symbol("HYG",   "STK", "SMART"),
    Symbol("LQD",   "STK", "SMART"),
]

# Auxiliary IV / RV series — same SPX index contract, different whatToShow
# These are pulled separately with whatToShow=OPTION_IMPLIED_VOLATILITY and
# HISTORICAL_VOLATILITY. TWS paper accounts return ~15 years (2011-05+) of
# both series, which is the actual ATM IV and realized vol the market was
# pricing/observing — much better than using VIX/100 as a synthetic proxy.
IV_HV_SYMBOLS: list[Symbol] = [
    Symbol("SPX",   "IND", "CBOE"),
]


def make_contract(s: Symbol):
    if s.sec_type == "IND":
        return Index(s.name, s.exchange, s.currency)
    if s.sec_type == "STK":
        return Stock(s.name, s.exchange, s.currency)
    raise ValueError(f"unsupported sec_type {s.sec_type} for {s.name}")


def fetch_one(
    ib: IB, s: Symbol, duration_str: str, whatToShow: str = "TRADES",
) -> pd.DataFrame:
    """Pull daily bars for a single symbol, return as DataFrame.

    endDateTime="" means "now" — TWS returns bars ending today, going back the
    full duration_str. We accept the ib_async limitation that explicit historical
    end-dates throw a format error and just slice the resulting frame locally.
    """
    contract = make_contract(s)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration_str,
        barSizeSetting="1 day",
        whatToShow=whatToShow,
        useRTH=True,
        formatDate=1,
    )
    if not bars:
        raise RuntimeError(f"empty response for {s.name}")
    df = util.df(bars)
    df["symbol"] = s.name
    df["sec_type"] = s.sec_type
    df["exchange"] = s.exchange
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def save(df: pd.DataFrame, symbol: str, out_dir: Path) -> Path:
    out_path = out_dir / f"{symbol}.parquet"
    df.to_parquet(out_path)
    return out_path


def fetch_all(duration_str: str = "15 Y") -> dict[str, pd.DataFrame]:
    """Connect once, pull everything in UNIVERSE, save each to parquet, return
    dict of dataframes."""
    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID_BASE, timeout=30)
    ib.reqMarketDataType(3)  # delayed feed; required for paper account
    log.info("connected to TWS at %s:%d (delayed feed enabled)", IBKR_HOST, IBKR_PORT)

    results: dict[str, pd.DataFrame] = {}
    try:
        for s in UNIVERSE:
            try:
                df = fetch_one(ib, s, duration_str, whatToShow="TRADES")
            except Exception as e:
                log.error("FAIL %-6s %s: %s", s.name, type(e).__name__, str(e)[:120])
                continue
            path = save(df, s.name, DATA_RAW_DIR)
            results[s.name] = df
            log.info("OK   %-6s %5d bars %s -> %s saved %s",
                     s.name, len(df), df.index.min().date(), df.index.max().date(),
                     path.name)

        # Auxiliary IV / RV series — same SPX index contract pulled with
        # different whatToShow values. Save with suffixed names.
        for s in IV_HV_SYMBOLS:
            for wts, suffix in [("OPTION_IMPLIED_VOLATILITY", "IV"),
                                 ("HISTORICAL_VOLATILITY", "HV")]:
                try:
                    df = fetch_one(ib, s, duration_str, whatToShow=wts)
                except Exception as e:
                    log.error("FAIL %s_%s %s: %s", s.name, suffix, type(e).__name__, str(e)[:120])
                    continue
                key = f"{s.name}_{suffix}"
                path = save(df, key, DATA_RAW_DIR)
                results[key] = df
                log.info("OK   %-8s %5d bars %s -> %s saved %s",
                         key, len(df), df.index.min().date(), df.index.max().date(),
                         path.name)
    finally:
        ib.disconnect()
        log.info("disconnected")

    return results


def main() -> int:
    p = argparse.ArgumentParser(description="Pull universe daily bars from TWS")
    p.add_argument("--duration", default="15 Y",
                   help="IBKR durationStr (e.g. '15 Y', '5 Y', '500 D'). Default 15 Y.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    results = fetch_all(args.duration)
    print(f"\nFetched {len(results)} / {len(UNIVERSE)} symbols")
    for name, df in results.items():
        print(f"  {name:6s} {len(df):5d} bars {df.index.min().date()} -> {df.index.max().date()}")
    return 0 if len(results) == len(UNIVERSE) else 1


if __name__ == "__main__":
    raise SystemExit(main())
