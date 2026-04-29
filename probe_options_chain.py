"""
IBKR options-chain feasibility probe for the SPX VRP project.

Goal: confirm shinybroker can pull historical SPX (and XSP) put-option bars
across the proposed backtest window (2015-2024), at strikes and expiries
representative of the strategy's design (~5% OTM, ~30-45 DTE).

Probes 6 historical dates spanning calm regimes, the COVID crisis, and
post-rate-hike periods. For each date:
  1. Pull SPX index spot via fetch_historical_data.
  2. Compute candidate strike (0.95 * spot, rounded to nearest 25).
  3. Compute candidate expiry (3rd Friday of the next-next calendar month).
  4. Try fetch_historical_data on the resulting put contract.

Reports per-date success/failure. If most dates succeed, IBKR backbone is
viable. If most fail, pivot to SPY chains or alternative data source before
sinking effort into the production fetcher.

Run AFTER starting TWS on the Windows host with API enabled (port 7497,
paper account logged in). The HOST below is the WSL-side address of the
Windows host — same one HW5 used.
"""
from __future__ import annotations

import datetime as dt
import sys

import shinybroker as sb

HOST = "172.29.208.1"
PORT = 7497
CLIENT_ID = 9999
TIMEOUT = 30

PROBE_DATES = [
    "20150615",
    "20180601",
    "20200115",
    "20200615",
    "20220615",
    "20240617",
]


def third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    offset = (4 - d.weekday()) % 7
    return d + dt.timedelta(days=offset + 14)


def target_expiry_for(end_date_str: str) -> str:
    d = dt.datetime.strptime(end_date_str, "%Y%m%d").date()
    target_month = d.month + 2
    target_year = d.year + (target_month - 1) // 12
    target_month = ((target_month - 1) % 12) + 1
    return third_friday(target_year, target_month).strftime("%Y%m%d")


def round_strike(price: float, increment: int = 25) -> int:
    return int(round(price / increment) * increment)


def fetch_index_spot(symbol: str, exchange: str, end_dt: str) -> float | None:
    contract = sb.Contract({
        "symbol": symbol,
        "secType": "IND",
        "exchange": exchange,
        "currency": "USD",
    })
    r = sb.fetch_historical_data(
        contract=contract,
        endDateTime=f"{end_dt} 23:59:59 US/Eastern",
        durationStr="5 D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        host=HOST, port=PORT, client_id=CLIENT_ID,
        timeout=TIMEOUT,
    )
    if r is None or "hst_dta" not in r or r["hst_dta"] is None:
        return None
    bars = r["hst_dta"]
    if len(bars) == 0:
        return None
    return float(bars["close"].iloc[-1])


def probe_put(symbol: str, exchange: str, strike: int, expiry: str, end_dt: str):
    contract = sb.Contract({
        "symbol": symbol,
        "secType": "OPT",
        "exchange": exchange,
        "currency": "USD",
        "right": "P",
        "strike": strike,
        "lastTradeDateOrContractMonth": expiry,
        "multiplier": "100",
    })
    try:
        r = sb.fetch_historical_data(
            contract=contract,
            endDateTime=f"{end_dt} 23:59:59 US/Eastern",
            durationStr="1 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            host=HOST, port=PORT, client_id=CLIENT_ID,
            timeout=TIMEOUT,
        )
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"

    if r is None:
        return False, "None response"
    if "hst_dta" not in r or r["hst_dta"] is None:
        return False, "no hst_dta key"
    bars = r["hst_dta"]
    if len(bars) == 0:
        return False, "0 bars"
    return True, f"{len(bars)} bars, close={float(bars['close'].iloc[-1]):.2f}"


def main() -> int:
    print(f"Probe started {dt.datetime.now().isoformat(timespec='seconds')}")
    print(f"TWS host {HOST}:{PORT}, client_id {CLIENT_ID}\n")

    overall = {"SPX": [], "XSP": []}

    for symbol, exchange in [("SPX", "CBOE"), ("XSP", "CBOE")]:
        print(f"=== {symbol} on {exchange} ===")
        for end_dt in PROBE_DATES:
            spot = fetch_index_spot(symbol, exchange, end_dt)
            if spot is None:
                print(f"  {end_dt}: spot fetch failed")
                overall[symbol].append((end_dt, False, "no spot"))
                continue
            strike = round_strike(spot * 0.95, increment=25 if symbol == "SPX" else 5)
            expiry = target_expiry_for(end_dt)
            ok, detail = probe_put(symbol, exchange, strike, expiry, end_dt)
            status = "OK " if ok else "FAIL"
            print(f"  {end_dt}: spot={spot:.2f} strike={strike} expiry={expiry} {status} ({detail})")
            overall[symbol].append((end_dt, ok, detail))
        print()

    print("=== Summary ===")
    for symbol, results in overall.items():
        n_ok = sum(1 for _, ok, _ in results if ok)
        print(f"  {symbol}: {n_ok}/{len(results)} probe dates returned data")

    spx_ok = sum(1 for _, ok, _ in overall["SPX"] if ok)
    xsp_ok = sum(1 for _, ok, _ in overall["XSP"] if ok)
    print()
    if xsp_ok >= 4:
        print("VERDICT: XSP feasible. Proceed with XSP as primary instrument.")
        return 0
    if spx_ok >= 4:
        print("VERDICT: XSP coverage thin but SPX feasible. Pivot to SPX (lose 1/10 sizing convenience, keep Section 1256).")
        return 0
    print("VERDICT: Both SPX and XSP coverage thin via IBKR. Pivot to SPY chains (lose Section 1256 narrative) or external data source.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
