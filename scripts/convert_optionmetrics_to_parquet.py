"""
One-time conversion: strat2.2.csv.gz → per-ticker parquet files.

Why this exists:
  Loading strat2.2.csv.gz (~2.4GB compressed, ~68.4M rows) into pandas
  exhausts WSL memory and crashes the session. DuckDB streams the
  decompression and partitioning in C++ without materializing the whole
  DataFrame, so it completes in <2 minutes without OOM.

  Once these per-ticker parquets exist, every downstream consumer
  (multi-instrument backtests, ML training, ablation runs) reads a
  ~50-200 MB parquet per instrument, which loads in <1s and stays in
  memory comfortably.

Output layout:
  data/processed/options_by_ticker/{TICKER}.parquet
    (one file per ticker; flat — no Hive subdirectories)

Schema preserved from CSV plus added columns:
  - All original columns: ticker, date, exdate, cp_flag, strike_price,
    best_bid, best_offer, impl_volatility, delta, gamma, vega, theta,
    volume, open_interest, ...
  - Added `strike` = CAST(strike_price AS DOUBLE) / 1000.0  (matches the
    pandas-loader convention in src/strategy/optionmetrics_pricer.py)
  - `date` and `exdate` cast to DATE (parquet logical type)

Run once. Idempotent (re-running with parquets in place is a no-op
unless --force).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/convert_optionmetrics_to_parquet.py
  PYTHONPATH=. .venv/bin/python scripts/convert_optionmetrics_to_parquet.py --force
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import duckdb


log = logging.getLogger(__name__)


CSV_PATH = "strat2.2.csv.gz"
OUTPUT_DIR = Path("data/processed/options_by_ticker")
STAGING_DIR = OUTPUT_DIR.parent / "_options_by_ticker_staging"


# Per-ticker expected row counts from sniff pass (2026-05-03).
# Used by the parity check post-conversion.
EXPECTED_ROW_COUNTS = {
    "AAPL": 2_491_630,
    "GLD": 3_545_675,
    "GOOGL": 4_624_094,
    "JNJ": 1_387_153,
    "JPM": 1_587_732,
    "KO": 1_212_308,
    "MSFT": 1_882_157,
    "NDX": 15_451_063,
    "PEP": 1_327_336,
    "PG": 1_452_732,
    "RUT": 8_841_879,
    "SPX": 20_657_223,
    "TLT": 2_419_203,
    "WMT": 1_474_655,
}
EXPECTED_TOTAL = sum(EXPECTED_ROW_COUNTS.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing per-ticker parquets",
    )
    parser.add_argument(
        "--csv",
        default=CSV_PATH,
        help=f"Input CSV path (default: {CSV_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    staging_dir = output_dir.parent / "_options_by_ticker_staging"

    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # Idempotent: skip if all 14 expected parquets already exist
    existing = {p.stem for p in output_dir.glob("*.parquet")}
    if not args.force and existing >= set(EXPECTED_ROW_COUNTS):
        log.info(
            "All %d per-ticker parquets already exist in %s — skipping conversion. "
            "Use --force to re-convert.",
            len(EXPECTED_ROW_COUNTS), output_dir,
        )
        return 0

    log.info("Connecting to DuckDB...")
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=4")

    # Cleanup any stale staging directory
    if staging_dir.exists():
        log.info("Cleaning stale staging dir: %s", staging_dir)
        shutil.rmtree(staging_dir)

    # ---------------------------------------------------------------
    # Pass 1: stream-decompress CSV, partition by ticker, write to
    # staging Hive layout. Adds `strike` derived column and casts
    # date/exdate to DATE type.
    # ---------------------------------------------------------------
    log.info("Pass 1/2: streaming CSV → Hive-partitioned parquets in %s", staging_dir)
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT
                ticker,
                CAST(date AS DATE) AS date,
                CAST(exdate AS DATE) AS exdate,
                cp_flag,
                strike_price,
                CAST(strike_price AS DOUBLE) / 1000.0 AS strike,
                best_bid,
                best_offer,
                impl_volatility,
                delta,
                gamma,
                vega,
                theta,
                volume,
                open_interest,
                CAST(date AS DATE) - CAST(exdate AS DATE) AS dte_signed_neg
            FROM read_csv_auto('{csv_path}')
        )
        TO '{staging_dir}'
        (FORMAT PARQUET,
         COMPRESSION SNAPPY,
         PARTITION_BY (ticker),
         OVERWRITE_OR_IGNORE)
    """)
    elapsed = time.time() - t0
    log.info("Pass 1 done in %.0fs", elapsed)

    # ---------------------------------------------------------------
    # Pass 2: coalesce each Hive partition into a single flat parquet.
    # DuckDB's PARTITION_BY may emit multiple .parquet files per ticker
    # depending on row volume; we want one flat file per ticker so the
    # loader can do `read_parquet(f"{ticker}.parquet")` without globbing.
    # ---------------------------------------------------------------
    log.info("Pass 2/2: coalescing Hive partitions to flat layout in %s", output_dir)
    t0 = time.time()
    actual_counts: dict[str, int] = {}

    for ticker_dir in sorted(staging_dir.iterdir()):
        if not ticker_dir.is_dir() or not ticker_dir.name.startswith("ticker="):
            continue
        ticker = ticker_dir.name.split("=", 1)[1]
        out_path = output_dir / f"{ticker}.parquet"

        # Coalesce all partition files into one parquet via DuckDB
        con.execute(f"""
            COPY (SELECT * FROM read_parquet('{ticker_dir}/*.parquet'))
            TO '{out_path}'
            (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)

        n_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{out_path}')"
        ).fetchone()[0]
        actual_counts[ticker] = n_rows
        size_mb = out_path.stat().st_size / 1024 / 1024
        log.info(
            "  %s: %s rows, %.1f MB → %s",
            ticker, f"{n_rows:,}", size_mb, out_path,
        )

    elapsed = time.time() - t0
    log.info("Pass 2 done in %.0fs", elapsed)

    # ---------------------------------------------------------------
    # Parity verification
    # ---------------------------------------------------------------
    log.info("Verifying parity vs CSV row counts...")
    fails = []
    for ticker, expected in EXPECTED_ROW_COUNTS.items():
        actual = actual_counts.get(ticker)
        if actual is None:
            fails.append(f"  MISSING: {ticker}")
        elif actual != expected:
            fails.append(
                f"  MISMATCH {ticker}: expected {expected:,}, got {actual:,} "
                f"(delta {actual - expected:+,})"
            )
        else:
            pass

    if fails:
        log.error("Parity FAILED on %d tickers:", len(fails))
        for line in fails:
            log.error(line)
        return 1

    total_actual = sum(actual_counts.values())
    log.info(
        "Parity PASS: %d tickers, %s total rows (matches expected %s)",
        len(actual_counts), f"{total_actual:,}", f"{EXPECTED_TOTAL:,}",
    )

    # ---------------------------------------------------------------
    # Cleanup staging
    # ---------------------------------------------------------------
    log.info("Cleaning staging directory: %s", staging_dir)
    shutil.rmtree(staging_dir)

    # Summary
    log.info("=" * 60)
    total_size_mb = sum(
        (output_dir / f"{t}.parquet").stat().st_size for t in EXPECTED_ROW_COUNTS
    ) / 1024 / 1024
    log.info(
        "Conversion complete. %d parquets, %.1f MB total in %s",
        len(EXPECTED_ROW_COUNTS), total_size_mb, output_dir,
    )
    log.info("Update loader to read from %s/{ticker}.parquet", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
