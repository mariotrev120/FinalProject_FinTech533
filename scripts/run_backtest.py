"""
End-to-end CLI runner. Loads inputs, runs one or all ablation modes,
prints a per-mode summary.

This is a thin orchestrator over `src.backtest`. All reusable logic lives
in src/.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_backtest.py
    PYTHONPATH=. .venv/bin/python scripts/run_backtest.py --mode naked --start 2018-01-01
"""
from __future__ import annotations

import argparse
import logging

from src.backtest.engine import run_backtest
from src.backtest.loader import load_inputs, summarize
from src.config import IS_START, OOS_END
from src.strategy.black_scholes import make_default_pricer


log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["naked", "ml_only", "halts_only", "full", "all"], default="all")
    p.add_argument("--start", default=IS_START)
    p.add_argument("--end", default=OOS_END)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    inputs = load_inputs()
    pricer = make_default_pricer()

    modes = ["naked", "ml_only", "halts_only", "full"] if args.mode == "all" else [args.mode]
    for m in modes:
        if m in ("ml_only", "full") and inputs.ml_probability is None:
            print(f"[skip {m}] no ML probability series wired in yet")
            continue
        log.info("running mode=%s on %s -> %s", m, args.start, args.end)
        result = run_backtest(inputs, pricer, mode=m, start=args.start, end=args.end)
        s = summarize(result, m)
        print(f"\n=== mode={m} ===")
        for k, v in s.items():
            if k == "fates":
                print(f"  {k}: {v}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
