"""
Head 3 per-instrument skew-direction trainer (Phase C).

Per PRE_COMMITMENT_VRP §5.1 Head 3:

  - Target (binary, per trade entry date):
        which side realizes higher loss probability over the next ~30d?
        i.e., did the put-side LOSE more than the call-side, or vice versa,
        in the realized holding window?
  - Output: dynamic delta selection from {0.10, 0.16, 0.25} per side,
    replacing the static 0.16/0.16. (THIS script trains the binary
    direction predictor; the dynamic-delta selection logic that consumes
    the predictor is engine-side and Phase D — once Head 3 is trained
    we wire its output into the engine's strike selection.)
  - Acceptance: Head 3 OOS log-loss must beat majority-class baseline on
    ≥ 4 of 5 instruments. If fails for an instrument → fall back to
    static 0.16/0.16 for that instrument. Static-fallback equity curve
    is reported as the headline for failing instruments.

Training data construction:
  Each iron condor produces a put-side Trade and a call-side Trade with
  shared iron_condor_id. For each IC pair, compute:
    put_loss  = max(0, -put_pnl_per_spread)    # how much the put lost
    call_loss = max(0, -call_pnl_per_spread)   # how much the call lost
    label = 1 if put_loss > call_loss else 0   # put hurt more?

  Index by put-side entry_date.

Memory discipline: same as Head 1 — one ticker's pricer at a time.

Output:
  - data/processed/head3_per_instrument/{TICKER}_p_skew.parquet
  - data/processed/head3_per_instrument/{TICKER}_skew_labels.parquet
  - data/processed/head3_per_instrument/acceptance_summary.csv
  - logs/head3_per_instrument.log

Usage:
  PYTHONPATH=. .venv/bin/python scripts/train_head3_per_instrument.py
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_inputs
from src.config import DATA_PROCESSED_DIR, IS_START, OOS_END, OOS_START, SEED
from src.models.xgboost_primary import (
    fit_xgb_calibrated,
    predict_calibrated_auto,
)
from src.strategy.optionmetrics_pricer import _TICKER_DF_CACHE, make_pricer_v2


log = logging.getLogger(__name__)


VRP_UNIVERSE: list[str] = ["SPX", "RUT", "NDX", "TLT", "GLD"]


def label_skew_direction(result) -> pd.DataFrame:
    """For each iron condor pair, label whether the put-side or
    call-side incurred the larger loss.

    label = 1  → put-side hurt more (skew bias toward downside risk)
    label = 0  → call-side hurt more (skew bias toward upside risk)

    Indexed by entry_date of the put-side trade.
    """
    pairs: dict[int, dict] = defaultdict(dict)
    for t in result.trades:
        if t.iron_condor_id is None:
            continue
        if t.pnl_per_spread is None:
            continue
        side = "put" if t.spread.right == "P" else "call"
        pairs[t.iron_condor_id][side] = t

    rows = []
    for ic_id, sides in pairs.items():
        if "put" not in sides or "call" not in sides:
            continue
        p, c = sides["put"], sides["call"]
        put_loss = max(0.0, -float(p.pnl_per_spread))
        call_loss = max(0.0, -float(c.pnl_per_spread))
        # Tie → drop (not informative for binary classifier)
        if put_loss == call_loss:
            continue
        rows.append({
            "entry_date": pd.Timestamp(p.entry_date),
            "ic_id": ic_id,
            "put_loss": put_loss,
            "call_loss": call_loss,
            "label": int(put_loss > call_loss),
        })
    return pd.DataFrame(rows).set_index("entry_date").sort_index()


def majority_class_log_loss(y: pd.Series) -> float:
    p = float(y.mean())
    if p <= 0 or p >= 1:
        return float("nan")
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def model_log_loss(y: pd.Series, p: pd.Series) -> float:
    y_t, y_p = y.align(p, join="inner")
    if len(y_t) == 0:
        return float("nan")
    eps = 1e-9
    pp = y_p.clip(eps, 1 - eps)
    return float(-(y_t * np.log(pp) + (1 - y_t) * np.log(1 - pp)).mean())


def label_for_ticker(ticker: str, inputs, start: str, end: str) -> pd.DataFrame:
    log.info("Labelling %s skew-direction with naked-mode IC backtest...", ticker)
    t0 = time.time()
    pricer = make_pricer_v2(ticker)
    log.info("  %s pricer loaded in %.1fs", ticker, time.time() - t0)
    t = time.time()
    result = run_backtest(
        inputs, pricer, mode="naked",
        start=start, end=end, use_iron_condor=True,
    )
    log.info("  %s backtest done in %.1fs (n_trades=%d)",
             ticker, time.time() - t, len(result.trades))
    labels = label_skew_direction(result)

    if ticker in _TICKER_DF_CACHE:
        del _TICKER_DF_CACHE[ticker]
    del pricer
    gc.collect()
    return labels


def train_head3_for_ticker(
    ticker: str,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    fold_starts: list[pd.Timestamp],
    fold_end: pd.Timestamp,
):
    log.info(
        "Training Head 3 (skew direction) for %s (%d labels, base rate %.1f%%)",
        ticker, len(labels), 100 * labels["label"].mean(),
    )
    out_preds: list[pd.Series] = []
    fold_starts = sorted(pd.Timestamp(fs) for fs in fold_starts)
    for i, fs in enumerate(fold_starts):
        fe = fold_starts[i + 1] if i + 1 < len(fold_starts) else fold_end
        train_labels = labels.loc[labels.index < fs]
        if len(train_labels) < 30 or train_labels["label"].nunique() < 2:
            continue
        train_idx = pd.DatetimeIndex(train_labels.index)
        prior_pos = features.index.searchsorted(train_idx, side="left") - 1
        valid = prior_pos >= 0
        train_idx, prior_pos = train_idx[valid], prior_pos[valid]
        if len(train_idx) == 0:
            continue
        X_train = features.iloc[prior_pos].copy()
        X_train.index = train_idx
        y_train = train_labels.loc[train_idx]["label"].astype(int)

        try:
            model, calibrator, cal_method = fit_xgb_calibrated(
                X_train, y_train, seed=SEED, method="auto",
                use_scale_pos_weight=True,
            )
        except Exception as e:
            log.error("  fold %s training failed: %s", fs.date(), e)
            continue

        fold_labels = labels.loc[(labels.index >= fs) & (labels.index < fe)]
        if len(fold_labels) == 0:
            continue
        pred_idx = pd.DatetimeIndex(fold_labels.index)
        prior_pos_pred = features.index.searchsorted(pred_idx, side="left") - 1
        valid_pred = prior_pos_pred >= 0
        pred_idx, prior_pos_pred = pred_idx[valid_pred], prior_pos_pred[valid_pred]
        if len(pred_idx) == 0:
            continue
        X_pred = features.iloc[prior_pos_pred].copy()
        X_pred.index = pred_idx
        preds = predict_calibrated_auto(model, calibrator, X_pred)
        out_preds.append(preds)
        log.info("  fold %s: trained %d (cal=%s), predicted %d",
                 fs.date(), len(X_train), cal_method, len(preds))

    if not out_preds:
        return pd.Series(dtype=float, name="p_skew"), {
            "ticker": ticker, "n_oos": 0,
            "log_loss_model": float("nan"),
            "log_loss_baseline": float("nan"),
            "passed": False,
        }
    p_skew = pd.concat(out_preds).sort_index().rename("p_skew")
    oos_labels = labels.loc[labels.index >= fold_starts[0]]["label"].astype(int)
    common = oos_labels.index.intersection(p_skew.index)
    y_oos = oos_labels.loc[common]
    p_oos = p_skew.loc[common]
    ll_model = model_log_loss(y_oos, p_oos)
    ll_baseline = majority_class_log_loss(y_oos)
    passed = (
        not np.isnan(ll_model) and not np.isnan(ll_baseline)
        and ll_model < ll_baseline
    )
    return p_skew, {
        "ticker": ticker, "n_oos": int(len(common)),
        "log_loss_model": ll_model, "log_loss_baseline": ll_baseline,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=VRP_UNIVERSE)
    args = parser.parse_args(argv)

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/head3_per_instrument.log", mode="w"),
        ],
    )

    out_dir = Path(DATA_PROCESSED_DIR) / "head3_per_instrument"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Head 3 per-instrument skew-direction trainer (Phase C)")
    log.info("=" * 70)

    inputs = load_inputs()
    features = pd.read_parquet(Path(DATA_PROCESSED_DIR) / "features.parquet")
    log.info("Features: %d rows × %d cols", *features.shape)

    fs_list = []
    fs = pd.Timestamp(OOS_START)
    while fs <= pd.Timestamp(OOS_END):
        fs_list.append(fs)
        fs = fs + pd.DateOffset(years=1)

    summaries = []
    for ticker in args.tickers:
        try:
            labels = label_for_ticker(
                ticker, inputs, start=IS_START, end=OOS_END,
            )
            log.info("  %s: %d skew labels, %s%% put-side-loss",
                     ticker, len(labels),
                     f"{100 * labels['label'].mean():.1f}" if len(labels) else "n/a")
            labels.to_parquet(out_dir / f"{ticker}_skew_labels.parquet")

            p_skew, summary = train_head3_for_ticker(
                ticker, features, labels,
                fold_starts=fs_list, fold_end=pd.Timestamp(OOS_END),
            )
            if len(p_skew) > 0:
                p_skew.to_frame().to_parquet(out_dir / f"{ticker}_p_skew.parquet")
            summary["base_rate"] = float(labels["label"].mean()) if len(labels) else float("nan")
            summary["n_labels"] = int(len(labels))
            summaries.append(summary)
            log.info(
                "  %s: log-loss model=%.4f vs baseline=%.4f → %s",
                ticker, summary["log_loss_model"], summary["log_loss_baseline"],
                "PASS" if summary["passed"] else "FAIL",
            )
        except Exception as e:
            log.exception("Ticker %s failed: %s", ticker, e)
            summaries.append({"ticker": ticker, "passed": False, "error": str(e)})

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "acceptance_summary.csv", index=False)

    log.info("=" * 70)
    log.info("ACCEPTANCE GATE (PRE_COMMITMENT_VRP §5.1):")
    log.info("=" * 70)
    n_pass = int(summary_df.get("passed", pd.Series(dtype=bool)).sum())
    n_total = len(summary_df)
    log.info("  PASS count: %d / %d (need ≥ 4 of 5 to activate dynamic delta)",
             n_pass, n_total)
    if n_pass >= 4:
        log.info("  GATE: PASS — dynamic delta selection activated for passing names.")
    else:
        log.info("  GATE: FAIL — fall back to static 0.16/0.16 for failing names.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
