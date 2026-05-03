"""
Head 1 per-instrument trade-quality trainer (Phase B).

For each ticker in the VRP universe, runs a labelling backtest and
trains an XGBoost classifier with auto-selected calibration (Platt for
small calibration sets per Niculescu-Mizil 2005; isotonic for n>=200)
and class-imbalance handling via scale_pos_weight.

Per PRE_COMMITMENT_VRP §5.1 Head 1:
  - Per-instrument target: pnl_per_spread > 0 over the realized
    holding window (PT/SL/time/emergency/eos).
  - Walk-forward annual folds.
  - Output: calibrated p_quality per (ticker, entry_date), used in
    ml_only / full modes for the continuous sizing curve.

Acceptance criterion (per §5.1):
  Head 1 must achieve OOS log-loss strictly below the within-fold
  majority-class baseline. Failure on >= 3 of 5 instruments → drop the
  Head 1 sizing curve and use base sizing per fallback rule. Negative
  finding reported either way.

Memory discipline: loads one ticker's pricer at a time, runs label
backtest, frees DataFrame, trains, saves. No two tickers' OptionMetrics
DataFrames in memory simultaneously.

Output:
  - data/processed/head1_per_instrument/{TICKER}_p_quality.parquet
  - data/processed/head1_per_instrument/{TICKER}_labels.parquet
  - data/processed/head1_per_instrument/acceptance_summary.csv
  - logs/head1_per_instrument.log

Usage:
  PYTHONPATH=. .venv/bin/python scripts/train_head1_per_instrument.py
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.loader import load_inputs
from src.config import DATA_PROCESSED_DIR, IS_END, IS_START, OOS_END, OOS_START, SEED
from src.models.label_trades import label_trades_dataframe
from src.models.xgboost_primary import (
    fit_xgb_calibrated,
    predict_calibrated_auto,
)
from src.strategy.optionmetrics_pricer import _TICKER_DF_CACHE, make_pricer_v2


log = logging.getLogger(__name__)


VRP_UNIVERSE: list[str] = ["SPX", "RUT", "NDX", "TLT", "GLD"]


def majority_class_log_loss(y_true: pd.Series) -> float:
    """Log-loss baseline: predict the majority-class probability for
    every observation. This is the §5.1 acceptance comparator."""
    p = float(y_true.mean())   # base rate of class 1
    if p <= 0 or p >= 1:
        return float("nan")
    # log-loss = -mean(y log p + (1-y) log(1-p))
    return float(
        -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean()
    )


def model_log_loss(y_true: pd.Series, y_prob: pd.Series) -> float:
    y_t, y_p = y_true.align(y_prob, join="inner")
    if len(y_t) == 0:
        return float("nan")
    eps = 1e-9
    p = y_p.clip(eps, 1 - eps)
    return float(-(y_t * np.log(p) + (1 - y_t) * np.log(1 - p)).mean())


def label_trades_for_ticker(
    ticker: str, inputs, start: str, end: str,
) -> tuple[pd.DataFrame, dict]:
    """Run a naked-mode IC backtest for `ticker`, return labelled trades
    + summary dict."""
    log.info("Labelling %s with naked-mode IC backtest...", ticker)
    t0 = time.time()
    pricer = make_pricer_v2(ticker)
    log.info("  %s pricer loaded in %.1fs", ticker, time.time() - t0)

    t = time.time()
    result = run_backtest(
        inputs, pricer, mode="naked",
        start=start, end=end, use_iron_condor=True,
    )
    log.info("  %s naked backtest done in %.1fs (n_trades=%d)",
             ticker, time.time() - t, len(result.trades))

    labels = label_trades_dataframe(result)
    summary = {
        "ticker": ticker,
        "n_trades": len(labels),
        "win_rate": float(labels["win"].mean()) if len(labels) else float("nan"),
    }

    # Free the per-ticker DataFrame
    if ticker in _TICKER_DF_CACHE:
        del _TICKER_DF_CACHE[ticker]
    del pricer
    gc.collect()

    return labels, summary


def train_head1_for_ticker(
    ticker: str,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    fold_starts: list[pd.Timestamp],
    fold_end: pd.Timestamp,
) -> tuple[pd.Series, dict]:
    """Annual walk-forward training of Head 1 for one ticker.

    For each fold start fs:
      1. training labels = entries with index < fs
      2. fit XGBoost + auto-calibration
      3. predict calibrated probabilities for entries in [fs, next_fs)
    Concatenate predictions across folds → per-trade p_quality.

    Returns (p_quality_series, summary_dict).

    Per §5.1 acceptance: model OOS log-loss strictly below majority-class
    baseline → "pass." Otherwise "fail" → caller falls back to base sizing.
    """
    log.info("Training Head 1 for %s (%d labels, %d features)",
             ticker, len(labels), len(features.columns))
    out_preds: list[pd.Series] = []

    fold_starts = sorted(pd.Timestamp(fs) for fs in fold_starts)
    for i, fs in enumerate(fold_starts):
        fe = fold_starts[i + 1] if i + 1 < len(fold_starts) else fold_end

        train_labels = labels.loc[labels.index < fs]
        if len(train_labels) < 30:
            log.warning("  fold %s: only %d training labels; skipping",
                        fs.date(), len(train_labels))
            continue
        if train_labels["win"].nunique() < 2:
            log.warning("  fold %s: training labels are single-class; skipping",
                        fs.date())
            continue

        # Pair each training label with PRIOR-trading-day features
        # (Bug #1 fix: features at Friday close, not Monday close).
        train_idx = pd.DatetimeIndex(train_labels.index)
        prior_pos = features.index.searchsorted(train_idx, side="left") - 1
        valid = prior_pos >= 0
        train_idx = train_idx[valid]
        prior_pos = prior_pos[valid]
        if len(train_idx) == 0:
            continue
        X_train = features.iloc[prior_pos]
        X_train.index = train_idx
        y_train = train_labels.loc[train_idx]["win"].astype(int)

        try:
            model, calibrator, cal_method = fit_xgb_calibrated(
                X_train, y_train, seed=SEED, method="auto",
                use_scale_pos_weight=True,
            )
        except Exception as e:
            log.error("  fold %s training failed: %s", fs.date(), e)
            continue

        # Predict for trade entries in [fs, fe)
        fold_labels = labels.loc[(labels.index >= fs) & (labels.index < fe)]
        if len(fold_labels) == 0:
            continue
        pred_idx = pd.DatetimeIndex(fold_labels.index)
        prior_pos_pred = features.index.searchsorted(pred_idx, side="left") - 1
        valid_pred = prior_pos_pred >= 0
        pred_idx = pred_idx[valid_pred]
        prior_pos_pred = prior_pos_pred[valid_pred]
        if len(pred_idx) == 0:
            continue
        X_pred = features.iloc[prior_pos_pred]
        X_pred.index = pred_idx
        preds = predict_calibrated_auto(model, calibrator, X_pred)
        out_preds.append(preds)
        log.info(
            "  fold %s: trained on %d (%d wins, %d losses), cal=%s, "
            "predicted %d trades",
            fs.date(), len(X_train),
            int((y_train == 1).sum()), int((y_train == 0).sum()),
            cal_method, len(preds),
        )

    if not out_preds:
        return pd.Series(dtype=float, name="p_quality"), {
            "ticker": ticker, "n_oos": 0,
            "log_loss_model": float("nan"),
            "log_loss_baseline": float("nan"),
            "passed": False,
        }

    p_quality = pd.concat(out_preds).sort_index().rename("p_quality")
    # Acceptance: OOS log-loss vs majority-class on the same OOS labels
    oos_labels = labels.loc[labels.index >= fold_starts[0]]["win"].astype(int)
    common = oos_labels.index.intersection(p_quality.index)
    y_oos = oos_labels.loc[common]
    p_oos = p_quality.loc[common]
    ll_model = model_log_loss(y_oos, p_oos)
    ll_baseline = majority_class_log_loss(y_oos)
    passed = (
        not np.isnan(ll_model)
        and not np.isnan(ll_baseline)
        and ll_model < ll_baseline
    )

    return p_quality, {
        "ticker": ticker,
        "n_oos": int(len(common)),
        "log_loss_model": ll_model,
        "log_loss_baseline": ll_baseline,
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
            logging.FileHandler("logs/head1_per_instrument.log", mode="w"),
        ],
    )

    out_dir = Path(DATA_PROCESSED_DIR) / "head1_per_instrument"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Head 1 per-instrument trainer (Phase B-1)")
    log.info("=" * 70)
    log.info("Universe: %s", args.tickers)

    inputs = load_inputs()
    features = pd.read_parquet(Path(DATA_PROCESSED_DIR) / "features.parquet")
    log.info("Features: %d rows × %d cols", *features.shape)

    # Annual fold starts within OOS
    fs_list = []
    fs = pd.Timestamp(OOS_START)
    while fs <= pd.Timestamp(OOS_END):
        fs_list.append(fs)
        fs = fs + pd.DateOffset(years=1)

    summaries: list[dict] = []
    for ticker in args.tickers:
        try:
            labels, lbl_summary = label_trades_for_ticker(
                ticker, inputs, start=IS_START, end=OOS_END,
            )
            log.info("  %s: %d trades labelled, win rate %.1f%%",
                     ticker, lbl_summary["n_trades"],
                     100 * lbl_summary["win_rate"])
            labels.to_parquet(out_dir / f"{ticker}_labels.parquet")

            p_quality, head1_summary = train_head1_for_ticker(
                ticker, features, labels,
                fold_starts=fs_list, fold_end=pd.Timestamp(OOS_END),
            )
            if len(p_quality) > 0:
                p_quality.to_frame().to_parquet(
                    out_dir / f"{ticker}_p_quality.parquet"
                )

            row = {**lbl_summary, **head1_summary}
            summaries.append(row)
            log.info(
                "  %s: log-loss model=%.4f vs baseline=%.4f → %s",
                ticker, head1_summary["log_loss_model"],
                head1_summary["log_loss_baseline"],
                "PASS" if head1_summary["passed"] else "FAIL",
            )
        except Exception as e:
            log.exception("Ticker %s training pipeline failed: %s", ticker, e)
            summaries.append({"ticker": ticker, "passed": False, "error": str(e)})

    summary_df = pd.DataFrame(summaries)
    summary_path = out_dir / "acceptance_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Saved acceptance summary to %s", summary_path)

    log.info("=" * 70)
    log.info("ACCEPTANCE GATE (PRE_COMMITMENT_VRP §5.1):")
    log.info("=" * 70)
    n_pass = int(summary_df.get("passed", pd.Series(dtype=bool)).sum())
    n_total = len(summary_df)
    log.info("  Per-instrument PASS count: %d / %d", n_pass, n_total)
    log.info("  Required: PASS on > %d (failure on ≥ 3 of %d → drop Head 1)",
             n_total - 3, n_total)
    if n_total - n_pass >= 3:
        log.info(
            "  GATE: FAIL — Head 1 sizing curve dropped per §5.1 fallback. "
            "Use base sizing in ml_only/full modes."
        )
    else:
        log.info(
            "  GATE: PASS — Head 1 sizing curve activated for passing names; "
            "fallback to base sizing for failing names."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
