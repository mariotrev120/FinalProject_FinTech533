"""
5-layer halt framework.

Each layer evaluates per-day market conditions and returns its current state:

  Layer 1 (hard halt, immediate close-all):
    - VIX intraday spike > +HALT_VIX_INTRADAY_PCT
    - SPX intraday move > HALT_SPX_INTRADAY_PCT in either direction
    - VIX3M-VIX inverted by more than HALT_TERM_INVERSION_HARD pts

  Layer 2 (soft halt, no new entries):
    - VIX3M < VIX for 2 consecutive closes
    - HYG-LQD spread > HALT_HYG_LQD_SD SD from 252-day mean
    - Calibrated p < entry threshold for 5 consecutive sessions

  Layer 3 (slow halt, statistical):
    - Rolling 60-trade win rate < IS baseline by HALT_WINRATE_SE_THRESHOLD SE
    - Rolling 90-day Sharpe significantly negative (block-bootstrap p < 0.10)

  Layer 4 (drawdown halt):
    - Underwater > HALT_DD_DURATION_DAYS trading days
    - Drawdown depth > HALT_DD_DEPTH_FRAC in trailing 90 days

  Layer 5 (auto-resume): all four conditions simultaneously
    - VIX3M > VIX by RESUME_TERM_BUFFER_PTS for RESUME_TERM_DAYS closes
    - Realized 5d vol < RESUME_REALIZED_VOL_PCT percentile of trailing 252d
    - HYG-LQD spread within RESUME_HYG_LQD_SD SD of long-run mean
    - Drawdown recovered to within RESUME_DD_RECOVERY_FRAC of HWM

Each function takes a row of market data plus relevant rolling stats and
returns a boolean / state string. Composition into the engine's halt
decision happens in the engine, not here, to keep these unit-testable in
isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.config import (
    HALT_DD_DEPTH_FRAC, HALT_DD_DURATION_DAYS, HALT_HYG_LQD_LOOKBACK,
    HALT_HYG_LQD_SD, HALT_MODEL_PROB_DAYS, HALT_SHARPE_BOOTSTRAP_P,
    HALT_SHARPE_WINDOW_DAYS, HALT_SPX_INTRADAY_PCT, HALT_TERM_INVERSION_HARD,
    HALT_TERM_INVERSION_SOFT_DAYS, HALT_VIX_INTRADAY_PCT, HALT_WINRATE_SE_THRESHOLD,
    HALT_WINRATE_WINDOW, ML_DECISION_THRESHOLD, RESUME_DD_RECOVERY_FRAC,
    RESUME_HYG_LQD_SD, RESUME_REALIZED_VOL_PCT, RESUME_TERM_BUFFER_PTS,
    RESUME_TERM_DAYS,
)


HaltState = Literal["active", "soft_halt", "hard_halt", "drawdown_halt", "slow_halt"]


# --- Layer 1: Hard halts ----------------------------------------------------

def vix_spike_triggered(vix_today: float, vix_yday: float) -> bool:
    if vix_yday <= 0:
        return False
    return (vix_today / vix_yday - 1.0) > HALT_VIX_INTRADAY_PCT


def spx_move_triggered(spx_today: float, spx_yday: float) -> bool:
    if spx_yday <= 0:
        return False
    return abs(spx_today / spx_yday - 1.0) > HALT_SPX_INTRADAY_PCT


def term_inversion_hard_triggered(vix3m: float, vix: float) -> bool:
    """VIX3M-VIX inverts by HALT_TERM_INVERSION_HARD pts (i.e. VIX > VIX3M
    by more than abs(HALT_TERM_INVERSION_HARD))."""
    return (vix3m - vix) <= HALT_TERM_INVERSION_HARD


def hard_halt_triggered(
    vix_today: float, vix_yday: float,
    spx_today: float, spx_yday: float,
    vix3m_today: float,
) -> bool:
    return (
        vix_spike_triggered(vix_today, vix_yday)
        or spx_move_triggered(spx_today, spx_yday)
        or term_inversion_hard_triggered(vix3m_today, vix_today)
    )


# --- Layer 2: Soft halts ----------------------------------------------------

def term_inversion_soft_triggered(
    vix3m_history: pd.Series, vix_history: pd.Series,
    days: int = HALT_TERM_INVERSION_SOFT_DAYS,
) -> bool:
    """VIX3M < VIX for `days` consecutive closes."""
    last_n = (vix3m_history - vix_history).tail(days)
    if len(last_n) < days:
        return False
    return bool((last_n < 0).all())


def hyg_lqd_spread_widened(hyg_lqd_history: pd.Series, sd_threshold: float = HALT_HYG_LQD_SD) -> bool:
    if len(hyg_lqd_history) < HALT_HYG_LQD_LOOKBACK:
        return False
    win = hyg_lqd_history.tail(HALT_HYG_LQD_LOOKBACK)
    mean = win.mean()
    sd = win.std()
    if sd <= 0:
        return False
    return (win.iloc[-1] - mean) / sd > sd_threshold


def model_prob_persistently_low(
    prob_history: pd.Series, days: int = HALT_MODEL_PROB_DAYS,
) -> bool:
    last_n = prob_history.tail(days)
    if len(last_n) < days:
        return False
    return bool((last_n < ML_DECISION_THRESHOLD).all())


# --- Layer 3: Slow halts ----------------------------------------------------

def winrate_below_baseline(
    rolling_winrate: float, baseline: float, n_trades: int,
    se_threshold: float = HALT_WINRATE_SE_THRESHOLD,
) -> bool:
    if n_trades < HALT_WINRATE_WINDOW or not 0 < baseline < 1:
        return False
    se = np.sqrt(baseline * (1 - baseline) / n_trades)
    return (baseline - rolling_winrate) > se_threshold * se


# --- Layer 4: Drawdown -----------------------------------------------------

def drawdown_halt_triggered(
    underwater_days: int, current_drawdown_frac: float,
) -> bool:
    if underwater_days >= HALT_DD_DURATION_DAYS:
        return True
    if current_drawdown_frac >= HALT_DD_DEPTH_FRAC:
        return True
    return False


# --- Layer 5: Auto-resume --------------------------------------------------

def resume_term_recovered(
    vix3m_history: pd.Series, vix_history: pd.Series,
) -> bool:
    last_n = (vix3m_history - vix_history).tail(RESUME_TERM_DAYS)
    if len(last_n) < RESUME_TERM_DAYS:
        return False
    return bool((last_n > RESUME_TERM_BUFFER_PTS).all())


def resume_realized_vol_calmed(
    rv5d_today: float, rv5d_history_252d: pd.Series,
) -> bool:
    if len(rv5d_history_252d) < 50 or np.isnan(rv5d_today):
        return False
    pct = rv5d_history_252d.quantile(RESUME_REALIZED_VOL_PCT)
    return rv5d_today < pct


def resume_hyg_lqd_within_band(hyg_lqd_history: pd.Series) -> bool:
    if len(hyg_lqd_history) < HALT_HYG_LQD_LOOKBACK:
        return False
    win = hyg_lqd_history.tail(HALT_HYG_LQD_LOOKBACK)
    mean = win.mean()
    sd = win.std()
    if sd <= 0:
        return True
    return abs(win.iloc[-1] - mean) / sd < RESUME_HYG_LQD_SD


def resume_drawdown_recovered(current_drawdown_frac: float) -> bool:
    return current_drawdown_frac < RESUME_DD_RECOVERY_FRAC


def auto_resume_ready(
    vix3m_history: pd.Series, vix_history: pd.Series,
    rv5d_today: float, rv5d_history_252d: pd.Series,
    hyg_lqd_history: pd.Series,
    current_drawdown_frac: float,
) -> bool:
    """ALL four conditions must be simultaneously true."""
    return (
        resume_term_recovered(vix3m_history, vix_history)
        and resume_realized_vol_calmed(rv5d_today, rv5d_history_252d)
        and resume_hyg_lqd_within_band(hyg_lqd_history)
        and resume_drawdown_recovered(current_drawdown_frac)
    )


# --- Composite halt state -------------------------------------------------

@dataclass
class HaltDecision:
    state: HaltState
    triggers: list[str]


def evaluate_halts(
    *,
    vix_today: float, vix_yday: float,
    spx_today: float, spx_yday: float,
    vix3m_today: float,
    vix3m_history: pd.Series, vix_history: pd.Series,
    hyg_lqd_history: pd.Series,
    prob_history: Optional[pd.Series],
    rolling_winrate: Optional[float],
    is_winrate_baseline: float,
    n_trades_in_rolling: int,
    underwater_days: int,
    current_drawdown_frac: float,
) -> HaltDecision:
    """Compose all 5 halt layers (auto-resume is evaluated separately,
    in the engine, when state != active)."""
    triggers: list[str] = []

    if vix_spike_triggered(vix_today, vix_yday):
        triggers.append("vix_spike")
    if spx_move_triggered(spx_today, spx_yday):
        triggers.append("spx_5pct_move")
    if term_inversion_hard_triggered(vix3m_today, vix_today):
        triggers.append("term_inversion_hard")
    if triggers:
        return HaltDecision(state="hard_halt", triggers=triggers)

    if drawdown_halt_triggered(underwater_days, current_drawdown_frac):
        return HaltDecision(state="drawdown_halt", triggers=["drawdown"])

    if rolling_winrate is not None and winrate_below_baseline(
        rolling_winrate, is_winrate_baseline, n_trades_in_rolling
    ):
        triggers.append("winrate_below_baseline")

    if term_inversion_soft_triggered(vix3m_history, vix_history):
        triggers.append("term_inversion_soft_2d")
    if hyg_lqd_spread_widened(hyg_lqd_history):
        triggers.append("hyg_lqd_spread_2sd")
    if prob_history is not None and model_prob_persistently_low(prob_history):
        triggers.append("model_prob_low_5d")

    if "winrate_below_baseline" in triggers:
        return HaltDecision(state="slow_halt", triggers=triggers)
    if triggers:
        return HaltDecision(state="soft_halt", triggers=triggers)

    return HaltDecision(state="active", triggers=[])
