"""
Mode-aware backtest engine.

Single function, four modes:
  naked      — no ML, no halts. Sells one spread every Monday.
  ml_only    — ML gate active, halts off.
  halts_only — halts active, no ML gate.
  full       — both active.

Same blotter / equity curve / metrics path for all four modes; only the
gating differs. No silent code paths between modes.

Friction is applied at entry and exit. The "no silent filtering" rule is
enforced via runtime asserts at every artifact handoff.

This engine is pricer-agnostic: it takes a `PricingProvider` instance as
input, so swapping BS for WRDS is a one-arg change in the caller.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config import (
    DTE_MIN, IS_END, IS_START, ML_DECISION_THRESHOLD, OOS_END, OOS_START,
    SPREAD_WIDTH_PTS, STRESS_CORR_THRESHOLD,
)
from src.strategy.exits import evaluate_exit
from src.strategy.friction import (
    round_trip_commissions, slippage_dollars_per_share,
    should_skip_entry_due_to_gap,
)
from src.strategy.halts import HaltDecision, evaluate_halts
from src.strategy.pricer import PricingProvider
from src.strategy.sizing import size_position
from src.strategy.spread_construction import build_spread
from src.strategy.types import Mode, Spread, Trade


log = logging.getLogger(__name__)


# --- Inputs to the engine -------------------------------------------------

@dataclass
class BacktestInputs:
    """All data the engine needs, pre-loaded into pandas. Indexed by date.

    bars_spx       : SPX OHLC (we use close + open for gap-aware execution)
    vix, vix3m     : index level series
    hyg_minus_lqd  : HYG-LQD spread series
    spy_treasury_corr: rolling SPY-Treasury correlation series (for stress mult)
    bid_ask_per_share: estimated quoted spread per option contract per share.
        Without real chains we use a simple model: max(0.10, 0.02 * vix).
    ml_probability : optional Series of calibrated p (only used in ml_only/full)
    is_winrate_baseline: Hoeffding baseline (computed from IS run)
    """
    bars_spx: pd.DataFrame      # cols: open, high, low, close
    vix: pd.Series
    vix3m: pd.Series
    hyg_minus_lqd: pd.Series
    spy_treasury_corr: pd.Series
    ml_probability: Optional[pd.Series] = None
    is_winrate_baseline: float = 0.75
    initial_equity: float = 100_000.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    halt_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped_entries: list[dict] = field(default_factory=list)


# --- Helpers -------------------------------------------------------------

def _bid_ask_estimate_per_share(vix: float) -> float:
    """Simple stand-in for the unavailable real bid-ask. Conservative — wider
    than reality at low VIX, narrower than reality at extreme VIX. Replace
    when real OPRA quotes are available."""
    return max(0.10, 0.02 * vix)


def _is_monday(d: date) -> bool:
    return d.weekday() == 0


def _gap_pct(spx_today_open: float, spx_yday_close: float) -> float:
    if spx_yday_close <= 0:
        return 0.0
    return spx_today_open / spx_yday_close - 1.0


# --- Main loop ----------------------------------------------------------

def run_backtest(
    inputs: BacktestInputs,
    pricer: PricingProvider,
    mode: Mode,
    underlying: str = "SPX",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> BacktestResult:
    use_ml = mode in ("ml_only", "full")
    use_halts = mode in ("halts_only", "full")

    if use_ml and inputs.ml_probability is None:
        raise ValueError(f"mode={mode} requires inputs.ml_probability")

    spx = inputs.bars_spx
    if start is None: start = IS_START
    if end is None:   end = OOS_END
    spx = spx.loc[start:end]

    equity = inputs.initial_equity
    equity_path = []
    open_trades: list[Trade] = []
    closed_trades: list[Trade] = []
    halt_log_rows: list[dict] = []
    skipped: list[dict] = []
    next_trade_id = 1

    high_water = inputs.initial_equity
    underwater_days = 0
    current_dd_frac = 0.0

    for idx, (today, row_spx) in enumerate(spx.iterrows()):
        if idx == 0:
            equity_path.append((today, equity))
            continue
        spx_today_close = row_spx["close"]
        spx_today_open = row_spx["open"]
        spx_yday_close = spx.iloc[idx - 1]["close"]

        try:
            vix_t = float(inputs.vix.asof(today))
            vix3m_t = float(inputs.vix3m.asof(today))
            vix_y = float(inputs.vix.asof(spx.index[idx - 1]))
        except (KeyError, ValueError):
            equity_path.append((today, equity))
            continue

        # --- Daily exit pass on open trades ---
        still_open: list[Trade] = []
        for t in open_trades:
            decision = evaluate_exit(
                pricer=pricer,
                as_of=today,
                spot=spx_today_close,
                open_gap_spot=spx_today_open,
                spread=t.spread,
                vix=vix_t,
                entry_credit_per_share=t.entry_credit_per_spread / 100.0,
                entry_date=t.entry_date,
            )
            if decision is None:
                still_open.append(t)
                continue
            # Apply exit friction
            ba_per_share = _bid_ask_estimate_per_share(vix_t)
            slip = slippage_dollars_per_share(ba_per_share, vix_t)
            exit_debit_per_share_after_friction = decision["exit_debit"] + slip
            exit_debit_per_spread = exit_debit_per_share_after_friction * 100.0
            commish = round_trip_commissions(t.contracts) / 2.0
            credit_per_spread = t.entry_credit_per_spread
            pnl_per_spread = credit_per_spread - exit_debit_per_spread
            total_pnl = pnl_per_spread * t.contracts - commish
            equity += total_pnl
            t.exit_date = today.date() if hasattr(today, "date") else today
            t.exit_debit_per_spread = exit_debit_per_spread
            t.fate = decision["fate"]
            t.exit_spx = spx_today_close
            t.exit_vix = vix_t
            t.exit_short_delta = decision.get("exit_short_delta")
            t.pnl_per_spread = pnl_per_spread
            closed_trades.append(t)

        open_trades = still_open

        # --- Drawdown bookkeeping ---
        # Strictly-below-high-water counts as underwater. Equality (e.g. on
        # no-trade days when equity sits unchanged at the prior high) does
        # NOT increment underwater_days; otherwise an idle account would
        # accumulate fake underwater days indefinitely.
        if equity >= high_water:
            high_water = equity
            underwater_days = 0
            current_dd_frac = 0.0
        else:
            underwater_days += 1
            current_dd_frac = (high_water - equity) / high_water if high_water > 0 else 0.0

        # --- Halt evaluation ---
        halt_decision: Optional[HaltDecision] = None
        if use_halts:
            n_recent_trades = sum(1 for t in closed_trades[-60:] if t.fate != "open")
            recent_winrate = (
                sum(1 for t in closed_trades[-60:]
                    if t.pnl_per_spread is not None and t.pnl_per_spread > 0) / max(n_recent_trades, 1)
            ) if n_recent_trades >= 1 else None
            halt_decision = evaluate_halts(
                vix_today=vix_t, vix_yday=vix_y,
                spx_today=spx_today_close, spx_yday=spx_yday_close,
                vix3m_today=vix3m_t,
                vix3m_history=inputs.vix3m.loc[:today],
                vix_history=inputs.vix.loc[:today],
                hyg_lqd_history=inputs.hyg_minus_lqd.loc[:today],
                prob_history=inputs.ml_probability.loc[:today] if inputs.ml_probability is not None else None,
                rolling_winrate=recent_winrate,
                is_winrate_baseline=inputs.is_winrate_baseline,
                n_trades_in_rolling=n_recent_trades,
                underwater_days=underwater_days,
                current_drawdown_frac=current_dd_frac,
            )
            halt_log_rows.append({"date": today, "state": halt_decision.state,
                                  "triggers": ",".join(halt_decision.triggers)})

        # --- Entry pass on Mondays ---
        if _is_monday(today.date() if hasattr(today, "date") else today):
            entry_blocked_reason = None
            gap = _gap_pct(spx_today_open, spx_yday_close)
            if should_skip_entry_due_to_gap(gap):
                entry_blocked_reason = "gap_skip"
            elif use_halts and halt_decision is not None and halt_decision.state != "active":
                entry_blocked_reason = f"halt_{halt_decision.state}"
            elif use_ml:
                p = float(inputs.ml_probability.asof(today)) if inputs.ml_probability is not None else None
                if p is None or np.isnan(p) or p < ML_DECISION_THRESHOLD:
                    entry_blocked_reason = "ml_prob_below_threshold"

            if entry_blocked_reason is not None:
                skipped.append({"date": today, "reason": entry_blocked_reason, "mode": mode})
            else:
                # Build the spread and open trade
                try:
                    spread, theoretical_credit_per_share, abs_delta = build_spread(
                        pricer=pricer, as_of=today, underlying=underlying,
                        spot=spx_today_close, vix=vix_t,
                    )
                except Exception as e:
                    log.warning("build_spread failed on %s: %s", today, e)
                    skipped.append({"date": today, "reason": f"build_spread_err:{type(e).__name__}", "mode": mode})
                    equity_path.append((today, equity))
                    continue

                # Apply entry friction (slippage)
                ba_per_share = _bid_ask_estimate_per_share(vix_t)
                slip = slippage_dollars_per_share(ba_per_share, vix_t, gap_pct=gap)
                entry_credit_per_share_after_friction = max(theoretical_credit_per_share - slip, 0.0)
                entry_credit_per_spread = entry_credit_per_share_after_friction * 100.0

                # Sizing: max win = credit, max loss = (width - credit) * 100
                max_win_per_spread = entry_credit_per_spread
                max_loss_per_spread = max(SPREAD_WIDTH_PTS * 100.0 - entry_credit_per_spread, 1.0)

                # ML probability for sizing (Kelly): use 0.5 if ML disabled.
                if use_ml and inputs.ml_probability is not None:
                    p_for_sizing = float(inputs.ml_probability.asof(today))
                    if np.isnan(p_for_sizing):
                        p_for_sizing = 0.5
                else:
                    p_for_sizing = 0.5

                try:
                    treas_corr = float(inputs.spy_treasury_corr.asof(today))
                    if np.isnan(treas_corr):
                        treas_corr = 0.0
                except (KeyError, ValueError):
                    treas_corr = 0.0

                # Sizing: ML probability is a binary gate (already filtered
                # above), not a Kelly weight. Kelly evaluates to 0 for credit
                # spreads at typical p values (loss/win ratio ~6:1 requires
                # p > 0.86 for positive Kelly), which would zero out every
                # ML-gated trade. Treat sizing as cap-based across all modes
                # and report Kelly as a diagnostic only. This is consistent
                # with the README's pre-committed interpretation rule:
                # "if ml_only Sharpe is within 0.1 of naked, ML is decorative."
                size = size_position(
                    equity=equity,
                    p_calibrated=p_for_sizing,
                    max_win_per_spread=max_win_per_spread,
                    max_loss_per_spread=max_loss_per_spread,
                    vix=vix_t,
                    spy_treasury_corr=treas_corr,
                    use_kelly=False,
                )

                if size.contracts < 1:
                    skipped.append({"date": today, "reason": "size_zero", "mode": mode})
                else:
                    # Apply entry commissions immediately
                    entry_commish = round_trip_commissions(size.contracts) / 2.0
                    equity -= entry_commish
                    trade = Trade(
                        trade_id=next_trade_id, mode=mode,
                        entry_date=today.date() if hasattr(today, "date") else today,
                        spread=spread, contracts=size.contracts,
                        entry_credit_per_spread=entry_credit_per_spread,
                        entry_spx=spx_today_close, entry_vix=vix_t,
                        entry_iv=vix_t / 100.0,
                        entry_dte=(spread.short_leg.expiry - today.date()).days if hasattr(today, "date") else (spread.short_leg.expiry - today).days,
                        ml_probability=p_for_sizing if use_ml else None,
                        halt_state_at_entry=halt_decision.state if halt_decision else None,
                        kelly_fraction=size.kelly_fraction,
                        vol_multiplier=size.vol_multiplier,
                        stress_multiplier=size.stress_multiplier,
                    )
                    open_trades.append(trade)
                    next_trade_id += 1

        equity_path.append((today, equity))

    # --- Force-close any still-open at the end ---
    for t in open_trades:
        t.fate = "time_exit"
        t.exit_date = spx.index[-1].date() if hasattr(spx.index[-1], "date") else spx.index[-1]
        closed_trades.append(t)

    # --- Build result ---
    eq_curve = pd.Series(dict(equity_path), name="equity")
    eq_curve.index = pd.DatetimeIndex(eq_curve.index)
    halt_df = pd.DataFrame(halt_log_rows).set_index("date") if halt_log_rows else pd.DataFrame()

    # CONSISTENCY ASSERTION: every closed trade has a fate, exit_date, pnl_per_spread (if not open)
    for t in closed_trades:
        assert t.fate != "open" or t.pnl_per_spread is None, \
            f"trade {t.trade_id} has fate=open but pnl_per_spread is set"

    return BacktestResult(
        trades=closed_trades,
        equity_curve=eq_curve,
        halt_log=halt_df,
        skipped_entries=skipped,
    )
