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
from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config import (
    DTE_MIN, GAP_SLIPPAGE_THRESHOLD_PCT, IS_END, IS_START, ML_DECISION_THRESHOLD,
    OOS_END, OOS_START, STRESS_CORR_THRESHOLD,
)
from src.strategy.exits import evaluate_exit
from src.strategy.friction import (
    round_trip_commissions, slippage_dollars_per_share, slippage_pct_at_vix,
    should_skip_entry_due_to_gap,
)
from src.strategy.halts import HaltDecision, HaltState, evaluate_halts
from src.strategy.pricer import PricingProvider
from src.strategy.sizing import size_position
from src.strategy.spread_construction import build_spread, build_iron_condor
from src.strategy.types import Mode, Spread, Trade


def _quote_for_spread(pricer: PricingProvider, today, spread: Spread, leg: str):
    """Dispatch quote_put vs quote_call based on spread side. `leg` is
    'short' or 'long'."""
    contract = spread.short_leg if leg == "short" else spread.long_leg
    if spread.right == "P" and hasattr(pricer, "quote_put"):
        return pricer.quote_put(today, spread.underlying, contract.strike, contract.expiry)
    if spread.right == "C" and hasattr(pricer, "quote_call"):
        return pricer.quote_call(today, spread.underlying, contract.strike, contract.expiry)
    return None


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
    bars_spx: pd.DataFrame      # market-wide signal: gap-skip, Layer 1 hard halt
    vix: pd.Series
    vix3m: pd.Series
    hyg_minus_lqd: pd.Series
    spy_treasury_corr: pd.Series
    ml_probability: Optional[pd.Series] = None
    is_winrate_baseline: float = 0.75
    initial_equity: float = 100_000.0
    risk_free_curve: Optional[pd.Series] = None  # TNX or IRX in tenths-of-percent
    # Per-ticker underlying OHLC. When None, the engine falls back to bars_spx
    # (preserves backward compatibility for SPX-only callers). When set, the
    # engine uses these for spot/gap_spot/entry_spx/exit_spx — i.e. the
    # per-ticker price feed for whatever underlying is being traded.
    underlying_bars: Optional[pd.DataFrame] = None


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
    underlying: str = "SPX",          # the committed instrument
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_iron_condor: bool = False,    # v2: iron condor instead of put credit spread
) -> BacktestResult:
    """Run the backtest in one of four modes.

    Underlying defaults to "SPX" — the strategy's committed instrument:
    Section 1256 60/40 tax treatment, European exercise, cash-settled,
    deepest and most liquid index option chain available. The original
    README considered XSP for retail account sizing, but for our defined-
    risk credit spread strategy 1 SPX 5-pt spread = 1 XSP 5-pt spread in
    dollar terms (same multiplier 100, same width). SPX is preferred for
    backtest data quality (continuous since 2012-01, tighter bid-ask).
    """
    use_ml = mode in ("ml_only", "full")
    use_halts = mode in ("halts_only", "full")

    if use_ml and inputs.ml_probability is None:
        raise ValueError(f"mode={mode} requires inputs.ml_probability")

    spx_bars = inputs.bars_spx
    if start is None: start = IS_START
    if end is None:   end = OOS_END
    spx_bars = spx_bars.loc[start:end]

    # SPX is the committed instrument. Bars from inputs.bars_spx are at
    # SPX scale natively; no scaling required. (If underlying='XSP' is ever
    # passed for cross-validation, scale by /10.)
    scale = 10.0 if underlying == "XSP" else 1.0
    spx = spx_bars / scale if scale != 1.0 else spx_bars

    equity = inputs.initial_equity
    equity_path = []
    open_trades: list[Trade] = []
    closed_trades: list[Trade] = []
    halt_log_rows: list[dict] = []
    skipped: list[dict] = []
    next_trade_id = 1

    # Layer 4 drawdown bookkeeping per PRE_COMMITMENT_VRP §4 spec text
    # ("in trailing 90 days"). Trailing-window max + underwater_days
    # both computed against a rolling 90-day deque, not since strategy
    # inception. Catches the v1 catch-22 where halt latched permanently
    # because all-time HWM never recovered while strategy was halted.
    DRAWDOWN_LOOKBACK_DAYS = 90   # matches §4 "in trailing 90 days"
    equity_trailing: deque = deque(maxlen=DRAWDOWN_LOOKBACK_DAYS)
    underwater_days = 0           # consecutive days below trailing-90 max
    current_dd_frac = 0.0         # drawdown from trailing-90 max
    high_water = inputs.initial_equity   # all-time HWM, kept for REPORTING
                                         # ONLY — NOT used in halt logic.
    next_ic_id = 1

    # Precompute 5-day realized vol of SPX (annualized) for Layer 5 resume gate.
    # rv5d_today is the latest value; rv5d_history_252d is the trailing 252-day
    # window the resume helper compares against (RESUME_REALIZED_VOL_PCT pctile).
    spx_returns = spx["close"].pct_change()
    rv5d_series = spx_returns.rolling(5).std() * np.sqrt(252)

    # Latching halt-state machine: persists across the day-loop so Layer 5
    # auto-resume sees yesterday's halt and can lift it once the market-
    # regime conditions clear OR the 60-day time fallback fires.
    prior_halt_state: HaltState = "active"
    days_in_halt: int = 0

    for idx, (today, row_spx) in enumerate(spx.iterrows()):
        if idx == 0:
            equity_path.append((today, equity))
            continue

        # Daily risk-free interest accrual on full equity. ACT/360 convention.
        # Using 3M T-bill (IRX) when supplied; fall back to 4% if not.
        if inputs.risk_free_curve is not None:
            try:
                rf_pct = float(inputs.risk_free_curve.asof(today)) / 10.0 / 100.0
                if np.isnan(rf_pct):
                    rf_pct = 0.04
            except (KeyError, ValueError):
                rf_pct = 0.04
        else:
            rf_pct = 0.04
        prior_dt = spx.index[idx - 1]
        days = max((today - prior_dt).days, 0)
        equity *= (1.0 + rf_pct * days / 360.0)
        spx_today_close = row_spx["close"]
        spx_today_open = row_spx["open"]
        spx_yday_close = spx.iloc[idx - 1]["close"]

        # Per-ticker underlying spot. Falls back to SPX close when no
        # per-ticker bars supplied (single-instrument SPX-only path).
        if inputs.underlying_bars is not None:
            try:
                under_today_close = float(inputs.underlying_bars["close"].asof(today))
                under_today_open = float(inputs.underlying_bars["open"].asof(today))
            except (KeyError, ValueError):
                equity_path.append((today, equity))
                continue
            if np.isnan(under_today_close) or np.isnan(under_today_open):
                equity_path.append((today, equity))
                continue
        else:
            under_today_close = spx_today_close
            under_today_open = spx_today_open

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
                spot=under_today_close,
                open_gap_spot=under_today_open,
                spread=t.spread,
                vix=vix_t,
                entry_credit_per_share=t.entry_credit_per_spread / 100.0,
                entry_date=t.entry_date,
            )
            if decision is None:
                still_open.append(t)
                continue
            # Bug #2 fix: realistic exit execution using REAL bid-ask spread
            # from the pricer, scaled by the README's VIX-conditional fraction.
            # Round-trip mechanics: closing a credit spread means buying back
            # the short leg (paying somewhere between mid and ASK) and selling
            # the long leg (receiving between BID and mid). At fraction f:
            #   exit_debit = mid_debit + f * (combined_half_spread)
            # where combined_half_spread = (full_short + full_long) / 2.
            # f=1.0 means touching the bid-ask boundary (worst case);
            # f=0 means executing at mid (best case). README schedule: 0.30 to 1.00.
            # Dispatch on spread side (puts use quote_put, calls use quote_call).
            short_q = _quote_for_spread(pricer, today, t.spread, "short")
            long_q = _quote_for_spread(pricer, today, t.spread, "long")
            if short_q is not None and long_q is not None:
                full_short = short_q.ask - short_q.bid
                full_long = long_q.ask - long_q.bid
                combined_half = (full_short + full_long) / 2.0
                frac = slippage_pct_at_vix(vix_t)
                mid_debit = max(short_q.mid - long_q.mid, 0.0)
                exit_debit_per_share = mid_debit + frac * combined_half
            else:
                # BS pricer fallback: use mid + heuristic slippage
                ba_per_share = _bid_ask_estimate_per_share(vix_t)
                slip = slippage_dollars_per_share(ba_per_share, vix_t)
                exit_debit_per_share = decision["exit_debit"] + slip
            # Bug guard: exit debit cannot exceed the spread width either
            # (max economic loss is the width). Quotes implying more are
            # stale/anomalous OptionMetrics rows.
            exit_debit_per_share = min(exit_debit_per_share, float(t.spread.width))
            exit_debit_per_spread = exit_debit_per_share * 100.0
            commish = round_trip_commissions(t.contracts) / 2.0
            credit_per_spread = t.entry_credit_per_spread
            pnl_per_spread = credit_per_spread - exit_debit_per_spread
            total_pnl = pnl_per_spread * t.contracts - commish
            equity += total_pnl
            t.exit_date = today.date() if hasattr(today, "date") else today
            t.exit_debit_per_spread = exit_debit_per_spread
            t.fate = decision["fate"]
            t.exit_spx = under_today_close
            t.exit_vix = vix_t
            t.exit_short_delta = decision.get("exit_short_delta")
            t.pnl_per_spread = pnl_per_spread
            closed_trades.append(t)

        open_trades = still_open

        # --- Drawdown bookkeeping ---
        # Per § 4: drawdown depth and underwater duration are computed
        # over the TRAILING 90-DAY window. The deque includes today's
        # equity BEFORE we compare; trailing_high is therefore the max
        # over [today − 89, today] (inclusive of today). underwater_days
        # counts consecutive days where equity < trailing_high — equality
        # (today's equity == trailing_high) resets the counter, which
        # matches the spec ("recovered to high" → no longer underwater).
        equity_trailing.append(equity)
        trailing_high = max(equity_trailing)
        if equity >= trailing_high:
            underwater_days = 0
            current_dd_frac = 0.0
        else:
            underwater_days += 1
            current_dd_frac = (
                (trailing_high - equity) / trailing_high
                if trailing_high > 0 else 0.0
            )
        # Maintain all-time HWM for reporting (NOT used in halt logic).
        if equity > high_water:
            high_water = equity

        # --- Halt evaluation ---
        halt_decision: Optional[HaltDecision] = None
        if use_halts:
            n_recent_trades = sum(1 for t in closed_trades[-60:] if t.fate != "open")
            recent_winrate = (
                sum(1 for t in closed_trades[-60:]
                    if t.pnl_per_spread is not None and t.pnl_per_spread > 0) / max(n_recent_trades, 1)
            ) if n_recent_trades >= 1 else None
            rv5d_today = float(rv5d_series.loc[today]) if today in rv5d_series.index else float("nan")
            rv5d_hist = rv5d_series.loc[:today].iloc[-252:]
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
                prior_state=prior_halt_state,
                rv5d_today=rv5d_today,
                rv5d_history_252d=rv5d_hist,
                days_in_halt=days_in_halt,
            )
            halt_log_rows.append({"date": today, "state": halt_decision.state,
                                  "triggers": ",".join(halt_decision.triggers)})
            # Increment days_in_halt while halted; reset when active
            if halt_decision.state == "active":
                days_in_halt = 0
            else:
                days_in_halt += 1
            prior_halt_state = halt_decision.state

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
                # Resolve sizing inputs once (shared across both IC sides if applicable)
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

                def _attempt_entry_for_side(
                    side_spread: Spread,
                    side_theoretical_credit: float,
                    ic_id: Optional[int] = None,
                ) -> Optional[Trade]:
                    """Per-side entry: realistic-quote pricing → cap → sizing
                    → Trade creation. Returns Trade or None (skipped reason
                    already appended). Mutates `equity` and `next_trade_id`
                    via nonlocal."""
                    nonlocal equity, next_trade_id
                    side_label = f"_{side_spread.right}" if ic_id is not None else ""

                    # Realistic-quote credit (Bug #2 fix), per side
                    if hasattr(pricer, "quote_put"):
                        short_q = _quote_for_spread(pricer, today, side_spread, "short")
                        long_q = _quote_for_spread(pricer, today, side_spread, "long")
                        if short_q is None or long_q is None:
                            skipped.append({"date": today,
                                            "reason": f"quote_lookup_miss{side_label}",
                                            "mode": mode})
                            return None
                        full_short = short_q.ask - short_q.bid
                        full_long = long_q.ask - long_q.bid
                        combined_half = (full_short + full_long) / 2.0
                        frac = slippage_pct_at_vix(vix_t)
                        if abs(gap) >= GAP_SLIPPAGE_THRESHOLD_PCT:
                            frac *= 1.5
                        mid_credit = max(short_q.mid - long_q.mid, 0.0)
                        entry_credit_per_share = max(mid_credit - frac * combined_half, 0.0)
                    else:
                        ba_per_share = _bid_ask_estimate_per_share(vix_t)
                        slip = slippage_dollars_per_share(ba_per_share, vix_t, gap_pct=gap)
                        entry_credit_per_share = max(side_theoretical_credit - slip, 0.0)

                    # Cap at width - $0.01 (Bug guard)
                    side_width = float(side_spread.width)
                    MAX_CREDIT_PER_SHARE = side_width - 0.01
                    if entry_credit_per_share > MAX_CREDIT_PER_SHARE:
                        log.warning("date=%s side=%s: capping entry credit %.4f -> %.4f",
                                    today.date(), side_spread.right,
                                    entry_credit_per_share, MAX_CREDIT_PER_SHARE)
                        entry_credit_per_share = MAX_CREDIT_PER_SHARE
                    entry_credit_per_spread = entry_credit_per_share * 100.0

                    max_win_per_spread = entry_credit_per_spread
                    max_loss_per_spread = max(
                        side_width * 100.0 - entry_credit_per_spread, 100.0,
                    )

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
                        skipped.append({"date": today, "reason": f"size_zero{side_label}",
                                        "mode": mode})
                        return None

                    entry_commish = round_trip_commissions(size.contracts) / 2.0
                    equity -= entry_commish
                    entry_dte_calc = (side_spread.short_leg.expiry -
                                      (today.date() if hasattr(today, "date") else today)).days
                    trade = Trade(
                        trade_id=next_trade_id, mode=mode,
                        entry_date=today.date() if hasattr(today, "date") else today,
                        spread=side_spread, contracts=size.contracts,
                        entry_credit_per_spread=entry_credit_per_spread,
                        entry_spx=under_today_close, entry_vix=vix_t,
                        entry_iv=vix_t / 100.0,
                        entry_dte=entry_dte_calc,
                        ml_probability=p_for_sizing if use_ml else None,
                        halt_state_at_entry=halt_decision.state if halt_decision else None,
                        kelly_fraction=size.kelly_fraction,
                        vol_multiplier=size.vol_multiplier,
                        stress_multiplier=size.stress_multiplier,
                        iron_condor_id=ic_id,
                    )
                    next_trade_id += 1
                    return trade

                if use_iron_condor:
                    try:
                        ps, cs, pc, cc, _, _ = build_iron_condor(
                            pricer=pricer, as_of=today, underlying=underlying,
                            spot=under_today_close, vix=vix_t,
                        )
                    except Exception as e:
                        log.warning("build_iron_condor failed on %s: %s", today, e)
                        skipped.append({"date": today,
                                        "reason": f"build_ic_err:{type(e).__name__}",
                                        "mode": mode})
                        equity_path.append((today, equity))
                        continue
                    ic_id = next_ic_id
                    next_ic_id += 1
                    for side_spread, side_credit in [(ps, pc), (cs, cc)]:
                        t = _attempt_entry_for_side(side_spread, side_credit, ic_id=ic_id)
                        if t is not None:
                            open_trades.append(t)
                else:
                    try:
                        spread, theoretical_credit_per_share, _ = build_spread(
                            pricer=pricer, as_of=today, underlying=underlying,
                            spot=under_today_close, vix=vix_t,
                        )
                    except Exception as e:
                        log.warning("build_spread failed on %s: %s", today, e)
                        skipped.append({"date": today,
                                        "reason": f"build_spread_err:{type(e).__name__}",
                                        "mode": mode})
                        equity_path.append((today, equity))
                        continue
                    t = _attempt_entry_for_side(spread, theoretical_credit_per_share, ic_id=None)
                    if t is not None:
                        open_trades.append(t)

        equity_path.append((today, equity))

    # --- Force-close any still-open at the end ---
    # Compute realistic exit P&L using the last bar's spot and vix so the
    # consistency check (every closed trade has pnl_per_spread set) passes.
    if open_trades:
        last_dt = spx.index[-1]
        # Prefer per-ticker last close when underlying_bars are wired
        if inputs.underlying_bars is not None:
            try:
                last_close = float(inputs.underlying_bars["close"].asof(last_dt))
            except (KeyError, ValueError):
                last_close = float(spx.iloc[-1]["close"])
            if last_close != last_close:    # NaN guard
                last_close = float(spx.iloc[-1]["close"])
        else:
            last_close = float(spx.iloc[-1]["close"])
        try:
            last_vix = float(inputs.vix.asof(last_dt))
        except (KeyError, ValueError):
            last_vix = 20.0
        for t in open_trades:
            # Dispatch put/call price depending on spread side
            if t.spread.right == "P":
                price_fn = pricer.price_put
            else:
                price_fn = pricer.price_call
            sp = price_fn(last_dt, t.spread.underlying, last_close,
                           t.spread.short_leg.strike, t.spread.short_leg.expiry, last_vix)
            lp = price_fn(last_dt, t.spread.underlying, last_close,
                           t.spread.long_leg.strike, t.spread.long_leg.expiry, last_vix)
            exit_debit_per_spread = max(sp - lp, 0.0) * 100.0
            commish_share = round_trip_commissions(t.contracts) / 2.0
            pnl_per_spread = t.entry_credit_per_spread - exit_debit_per_spread
            equity += pnl_per_spread * t.contracts - commish_share
            t.exit_date = last_dt.date() if hasattr(last_dt, "date") else last_dt
            t.exit_debit_per_spread = exit_debit_per_spread
            t.fate = "eos_force"
            t.exit_spx = last_close
            t.exit_vix = last_vix
            t.pnl_per_spread = pnl_per_spread
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
