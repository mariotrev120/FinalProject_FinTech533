"""
Comprehensive portfolio reporting metrics — HW3/HW4 spec.

Equations are written explicitly so they can be verified against the
homework expected values. All functions take pandas Series and return
scalar floats. Returns are simple daily returns; annualization uses 252
trading days per year unless explicitly stated.

Distinction between ACT/360 (money-market accrual, used in engine for
interest accrual) and 1/252 (Sharpe / vol annualization) is preserved
explicitly.

  - Engine interest accrual uses ACT/360 (standard money-market convention)
  - Sharpe / Sortino / annualization here use trading-day convention 1/252
  - Risk-free rate from IRX is /10/100 to decimal annual rate, then
    daily-equivalent for Sharpe = annual / 252

Returns from this module are raw decimals — multiply by 100 for percent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DATA_RAW_DIR


# --- Helpers --------------------------------------------------------------

def _aligned(*series: pd.Series) -> tuple[pd.Series, ...]:
    """Inner-join series on shared dates, drop any rows with NaN in any."""
    if not series:
        return ()
    df = pd.concat(series, axis=1, join="inner").dropna()
    return tuple(df.iloc[:, i] for i in range(len(series)))


def _irx_annual_rate(irx_curve: pd.Series) -> pd.Series:
    """CBOE IRX (10x convention, e.g. 43.96 = 4.396%) -> decimal annual rate."""
    return irx_curve / 10.0 / 100.0


# --- Return / risk metrics ------------------------------------------------

def geometric_mean_return(
    returns: pd.Series, periods_per_year: int = 252,
) -> float:
    """Annualized geometric mean return.

    Equation:
        GMR_per_period = (prod(1 + r_i))^(1/n) - 1
        GMR_annualized = (1 + GMR_per_period)^periods_per_year - 1

    Equivalent closed form:
        GMR_annualized = (E_final / E_start)^(periods_per_year / n) - 1
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    cumulative = float((1.0 + r).prod())
    if cumulative <= 0:
        return float("nan")
    n = len(r)
    gmr_per_period = cumulative ** (1.0 / n) - 1.0
    return (1.0 + gmr_per_period) ** periods_per_year - 1.0


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = 252,
) -> float:
    """Annualized standard deviation of returns. sigma_ann = sigma_daily * sqrt(252)."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def downside_deviation(
    returns: pd.Series, target: float = 0.0, periods_per_year: int = 252,
) -> float:
    """Annualized downside deviation per HW5 convention: std of returns BELOW
    the target threshold.

    Equation (HW5 metrics.py):
        downside = r[r < target] - target
        d_std    = std(downside, ddof=1)
        DD_ann   = d_std * sqrt(periods_per_year)

    Default target = 0 (zero-return floor). Set target to daily rf for the
    rf-adjusted Sortino convention.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    below = r[r < target] - target
    if len(below) < 2:
        return 0.0
    return float(below.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_annualized(
    returns: pd.Series, ann_rf: float, periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe = (GMR_ann - ann_rf) / sigma_ann.

    Uses geometric (not arithmetic) mean for numerator — this matches what
    a buy-and-hold investor actually realizes over the window.
    """
    gmr = geometric_mean_return(returns, periods_per_year)
    vol = annualized_volatility(returns, periods_per_year)
    if vol <= 0 or np.isnan(gmr):
        return float("nan")
    return (gmr - ann_rf) / vol


def sortino_ratio(
    returns: pd.Series, ann_rf: float, periods_per_year: int = 252,
) -> float:
    """Annualized Sortino per HW5 convention.

    Equation (HW5 metrics.py):
        rf_daily = ann_rf / periods_per_year
        excess   = r - rf_daily
        downside = r[r < rf_daily] - rf_daily      (only sub-rf returns)
        d_std    = std(downside, ddof=1)
        Sortino  = mean(excess) / d_std * sqrt(periods_per_year)

    NOTE: HW5 uses ARITHMETIC mean of excess daily returns in the numerator
    (not GMR). This differs from some textbook treatments but matches the
    course convention.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_daily = ann_rf / periods_per_year
    excess = r - rf_daily
    downside = r[r < rf_daily] - rf_daily
    if len(downside) < 2:
        return float("nan")
    d_std = float(downside.std(ddof=1))
    if d_std == 0:
        return float("nan")
    return float(excess.mean() / d_std * np.sqrt(periods_per_year))


def calmar_ratio(returns: pd.Series, equity: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar = annualized return / |max drawdown|.

    Drawdown is taken as POSITIVE magnitude in the denominator.
    """
    gmr = geometric_mean_return(returns, periods_per_year)
    mdd = abs(max_drawdown_signed(equity))
    if mdd <= 0 or np.isnan(gmr):
        return float("nan")
    return gmr / mdd


# --- Drawdown ------------------------------------------------------------

def max_drawdown_signed(equity: pd.Series) -> float:
    """Maximum drawdown as a SIGNED fraction (negative number).

    Equation:
        DD_t        = (CumMax_t - E_t) / CumMax_t
        MaxDD_signed = -max_t(DD_t)
    """
    eq = equity.dropna()
    if len(eq) < 2:
        return 0.0
    cummax = eq.cummax()
    dd = (cummax - eq) / cummax
    return float(-dd.max())


def max_drawdown_duration_days(equity: pd.Series) -> int:
    """Longest run of consecutive trading days underwater (E_t < cummax).

    Returns the LONGEST drawdown episode in days (not the duration of the
    deepest drawdown — those can differ).
    """
    eq = equity.dropna()
    if len(eq) < 2:
        return 0
    underwater = eq < eq.cummax()
    if not underwater.any():
        return 0
    # Identify contiguous underwater episodes
    groups = (underwater != underwater.shift()).cumsum()
    durations = underwater.groupby(groups).sum()
    return int(durations.max())


def underwater_curve(equity: pd.Series) -> pd.Series:
    """Time series of drawdown depth (POSITIVE values, fraction of HWM lost).

    DD_t = (CumMax_t - E_t) / CumMax_t

    Returns a Series indexed the same as `equity`. 0 means at HWM.
    """
    eq = equity.dropna()
    cummax = eq.cummax()
    return (cummax - eq) / cummax


def time_in_drawdown_pct(equity: pd.Series) -> float:
    """Fraction of trading days the strategy spent underwater (E < cummax)."""
    eq = equity.dropna()
    if len(eq) < 2:
        return 0.0
    return float((eq < eq.cummax()).mean())


# --- Beta / alpha (CAPM) -------------------------------------------------

def beta_to_market(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_daily: Optional[pd.Series] = None,   # noqa: ARG001 — kept for API compat
) -> float:
    """OLS-regression beta per HW3 convention.

    Equation (HW3 polyfit-based):
        coeffs = polyfit(market_daily_returns, portfolio_daily_returns, 1)
        beta   = coeffs[0]   (slope of OLS regression line)

    Equivalently: beta = cov(R_p, R_m) / var(R_m) with raw (non-excess)
    returns. HW3 does NOT subtract risk-free before regression.
    """
    p, m = _aligned(portfolio_returns, market_returns)
    if len(p) < 3:
        return float("nan")
    var_m = float(m.var(ddof=1))
    if var_m == 0:
        return float("nan")
    return float(p.cov(m) / var_m)


def alpha_annualized(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """OLS-regression alpha per HW3 convention, annualized.

    Equation (HW3 polyfit-based):
        coeffs = polyfit(R_m_daily, R_p_daily, 1)   # [slope, intercept]
        beta   = coeffs[0]
        alpha_daily = coeffs[1]
        alpha_annual = alpha_daily * periods_per_year

    Equivalently:
        alpha_daily = mean(R_p) - beta * mean(R_m)
        alpha_annual = alpha_daily * 252

    HW3 does NOT use Jensen's alpha (no rf adjustment). This is the simpler
    OLS-regression alpha used in the course.

    Returns the ANNUALIZED alpha as a decimal (0.209 = 20.9%).
    """
    p, m = _aligned(portfolio_returns, market_returns)
    if len(p) < 3:
        return float("nan")
    beta = beta_to_market(portfolio_returns, market_returns)
    if np.isnan(beta):
        return float("nan")
    alpha_daily = float(p.mean() - beta * m.mean())
    return alpha_daily * periods_per_year


# --- Trade-level ----------------------------------------------------------

def average_return_per_trade(
    trade_pnls: list[float], starting_equity: float,
) -> float:
    """Mean trade return as fraction of starting equity."""
    if not trade_pnls or starting_equity <= 0:
        return 0.0
    rets = np.array(trade_pnls) / starting_equity
    return float(rets.mean())


def trades_per_year(n_trades: int, equity_index: pd.DatetimeIndex) -> float:
    """n_trades / years_in_window where years = (last - first) / 365.25."""
    if n_trades == 0 or len(equity_index) < 2:
        return 0.0
    span_days = (equity_index[-1] - equity_index[0]).days
    if span_days <= 0:
        return 0.0
    return n_trades / (span_days / 365.25)


# --- Per-year breakdown --------------------------------------------------

def per_year_breakdown(
    equity: pd.Series, ann_rf_per_year: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Year-by-year return, vol, Sharpe, max DD.

    Returns DataFrame indexed by year.
    """
    eq = equity.dropna()
    if len(eq) < 2:
        return pd.DataFrame()
    daily_ret = eq.pct_change().dropna()
    rows = []
    for year, group in daily_ret.groupby(daily_ret.index.year):
        if len(group) < 5:
            continue
        eq_year = eq.loc[group.index]
        gmr = geometric_mean_return(group, 252)
        vol = annualized_volatility(group, 252)
        if ann_rf_per_year is not None and year in ann_rf_per_year.index:
            rf_y = float(ann_rf_per_year.loc[year])
        else:
            rf_y = 0.04
        sharpe_y = (gmr - rf_y) / vol if vol > 0 else float("nan")
        rows.append({
            "year": year,
            "n_days": len(group),
            "gmr_pct": gmr * 100,
            "vol_pct": vol * 100,
            "ann_rf_pct": rf_y * 100,
            "sharpe": sharpe_y,
            "max_dd_pct": max_drawdown_signed(eq_year) * 100,
        })
    return pd.DataFrame(rows).set_index("year")


# --- After-tax Sharpe ----------------------------------------------------

def section_1256_after_tax_pnl(
    annual_pnl: float, ltcg_rate: float = 0.20, stcg_rate: float = 0.37,
    ltcg_frac: float = 0.60, stcg_frac: float = 0.40,
) -> float:
    """Section 1256 60/40 split. Default rates: 20% LTCG, 37% top-bracket STCG."""
    return annual_pnl * (1 - (ltcg_frac * ltcg_rate + stcg_frac * stcg_rate))


def short_term_after_tax_pnl(
    annual_pnl: float, stcg_rate: float = 0.37,
) -> float:
    """Standard short-term capital gains. ETF/equity options if held < 1 year."""
    return annual_pnl * (1 - stcg_rate)


# --- Combined report ------------------------------------------------------

@dataclass
class PortfolioReport:
    """Bundle of HW3/HW4-style portfolio metrics + extensions."""
    # Headline
    sharpe_annualized: float             # (GMR - rf) / vol — RF-adjusted (canonical)
    sharpe_raw: float                    # GMR / vol — no RF subtraction (no-RF view)
    sortino_annualized: float
    calmar: float

    # Returns
    geometric_mean_return: float          # annualized GMR (the actual realized return)
    annualized_risk_free: float           # avg annualized rf over the window (IRX-derived)
    annualized_excess_return: float       # GMR - ann_rf (what you earned ABOVE cash)
    beats_rf: bool                        # True if GMR > ann_rf

    # Risk
    annualized_volatility: float
    downside_deviation_annualized: float

    # Drawdown
    max_drawdown: float                    # signed (negative)
    max_drawdown_duration_days: int
    time_in_drawdown_pct: float

    # OLS-regression CAPM (HW3 convention)
    beta: float
    alpha: float                            # annualized OLS intercept × 252

    # Trade-level
    avg_return_per_trade: float
    total_trades: int
    trades_per_year: float
    win_rate: float

    # Deflated Sharpe Ratio (Bailey & López de Prado 2014). NaN if not computed
    # (when n_trials / sharpe_var_across_trials weren't supplied to build_portfolio_report).
    deflated_sharpe_psr: float = float("nan")
    deflated_sharpe_n_trials: int = 0
    deflated_sharpe_benchmark_annualized: float = float("nan")

    # Per-year (DataFrame) and underwater curve
    per_year: pd.DataFrame = field(default_factory=pd.DataFrame)
    underwater: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def to_dict(self) -> dict:
        return {
            "sharpe_annualized": self.sharpe_annualized,
            "sharpe_raw": self.sharpe_raw,
            "sortino_annualized": self.sortino_annualized,
            "calmar": self.calmar,
            "geometric_mean_return": self.geometric_mean_return,
            "annualized_risk_free": self.annualized_risk_free,
            "annualized_excess_return": self.annualized_excess_return,
            "beats_rf": self.beats_rf,
            "annualized_volatility": self.annualized_volatility,
            "downside_deviation_annualized": self.downside_deviation_annualized,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "time_in_drawdown_pct": self.time_in_drawdown_pct,
            "beta": self.beta,
            "alpha": self.alpha,
            "avg_return_per_trade": self.avg_return_per_trade,
            "total_trades": self.total_trades,
            "trades_per_year": self.trades_per_year,
            "win_rate": self.win_rate,
        }

    def headline(self) -> str:
        """Single multi-line string suitable for printing or dropping into the
        writeup. Makes the rf comparison front-and-center."""
        beats = "YES" if self.beats_rf else "NO"
        if not np.isnan(self.deflated_sharpe_psr):
            dsr_line = (
                f"Deflated Sharpe (PSR, N={self.deflated_sharpe_n_trials} trials): "
                f"{self.deflated_sharpe_psr:>8.4f}\n"
                f"DSR benchmark (Sharpe under H0):     "
                f"{self.deflated_sharpe_benchmark_annualized:>+8.4f} (ann)\n"
            )
        else:
            dsr_line = "Deflated Sharpe (PSR):               (not computed)\n"
        return (
            f"Sharpe (annualized, rf-adjusted):    {self.sharpe_annualized:>8.4f}\n"
            f"Sharpe (raw, no rf subtraction):     {self.sharpe_raw:>8.4f}\n"
            f"{dsr_line}"
            f"Sortino (annualized):                {self.sortino_annualized:>8.4f}\n"
            f"Calmar (return / |max DD|):          {self.calmar:>8.4f}\n"
            f"---\n"
            f"Geometric mean return (ann):         {self.geometric_mean_return*100:>8.2f}%\n"
            f"Annualized risk-free (IRX):          {self.annualized_risk_free*100:>8.2f}%\n"
            f"Excess return over cash:             {self.annualized_excess_return*100:>+8.2f}%   (beats RF: {beats})\n"
            f"---\n"
            f"Annualized volatility:               {self.annualized_volatility*100:>8.2f}%\n"
            f"Downside deviation (ann):            {self.downside_deviation_annualized*100:>8.2f}%\n"
            f"Max drawdown:                        {self.max_drawdown*100:>+8.2f}%\n"
            f"Max DD duration (days):              {self.max_drawdown_duration_days:>8d}\n"
            f"Time underwater:                     {self.time_in_drawdown_pct*100:>8.2f}%\n"
            f"---\n"
            f"Beta (vs SPY, OLS):                  {self.beta:>8.4f}\n"
            f"Alpha (annualized OLS intercept):    {self.alpha*100:>+8.2f}%\n"
            f"---\n"
            f"Total trades:                        {self.total_trades:>8d}\n"
            f"Trades per year:                     {self.trades_per_year:>8.2f}\n"
            f"Win rate:                            {self.win_rate*100:>8.2f}%\n"
            f"Avg return per trade:                {self.avg_return_per_trade*100:>+8.4f}%\n"
        )


def build_portfolio_report(
    equity_curve: pd.Series,
    trade_pnls: list[float],
    starting_equity: float = 100_000.0,
    benchmark_close: Optional[pd.Series] = None,
    irx_curve: Optional[pd.Series] = None,
    periods_per_year: int = 252,
    dsr_n_trials: Optional[int] = None,
    dsr_sharpe_var_per_period: Optional[float] = None,
) -> PortfolioReport:
    """Compute the full HW3/HW4-style portfolio report from a backtest run.

    equity_curve: pd.Series of equity values indexed by trading date.
    trade_pnls:   list of total realized P&L per trade (dollars).
    benchmark_close: pd.Series of benchmark daily closes (default loads SPY
        from data/raw/SPY.parquet).
    irx_curve:    pd.Series of CBOE IRX (10x convention) indexed by date
        (default loads from data/raw/IRX.parquet).
    """
    daily_ret = equity_curve.pct_change().dropna()

    # Risk-free: average annualized over the window
    if irx_curve is None:
        try:
            irx_curve = pd.read_parquet(DATA_RAW_DIR / "IRX.parquet")["close"]
        except FileNotFoundError:
            irx_curve = pd.Series(0.04 * 1000.0, index=daily_ret.index, name="IRX")
    annual_rf_series = _irx_annual_rate(irx_curve).reindex(daily_ret.index, method="ffill")
    ann_rf = float(annual_rf_series.mean()) if len(annual_rf_series.dropna()) > 0 else 0.04
    rf_daily_for_sharpe = annual_rf_series / periods_per_year

    # Headline returns / risk
    gmr = geometric_mean_return(daily_ret, periods_per_year)
    vol = annualized_volatility(daily_ret, periods_per_year)
    sharpe = sharpe_annualized(daily_ret, ann_rf, periods_per_year)
    sortino = sortino_ratio(daily_ret, ann_rf, periods_per_year)
    calmar = calmar_ratio(daily_ret, equity_curve, periods_per_year)
    dd_daily = downside_deviation(daily_ret, target=0.0,
                                    periods_per_year=periods_per_year)

    # OLS regression alpha/beta vs benchmark (SPY by default)
    # Per HW3 convention: simple polyfit on raw (non-excess) returns.
    if benchmark_close is None:
        try:
            benchmark_close = pd.read_parquet(DATA_RAW_DIR / "SPY.parquet")["close"]
        except FileNotFoundError:
            benchmark_close = None
    if benchmark_close is not None:
        bench_ret = benchmark_close.pct_change().dropna()
        beta = beta_to_market(daily_ret, bench_ret)
        alpha = alpha_annualized(daily_ret, bench_ret, periods_per_year)
    else:
        beta = float("nan")
        alpha = float("nan")

    # Trade-level
    n_trades = len(trade_pnls)
    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = wins / n_trades if n_trades > 0 else 0.0

    # Per-year breakdown
    ann_rf_by_year = annual_rf_series.groupby(annual_rf_series.index.year).mean()
    per_year = per_year_breakdown(equity_curve, ann_rf_by_year)

    sharpe_raw = (gmr / vol) if (vol > 0 and not np.isnan(gmr)) else float("nan")

    # DSR (Bailey & López de Prado 2014) — optional
    dsr_psr = float("nan")
    dsr_bench_ann = float("nan")
    dsr_n = 0
    if dsr_n_trials is not None and dsr_sharpe_var_per_period is not None:
        from src.metrics.deflated_sharpe import deflated_sharpe_ratio
        dsr_rep = deflated_sharpe_ratio(
            returns=daily_ret,
            n_trials=dsr_n_trials,
            sharpe_var_across_trials_per_period=dsr_sharpe_var_per_period,
            periods_per_year=periods_per_year,
        )
        dsr_psr = dsr_rep.psr
        dsr_bench_ann = dsr_rep.sharpe_benchmark_annualized
        dsr_n = dsr_rep.n_trials

    return PortfolioReport(
        sharpe_annualized=sharpe,
        sharpe_raw=sharpe_raw,
        sortino_annualized=sortino,
        calmar=calmar,
        geometric_mean_return=gmr,
        annualized_risk_free=ann_rf,
        annualized_excess_return=gmr - ann_rf,
        beats_rf=bool(gmr > ann_rf) if not np.isnan(gmr) else False,
        annualized_volatility=vol,
        downside_deviation_annualized=dd_daily,
        max_drawdown=max_drawdown_signed(equity_curve),
        max_drawdown_duration_days=max_drawdown_duration_days(equity_curve),
        time_in_drawdown_pct=time_in_drawdown_pct(equity_curve),
        beta=beta,
        alpha=alpha,
        avg_return_per_trade=average_return_per_trade(trade_pnls, starting_equity),
        total_trades=n_trades,
        trades_per_year=trades_per_year(n_trades, daily_ret.index),
        win_rate=win_rate,
        deflated_sharpe_psr=dsr_psr,
        deflated_sharpe_n_trials=dsr_n,
        deflated_sharpe_benchmark_annualized=dsr_bench_ann,
        per_year=per_year,
        underwater=underwater_curve(equity_curve),
    )
