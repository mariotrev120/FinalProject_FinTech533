"""
Shared pytest fixtures used across all test modules.

Fixtures are divided into three groups:
  1. Primitive data builders  — raw numpy / pandas building blocks
  2. Market data fixtures     — realistic multi-asset synthetic histories
  3. Strategy object fixtures — pre-built spread / trade / options-chain dicts
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. Primitive builders
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded RNG for all random data generation."""
    return np.random.default_rng(42)


@pytest.fixture
def biz_dates():
    """300 business days starting 2010-01-04."""
    return pd.bdate_range(start="2010-01-04", periods=300)


@pytest.fixture
def long_biz_dates():
    """500 business days — enough to clear the 200-day MA warmup."""
    return pd.bdate_range(start="2010-01-04", periods=500)


# ---------------------------------------------------------------------------
# 2. Market data fixtures
# ---------------------------------------------------------------------------

def _build_market_data(dates, rng):
    """Internal helper — builds a full synthetic market-data dict."""
    n = len(dates)

    spy = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.010, n)))
    tlt = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.005, n)))
    gld = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.008, n)))
    hyg = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.004, n)))
    lqd = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.003, n)))
    vix  = np.clip(rng.normal(18, 4, n), 9, 80)
    vix3m = np.clip(vix + rng.normal(2, 1, n), 9, 80)
    vvix = np.clip(rng.normal(85, 15, n), 50, 200)
    irx  = np.clip(rng.normal(0.50, 0.20, n), 0.01, 3.0)
    fvx  = np.clip(rng.normal(1.50, 0.30, n), 0.10, 5.0)
    tnx  = np.clip(rng.normal(2.50, 0.40, n), 0.10, 6.0)
    tyx  = np.clip(rng.normal(3.00, 0.40, n), 0.10, 6.0)

    def s(arr, name):
        return pd.Series(arr, index=dates, name=name)

    return {
        "SPY":  s(spy,  "SPY"),
        "TLT":  s(tlt,  "TLT"),
        "GLD":  s(gld,  "GLD"),
        "HYG":  s(hyg,  "HYG"),
        "LQD":  s(lqd,  "LQD"),
        "VIX":  s(vix,  "VIX"),
        "VIX3M": s(vix3m, "VIX3M"),
        "VVIX": s(vvix, "VVIX"),
        "IRX":  s(irx,  "IRX"),
        "FVX":  s(fvx,  "FVX"),
        "TNX":  s(tnx,  "TNX"),
        "TYX":  s(tyx,  "TYX"),
    }


@pytest.fixture
def market_data(biz_dates, rng):
    """300-day synthetic market data dict."""
    return _build_market_data(biz_dates, rng)


@pytest.fixture
def long_market_data(long_biz_dates, rng):
    """500-day synthetic market data dict (clears 200d MA warmup)."""
    return _build_market_data(long_biz_dates, rng)


@pytest.fixture
def stress_market_data(long_biz_dates, rng):
    """
    500-day market data with a stress episode injected at day 400.
    VIX spikes to 50, term structure inverts (VIX > VIX3M by 5pts),
    HYG-LQD spread blows out, SPY-TLT correlation turns positive.
    Used by halt tests.
    """
    data = _build_market_data(long_biz_dates, rng)

    # Inject stress at rows 400-420
    for key in ["VIX"]:
        arr = data[key].values.copy()
        arr[400:421] = 50.0
        data[key] = pd.Series(arr, index=long_biz_dates, name=key)

    arr_3m = data["VIX3M"].values.copy()
    arr_3m[400:421] = 45.0  # VIX3M < VIX → backwardation
    data["VIX3M"] = pd.Series(arr_3m, index=long_biz_dates, name="VIX3M")

    # SPY and TLT move together during stress
    arr_spy = data["SPY"].values.copy()
    arr_tlt = data["TLT"].values.copy()
    stress_move = np.linspace(0, -0.15, 21)
    arr_spy[400:421] *= (1 + stress_move)
    arr_tlt[400:421] *= (1 + stress_move)  # both fall together → positive correlation
    data["SPY"] = pd.Series(arr_spy, index=long_biz_dates, name="SPY")
    data["TLT"] = pd.Series(arr_tlt, index=long_biz_dates, name="TLT")

    return data


# ---------------------------------------------------------------------------
# 3. Strategy object fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_options_chain():
    """
    Minimal synthetic XSP options chain.
    Spot ≈ 4000. Puts at 5 different strikes spanning 5%–15% OTM.
    delta column uses negative convention (put deltas are negative).
    """
    spot = 4000.0
    rows = [
        {"strike": 3800, "right": "P", "delta": -0.07, "bid": 2.50, "ask": 3.00, "dte": 35, "expiry": "20100219"},
        {"strike": 3850, "right": "P", "delta": -0.10, "bid": 3.20, "ask": 3.80, "dte": 35, "expiry": "20100219"},
        {"strike": 3900, "right": "P", "delta": -0.14, "bid": 4.50, "ask": 5.20, "dte": 35, "expiry": "20100219"},
        {"strike": 3920, "right": "P", "delta": -0.16, "bid": 5.10, "ask": 5.90, "dte": 35, "expiry": "20100219"},
        {"strike": 3950, "right": "P", "delta": -0.22, "bid": 7.00, "ask": 8.00, "dte": 35, "expiry": "20100219"},
        {"strike": 3975, "right": "P", "delta": -0.25, "bid": 8.50, "ask": 9.50, "dte": 35, "expiry": "20100219"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sample_trade():
    """A representative open trade dict as produced by the backtest engine."""
    return {
        "entry_date": "2010-01-04",
        "entry_credit": 2.00,       # $200 per contract (2.00 pts × $100 multiplier)
        "short_strike": 3920,
        "long_strike": 3915,
        "spread_width": 5,
        "expiry": "20100219",
        "dte_at_entry": 35,
        "num_contracts": 1,
        "max_gain": 200.0,          # credit × 100
        "max_loss": 300.0,          # (width - credit) × 100
    }


@pytest.fixture
def winning_bar(sample_trade):
    """A price bar where the spread has decayed to 50% of original credit → profit target."""
    return {
        "date": "2010-01-25",
        "current_spread_value": 1.00,   # 50% of 2.00 entry credit
        "open_spread_value": 1.02,
        "short_delta": -0.08,           # moved more OTM; delta shrunk
        "dte_remaining": 25,
        "vix": 16.0,
    }


@pytest.fixture
def stop_bar(sample_trade):
    """A bar where the spread has reached 200% of credit → stop loss."""
    return {
        "date": "2010-01-15",
        "current_spread_value": 4.00,   # 200% of 2.00 entry credit
        "open_spread_value": 4.10,
        "short_delta": -0.38,
        "dte_remaining": 29,
        "vix": 30.0,
    }


@pytest.fixture
def time_exit_bar(sample_trade):
    """A bar at exactly 21 DTE with no other exit triggered."""
    return {
        "date": "2010-01-29",
        "current_spread_value": 1.60,   # between stop and target
        "open_spread_value": 1.62,
        "short_delta": -0.12,
        "dte_remaining": 21,
        "vix": 17.0,
    }


@pytest.fixture
def emergency_bar(sample_trade):
    """A bar where the short leg delta has blown through 0.50 → emergency exit."""
    return {
        "date": "2010-01-12",
        "current_spread_value": 3.50,
        "open_spread_value": 3.80,
        "short_delta": -0.55,           # past emergency threshold
        "dte_remaining": 28,
        "vix": 35.0,
    }


@pytest.fixture
def gap_bar(sample_trade):
    """
    A bar where the market gapped through the stop level at open.
    open_spread_value (4.80) is worse than the theoretical stop trigger (4.00).
    """
    return {
        "date": "2010-01-11",
        "current_spread_value": 4.80,
        "open_spread_value": 4.80,      # gapped past 4.00 stop
        "short_delta": -0.42,
        "dte_remaining": 30,
        "vix": 32.0,
    }


# ---------------------------------------------------------------------------
# 4. ML / model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_classification_data(rng):
    """
    Simple binary classification dataset for model smoke tests.
    16 features (matching strategy feature count), 300 samples, ~70% positive class.
    """
    n = 300
    X = pd.DataFrame(
        rng.normal(0, 1, size=(n, 16)),
        columns=[
            "vix", "vix3m", "vix_term_spread", "vvix",
            "vrp_30d", "vrp_60d",
            "yield_level", "yield_slope", "yield_curvature", "yield_inflection",
            "spy_tlt_corr_20d", "spy_gld_corr_20d", "hyg_lqd_spread",
            "spy_return_20d", "spy_dist_200ma", "ma_cross_state",
        ],
    )
    # Signal: high VIX spread + positive VRP → more likely to be profitable
    logit = 0.5 * X["vix_term_spread"] + 0.3 * X["vrp_30d"] - 0.2 * X["vix"]
    prob = 1 / (1 + np.exp(-logit))
    y = pd.Series((prob > 0.5).astype(int), name="outcome")
    return X, y


@pytest.fixture
def two_regime_series(rng):
    """
    500-day time series with two clear regimes for HMM tests.
    Regime 0 (Calm): low vol, mean-reverting.
    Regime 1 (Stressed): high vol, trending down.
    True state labels returned for validation.
    """
    n = 500
    true_states = np.zeros(n, dtype=int)
    true_states[200:300] = 1  # stress window

    vix = np.where(true_states == 0,
                   rng.normal(15, 2, n),
                   rng.normal(35, 5, n))
    term_spread = np.where(true_states == 0,
                           rng.normal(2, 0.5, n),
                           rng.normal(-2, 1, n))

    X = pd.DataFrame({"vix": vix, "vix_term_spread": term_spread})
    return X, true_states


# ---------------------------------------------------------------------------
# 5. Halt framework fixtures
# ---------------------------------------------------------------------------

def _base_market_state():
    """Normal (non-stress) market state."""
    return {
        "vix": 16.0,
        "vix_prev_close": 15.5,
        "vix3m": 18.0,
        "spx_intraday_pct": 0.003,      # +0.3%
        "hyg_lqd_spread_zscore": 0.5,   # within 2 SD
        "model_probability": 0.62,
        "consecutive_low_prob_days": 0,
        "consecutive_vix_inversion_days": 0,
        "realized_vol_5d": 0.10,
        "realized_vol_252d_pct80": 0.15,
        "hwm_drawdown_pct": -0.02,       # 2% drawdown — minor
        "underwater_days": 5,
    }


@pytest.fixture
def normal_market_state():
    return _base_market_state()


@pytest.fixture
def vix_spike_state():
    """VIX spikes 50% intraday from prior close — hard halt trigger."""
    s = _base_market_state()
    s["vix"] = 30.0
    s["vix_prev_close"] = 20.0  # +50% spike
    return s


@pytest.fixture
def spx_crash_state():
    """SPX drops 6% intraday — hard halt trigger."""
    s = _base_market_state()
    s["spx_intraday_pct"] = -0.06
    return s


@pytest.fixture
def term_structure_inversion_state():
    """VIX > VIX3M by 3 points — hard halt trigger."""
    s = _base_market_state()
    s["vix"] = 25.0
    s["vix3m"] = 22.0  # VIX - VIX3M = +3 > threshold of 2
    return s


@pytest.fixture
def soft_halt_credit_stress_state():
    """HYG-LQD spread > 2 SD — Layer 2 soft halt trigger."""
    s = _base_market_state()
    s["hyg_lqd_spread_zscore"] = 2.5
    return s


@pytest.fixture
def soft_halt_low_prob_state():
    """Model probability < 0.55 for 5 consecutive sessions."""
    s = _base_market_state()
    s["model_probability"] = 0.52
    s["consecutive_low_prob_days"] = 5
    return s


@pytest.fixture
def drawdown_halt_state():
    """Portfolio 16% underwater — Layer 4 drawdown depth trigger."""
    s = _base_market_state()
    s["hwm_drawdown_pct"] = -0.16
    s["underwater_days"] = 30
    return s


@pytest.fixture
def duration_halt_state():
    """Portfolio underwater for 92 trading days — Layer 4 duration trigger."""
    s = _base_market_state()
    s["hwm_drawdown_pct"] = -0.08
    s["underwater_days"] = 92
    return s


@pytest.fixture
def full_resume_state():
    """All four Layer 5 auto-resume conditions simultaneously satisfied."""
    return {
        "vix": 14.0,
        "vix_prev_close": 13.8,
        "vix3m": 17.0,                      # VIX3M > VIX by 3 pts
        "vix3m_above_vix_by2_consecutive": 5,  # 5 consecutive days
        "spx_intraday_pct": 0.002,
        "hyg_lqd_spread_zscore": 0.8,       # within 1 SD
        "model_probability": 0.63,
        "consecutive_low_prob_days": 0,
        "consecutive_vix_inversion_days": 0,
        "realized_vol_5d": 0.08,
        "realized_vol_252d_pct80": 0.14,    # 5d vol < 80th pct
        "hwm_drawdown_pct": -0.025,         # within 3% of HWM
        "underwater_days": 0,
    }


@pytest.fixture
def trade_history_with_losses():
    """60 trades where win rate is well below typical in-sample baseline."""
    rng = np.random.default_rng(7)
    outcomes = rng.choice([0, 1], size=60, p=[0.45, 0.55])  # 55% win rate
    return [
        {"entry_date": f"2019-{(i // 20) + 1:02d}-{(i % 20) + 1:02d}",
         "exit_date": f"2019-{(i // 20) + 1:02d}-{(i % 20) + 5:02d}",
         "outcome": int(outcomes[i]),
         "pnl": 150.0 if outcomes[i] else -300.0}
        for i in range(60)
    ]


@pytest.fixture
def strong_trade_history():
    """60 trades where win rate matches or exceeds in-sample baseline (~75%)."""
    rng = np.random.default_rng(99)
    outcomes = rng.choice([0, 1], size=60, p=[0.25, 0.75])
    return [
        {"entry_date": f"2019-{(i // 20) + 1:02d}-{(i % 20) + 1:02d}",
         "exit_date": f"2019-{(i // 20) + 1:02d}-{(i % 20) + 5:02d}",
         "outcome": int(outcomes[i]),
         "pnl": 150.0 if outcomes[i] else -300.0}
        for i in range(60)
    ]
