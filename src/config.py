"""
Global configuration for the SPX VRP project.

Single source of truth for the random seed, IBKR connection parameters, data
windows, decision threshold, and pre-committed parameters. Anything that needs
to be reproducible across runs reads from this module.

Frozen at PRE_COMMITMENT.md commit time. Do not edit after that timestamp.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

# --- Reproducibility -------------------------------------------------------
SEED: int = 42

# --- IBKR / TWS -----------------------------------------------------------
def discover_ibkr_host() -> str:
    """WSL gateway IP = Windows host where TWS runs. Discovered at runtime
    because WSL distro IPs rotate but the gateway stays consistent."""
    try:
        out = subprocess.check_output(
            "ip route show default | awk '{print $3}'",
            shell=True, text=True, timeout=5,
        ).strip()
        return out or "127.0.0.1"
    except Exception:
        return "127.0.0.1"


IBKR_HOST: str = discover_ibkr_host()
IBKR_PORT: int = 7497
IBKR_CLIENT_ID_BASE: int = 100

# --- Backtest windows -----------------------------------------------------
# Original README plan was 2010-2017 IS / 2018-2024 OOS. TWS paper account
# limits index history starts: VIX3M/yields begin 2011-05, VVIX begins
# 2012-03. The binding constraint is VVIX, so IS starts 2012-03-26.
# Net split: ~5.75 years IS / 7 years OOS. Legitimate windows for
# walk-forward methodology.
IS_START: str = "2012-03-26"
IS_END:   str = "2017-12-31"
OOS_START: str = "2018-01-01"
OOS_END:   str = "2024-12-31"

# --- Strategy parameters (PRE-COMMIT, frozen across folds) ---------------
ENTRY_DELTA_TARGET: float = 0.16     # 1-sigma OTM short strike
SPREAD_WIDTH_PTS: int = 5
DTE_MIN: int = 30
DTE_MAX: int = 45
PROFIT_TARGET_FRAC: float = 0.50      # close at 50% of max profit
STOP_LOSS_MULT: float = 2.00          # 200% of credit received
TIME_EXIT_DTE: int = 21
EMERGENCY_DELTA: float = 0.50

# --- ML gate ---------------------------------------------------------------
ML_DECISION_THRESHOLD: float = 0.55   # calibrated XGBoost p

# --- Sizing ---------------------------------------------------------------
KELLY_CAP: float = 0.25               # quarter Kelly
VOL_SCALE_PIVOT: float = 15.0         # vol_mult = VOL_SCALE_PIVOT / VIX_t
STRESS_CORR_THRESHOLD: float = 0.50   # SPY-TLT 20d corr trigger
STRESS_MULTIPLIER: float = 0.50       # halve size when triggered
TRADE_RISK_CAP_FRAC: float = 0.01     # 1% of equity per trade

# --- Halt thresholds (PRE-COMMIT, calibrated on IS only) -----------------
HALT_VIX_INTRADAY_PCT: float = 0.40       # +40% from prior close
HALT_SPX_INTRADAY_PCT: float = 0.05       # 5% in either direction
HALT_TERM_INVERSION_HARD: float = -2.0    # VIX > VIX3M by >2pts
HALT_TERM_INVERSION_SOFT_DAYS: int = 2    # VIX3M < VIX for 2 closes
HALT_HYG_LQD_SD: float = 2.0
HALT_HYG_LQD_LOOKBACK: int = 252
HALT_MODEL_PROB_DAYS: int = 5
HALT_WINRATE_WINDOW: int = 60             # rolling trades
HALT_WINRATE_SE_THRESHOLD: float = 2.0
HALT_SHARPE_WINDOW_DAYS: int = 90
HALT_SHARPE_BOOTSTRAP_P: float = 0.10
HALT_DD_DURATION_DAYS: int = 90
HALT_DD_DEPTH_FRAC: float = 0.15

# Auto-resume (ALL four required)
RESUME_TERM_BUFFER_PTS: float = 2.0
RESUME_TERM_DAYS: int = 5
RESUME_REALIZED_VOL_PCT: float = 0.80     # below 80th percentile of trailing 252d
RESUME_HYG_LQD_SD: float = 1.0
RESUME_DD_RECOVERY_FRAC: float = 0.03

# --- Friction ------------------------------------------------------------
COMMISSION_PER_LEG: float = 0.65
COMMISSION_MIN_PER_ORDER: float = 1.00
REG_FEES_PER_LEG: float = 0.05
SLIPPAGE_PCT_BY_VIX: dict[tuple[float, float], float] = {
    (0.0, 20.0):  0.30,
    (20.0, 30.0): 0.50,
    (30.0, 40.0): 0.75,
    (40.0, 999.0): 1.00,
}
GAP_SLIPPAGE_THRESHOLD_PCT: float = 0.015     # 50% extra slippage above this
GAP_SKIP_THRESHOLD_PCT: float = 0.030          # skip entry above this
TAX_LTCG_FRAC: float = 0.60                    # Section 1256 60/40
TAX_STCG_FRAC: float = 0.40

# --- Walk-forward refit cadence -----------------------------------------
REFIT_CADENCE: str = "annual"                  # December 31
PARAM_BLEND_ALPHA: float = 0.70                # exponential blending across refits

# --- Paths --------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
