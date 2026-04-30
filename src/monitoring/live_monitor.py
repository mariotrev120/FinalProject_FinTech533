"""
Live performance monitoring framework.

STATUS: documentation-only placeholder. The assignment requires a *backtest*
and the writeup answers Vestal's two rubric questions ("how do you know
performance is in line" / "how do you know it stopped working") as a
forward-looking framework, not as code that actually executes against a
live trading account.

When the strategy is deployed for real, this module becomes the daily-update
loop:

  - Pull the most recent closed trade outcome
  - Update the rolling 60-trade win rate against the in-sample baseline
  - Compute the Hoeffding lower bound at alpha=0.10 (see src/metrics/hoeffding.py)
  - Recompute the rolling 90-day Sharpe with block-bootstrap CI
    (see src/metrics/bootstrap.py)
  - Compare predicted-prob bins to realized win rates (calibration drift)
  - Compare current XGBoost feature importance to original training importance

Output would be a daily HTML/JSON dashboard reporting:
  Status: green / yellow / red
  Latest Hoeffding-bound win rate vs baseline
  Latest 90d Sharpe with CI
  Top 3 features whose importance has drifted most
  Halt state today (active / soft / hard / drawdown)

The IS-baseline numbers needed by this loop are fully computed by the
existing backtest pipeline (see scripts/component_attribution.py and
src/metrics/hoeffding.py); this module would just thread them into a daily
job. Not implemented in this build — the writeup describes it as the
forward-looking framework the assignment asks for.
"""
from __future__ import annotations

# No code yet. See module docstring for the intended interface.
