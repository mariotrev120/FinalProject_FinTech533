"""
Flask web application for the SPX VRP Strategy dashboard.

Routes:
    /           → Home  (static strategy overview)
    /results    → Results (dynamic trading metrics and blotter)
    /tests      → Tests  (pytest status dashboard)

Data:
    Dynamic content is read from JSON files in website/data/ at request time.
    Missing files produce graceful placeholder UIs — no errors raised.
"""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, render_template, request

app = Flask(__name__)

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Constants shared across templates
# ---------------------------------------------------------------------------

STRESS_EVENTS = [
    {"date": "2015-08-24", "label": "China Devaluation"},
    {"date": "2018-02-05", "label": "Volmageddon"},
    {"date": "2018-12-24", "label": "Q4 2018 Selloff"},
    {"date": "2020-03-16", "label": "COVID Crash"},
    {"date": "2023-03-10", "label": "SVB / Regional Banking"},
]

MODES = ["naked", "ml_only", "halts_only", "full"]

MODE_LABELS = {
    "naked":      "Naked Baseline",
    "ml_only":    "ML Gate Only",
    "halts_only": "Halts Only",
    "full":       "Full Strategy",
}

MODE_DESCRIPTIONS = {
    "naked":      "Sells one spread every Monday with no ML filter and no halt rules. Pure unfiltered baseline.",
    "ml_only":    "XGBoost gate active (p ≥ 0.55 to enter). All halt rules disabled. Isolates ML contribution.",
    "halts_only": "No ML filter — every Monday is a candidate entry. All five halt layers active. Isolates halt framework contribution.",
    "full":       "Both ML gate and all five halt layers active. The strategy as designed.",
}

METRIC_DEFINITIONS = {
    "sharpe":             "Annualized Sharpe Ratio (RFR = 4%)",
    "win_rate":           "Percentage of trades closing at profit target",
    "total_return":       "Cumulative net return over the full period",
    "max_drawdown":       "Largest peak-to-trough decline",
    "avg_trade_duration": "Mean calendar days per trade",
    "total_trades":       "Total closed trades in the period",
    "sortino":            "Sortino Ratio (penalizes downside deviation only)",
    "profit_factor":      "Gross profit divided by gross loss",
    "avg_return_per_trade": "Mean net P&L per trade in dollars",
}

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_json(filename: str) -> dict | list | None:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_metrics() -> dict | None:
    return _load_json("metrics.json")  # type: ignore[return-value]


def load_blotter() -> list[dict]:
    data = _load_json("blotter.json")
    return data if isinstance(data, list) else []


def load_test_results() -> dict | None:
    return _load_json("test_results.json")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Test name formatting helpers
# ---------------------------------------------------------------------------

def _parse_nodeid(nodeid: str) -> dict[str, str]:
    """
    'tests/test_features.py::TestFeatureSchema::test_exactly_16_features'
    → {module: 'Features', cls: 'Feature Schema', name: 'Exactly 16 Features'}
    """
    parts = nodeid.replace("tests/", "").split("::")
    module_raw = parts[0].replace("test_", "").replace(".py", "")
    module = module_raw.replace("_", " ").title()

    if len(parts) >= 3:
        cls_raw = parts[1].replace("Test", "", 1).strip()
        cls = " ".join(
            w.capitalize() for w in cls_raw.split("_") if w
        )
        name_raw = parts[2]
    else:
        cls = ""
        name_raw = parts[-1]

    name = " ".join(
        w.capitalize() for w in name_raw.replace("test_", "", 1).split("_") if w
    )
    return {"module": module, "cls": cls, "name": name}


def _group_tests(tests: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for t in tests:
        info = _parse_nodeid(t.get("nodeid", "unknown"))
        mod = info["module"]
        grouped.setdefault(mod, []).append({
            **t,
            "display_name": info["name"],
            "cls_name":     info["cls"],
        })
    return grouped


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("home.html", active="home")


@app.route("/results")
def results():
    metrics   = load_metrics()
    blotter   = load_blotter()
    mode      = request.args.get("mode", "full")
    if mode not in MODES:
        mode = "full"

    mode_metrics = None
    equity_curve: list[dict] = []
    if metrics:
        mode_metrics = (metrics.get("modes") or {}).get(mode)
        equity_curve = (metrics.get("equity_curves") or {}).get(mode, [])

    blotter_display = sorted(
        blotter,
        key=lambda x: x.get("entry_date", ""),
        reverse=True,
    )[:200]

    fate_counts: dict[str, int] = {}
    for trade in blotter:
        fate = trade.get("fate", "unknown")
        fate_counts[fate] = fate_counts.get(fate, 0) + 1

    strategy_status = (metrics or {}).get("strategy_status", "unknown")
    halt_layer      = (metrics or {}).get("halt_layer")
    last_updated    = (metrics or {}).get("last_updated", "")

    return render_template(
        "results.html",
        active           = "results",
        has_data         = metrics is not None,
        metrics          = metrics,
        mode_metrics     = mode_metrics,
        blotter          = blotter_display,
        fate_counts      = json.dumps(fate_counts),
        equity_curve     = json.dumps(equity_curve),
        stress_events    = json.dumps(STRESS_EVENTS),
        current_mode     = mode,
        modes            = MODES,
        mode_labels      = MODE_LABELS,
        mode_descriptions= MODE_DESCRIPTIONS,
        metric_defs      = METRIC_DEFINITIONS,
        strategy_status  = strategy_status,
        halt_layer       = halt_layer,
        last_updated     = last_updated,
    )


@app.route("/tests")
def tests_view():
    raw          = load_test_results()
    test_summary = None
    grouped      = {}
    generated_at = ""
    duration     = 0.0

    if raw:
        test_summary = raw.get("summary", {})
        duration     = round(raw.get("duration", 0), 1)
        generated_at = raw.get("created", "")
        grouped      = _group_tests(raw.get("tests", []))

    return render_template(
        "tests.html",
        active       = "tests",
        has_data     = raw is not None,
        test_summary = test_summary,
        grouped      = grouped,
        generated_at = generated_at,
        duration     = duration,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
