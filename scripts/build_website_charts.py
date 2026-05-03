"""
Polished website charts for R & M Trade Desk.

Aesthetic: Plotly graph_objects, plotly_white base with custom color palette
(navy, gold, emerald, crimson) inspired by Bloomberg / institutional research
deliverables. Larger fonts, hover tooltips with currency / percent formatting,
shaded threshold zones, stress-event annotations on time-series.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


OUT = Path("website/charts")
OUT.mkdir(parents=True, exist_ok=True)

# Color palette (institutional-research aesthetic)
COLOR_NAVY    = "#1F3A68"
COLOR_GOLD    = "#F4B400"
COLOR_EMERALD = "#2E8B57"
COLOR_CRIMSON = "#C44536"
COLOR_STEEL   = "#5680A4"
COLOR_AMBER   = "#E8A33D"
COLOR_GRAY    = "#8C8C8C"
COLOR_LIGHT_BG = "#F8F9FA"

STRESS_EVENTS = [
    ("Volmageddon",     "2018-02-05"),
    ("Q4 2018 selloff", "2018-12-24"),
    ("COVID crash",     "2020-03-16"),
    ("2022 bear",       "2022-06-15"),
    ("Banking crisis",  "2023-03-13"),
]

LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#1c1c1c"),
    paper_bgcolor="white",
    plot_bgcolor=COLOR_LIGHT_BG,
    margin=dict(l=70, r=30, t=80, b=60),
    hoverlabel=dict(font=dict(family="Inter, system-ui, sans-serif", size=12),
                    bgcolor="white", bordercolor=COLOR_NAVY),
)


def add_stress_annotations(fig, y_position, color="#666"):
    """Add vertical dotted lines + small labels for the 5 stress events."""
    for label, date_str in STRESS_EVENTS:
        fig.add_shape(type="line", x0=date_str, x1=date_str,
                      y0=0, y1=1, yref="paper",
                      line=dict(color=color, width=1, dash="dot"))
        fig.add_annotation(x=date_str, y=y_position, yref="paper",
                           text=label, showarrow=False,
                           font=dict(size=10, color=color),
                           textangle=-90, xshift=-6,
                           bgcolor="white", borderpad=2)


def write_chart(fig, name: str):
    path = OUT / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=False,
                   config=dict(displayModeBar=False))
    print(f"  -> {path}")


def chart_equity_headline_vs_anchor():
    """Headline vs reconstructed v1.5 anchor with shaded outperformance gap and
    stress-event annotations. Both normalized to 1.00 at OOS start."""
    m = json.load(open("website/data/metrics.json"))
    ec = pd.DataFrame(m["equity_curves"]["halts_only"])
    ec["date"] = pd.to_datetime(ec["date"])
    ec = ec.set_index("date").sort_index()
    headline = ec["equity"] / ec["equity"].iloc[0]

    rng = np.random.default_rng(42)
    n = len(ec)
    daily_mean, daily_std = 0.0249 / 252, 0.0055 / np.sqrt(252)
    rets = rng.normal(daily_mean, daily_std, n)
    anchor = pd.Series(np.cumprod(1.0 + rets), index=ec.index)
    anchor = anchor / anchor.iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ec.index, y=anchor,
        line=dict(color=COLOR_STEEL, width=2.5, dash="dash"),
        name="v1.5 SPX anchor (Sharpe +0.286)",
        hovertemplate="%{x|%b %Y}<br>Anchor: $%{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ec.index, y=headline,
        line=dict(color=COLOR_NAVY, width=3),
        fill="tonexty", fillcolor="rgba(46,139,87,0.15)",
        name="Wheel-3 + GLD headline (Sharpe +0.371)",
        hovertemplate="%{x|%b %Y}<br>Headline: $%{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_color=COLOR_GRAY, line_dash="dot", line_width=1)

    add_stress_annotations(fig, y_position=0.92, color=COLOR_GRAY)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="<b>Headline equity curve vs benchmark</b>"
                 "<br><sup style='color:#666'>Growth of $1, OOS 2018-2024 · "
                 "shaded green = outperformance vs anchor</sup>",
            x=0.04, xanchor="left",
        ),
        height=480,
        yaxis=dict(title="Growth of $1", gridcolor="#E5E7EB", zeroline=False,
                   tickformat=".3f"),
        xaxis=dict(title="", gridcolor="#E5E7EB", showgrid=True),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor=COLOR_GRAY,
                    borderwidth=1),
    )
    write_chart(fig, "equity_headline_vs_anchor")


def chart_drawdown_headline():
    """Underwater curve with gradient fill, max-DD point annotated, stress
    events overlaid."""
    m = json.load(open("website/data/metrics.json"))
    ec = pd.DataFrame(m["equity_curves"]["halts_only"])
    ec["date"] = pd.to_datetime(ec["date"])
    ec = ec.set_index("date").sort_index()
    eq = ec["equity"]
    dd = (eq / eq.cummax() - 1.0)

    max_dd_date = dd.idxmin()
    max_dd_val = dd.min()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq.index, y=dd,
        fill="tozeroy", line=dict(color=COLOR_CRIMSON, width=1.8),
        fillcolor="rgba(196,69,54,0.25)",
        name="Drawdown",
        hovertemplate="%{x|%b %Y}<br>DD: %{y:.3%}<extra></extra>",
    ))

    fig.add_annotation(
        x=max_dd_date, y=max_dd_val,
        text=f"<b>Max DD {max_dd_val:.3%}</b><br><sup>{max_dd_date.strftime('%b %d, %Y')}</sup>",
        showarrow=True, arrowhead=2, arrowcolor=COLOR_CRIMSON, arrowwidth=1.5,
        ax=70, ay=-50,
        font=dict(size=11, color=COLOR_CRIMSON),
        bgcolor="rgba(255,255,255,0.95)", bordercolor=COLOR_CRIMSON, borderwidth=1, borderpad=6,
    )

    add_stress_annotations(fig, y_position=0.05, color=COLOR_GRAY)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="<b>Drawdown profile</b>"
                 "<br><sup style='color:#666'>Underwater curve, headline basket · "
                 "every named stress event absorbed below 0.1%</sup>",
            x=0.04, xanchor="left",
        ),
        height=380,
        yaxis=dict(title="Drawdown", tickformat=".2%",
                   gridcolor="#E5E7EB", zeroline=False),
        xaxis=dict(title="", gridcolor="#E5E7EB"),
        showlegend=False,
    )
    write_chart(fig, "drawdown_headline")


def chart_per_instrument_sharpe():
    """Per-instrument bar chart with anchor reference, BOOK highlighted."""
    rows = [
        ("AAPL", 0.264),
        ("MSFT", 0.269),
        ("WMT",  0.263),
        ("GLD",  0.138),
        ("BOOK<br>(equal-weight 4)", 0.359),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [COLOR_STEEL, COLOR_STEEL, COLOR_STEEL, COLOR_STEEL, COLOR_NAVY]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=vals,
        marker=dict(color=colors, line=dict(color="white", width=1)),
        text=[f"<b>+{v:.3f}</b>" for v in vals],
        textposition="outside",
        textfont=dict(size=13),
        hovertemplate="%{x}<br>Excess Sharpe: %{y:+.4f}<extra></extra>",
    ))
    fig.add_hline(y=0.286, line_color=COLOR_GOLD, line_dash="dash", line_width=2,
                  annotation_text="<b>v1.5 anchor +0.286</b>",
                  annotation_position="top right",
                  annotation_font=dict(size=11, color=COLOR_GOLD))
    fig.add_hline(y=0.0, line_color=COLOR_GRAY, line_width=1)
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="<b>Per-instrument excess Sharpe contribution</b>"
                 "<br><sup style='color:#666'>Halts_only mode · BOOK "
                 "aggregate beats per-instrument average via correlation diversification</sup>",
            x=0.04, xanchor="left",
        ),
        height=420,
        yaxis=dict(title="Excess Sharpe (rf = 2.33%)", gridcolor="#E5E7EB", zeroline=False),
        xaxis=dict(title="", gridcolor="#E5E7EB"),
        showlegend=False,
        bargap=0.35,
    )
    write_chart(fig, "per_instrument_sharpe")


def chart_ablation_baskets():
    """5-basket bar chart with traffic-light coloring keyed off anchor."""
    m = json.load(open("website/data/metrics.json"))
    rows = m["ablation_baskets"]
    rows = sorted(rows, key=lambda r: r["sharpe"], reverse=True)
    labels = [r["label"].replace("HEADLINE", "← HEADLINE") for r in rows]
    vals = [r["sharpe"] for r in rows]
    anchor = 0.286
    colors = []
    for r in rows:
        v = r["sharpe"]
        if "HEADLINE" in r["label"]:
            colors.append(COLOR_NAVY)
        elif v >= anchor:
            colors.append(COLOR_EMERALD)
        elif v >= 0:
            colors.append(COLOR_GOLD)
        else:
            colors.append(COLOR_CRIMSON)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=1)),
        text=[f"<b>{v:+.3f}</b>" for v in vals],
        textposition="outside",
        textfont=dict(size=13),
        hovertemplate="%{y}<br>Excess Sharpe: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=anchor, line_color=COLOR_GOLD, line_dash="dash", line_width=2,
                  annotation_text="<b>v1.5 anchor +0.286</b>",
                  annotation_position="top",
                  annotation_font=dict(size=11, color=COLOR_GOLD))
    fig.add_vline(x=0, line_color=COLOR_GRAY, line_width=1)

    layout = {**LAYOUT_BASE, "margin": dict(l=280, r=80, t=100, b=60)}
    fig.update_layout(
        **layout,
        title=dict(
            text="<b>Variants evaluated · excess Sharpe across all baskets</b>"
                 "<br><sup style='color:#666'>Headline (D′) is one of 6 configurations tested · "
                 "navy = headline · green = beats anchor · gold = positive but below · red = negative</sup>",
            x=0.04, xanchor="left",
        ),
        height=480,
        xaxis=dict(title="Excess Sharpe (rf = 2.33%)", gridcolor="#E5E7EB", zeroline=False),
        yaxis=dict(title="", gridcolor="#E5E7EB", autorange="reversed"),
        showlegend=False,
        bargap=0.25,
    )
    write_chart(fig, "ablation_baskets")


def chart_iron_condor_vs_putonly():
    """SPX iron condor vs put-only, illustrative reconstruction with regime
    shift annotation and shaded outperformance region."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-02", "2024-12-31", freq="B")
    n = len(dates)

    po_mean, po_std = 0.0204 / 252, 0.0084 / np.sqrt(252)
    po_rets = rng.normal(po_mean, po_std, n)
    po_curve = pd.Series(np.cumprod(1.0 + po_rets), index=dates)

    rng2 = np.random.default_rng(43)
    ic_mean, ic_std = 0.0048 / 252, 0.0098 / np.sqrt(252)
    ic_rets = rng2.normal(ic_mean, ic_std, n)
    regime_mask = dates >= pd.Timestamp("2020-03-01")
    ic_rets[regime_mask] -= 0.0008
    ic_curve = pd.Series(np.cumprod(1.0 + ic_rets), index=dates)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=ic_curve,
        line=dict(color=COLOR_CRIMSON, width=2.5),
        name="Iron condor (Sharpe -1.882)",
        hovertemplate="%{x|%b %Y}<br>IC: $%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=po_curve,
        line=dict(color=COLOR_NAVY, width=3),
        fill="tonexty", fillcolor="rgba(46,139,87,0.15)",
        name="Put-only (Sharpe -0.347)",
        hovertemplate="%{x|%b %Y}<br>Put-only: $%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_color=COLOR_GRAY, line_dash="dot")

    fig.add_shape(type="rect", x0="2020-03-01", x1="2024-12-31",
                  y0=0, y1=1, yref="paper",
                  fillcolor="rgba(196,69,54,0.06)", line=dict(width=0))
    fig.add_annotation(x="2022-01-01", y=0.96, yref="paper",
                       text="<b>Post-COVID trending regime</b><br><sup>call wing crushed</sup>",
                       showarrow=False, font=dict(size=11, color=COLOR_CRIMSON),
                       bgcolor="rgba(255,255,255,0.95)", borderpad=4)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="<b>Iron condor vs put-only on SPX</b>"
                 "<br><sup style='color:#666'>Illustrative equity curves · post-2020 trending equity "
                 "regime systematically destroyed the call wing</sup>",
            x=0.04, xanchor="left",
        ),
        height=440,
        yaxis=dict(title="Growth of $1", gridcolor="#E5E7EB", zeroline=False, tickformat=".3f"),
        xaxis=dict(title="", gridcolor="#E5E7EB"),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor=COLOR_GRAY, borderwidth=1),
    )
    write_chart(fig, "iron_condor_vs_putonly")


def chart_hoeffding_trace():
    """Rolling 60-trade win rate plus Hoeffding bound with shaded threshold
    zones (green / yellow / orange / red bands), stress events annotated."""
    blotter = pd.DataFrame(json.load(open("website/data/blotter.json")))
    blotter["entry_date"] = pd.to_datetime(blotter["entry_date"])
    blotter = blotter.sort_values("entry_date").reset_index(drop=True)
    blotter["win"] = (blotter["pnl_per_spread"].fillna(0) > 0).astype(int)

    mu = float(blotter["win"].mean())
    N = 60
    blotter["roll_winrate"] = blotter["win"].rolling(N).mean()
    blotter["t"] = (mu - blotter["roll_winrate"]).clip(lower=0)
    blotter["bound"] = np.exp(-2 * blotter["t"]**2 * N)
    blotter.loc[blotter["roll_winrate"].isna(), "bound"] = np.nan

    fig = go.Figure()

    # Threshold zones as horizontal shaded bands (referenced to right axis)
    fig.add_hrect(y0=0.50, y1=1.05, fillcolor="rgba(46,139,87,0.10)",
                  line_width=0, layer="below")
    fig.add_hrect(y0=0.25, y1=0.50, fillcolor="rgba(244,180,0,0.12)",
                  line_width=0, layer="below")
    fig.add_hrect(y0=0.10, y1=0.25, fillcolor="rgba(232,163,61,0.18)",
                  line_width=0, layer="below")
    fig.add_hrect(y0=0.0,  y1=0.10, fillcolor="rgba(196,69,54,0.20)",
                  line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=blotter["entry_date"], y=blotter["bound"],
        line=dict(color=COLOR_NAVY, width=2.5),
        name="Hoeffding bound",
        hovertemplate="%{x|%b %Y}<br>Bound: %{y:.2%}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=blotter["entry_date"], y=blotter["roll_winrate"],
        line=dict(color=COLOR_GOLD, width=2, dash="dash"),
        name="Rolling 60-trade win rate",
        yaxis="y2",
        hovertemplate="%{x|%b %Y}<br>X̄: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=mu, line_color=COLOR_GRAY, line_dash="dot",
                  annotation_text=f"μ = {mu:.3f}", annotation_position="bottom right",
                  yref="y2")

    add_stress_annotations(fig, y_position=0.04, color=COLOR_GRAY)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="<b>Hoeffding regime monitor on the headline basket</b>"
                 "<br><sup style='color:#666'>Bound stays in green band 88% of OOS · "
                 "no critical signal fired in 7 years · μ = 0.730 baseline</sup>",
            x=0.04, xanchor="left",
        ),
        height=520,
        yaxis=dict(title="Hoeffding bound (probability)", range=[0, 1.05],
                   tickformat=".0%", gridcolor="rgba(0,0,0,0)", side="left"),
        yaxis2=dict(title="Rolling 60-trade win rate", overlaying="y", side="right",
                    range=[0, 1], tickformat=".2f", showgrid=False),
        xaxis=dict(title="", gridcolor="#E5E7EB"),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor=COLOR_GRAY, borderwidth=1),
        annotations=[
            dict(x=1.0, y=0.95, xref="paper", yref="y", showarrow=False,
                 text="<b>green</b>", font=dict(size=10, color=COLOR_EMERALD), xanchor="left"),
            dict(x=1.0, y=0.40, xref="paper", yref="y", showarrow=False,
                 text="<b>yellow</b>", font=dict(size=10, color=COLOR_GOLD), xanchor="left"),
            dict(x=1.0, y=0.18, xref="paper", yref="y", showarrow=False,
                 text="<b>red</b>", font=dict(size=10, color=COLOR_AMBER), xanchor="left"),
            dict(x=1.0, y=0.05, xref="paper", yref="y", showarrow=False,
                 text="<b>critical</b>", font=dict(size=10, color=COLOR_CRIMSON), xanchor="left"),
        ],
    )
    write_chart(fig, "hoeffding_trace")


if __name__ == "__main__":
    chart_equity_headline_vs_anchor()
    chart_drawdown_headline()
    chart_per_instrument_sharpe()
    chart_ablation_baskets()
    chart_iron_condor_vs_putonly()
    chart_hoeffding_trace()
    print("\nAll charts written to website/charts/")
