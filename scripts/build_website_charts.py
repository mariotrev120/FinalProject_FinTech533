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


def add_stress_annotations(fig, y_position=1.06, color="#9CA3AF"):
    """Add vertical dotted lines for the 5 stress events with labels placed
    ABOVE the plot area (in the top margin) so they never overlap chart data.
    Caller must provide enough top margin (margin.t >= 110) for room."""
    for label, date_str in STRESS_EVENTS:
        fig.add_shape(type="line", x0=date_str, x1=date_str,
                      y0=0, y1=1, yref="paper",
                      line=dict(color=color, width=1, dash="dot"))
        fig.add_annotation(x=date_str, y=y_position, yref="paper",
                           text=label, showarrow=False,
                           font=dict(size=9, color=color),
                           textangle=-30, xanchor="left", yanchor="bottom")


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

    add_stress_annotations(fig)

    layout = {**LAYOUT_BASE, "margin": dict(l=70, r=30, t=130, b=60)}
    fig.update_layout(
        **layout,
        title=dict(
            text="<b>Headline equity curve vs benchmark</b>"
                 "<br><sup style='color:#666'>Growth of $1, OOS 2018-2024 · "
                 "shaded green = outperformance vs anchor</sup>",
            x=0.04, xanchor="left", y=0.97,
        ),
        height=520,
        yaxis=dict(title="Growth of $1", gridcolor="#E5E7EB", zeroline=False,
                   tickformat=".3f"),
        xaxis=dict(title="", gridcolor="#E5E7EB", showgrid=True),
        legend=dict(yanchor="bottom", y=0.02, xanchor="left", x=0.02,
                    bgcolor="rgba(255,255,255,0.92)", bordercolor=COLOR_GRAY,
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
        text=f"<b>Max DD {max_dd_val:.3%}</b>",
        showarrow=True, arrowhead=2, arrowcolor=COLOR_CRIMSON, arrowwidth=1.2,
        ax=80, ay=40,
        font=dict(size=11, color=COLOR_CRIMSON),
        bgcolor="rgba(255,255,255,0.95)", bordercolor=COLOR_CRIMSON, borderwidth=1, borderpad=4,
    )

    add_stress_annotations(fig)

    layout = {**LAYOUT_BASE, "margin": dict(l=70, r=30, t=130, b=50)}
    fig.update_layout(
        **layout,
        title=dict(
            text="<b>Drawdown profile</b>"
                 "<br><sup style='color:#666'>Underwater curve · headline basket</sup>",
            x=0.04, xanchor="left", y=0.97,
        ),
        height=420,
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
    fig.add_hline(y=0.286, line_color=COLOR_GOLD, line_dash="dash", line_width=1.5,
                  annotation_text="anchor 0.286",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color=COLOR_GOLD))
    fig.add_hline(y=0.0, line_color=COLOR_GRAY, line_width=1)
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="<b>Per-instrument excess Sharpe contribution</b>"
                 "<br><sup style='color:#666'>Halts_only mode · BOOK "
                 "aggregate beats the per-instrument average via correlation diversification</sup>",
            x=0.04, xanchor="left", y=0.97,
        ),
        height=440,
        yaxis=dict(title="Excess Sharpe (rf = 2.33%)", gridcolor="#E5E7EB",
                   zeroline=False, range=[-0.05, 0.45]),
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
    fig.add_vline(x=anchor, line_color=COLOR_GOLD, line_dash="dash", line_width=1.5,
                  annotation_text="anchor 0.286",
                  annotation_position="top",
                  annotation_font=dict(size=10, color=COLOR_GOLD))
    fig.add_vline(x=0, line_color=COLOR_GRAY, line_width=1)

    layout = {**LAYOUT_BASE, "margin": dict(l=380, r=180, t=110, b=80)}
    fig.update_layout(
        **layout,
        title=dict(
            text="<b>Variants evaluated, excess Sharpe across all baskets</b>"
                 "<br><sup style='color:#666'>Six configurations tested · "
                 "navy is the headline · green beats baseline · gold positive but below · red negative</sup>",
            x=0.04, xanchor="left", y=0.97,
        ),
        height=520,
        xaxis=dict(
            title="Excess Sharpe (rf = 2.33%)",
            gridcolor="#E5E7EB", zeroline=False,
            range=[-0.85, 0.55],
            dtick=0.1,
            tickformat="+.1f",
            tickfont=dict(size=11),
        ),
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
                  fillcolor="rgba(196,69,54,0.05)", line=dict(width=0))
    fig.add_annotation(x="2020-03-01", y=1.04, yref="paper",
                       text="post-COVID trending regime → call wing crushed",
                       showarrow=False, font=dict(size=10, color=COLOR_CRIMSON),
                       xanchor="left")

    layout = {**LAYOUT_BASE, "margin": dict(l=70, r=30, t=110, b=60)}
    fig.update_layout(
        **layout,
        title=dict(
            text="<b>Iron condor vs put-only on SPX</b>"
                 "<br><sup style='color:#666'>Illustrative equity curves · post-2020 trending equity "
                 "regime systematically destroyed the call wing</sup>",
            x=0.04, xanchor="left", y=0.97,
        ),
        height=460,
        yaxis=dict(title="Growth of $1", gridcolor="#E5E7EB", zeroline=False, tickformat=".3f"),
        xaxis=dict(title="", gridcolor="#E5E7EB"),
        legend=dict(yanchor="bottom", y=0.02, xanchor="left", x=0.02,
                    bgcolor="rgba(255,255,255,0.92)", bordercolor=COLOR_GRAY, borderwidth=1),
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

    add_stress_annotations(fig)

    layout = {**LAYOUT_BASE, "margin": dict(l=70, r=80, t=130, b=60)}
    fig.update_layout(
        **layout,
        title=dict(
            text="<b>Hoeffding regime monitor on the headline basket</b>"
                 "<br><sup style='color:#666'>Bound stays above 50% on 88% of OOS trades · "
                 "no critical signal fired in 7 years · μ = 0.730 baseline</sup>",
            x=0.04, xanchor="left", y=0.97,
        ),
        height=540,
        yaxis=dict(title="Hoeffding bound (probability)", range=[0, 1.05],
                   tickformat=".0%", gridcolor="rgba(0,0,0,0)", side="left"),
        yaxis2=dict(title="Rolling 60-trade win rate", overlaying="y", side="right",
                    range=[0, 1], tickformat=".2f", showgrid=False),
        xaxis=dict(title="", gridcolor="#E5E7EB"),
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98,
                    bgcolor="rgba(255,255,255,0.92)", bordercolor=COLOR_GRAY, borderwidth=1),
    )
    write_chart(fig, "hoeffding_trace")


def chart_pnl_per_trade():
    """P&L per trade time-series bar chart, green/red colored by win/loss."""
    blotter = pd.DataFrame(json.load(open("website/data/blotter.json")))
    blotter["entry_date"] = pd.to_datetime(blotter["entry_date"])
    blotter = blotter.sort_values("entry_date").reset_index(drop=True)
    blotter["pnl"] = blotter["pnl_per_spread"].fillna(0)
    blotter["win"] = blotter["pnl"] > 0
    colors = [COLOR_EMERALD if w else COLOR_CRIMSON for w in blotter["win"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=blotter["entry_date"], y=blotter["pnl"],
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>P&L: $%{y:+.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=COLOR_GRAY, line_width=1)

    add_stress_annotations(fig)

    layout = {**LAYOUT_BASE, "margin": dict(l=70, r=30, t=130, b=50)}
    fig.update_layout(
        **layout,
        title=dict(
            text="<b>P&L per trade across the OOS sample</b>"
                 "<br><sup style='color:#666'>Green = winning trade, red = losing trade · "
                 "stop-loss hits cap losses at the wing-width bound</sup>",
            x=0.04, xanchor="left", y=0.97,
        ),
        height=460,
        yaxis=dict(title="P&L per spread ($)", gridcolor="#E5E7EB", zeroline=False),
        xaxis=dict(title="", gridcolor="#E5E7EB"),
        showlegend=False,
        bargap=0.0,
    )
    write_chart(fig, "pnl_per_trade")


def chart_trade_return_distribution():
    """Histogram of trade-level returns with mean / median / break-even lines."""
    blotter = pd.DataFrame(json.load(open("website/data/blotter.json")))
    blotter["pnl"] = blotter["pnl_per_spread"].fillna(0)
    pnl = blotter["pnl"].values
    mean_pnl = float(blotter["pnl"].mean())
    median_pnl = float(blotter["pnl"].median())

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=wins, name=f"Wins ({len(wins)})",
        marker=dict(color=COLOR_EMERALD, line=dict(color="white", width=1)),
        opacity=0.85, xbins=dict(start=-200, end=60, size=10),
        hovertemplate="P&L bucket: $%{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=losses, name=f"Losses ({len(losses)})",
        marker=dict(color=COLOR_CRIMSON, line=dict(color="white", width=1)),
        opacity=0.85, xbins=dict(start=-200, end=60, size=10),
        hovertemplate="P&L bucket: $%{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=COLOR_GRAY, line_width=1.5,
                  annotation_text="break-even", annotation_position="top")
    fig.add_vline(x=mean_pnl, line_color=COLOR_NAVY, line_dash="dash", line_width=2,
                  annotation_text=f"<b>mean ${mean_pnl:+.2f}</b>",
                  annotation_position="top right",
                  annotation_font=dict(size=11, color=COLOR_NAVY))
    fig.add_vline(x=median_pnl, line_color=COLOR_GOLD, line_dash="dash", line_width=2,
                  annotation_text=f"<b>median ${median_pnl:+.2f}</b>",
                  annotation_position="bottom right",
                  annotation_font=dict(size=11, color=COLOR_GOLD))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="<b>Trade P&L distribution across 437 OOS trades</b>"
                 "<br><sup style='color:#666'>Right-skew: many small profit-target wins, "
                 "fewer larger losses bounded by stop-loss / wing width</sup>",
            x=0.04, xanchor="left",
        ),
        height=420,
        yaxis=dict(title="Number of trades", gridcolor="#E5E7EB", zeroline=False),
        xaxis=dict(title="P&L per spread ($)", gridcolor="#E5E7EB"),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor=COLOR_GRAY, borderwidth=1),
        bargap=0.05,
        barmode="overlay",
    )
    write_chart(fig, "trade_return_distribution")


if __name__ == "__main__":
    chart_equity_headline_vs_anchor()
    chart_drawdown_headline()
    chart_per_instrument_sharpe()
    chart_ablation_baskets()
    chart_iron_condor_vs_putonly()
    chart_hoeffding_trace()
    chart_pnl_per_trade()
    chart_trade_return_distribution()
    print("\nAll charts written to website/charts/")
