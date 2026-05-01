/**
 * SPX VRP Strategy — Plotly chart initializers
 *
 * All charts use a shared dark theme matching theme.css.
 * Data is embedded in the HTML as JSON by the Flask template
 * and passed to these functions at page load.
 */

// ---------------------------------------------------------------------------
// Shared Plotly layout defaults (dark finance theme)
// ---------------------------------------------------------------------------
const DARK_LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor:  "#0d1117",
  font: { family: "'Inter', system-ui, sans-serif", color: "#94a3b8", size: 11 },
  margin: { l: 52, r: 20, t: 20, b: 44 },
  xaxis: {
    gridcolor:    "#1e2d40",
    linecolor:    "#1e2d40",
    tickcolor:    "#1e2d40",
    tickfont:     { color: "#64748b", size: 10 },
    zeroline:     false,
    showspikes:   true,
    spikecolor:   "#3b82f6",
    spikethickness: 1,
    spikedash:    "dot",
  },
  yaxis: {
    gridcolor:    "#1e2d40",
    linecolor:    "#1e2d40",
    tickcolor:    "#1e2d40",
    tickfont:     { color: "#64748b", size: 10 },
    zeroline:     true,
    zerolinecolor:"#1e2d40",
    showspikes:   true,
    spikecolor:   "#3b82f6",
    spikethickness: 1,
    spikedash:    "dot",
  },
  hoverlabel: {
    bgcolor:      "#0d1117",
    bordercolor:  "#1e2d40",
    font:         { color: "#e2e8f0", size: 11 },
  },
  legend: {
    bgcolor:      "transparent",
    bordercolor:  "#1e2d40",
    font:         { color: "#94a3b8", size: 11 },
  },
};

const PLOTLY_CONFIG = {
  displayModeBar:  true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "toImage", "autoScale2d"],
  displaylogo:     false,
  responsive:      true,
};

// ---------------------------------------------------------------------------
// Equity curve
// ---------------------------------------------------------------------------
function renderEquityCurve(elementId, equityCurveData, stressEvents) {
  const el = document.getElementById(elementId);
  if (!el) return;

  let dates, navValues;
  const hasData = Array.isArray(equityCurveData) && equityCurveData.length > 0;

  if (hasData) {
    dates     = equityCurveData.map(d => d.date);
    navValues = equityCurveData.map(d => d.nav);
  } else {
    // Placeholder: synthetic equity curve resembling a vol-selling strategy
    const n = 252 * 4; // 4 years of daily points
    dates = [];
    navValues = [];
    let nav   = 100000;
    let t     = new Date("2020-01-02");
    for (let i = 0; i < n; i++) {
      nav *= 1 + (Math.random() * 0.003 - 0.0005); // small upward drift
      // COVID crash analog
      if (i >= 290 && i <= 320) nav *= 0.993;
      dates.push(t.toISOString().slice(0, 10));
      navValues.push(+nav.toFixed(2));
      t.setDate(t.getDate() + 1);
      // skip weekends
      while (t.getDay() === 0 || t.getDay() === 6) t.setDate(t.getDate() + 1);
    }
  }

  // Build stress-event vertical shapes and annotations
  const shapes     = [];
  const annotations = [];
  (stressEvents || []).forEach(ev => {
    if (dates.indexOf(ev.date) === -1 && !hasData) return;
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: ev.date, x1: ev.date, y0: 0, y1: 1,
      line: { color: "rgba(239,68,68,0.35)", width: 1, dash: "dash" },
    });
    annotations.push({
      xref: "x", yref: "paper",
      x: ev.date, y: 1.02,
      text: ev.label, showarrow: false,
      font: { color: "#ef4444", size: 9 },
      xanchor: "left", textangle: -45,
    });
  });

  const trace = {
    x: dates,
    y: navValues,
    type: "scatter",
    mode: "lines",
    name: "Portfolio NAV",
    line: { color: "#3b82f6", width: 2 },
    fill: "tozeroy",
    fillcolor: "rgba(59,130,246,0.06)",
    hovertemplate: "<b>%{x}</b><br>NAV: $%{y:,.0f}<extra></extra>",
  };

  const layout = {
    ...DARK_LAYOUT,
    shapes,
    annotations,
    yaxis: {
      ...DARK_LAYOUT.yaxis,
      tickprefix: "$",
      tickformat: ",.0f",
    },
    margin: { l: 70, r: 20, t: 28, b: 44 },
  };

  if (!hasData) {
    layout.annotations = (layout.annotations || []).concat([{
      xref: "paper", yref: "paper",
      x: 0.5, y: 0.5,
      text: "PLACEHOLDER — awaiting backtest data",
      showarrow: false,
      font: { color: "rgba(148,163,184,0.25)", size: 14 },
      xanchor: "center",
    }]);
  }

  Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Fate breakdown bar chart
// ---------------------------------------------------------------------------
function renderFateChart(elementId, fateCounts) {
  const el = document.getElementById(elementId);
  if (!el) return;

  const FATE_COLORS = {
    profit_target:  "#22c55e",
    stop_loss:      "#ef4444",
    time_exit:      "#06b6d4",
    emergency_exit: "#f59e0b",
  };

  const FATE_LABELS = {
    profit_target:  "Profit Target",
    stop_loss:      "Stop Loss",
    time_exit:      "Time Exit (21 DTE)",
    emergency_exit: "Emergency Exit (Δ > 0.50)",
  };

  const isEmpty = !fateCounts || Object.keys(fateCounts).length === 0;

  const fates  = isEmpty ? ["profit_target","stop_loss","time_exit","emergency_exit"] : Object.keys(fateCounts);
  const counts = isEmpty ? [0, 0, 0, 0] : fates.map(f => fateCounts[f] || 0);
  const colors = fates.map(f => FATE_COLORS[f] || "#3b82f6");
  const labels = fates.map(f => FATE_LABELS[f] || f);

  const trace = {
    x: labels,
    y: counts,
    type: "bar",
    marker: {
      color: colors,
      opacity: 0.85,
      line: { color: "rgba(255,255,255,0.05)", width: 1 },
    },
    text: counts.map(c => c > 0 ? c.toString() : ""),
    textposition: "outside",
    textfont: { color: "#94a3b8", size: 10 },
    hovertemplate: "<b>%{x}</b><br>Count: %{y}<extra></extra>",
  };

  const layout = {
    ...DARK_LAYOUT,
    yaxis: {
      ...DARK_LAYOUT.yaxis,
      title: { text: "Trade Count", font: { color: "#64748b", size: 10 } },
      tickformat: "d",
    },
    xaxis: {
      ...DARK_LAYOUT.xaxis,
      tickfont: { color: "#94a3b8", size: 10 },
    },
    margin: { l: 48, r: 16, t: 28, b: 60 },
  };

  if (isEmpty) {
    layout.annotations = [{
      xref: "paper", yref: "paper",
      x: 0.5, y: 0.5,
      text: "No trade data yet",
      showarrow: false,
      font: { color: "rgba(148,163,184,0.25)", size: 13 },
      xanchor: "center",
    }];
  }

  Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Monthly returns heatmap (calendar-style bar chart as fallback)
// ---------------------------------------------------------------------------
function renderMonthlyReturns(elementId, blotterData) {
  const el = document.getElementById(elementId);
  if (!el) return;

  // Group net_pnl by year-month from blotter
  const monthly = {};
  (blotterData || []).forEach(trade => {
    if (!trade.exit_date) return;
    const ym = trade.exit_date.slice(0, 7); // "YYYY-MM"
    monthly[ym] = (monthly[ym] || 0) + (trade.net_pnl || 0);
  });

  const hasData = Object.keys(monthly).length > 0;

  let labels, values, colors;
  if (hasData) {
    labels = Object.keys(monthly).sort();
    values = labels.map(k => +monthly[k].toFixed(0));
    colors = values.map(v => v >= 0 ? "rgba(34,197,94,0.75)" : "rgba(239,68,68,0.75)");
  } else {
    labels = []; values = []; colors = [];
  }

  const trace = {
    x: labels, y: values, type: "bar",
    marker: { color: colors, line: { color: "rgba(255,255,255,0.04)", width: 1 } },
    hovertemplate: "<b>%{x}</b><br>P&L: $%{y:,.0f}<extra></extra>",
  };

  const layout = {
    ...DARK_LAYOUT,
    yaxis: {
      ...DARK_LAYOUT.yaxis,
      tickprefix: "$",
      tickformat: ",.0f",
      title: { text: "Net P&L ($)", font: { color: "#64748b", size: 10 } },
    },
    margin: { l: 70, r: 16, t: 20, b: 52 },
  };

  if (!hasData) {
    layout.annotations = [{
      xref: "paper", yref: "paper",
      x: 0.5, y: 0.5,
      text: "No trade data yet",
      showarrow: false,
      font: { color: "rgba(148,163,184,0.25)", size: 13 },
      xanchor: "center",
    }];
  }

  Plotly.newPlot(el, hasData ? [trace] : [], layout, PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Test pass/fail doughnut (Tests page summary)
// ---------------------------------------------------------------------------
function renderTestDonut(elementId, passed, failed, skipped) {
  const el = document.getElementById(elementId);
  if (!el) return;

  const total = passed + failed + skipped;
  if (total === 0) return;

  const trace = {
    values: [passed, failed, skipped].filter(v => v > 0),
    labels: ["Passed", "Failed", "Skipped"].filter((_, i) => [passed, failed, skipped][i] > 0),
    type: "pie",
    hole: 0.68,
    marker: {
      colors: ["#22c55e", "#ef4444", "#64748b"],
      line: { color: "#0d1117", width: 2 },
    },
    textinfo: "none",
    hovertemplate: "<b>%{label}</b>: %{value} (%{percent})<extra></extra>",
  };

  const pct = total > 0 ? Math.round((passed / total) * 100) : 0;

  const layout = {
    ...DARK_LAYOUT,
    margin: { l: 10, r: 10, t: 10, b: 10 },
    annotations: [{
      text: `${pct}%`,
      x: 0.5, y: 0.55,
      font: { size: 22, color: pct === 100 ? "#22c55e" : pct >= 80 ? "#f59e0b" : "#ef4444", family: "'JetBrains Mono', monospace" },
      showarrow: false,
    }, {
      text: "passing",
      x: 0.5, y: 0.38,
      font: { size: 10, color: "#64748b" },
      showarrow: false,
    }],
    showlegend: true,
    legend: {
      orientation: "h", x: 0.5, xanchor: "center", y: -0.08,
      font: { color: "#94a3b8", size: 10 },
    },
  };

  Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}
