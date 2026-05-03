# Algorithmic Trading Project Master Context

**Purpose:** This document serves as the architectural and technical baseline for algorithmic trading projects in this course. When beginning a new assignment, feed this document to your AI assistant or reference it to ensure environment consistency, correct data fetching protocols, and standardized metric evaluation.

---

## 1. Environment & Dependencies

**`requirements.txt`**

```text
pandas
numpy
scipy
statsmodels
matplotlib
plotly
seaborn
shinybroker
python-dotenv
```

---

## 2. Data Acquisition: The `shinybroker` Protocol

We exclusively use the `shinybroker` library to pull historical price data from the Interactive Brokers (IBKR) Trader Workstation (TWS). Follow these strict rules to avoid connection failures and corrupted data errors:

```python
import os
import shinybroker as sb
from dotenv import load_dotenv

load_dotenv()
host_ip = os.getenv("WINDOWS_HOST_IP", "127.0.0.1")
tws_port = int(os.getenv("TWS_PORT", 7497)) 
```

### Contract Formatting & 'SMART' Routing
IBKR relies on SmartRouting to aggregate historical price data. Explicitly targeting an exchange (like 'NASDAQ') will often result in a rejected API request due to missing unbundled data subscriptions.
* **Rule:** Always set the `exchange` parameter to `'SMART'`.

```python
# CORRECT IMPLEMENTATION
contract = sb.Contract({'symbol': 'TICKER', 'secType': 'STK', 'exchange': 'SMART', 'currency': 'USD'})

fetch_result = sb.fetch_historical_data(
    contract=contract, 
    barSizeSetting='1 day', 
    durationStr='2 Y', 
    host=host_ip, 
    port=tws_port
)
```

### Data Anomalies (The Pandas Array Error)
If `shinybroker` throws a `ValueError: All arrays must be of the same length` when passing data to Pandas, this means IBKR returned a corrupted historical bar (often missing volume data on a half-day).
* **Fix:** Do not attempt to fix the wrapper code. Simply bypass the corrupted data by adjusting the `durationStr` (e.g., from `'3 Y'` to `'2 Y'`) or switching to a highly liquid proxy asset like `SPY`.

---

## 3. Standardized Project Architecture

Do not use single, monolithic Jupyter Notebooks for final deployments. Break the project into modular Python files:

1. **`config.py`**: Contains all hardcoded parameters (lookbacks, starting capital, risk-free rate, sizing).
2. **`strategy.py`**: The mathematical core. Contains the signal generation and the loop that processes trades to build the Blotter and Ledger.
3. **`metrics.py`**: The evaluation engine. (Use the universal `metrics.py` template provided below).
4. **`main.py`**: The orchestrator. It connects to TWS, fetches the data, runs the strategy, and triggers the metrics report.

---

## 4. Backtesting Output Requirements

Regardless of the specific trading strategy (Breakout, Mean Reversion, Pairs Trading), the output must always include:

1. **`ledger.csv`**: A daily mark-to-market accounting of the portfolio (Date, Position, Cash, Market Value, NAV).
2. **`trades.csv`**: A log of executed trades (Entry/Exit Dates, Prices, Direction, Quantity, Outcome).
3. **Rounding:** Both CSVs must be rounded to exactly 2 decimal places before export to mimic real financial accounting.

---

## 5. Documentation & Tone (GitHub Pages)

When writing Markdown for output:
* **Tone:** Use an objective, graduate-level, first-person voice ("I define", "I calculate"). Remove dramatic modifiers and trading slang. 
* **Objectivity:** If a strategy loses money or fails a statistical test, report it factually. Frame losses as a consequence of market mechanics (e.g., regime changes, volatility expansion) and propose quantitative solutions.
* **Visuals:** Use Plotly `iframe` tags to embed interactive HTML charts into the Markdown.

```html
<iframe src="./data/equity_curve.html" width="100%" height="600px" frameborder="0"></iframe>
```
"""

metrics_content = """import pandas as pd
import numpy as np
import plotly.express as px
import os

def generate_universal_reports(blotter, ledger, risk_free_rate=0.0375, output_dir='./data'):
    \"\"\"
    A universal metrics engine that can be applied to any trading strategy.
    Expects a standard 'blotter' DataFrame and 'ledger' DataFrame.
    \"\"\"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Export CSVs (Rounded to 2 decimal places)
    if not blotter.empty:
        blotter.round(2).to_csv(f'{output_dir}/trades.csv', index=False)
    ledger.round(2).to_csv(f'{output_dir}/ledger.csv')
    
    # 2. Daily Returns & Portfolio Level Metrics
    ledger['log_ret'] = np.log(ledger['NAV'] / ledger['NAV'].shift(1))
    clean_ret = ledger['log_ret'].dropna()
    
    annualized_return = clean_ret.mean() * 252
    volatility = clean_ret.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
    
    # Max Drawdown
    peak = ledger['NAV'].cummax()
    drawdown = (ledger['NAV'] - peak) / peak
    max_dd = drawdown.min()
    
    # Sortino Ratio
    downside_ret = clean_ret[clean_ret < 0]
    downside_vol = downside_ret.std() * np.sqrt(252)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_vol if downside_vol != 0 else 0
    
    # 3. Trade-Level Metrics (Requires 'entry_price' and 'exit_price' in blotter)
    if not blotter.empty and 'entry_price' in blotter.columns and 'exit_price' in blotter.columns:
        # Standardize return calculation regardless of long/short
        # Assumes blotter has a 'direction' column ('Long' or 'Short')
        if 'direction' in blotter.columns:
            blotter['return_pct'] = np.where(
                blotter['direction'] == 'Long',
                (blotter['exit_price'] - blotter['entry_price']) / blotter['entry_price'],
                (blotter['entry_price'] - blotter['exit_price']) / blotter['entry_price']
            )
        else:
            # Fallback assuming long-only if direction isn't specified
            blotter['return_pct'] = (blotter['exit_price'] - blotter['entry_price']) / blotter['entry_price']
            
        avg_trade_return = blotter['return_pct'].mean()
        win_rate = len(blotter[blotter['return_pct'] > 0]) / len(blotter) if len(blotter) > 0 else 0
        
        winning_trades = blotter[blotter['return_pct'] > 0]['return_pct']
        losing_trades = blotter[blotter['return_pct'] <= 0]['return_pct']
        
        gross_profit = winning_trades.sum()
        gross_loss = abs(losing_trades.sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        avg_win = winning_trades.mean() if not winning_trades.empty else 0
        avg_loss = abs(losing_trades.mean()) if not losing_trades.empty else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        total_trades = len(blotter)
    else:
        # Fallback if no trades executed
        total_trades, win_rate, avg_trade_return, profit_factor, expectancy = 0, 0, 0, 0, 0

    # Print Markdown Table to Console
    print("\\nCopy and paste this markdown table into your write-up:\\n")
    print("| Metric | Value |")
    print("| :--- | :--- |")
    print(f"| Total Trades | {total_trades} |")
    print(f"| Win Rate | {win_rate:.2%} |")
    print(f"| Average Return per Trade | {avg_trade_return:.2%} |")
    print(f"| Sharpe Ratio (RFR {risk_free_rate}) | {sharpe_ratio:.4f} |")
    print(f"| Sortino Ratio | {sortino_ratio:.4f} |")
    print(f"| Max Drawdown | {max_dd:.2%} |")
    print(f"| Profit Factor | {profit_factor:.2f} |")
    print(f"| Expectancy (Per Trade) | {expectancy:.4f} |")
    print("\\n")
    
    # 4. Standard Visualizations
    if not blotter.empty and 'outcome' in blotter.columns:
        outcome_counts = blotter['outcome'].value_counts().reset_index()
        outcome_counts.columns = ['Outcome', 'Count']
        fig_hist = px.bar(
            outcome_counts, x='Outcome', y='Count', 
            title='Trade Outcomes', color='Outcome'
        )
        fig_hist.write_html(f'{output_dir}/outcome_histogram.html')
        
    fig_equity = px.line(ledger, x=ledger.index, y='NAV', title='Portfolio Equity Curve')
    fig_equity.write_html(f'{output_dir}/equity_curve.html')
    
    print(f"Exports complete. CSV data and Plotly HTML files saved to {output_dir}/")
"""

with open('AI_PROJECT_CONTEXT.md', 'w') as f:
    f.write(context_content)

with open('universal_metrics.py', 'w') as f:
    f.write(metrics_content)

with open('requirements.txt', 'w') as f:
    f.write("pandas\nnumpy\nscipy\nstatsmodels\nmatplotlib\nplotly\nseaborn\nshinybroker\npython-dotenv")

print("[file-tag: AI_PROJECT_CONTEXT.md]")
print("[file-tag: universal_metrics.py]")
print("[file-tag: requirements.txt]")

```
