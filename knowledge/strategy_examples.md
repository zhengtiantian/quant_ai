# Strategy Examples and Patterns

## Example 1: Earnings Surprise Momentum Strategy
**Description**: Buy stocks with positive earnings surprise + bullish LLM sentiment, hold 10–20 days.

**Strategy Spec**:
```json
{
  "name": "Earnings Surprise Momentum",
  "description": "Long on positive EPS surprise with bullish sentiment confirmation",
  "market": "US_EQUITY",
  "tasks": [
    {
      "taskId": "task_data",
      "type": "data_collection",
      "module": "quant_data.stock_collector.price_collector.collector",
      "dependencies": [],
      "parameters": {"symbols": "all", "lookback_days": 30}
    },
    {
      "taskId": "task_features",
      "type": "feature_engineering",
      "module": "quant_data.feature_builders.daily_symbol_features",
      "dependencies": ["task_data"],
      "parameters": {"indicators": ["surprise_pct_last", "eps_beat_flag", "llm_sent_7d", "days_to_earnings"]}
    },
    {
      "taskId": "task_signals",
      "type": "signal_generation",
      "module": "quant_data.research.score_daily_signals",
      "dependencies": ["task_features"],
      "parameters": {
        "rule": "surprise_pct_last > 0.05 AND eps_beat_flag == 1 AND llm_sent_7d > 0.3",
        "min_score": 0.6
      }
    },
    {
      "taskId": "task_risk",
      "type": "risk_management",
      "module": "quant_data.research.track_positions",
      "dependencies": ["task_signals"],
      "parameters": {"max_position_size": 0.1, "stop_loss": 0.05, "max_hold": 20}
    },
    {
      "taskId": "task_backtest",
      "type": "backtesting",
      "module": "quant_data.research.backtest_portfolio",
      "dependencies": ["task_risk"],
      "parameters": {"initial_cash": 100000, "fee_bps": 5, "window": "2y", "rebalance": "daily"}
    }
  ],
  "risk": {"max_position_size": 0.1, "stop_loss": 0.05, "max_drawdown": 0.2},
  "backtest": {"initial_cash": 100000, "fee_bps": 5, "window": "2y", "rebalance": "daily"}
}
```

**Python code pattern for signal generation**:
```python
from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb://mongo:27017/")
db = client["quant_data"]

# Load features
cursor = db.daily_symbol_features.find(
    {"date": {"$gte": "2024-01-01"}},
    {"symbol": 1, "date": 1, "surprise_pct_last": 1, "eps_beat_flag": 1, "llm_sent_7d": 1}
)
df = pd.DataFrame(list(cursor))

# Apply signal rule
df["signal"] = (
    (df["surprise_pct_last"] > 0.05) &
    (df["eps_beat_flag"] == 1) &
    (df["llm_sent_7d"] > 0.3)
).astype(int)

signals = df[df["signal"] == 1][["symbol", "date", "surprise_pct_last", "llm_sent_7d"]]
```

---

## Example 2: Analyst Upgrade + Institutional Accumulation
**Description**: Buy when analyst buy ratio rises AND institutions are accumulating.

**Signal rule**: `analyst_buy_ratio_chg_1m > 0.1 AND inst_holding_pct_chg > 0.02`

**Key parameters**:
- `analyst_buy_ratio_chg_1m > 0.1`: significant analyst upgrade momentum
- `inst_holding_pct_chg > 0.02`: 2% increase in institutional ownership (QoQ)
- Hold period: 30 days max
- Stop loss: 3%

---

## Example 3: Pre-Market Gap + Retail Sentiment Surge
**Description**: Capture pre-market momentum backed by retail enthusiasm.

**Signal rule**: `pm_gap > 0.02 AND retail_sent_score > 0.5 AND news_burst_20d > 2.0`

**Key parameters**:
- `pm_gap > 0.02`: 2%+ pre-market gap up
- `retail_sent_score > 0.5`: majority bullish on StockTwits
- `news_burst_20d > 2.0`: news volume 2x above 20-day average
- Hold period: 5 days (short-term momentum)
- Stop loss: 2%

---

## Example 4: Low-Risk Regime Strategy
**Description**: Conservative strategy that only activates in low-VIX environments.

**Signal rule**: `macro_risk_on == 1 AND composite_score > 0.65 AND macro_vix_pctile_252d < 0.3`

**Python code pattern for regime filter**:
```python
# Read macro indicators
macro_doc = db.macro_indicators.find_one(sort=[("date", -1)])
vix_percentile = macro_doc.get("macro_vix_pctile_252d", 0.5)
risk_on = macro_doc.get("macro_risk_on", 0)

if vix_percentile > 0.8 or risk_on == 0:
    print("Risk-off regime: skipping signal generation")
    exit()
```

---

## Common Parameter Defaults
- `max_position_size`: 0.1 (10% of portfolio per position)
- `stop_loss`: 0.02 (2%)
- `max_drawdown`: 0.2 (20% portfolio stop)
- `fee_bps`: 5 (5 basis points commission)
- `initial_cash`: 100000
- `rebalance`: "daily"
- `window`: "2y"
- `max_hold`: 60 (days)

## MongoDB Connection Pattern
```python
from pymongo import MongoClient
client = MongoClient("mongodb://mongo:27017/")
db = client["quant_data"]
```

## Backtest Code Pattern
```python
import pandas as pd
import numpy as np

def compute_returns(signals_df, prices_df, hold_days=20):
    """Compute strategy returns given signal dates and prices."""
    results = []
    for _, row in signals_df.iterrows():
        symbol = row["symbol"]
        entry_date = row["date"]
        price_series = prices_df[prices_df["symbol"] == symbol].set_index("date")["close"]
        if entry_date not in price_series.index:
            continue
        future_prices = price_series.loc[entry_date:].head(hold_days + 1)
        if len(future_prices) < 2:
            continue
        ret = (future_prices.iloc[-1] / future_prices.iloc[0]) - 1
        results.append({"symbol": symbol, "entry": entry_date, "return": ret})
    return pd.DataFrame(results)

# Sharpe ratio
def sharpe(returns, periods_per_year=252, rf=0.05):
    mu = returns.mean() * periods_per_year
    sigma = returns.std() * np.sqrt(periods_per_year)
    return (mu - rf) / sigma if sigma > 0 else 0
```
