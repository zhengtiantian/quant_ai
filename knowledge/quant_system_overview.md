# Quant Trade System — Architecture Overview

## System Purpose
An AI-driven equity signal platform that classifies financial news, engineers multi-factor features,
scores 100 US stocks daily, and publishes LONG signals to Kafka. A LangChain agent on top
takes natural language strategy descriptions and generates executable Python trading code.

## Data Pipeline (MongoDB: quant_data)

### Collections
- `articles` — raw financial news articles (840K+), fields: symbol, headline, body, published_at
- `article_sentiments` — LLM-tagged sentiment (llm_sentiment_final: bullish/bearish/neutral, llm_disagreement, confidence)
- `daily_symbol_features` — per-symbol per-date feature vectors (the main feature store)
- `daily_signals` — scored signals output, fields: symbol, date, composite_score, signal_type, quality_score
- `positions` — paper trading open/closed positions, fields: symbol, entry_date, entry_price, status, exit_reason
- `portfolio_performance` — backtest stats: sharpe, max_drawdown, win_rate, ann_return
- `macro_indicators` — VIX, TNX, DXY, SPY daily close prices
- `data_quality_checks` — pipeline health checks (news_volume, price_freshness, feature_coverage)

## Feature Engineering (daily_symbol_features fields)

### Sentiment Features
- `news_count_1d`, `news_count_7d`, `news_count_30d` — article volume
- `sentiment_score_7d`, `sentiment_score_30d` — rolling bullish/bearish ratio
- `llm_sent_7d`, `llm_sent_30d` — LLM-weighted sentiment
- `news_burst_20d` — unusual spike in news volume vs 20-day average
- `sentiment_momentum` — delta between 7d and 30d sentiment

### Earnings Features
- `surprise_pct_last` — (actual - consensus) / |consensus|, best single factor (IC=0.064)
- `eps_beat_flag` — binary: EPS beat current quarter
- `days_since_earnings`, `days_to_earnings` — calendar proximity

### Analyst Features
- `analyst_buy_ratio` — fraction of analysts with Buy rating
- `analyst_buy_ratio_chg_1m` — 1-month change in buy ratio (captures upgrades/downgrades)

### Institutional Features
- `inst_holding_pct_chg` — QoQ change in institutional ownership (13F)

### Pre/After-Market Features
- `ah_gap` — after-hours close vs regular close: (ah_last / reg_close) - 1
- `pm_gap` — pre-market gap

### Retail Sentiment
- `retail_sent_score` — StockTwits bull/bear ratio: (bull - bear) / (bull + bear)

### Macro / Regime Features
- `macro_vix_pctile_252d` — VIX percentile over 252 trading days
- `macro_risk_on` — 1 if SPY > 200MA AND macro_vix_pctile_252d < 0.5, else 0

## Signal Scoring (score_daily_signals.py)

### Factor Weights
| Factor | Weight | Notes |
|--------|--------|-------|
| surprise_pct_last | 2.0 | Best IC (0.064) |
| ah_gap | 1.2 | Highest D-series IC |
| analyst_buy_ratio_chg_1m | 0.9 | Captures upgrade momentum |
| analyst_buy_ratio | 0.7 | Absolute buy conviction |
| inst_holding_pct_chg | 0.6 | Institutional accumulation |
| llm_sent_7d | 0.8 | AI sentiment signal |
| news_burst_20d | 0.5 | Unusual attention spike |
| retail_sent_score | 0.3 | Retail confirmation |

### Regime Adjustment
- If macro_vix_pctile_252d > 0.8: multiply composite_score by 0.5 (high-fear regime)
- If macro_risk_on == 0: multiply by 0.7 (risk-off)

### Signal Output
- composite_score > 0.6 → signal_type = "LONG"
- composite_score 0.3–0.6 → signal_type = "WATCH"
- quality_score = weighted average of data freshness and coverage

## Position Management (track_positions.py)
Exit triggers: max_hold (60d), score_below_exit (< 0.25), earnings_miss, sentiment_reversal,
analyst_downgrade, inst_outflow. Results written to `positions` with status=closed.

## Backtest Performance (backtest_portfolio.py)
- Annualized return: +21.7% (after transaction costs)
- Sharpe ratio: 1.34
- Max drawdown: -11.2%
- Win rate: 58%
- Transaction cost model: 5 bps commission + slippage

## LangChain Agent (quant_langchain/main.py)
- Exposes FastAPI on port 8083
- /api/workflow/generate-spec: agent uses tools to gather context, then generates JSON strategy spec
- /api/workflow/generate-tasks: LLM generates Python code for each task in the spec
- /api/ask: RAG-augmented Q&A about the quant system
