# Factor Reference — IC Values, Weights, and Usage

## Information Coefficient (IC) by Factor
IC measures Spearman rank correlation between factor rank and forward 20-day return rank.
Industry threshold: IC > 0.05 is considered usable.

| Factor | IC (20d) | Weight | Source |
|--------|----------|--------|--------|
| surprise_pct_last | 0.064 | 2.0 | Earnings |
| ah_gap | 0.058 | 1.2 | Pre/After-market |
| analyst_buy_ratio_chg_1m | 0.051 | 0.9 | Analyst |
| llm_sent_7d | 0.047 | 0.8 | LLM Sentiment |
| analyst_buy_ratio | 0.043 | 0.7 | Analyst |
| inst_holding_pct_chg | 0.039 | 0.6 | Institutional |
| news_burst_20d | 0.033 | 0.5 | News Volume |
| retail_sent_score | 0.021 | 0.3 | Retail (StockTwits) |

## Factor Computation Details

### surprise_pct_last
`(actual_eps - consensus_eps) / abs(consensus_eps)`
- Values typically range from -0.5 to +0.5
- Signal threshold: > 0.05 (5% positive surprise)
- Post-earnings drift makes this the highest-IC factor

### ah_gap
`(after_hours_close / regular_close) - 1`
- Values range: -0.1 to +0.1 (±10%)
- Signal threshold: > 0.02 (2% after-hours gain)

### analyst_buy_ratio_chg_1m
`buy_ratio_now - buy_ratio_1m_ago`
- Captures upgrade/downgrade momentum
- Signal threshold: > 0.1 (10 percentage point improvement)

### llm_sent_7d
`rolling 7-day average of LLM sentiment scores`
- LLM scores: bullish=1, neutral=0, bearish=-1
- Weighted by article confidence score
- Signal threshold: > 0.3

### inst_holding_pct_chg
`(current_quarter_pct - last_quarter_pct)`
- QoQ change in 13F institutional ownership
- Signal threshold: > 0.02 (2% increase)

### news_burst_20d
`news_count_1d / rolling_mean_20d`
- Ratio of today's news count to 20-day average
- Signal threshold: > 2.0 (2x normal volume)

### retail_sent_score
`(bullish_count - bearish_count) / (bullish_count + bearish_count)`
- Ranges from -1 to +1
- Signal threshold: > 0.5

### macro_risk_on
`1 if (SPY_close > SPY_200MA) AND (macro_vix_pctile_252d < 0.5) else 0`
- Binary regime indicator
- When 0: multiply all scores by 0.7

## Composite Score Formula
```
composite_score = sum(factor_value * weight for each factor) / sum(weights)
```
Then apply regime adjustment:
- If macro_vix_pctile_252d > 0.8: score *= 0.5
- If macro_risk_on == 0: score *= 0.7

## Walk-Forward Validation Results
- Training window: 1 year rolling
- Validation window: 3 months OOS (out-of-sample)
- IC stability: best factors maintain IC > 0.04 across all validation windows
- Factor decay: IC drops ~40% at 45-day horizon vs 20-day

## Long-Short Portfolio (research/factor_analysis.py)
- Top quintile LONG vs bottom quintile SHORT
- Annualized spread: 18.3%
- Walk-forward Sharpe: 1.41

## Available Signals in daily_signals Collection
Fields returned by GET /api/signals/latest:
- `symbol`: ticker
- `date`: signal date
- `composite_score`: 0–1 float
- `signal_type`: "LONG" | "WATCH" | "NEUTRAL"
- `quality_score`: data freshness/coverage score
- `surprise_pct_last`: earnings surprise
- `ah_gap`: after-hours gap
- `analyst_buy_ratio_chg_1m`: analyst momentum
- `inst_holding_pct_chg`: institutional change
- `llm_sent_7d`: LLM sentiment
- `news_burst_20d`: news volume spike
