"""The user's loss budget: how much a bad day, or a report night, could cost, kept under a share of
the portfolio (RISK_BUDGET_PCT: 3% chosen by the user on 2026-11-02, then 5% the same day).

No model forecasts the direction of a night like INTC's 2026-10-30 (-29%); its size can be bounded, and
the bounds below were checked on the 102-stock daily bars, 2017-2026
(quant_data research/models/risk_budget.py):

  bad day     DAY_K x the EWMA (half-life 20) standard deviation of the portfolio's daily return. With
              DAY_K = 1.671, 5-stock portfolios lost more on 5.0% of days (4.9% for a chip basket):
              "a 1-in-20 bad day"
  bad night   EVENT_K x the typical report move. Over 3,481 reports the loss exceeded 2.635 x the
              median absolute move of the stock's previous 8 reports on 5% of nights. Live, the typical
              move comes from the options straddle (expected absolute move; the median of a normal move
              is 0.845 of its mean), else from the ATR
  budget      kept to 3% on a chip basket (INTC, TSM, STX, MU, AMD), the worst day went from -16.0% to
              -7.7%, max drawdown -52% to -43%, Sharpe 1.07 to 1.12, at 31.1% a year instead of 38.2%:
              less lost in 2022, much less made in 2025-26

Market state (VIX percentile and SPY's 5-day move) is shown with what followed similar weeks in
2016-2026. It is context, not a forecast: the 5-day mean after a fearful week (+0.86%, 66 weeks) is
not significantly different from all weeks (+0.26%); what changes reliably is the spread.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import portfolio as pf

BUDGET_PCT = float(os.getenv("RISK_BUDGET_PCT", "5"))   # the user moved it from 3% to 5% on 2026-11-02
DAY_K = 1.671
EVENT_K = 2.635
MEDIAN_OF_MEAN_ABS = 0.845          # median |x| / mean |x| for a normal move
REPORT_MOVE_PER_ATR = 1.6           # fallback: median report move ~ 1.6 x ATR% (4.36% vs ~2.7% ATR, 102 stocks)
HALFLIFE = 20

# What followed similar weeks for SPY (2016-2026, one per week; quant_data risk_budget / 11-02 analysis)
STATES = {
    "fearful": {"label": "fearful (VIX in its top 20% of the year, SPY down 3%+ in 5 days)", "label_zh": "恐慌（VIX 处于一年内最高的 20%，SPY 5 天跌超 3%）",
                "n": 66, "up5": 67, "mean5": 0.86, "p10_5": -4.6},
    "tense": {"label": "tense (VIX in its top 20% of the year)", "label_zh": "紧张（VIX 处于一年内最高的 20%）",
              "n": 152, "up5": 61, "mean5": 0.37, "p10_5": -3.9},
    "normal": {"label": "normal", "label_zh": "正常", "n": 564, "up5": 62, "mean5": 0.26, "p10_5": -2.4},
}
ELECTIONS = "On the 5 US election days 2016-2024 SPY rose each day and in 4 of 5 next sessions (5 cases: not a forecast)"
ELECTIONS_ZH = "2016–2024 年 5 次美国大选当天 SPY 都上涨，次日 4 涨 1 跌（只有 5 例，不是预测）"


def ewma_cov(symbols: List[str], chart_fn: Optional[Callable] = None, path=None, cutoff: Optional[str] = None) -> Optional[np.ndarray]:
    """Daily covariance with EWMA weights (half-life HALFLIFE sessions) over the dates all symbols have."""
    closes = {}
    for s in symbols:
        c = (chart_fn or pf.get_chart)(s, "6m", path).get("candles") or []
        closes[s] = {x["time"]: float(x["close"]) for x in c if not x.get("partial") and x.get("close") and (cutoff is None or x["time"] <= cutoff)}
    common = sorted(set.intersection(*(set(v) for v in closes.values()))) if closes else []
    if len(common) < 40:
        return None
    px = np.array([[closes[s][d] for s in symbols] for d in common])
    r = np.diff(np.log(px), axis=0)
    w = 0.5 ** (np.arange(len(r))[::-1] / HALFLIFE)
    w /= w.sum()
    mu = (w[:, None] * r).sum(0)
    d = r - mu
    return np.atleast_2d((w[:, None] * d).T @ d)


def day_loss(values: np.ndarray, cov: Optional[np.ndarray]) -> Optional[float]:
    """Dollars a 1-in-20 bad day loses on these holdings."""
    if cov is None:
        return None
    return float(DAY_K * math.sqrt(max(values @ cov @ values, 0.0)))


def report_move(symbol: str, report_date, price: Optional[float], atr_pct: Optional[float],
                implied_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """The typical (median absolute) move of this report, from the options straddle, else from the ATR."""
    implied = None
    try:
        from . import events
        implied = (implied_fn or events.implied_move)(symbol, report_date, price)
    except Exception:  # noqa: BLE001 - no options quote: fall back to the ATR
        implied = None
    if implied and implied.get("implied_move_pct"):
        return {"typical_pct": round(implied["implied_move_pct"] * MEDIAN_OF_MEAN_ABS, 2), "source": "options",
                "straddle_pct": implied["implied_move_pct"], "expiry": implied.get("expiry")}
    if atr_pct:
        return {"typical_pct": round(atr_pct * REPORT_MOVE_PER_ATR, 2), "source": "atr"}
    return {"typical_pct": None, "source": None}


def night_loss_pct(typical_pct: Optional[float]) -> Optional[float]:
    return round(EVENT_K * typical_pct, 1) if typical_pct else None


def market_state(path=None, features_fn: Optional[Callable] = None, chart_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """VIX percentile (latest feature row) and SPY's 5-session move, with what followed similar weeks."""
    vix = pct = None
    try:
        row = (features_fn or _latest_macro)()
        vix, pct = row.get("macro_vix"), row.get("macro_vix_pctile_252d")
    except Exception:  # noqa: BLE001 - mongo down: state from SPY alone
        pass
    spy5 = None
    try:
        c = [x for x in (chart_fn or pf.get_chart)("SPY", "1m", path).get("candles") or [] if x.get("close")]
        if len(c) >= 6:
            spy5 = round((c[-1]["close"] / c[-6]["close"] - 1) * 100, 2)
    except Exception:  # noqa: BLE001
        pass
    key = "fearful" if (pct or 0) >= 0.8 and (spy5 or 0) <= -3 else "tense" if (pct or 0) >= 0.8 else "normal"
    return {"state": key, "vix": None if vix is None else round(float(vix), 1), "vix_pctile": None if pct is None else round(float(pct), 2),
            "spy_5d_pct": spy5, **STATES[key]}


def _latest_macro() -> Dict[str, Any]:
    from . import advisor
    return advisor._db()["daily_symbol_features"].find_one(
        {"symbol": "AAPL", "trade_date": {"$type": "string", "$regex": "^20"}}, sort=[("trade_date", -1)],
        projection={"_id": 0, "trade_date": 1, "macro_vix": 1, "macro_vix_pctile_252d": 1}) or {}
