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

import json
import math
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from . import portfolio as pf

BUDGET_PCT = float(os.getenv("RISK_BUDGET_PCT", "5"))   # the user moved it from 3% to 5% on 2026-11-02
DAY_K = 1.671
EVENT_K = 2.635
MEDIAN_OF_MEAN_ABS = 0.845          # median |x| / mean |x| for a normal move
REPORT_MOVE_PER_ATR = 1.6           # fallback: median report move ~ 1.6 x ATR% (4.36% vs ~2.7% ATR, 102 stocks)
HALFLIFE = 20
ADVICE_FILE = Path(os.getenv("ADVICE_FILE", "/Volumes/Data4T/daily_advice/latest.json"))
LIVE_MACRO_TTL = 300

# Deterministic thresholds for the Market & Macro Weather Banner (context & risk guardrail only, NOT a forecast).
VIX_HIGH = 25.0
VIX_EXTREME = 30.0
SPY_5D_PANIC = -3.0
SPX_1D_PANIC = -1.5
SOX_1D_STRESS = -3.0
SOX_1D_CRITICAL = -5.0
SOXX_5D_STRESS = -6.0
US10Y_HIGH = 4.75
YIELD_CHG60_HEADWIND = 0.30
DXY_CHG60_HEADWIND = 2.0
CREDIT_CHG_HEADWIND = -1.0
OIL_WTI_HIGH = 85.0
OIL_BRENT_HIGH = 90.0
OIL_CHG20_SHOCK = 15.0
OIL_CHG60_SHOCK = 20.0
UPCOMING_MACRO_DAYS = 14

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


def _chart_ret(symbol: str, range_key: str, sessions: int, chart_fn: Callable, path=None) -> Optional[float]:
    try:
        c = [x for x in chart_fn(symbol, range_key, path).get("candles") or [] if x.get("close")]
        if len(c) >= sessions + 1 and c[-sessions - 1]["close"]:
            return round((float(c[-1]["close"]) / float(c[-sessions - 1]["close"]) - 1.0) * 100.0, 2)
    except Exception:  # noqa: BLE001
        pass
    return None


def _credit_changes(chart_fn: Callable, path=None) -> Dict[str, Optional[float]]:
    """20- and 60-session log-ratio change (in %) of HYG / IEF as a live credit-spread proxy."""
    out: Dict[str, Optional[float]] = {"credit_chg20": None, "credit_chg60": None}
    try:
        hyg = {x["time"]: float(x["close"]) for x in (chart_fn("HYG", "6m", path).get("candles") or []) if x.get("close")}
        ief = {x["time"]: float(x["close"]) for x in (chart_fn("IEF", "6m", path).get("candles") or []) if x.get("close")}
        common = sorted(set(hyg) & set(ief))
        if len(common) >= 21:
            ratios = [math.log(hyg[d] / ief[d]) for d in common if hyg[d] > 0 and ief[d] > 0]
            if len(ratios) >= 21:
                out["credit_chg20"] = round((ratios[-1] - ratios[-21]) * 100.0, 2)
            if len(ratios) >= 61:
                out["credit_chg60"] = round((ratios[-1] - ratios[-61]) * 100.0, 2)
    except Exception:  # noqa: BLE001
        pass
    return out


def _live_cnbc_snapshot(path=None) -> Dict[str, Dict[str, Any]]:
    """Live VIX, SPX, SOX, 10Y/2Y Treasury yields, DXY, and WTI/Brent oil from CNBC via global_markets cache."""
    try:
        from . import global_markets
        syms = [".VIX", ".SPX", ".SOX", "US10Y", "US2Y", ".DXY", "@CL.1", "@LCO.1"]
        stored = global_markets._cached_cnbc(syms, path, max_age=LIVE_MACRO_TTL)
        out: Dict[str, Dict[str, Any]] = {}
        for s in syms:
            row = stored.get(s)
            if row and row.get("data"):
                out[s] = json.loads(row["data"])
        return out
    except Exception:  # noqa: BLE001
        return {}


def _macro_hist_baseline(advice_path: Path = ADVICE_FILE) -> Dict[str, Any]:
    """Latest 60-session macro baseline (rates, dollar, credit, oil) written into daily_advice latest.json."""
    out: Dict[str, Any] = {
        "as_of": None, "y10": None, "y10_60d_ago": None, "y10_chg60": None,
        "y2": None, "y2_60d_ago": None, "y2_chg60": None,
        "dxy": None, "dxy_60d_ago": None, "dollar_chg60": None,
        "credit_chg20": None, "credit_chg60": None,
        "oil": None, "oil_20d_ago": None, "oil_60d_ago": None, "oil_chg20": None, "oil_chg60": None,
        "brent": None, "brent_chg20": None, "brent_chg60": None,
    }
    try:
        if not advice_path.exists():
            return out
        raw = json.loads(advice_path.read_text()).get("macro") or {}
        if raw.get("as_of"):
            out["as_of"] = str(raw["as_of"])[:10]
        for k in ("y10", "y10_60d_ago", "y10_chg60", "y2", "y2_60d_ago", "y2_chg60",
                  "dxy", "dxy_60d_ago", "dollar_chg60", "credit_chg20", "credit_chg60",
                  "oil", "oil_20d_ago", "oil_60d_ago", "oil_chg20", "oil_chg60",
                  "brent", "brent_chg20", "brent_chg60"):
            v = raw.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out[k] = round(float(v), 2)
    except Exception:  # noqa: BLE001
        pass
    return out


def _topic_sig(text: str) -> set:
    """Alphanumeric / CJK tokens for deduplicating near-identical same-day policy headlines."""
    return set(re.findall(r"[a-z0-9]{3,}|[\u4e00-\u9fff]{2,}", (text or "").lower()))


def _upcoming_macro_policy(path=None, today: Optional[date] = None, days: int = UPCOMING_MACRO_DAYS) -> List[Dict[str, Any]]:
    """Fast local read of upcoming high-impact macro releases (FOMC/CPI/NFP/OpEx/Elections)
    and MACRO policy deadlines (tariffs, export controls, AI/chip regulation) within `days`."""
    cur = today or datetime.now(ZoneInfo("America/New_York")).date()
    end = cur + timedelta(days=days)
    items: List[Dict[str, Any]] = []

    try:
        from . import catalysts
        for m in catalysts.OFFICIAL_MACRO_SCHEDULE:
            d = date.fromisoformat(m["date"])
            if cur <= d <= end:
                zh_raw = m.get("note_zh") or m["event"]
                zh_short = re.split(r"[（(:：]", zh_raw, maxsplit=1)[0].strip() or zh_raw
                items.append({
                    "date": m["date"], "days_away": (d - cur).days, "kind": "macro",
                    "impact": "high", "title_en": m["event"], "title_zh": zh_short,
                })
    except Exception:  # noqa: BLE001
        pass

    try:
        conn = pf._conn(path)
        try:
            rows = conn.execute(
                "SELECT event_date, kind, impact_level, title_zh, title_en FROM collected_catalysts "
                "WHERE symbol = 'MACRO' AND kind IN ('policy', 'macro') AND event_date >= ? AND event_date <= ? "
                "ORDER BY event_date ASC, CASE impact_level WHEN 'high' THEN 0 ELSE 1 END, id ASC",
                (cur.isoformat(), end.isoformat()),
            ).fetchall()
        finally:
            conn.close()
        for r in rows:
            d = date.fromisoformat(r["event_date"])
            t_en = (r["title_en"] or r["title_zh"] or "").strip()
            t_zh = (r["title_zh"] or r["title_en"] or "").strip()
            if not t_en and not t_zh:
                continue
            items.append({
                "date": r["event_date"], "days_away": (d - cur).days, "kind": r["kind"],
                "impact": r["impact_level"] or "medium", "title_en": t_en, "title_zh": t_zh,
            })
    except Exception:  # noqa: BLE001 - table may not exist in isolated test DBs
        pass

    items.sort(key=lambda x: (x["date"], 0 if x["kind"] == "macro" else 1, 0 if x["impact"] == "high" else 1))
    deduped: List[Dict[str, Any]] = []
    for it in items:
        sig = _topic_sig(it["title_en"]) | _topic_sig(it["title_zh"])
        dup = False
        for prev in deduped:
            if prev["date"] != it["date"] or prev["kind"] != it["kind"]:
                continue
            psig = _topic_sig(prev["title_en"]) | _topic_sig(prev["title_zh"])
            if sig and psig and (len(sig & psig) / min(len(sig), len(psig)) >= 0.45):
                dup = True
                break
        if not dup:
            deduped.append(it)
        if len(deduped) >= 5:
            break
    return deduped


def market_state(path=None, features_fn: Optional[Callable] = None, chart_fn: Optional[Callable] = None,
                 live_fn: Optional[Callable] = None, macro_hist_fn: Optional[Callable] = None,
                 events_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """Live market panic, semiconductor sector stress and macro headwind weather.

    Context and risk-guardrail input only — NOT a return forecast. Backtests (1990-2026) showed
    neither VIX nor macro timing rules reliably beat holding, while short-term tail volatility
    and dip-buying follow-through risk jump sharply during panic / macro headwind regimes.
    """
    c_fn = chart_fn or pf.get_chart
    live = {}
    try:
        live = (live_fn or _live_cnbc_snapshot)(path) or {}
    except Exception:  # noqa: BLE001
        live = {}

    vix_mongo = pct = None
    try:
        row = (features_fn or _latest_macro)() or {}
        vix_mongo, pct = row.get("macro_vix"), row.get("macro_vix_pctile_252d")
    except Exception:  # noqa: BLE001 - mongo down: state from live CNBC & SPY
        pass

    live_vix = (live.get(".VIX") or {}).get("price")
    vix = float(live_vix) if live_vix is not None else (float(vix_mongo) if vix_mongo is not None else None)
    vix_source = "cnbc_live" if live_vix is not None else ("mongo" if vix_mongo is not None else None)
    spx_1d = (live.get(".SPX") or {}).get("change_pct")
    sox_1d = (live.get(".SOX") or {}).get("change_pct")
    us10y = (live.get("US10Y") or {}).get("price")
    us2y = (live.get("US2Y") or {}).get("price")
    dxy = (live.get(".DXY") or {}).get("price")
    wti = (live.get("@CL.1") or {}).get("price")
    brent = (live.get("@LCO.1") or {}).get("price")

    spy5 = _chart_ret("SPY", "1m", 5, c_fn, path)
    soxx5 = _chart_ret("SOXX", "1m", 5, c_fn, path)

    hist = {}
    try:
        hist = (macro_hist_fn or _macro_hist_baseline)() or {}
    except Exception:  # noqa: BLE001
        hist = {}
    credit_live = _credit_changes(c_fn, path)

    y10_val = round(float(us10y), 2) if us10y is not None else hist.get("y10")
    y2_val = round(float(us2y), 2) if us2y is not None else hist.get("y2")
    dxy_val = round(float(dxy), 2) if dxy is not None else hist.get("dxy")
    wti_val = round(float(wti), 2) if wti is not None else hist.get("oil")
    brent_val = round(float(brent), 2) if brent is not None else hist.get("brent")
    y10_chg60 = (round(y10_val - hist["y10_60d_ago"], 2)
                 if (y10_val is not None and hist.get("y10_60d_ago") is not None)
                 else hist.get("y10_chg60"))
    y2_chg60 = (round(y2_val - hist["y2_60d_ago"], 2)
                if (y2_val is not None and hist.get("y2_60d_ago") is not None)
                else hist.get("y2_chg60"))
    dxy_chg60 = (round((dxy_val / hist["dxy_60d_ago"] - 1.0) * 100.0, 2)
                 if (dxy_val is not None and hist.get("dxy_60d_ago"))
                 else hist.get("dollar_chg60"))
    if dxy_chg60 is None:
        dxy_chg60 = _chart_ret("UUP", "6m", 60, c_fn, path)
    oil_chg20 = (round((wti_val / hist["oil_20d_ago"] - 1.0) * 100.0, 2)
                 if (wti_val is not None and hist.get("oil_20d_ago"))
                 else hist.get("oil_chg20"))
    oil_chg60 = (round((wti_val / hist["oil_60d_ago"] - 1.0) * 100.0, 2)
                 if (wti_val is not None and hist.get("oil_60d_ago"))
                 else hist.get("oil_chg60"))
    if oil_chg60 is None:
        oil_chg60 = _chart_ret("USO", "6m", 60, c_fn, path)
    credit_chg20 = credit_live["credit_chg20"] if credit_live["credit_chg20"] is not None else hist.get("credit_chg20")
    credit_chg60 = credit_live["credit_chg60"] if credit_live["credit_chg60"] is not None else hist.get("credit_chg60")

    high_vix = ((pct or 0.0) >= 0.8) or (vix is not None and vix >= VIX_HIGH)
    broad_drop = (spy5 is not None and spy5 <= SPY_5D_PANIC) or (spx_1d is not None and spx_1d <= SPX_1D_PANIC)
    key = "fearful" if (high_vix and broad_drop) else "tense" if high_vix else "normal"

    sector_stress = bool((sox_1d is not None and sox_1d <= SOX_1D_STRESS) or (soxx5 is not None and soxx5 <= SOXX_5D_STRESS))

    rates_headwind = bool(
        (y10_val is not None and y10_val >= US10Y_HIGH)
        or (y2_chg60 is not None and y2_chg60 >= YIELD_CHG60_HEADWIND)
        or (y10_chg60 is not None and y10_chg60 >= YIELD_CHG60_HEADWIND)
    )
    dollar_headwind = bool(dxy_chg60 is not None and dxy_chg60 >= DXY_CHG60_HEADWIND)
    credit_headwind = bool(
        (credit_chg20 is not None and credit_chg20 <= CREDIT_CHG_HEADWIND)
        or (credit_chg60 is not None and credit_chg60 <= CREDIT_CHG_HEADWIND)
    )
    oil_headwind = bool(
        (wti_val is not None and wti_val >= OIL_WTI_HIGH)
        or (brent_val is not None and brent_val >= OIL_BRENT_HIGH)
        or (oil_chg20 is not None and oil_chg20 >= OIL_CHG20_SHOCK)
        or (oil_chg60 is not None and oil_chg60 >= OIL_CHG60_SHOCK)
    )
    macro_flags = [
        name for name, on in (("rates", rates_headwind), ("dollar", dollar_headwind), ("credit", credit_headwind), ("oil", oil_headwind))
        if on
    ]
    macro_level = "heavy" if len(macro_flags) >= 2 else "moderate" if len(macro_flags) == 1 else "calm"

    if (key == "fearful" and (sector_stress or macro_level == "heavy")) or (vix is not None and vix >= VIX_EXTREME) or (sox_1d is not None and sox_1d <= SOX_1D_CRITICAL):
        severity = "critical"
    elif key in ("fearful", "tense") or sector_stress or macro_level == "heavy":
        severity = "warn"
    else:
        severity = "info"

    tighten_buys = bool(key == "fearful" or sector_stress)

    # Bilingual headline & detail
    tags_en, tags_zh = [], []
    if key == "fearful":
        tags_en.append("Market panic")
        tags_zh.append("市场恐慌")
    elif key == "tense":
        tags_en.append("Elevated volatility")
        tags_zh.append("波动升温")
    if sector_stress:
        tags_en.append("Chip sector selloff")
        tags_zh.append("半导体踩踏")
    if macro_level == "heavy":
        tags_en.append("Heavy macro headwind")
        tags_zh.append("宏观紧缩重压")
    elif macro_level == "moderate":
        tags_en.append("Mild macro pressure")
        tags_zh.append("温和宏观压力")

    title_en = " + ".join(tags_en) if tags_en else "Market normal"
    title_zh = " + ".join(tags_zh) if tags_zh else "市场状态正常"

    pills: List[Dict[str, Any]] = []
    if vix is not None:
        pills.append({"id": "vix", "label": "VIX", "value": f"{vix:.1f}", "stress": vix >= VIX_HIGH or (pct or 0) >= 0.8})
    if spx_1d is not None:
        pills.append({"id": "spx1d", "label": "SPX 1d", "value": f"{spx_1d:+.2f}%", "stress": spx_1d <= SPX_1D_PANIC})
    if spy5 is not None:
        pills.append({"id": "spy5d", "label": "SPY 5d", "value": f"{spy5:+.2f}%", "stress": spy5 <= SPY_5D_PANIC})
    if sox_1d is not None:
        pills.append({"id": "sox1d", "label": "SOX 1d", "value": f"{sox_1d:+.2f}%", "stress": sox_1d <= SOX_1D_STRESS})
    if soxx5 is not None:
        pills.append({"id": "soxx5d", "label": "SOXX 5d", "value": f"{soxx5:+.2f}%", "stress": soxx5 <= SOXX_5D_STRESS})
    if y10_val is not None:
        chg = f" ({y10_chg60:+.2f}pp 60d)" if y10_chg60 is not None else ""
        pills.append({"id": "us10y", "label": "US10Y", "value": f"{y10_val:.2f}%{chg}", "stress": rates_headwind})
    if y2_val is not None or y2_chg60 is not None:
        val = f"{y2_val:.2f}%" if y2_val is not None else ""
        chg = f"{' ' if val else ''}({y2_chg60:+.2f}pp 60d)" if y2_chg60 is not None else ""
        pills.append({"id": "us2y", "label": "US2Y", "value": f"{val}{chg}", "stress": bool(y2_chg60 is not None and y2_chg60 >= YIELD_CHG60_HEADWIND)})
    if dxy_val is not None or dxy_chg60 is not None:
        val = f"{dxy_val:.1f}" if dxy_val is not None else ""
        chg = f"{' ' if val else ''}({dxy_chg60:+.1f}% 60d)" if dxy_chg60 is not None else ""
        pills.append({"id": "dxy", "label": "DXY", "value": f"{val}{chg}", "stress": dollar_headwind})
    if credit_chg60 is not None or credit_chg20 is not None:
        cval = credit_chg60 if credit_chg60 is not None else credit_chg20
        win = "60d" if credit_chg60 is not None else "20d"
        pills.append({"id": "credit", "label": "HYG/IEF", "value": f"{cval:+.2f}% ({win})", "stress": credit_headwind})
    if wti_val is not None or brent_val is not None:
        parts = []
        if wti_val is not None:
            parts.append(f"WTI ${wti_val:.1f}")
        if brent_val is not None:
            parts.append(f"Brent ${brent_val:.1f}")
        chg = ""
        if oil_chg20 is not None and oil_chg20 >= OIL_CHG20_SHOCK:
            chg = f" ({oil_chg20:+.1f}% 20d)"
        elif oil_chg60 is not None:
            chg = f" ({oil_chg60:+.1f}% 60d)"
        pills.append({"id": "oil", "label": "Oil", "value": f"{' / '.join(parts)}{chg}", "stress": oil_headwind})

    upcoming: List[Dict[str, Any]] = []
    try:
        upcoming = (events_fn or _upcoming_macro_policy)(path) or []
    except Exception:  # noqa: BLE001
        upcoming = []

    if severity == "critical":
        detail_en = ("Multiple stress signals active (volatility, chips, and/or tightening macro). "
                     "Buy guardrails are tightened (pause dip-adds / 3x ATR spacing). "
                     "Context only, not a direction forecast: 1990-2026 tests show macro/panic timing does not beat holding, "
                     "but short-term tail swings widen sharply and catching falling knives is high-risk.")
        detail_zh = ("多项压力共振（高波动、半导体急跌及/或宏观利率美元偏紧）。"
                     "买入护栏已自动收紧（暂停补仓 / 补仓间距放宽至 3×ATR）。"
                     "仅作风险背景提示，不预测涨跌：1990–2026 回测表明宏观/恐慌择时未能跑赢持有，但短期尾部波动显著放大、逆势抄底接飞刀风险极高。")
    elif severity == "warn":
        detail_en = ("Elevated market/sector volatility or macro tightening pressure. "
                     + ("Buy guardrails tightened (pause dip-adds / 3x ATR spacing). " if tighten_buys else "")
                     + "Context only, not a direction forecast — watch position sizing and avoid chasing intraday bounces.")
        detail_zh = ("市场/板块波动升温或宏观偏紧。"
                     + ("买入护栏已自动收紧（暂停补仓 / 3×ATR 间距）。" if tighten_buys else "")
                     + "仅作风险背景提示，不作涨跌预测——注意仓位集中度，避免盘中追高或急着抄底。")
    else:
        detail_en = "Volatility, chip sector and macro conditions are within normal ranges. Standard risk rules apply."
        detail_zh = "波动率、半导体板块与宏观压力均处于常规区间，按标准风控规则执行。"

    macro_obj = {
        "level": macro_level, "flags": macro_flags, "headwind_count": len(macro_flags),
        "us10y": y10_val, "us2y": y2_val, "dxy": dxy_val,
        "y10_chg60": y10_chg60, "y2_chg60": y2_chg60, "dxy_chg60": dxy_chg60,
        "credit_chg20": credit_chg20, "credit_chg60": credit_chg60,
        "oil": wti_val, "brent": brent_val, "oil_chg20": oil_chg20, "oil_chg60": oil_chg60,
        "hist_as_of": hist.get("as_of"),
    }
    return {
        "state": key,
        "vix": None if vix is None else round(vix, 1),
        "vix_source": vix_source,
        "vix_pctile": None if pct is None else round(float(pct), 2),
        "spy_5d_pct": spy5,
        "spx_1d_pct": None if spx_1d is None else round(float(spx_1d), 2),
        "sox_1d_pct": None if sox_1d is None else round(float(sox_1d), 2),
        "soxx_5d_pct": soxx5,
        "sector_stress": sector_stress,
        "macro": macro_obj,
        "severity": severity,
        "tighten_buys": tighten_buys,
        "title": {"en": title_en, "zh": title_zh},
        "detail": {"en": detail_en, "zh": detail_zh},
        "pills": pills,
        "upcoming": upcoming,
        **STATES[key],
    }


def _latest_macro() -> Dict[str, Any]:
    from . import advisor
    return advisor._db()["daily_symbol_features"].find_one(
        {"symbol": "AAPL", "trade_date": {"$type": "string", "$regex": "^20"}}, sort=[("trade_date", -1)],
        projection={"_id": 0, "trade_date": 1, "macro_vix": 1, "macro_vix_pctile_252d": 1}) or {}
