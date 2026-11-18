"""Risk factors rewritten (PROJECT_PLAN Q.17.5, 2026-11-18): the flag on a holding or a watched stock whose
latest 10-K / 10-Q changed its risk factors more than 80% of filings do.

quant_data's filing_changes job (research/models/filing_changes.py, weekdays after the SEC's cut-off)
compares each filing's Item 1A with the same form a year earlier and writes latest.json; this module
reads it. In the Q.17.5 test the least-similar fifth trailed its sector over the next 60 sessions by
1.9% in sample (2018-2023) and 3.2% out of sample (2024-2026), over 120 by 3.5% and 6.1%, when the risk
text grew or was rewritten (not when it shrank). It is a base rate for one kind of filing, not a forecast for this
stock, and it does not change the action.

Each flag is logged once as a blocking hint (hint_scores, kind "filing_change") on the first session the
filing could be traded on, so Q.17.0 scores it like the other blocks: a fall afterwards is a loss avoided,
a rise a gain missed.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

FILE = Path(os.getenv("FILING_CHANGES_FILE", "/Volumes/Data4T/filing_changes/latest.json"))
FLAG_DAYS = 180                  # a flag older than this is not shown (the test looked 120 sessions ahead)
NY = ZoneInfo("America/New_York")
_cache: Dict[str, Any] = {"mtime": None, "data": {}}

LABELS = {
    "flag": {"en": "Risk factors rewritten", "zh": "风险因素大改"},
    "grown": {"en": "risk text grown", "zh": "风险披露大幅增加"},
    "rewritten": {"en": "risk text rewritten", "zh": "风险披露改写"},
    "explain": {"en": "The latest {form} ({date}) changed its risk factors more than 80% of filings do (similarity {sim} against a year "
                      "earlier, cut {cut}). Filings like this trailed their sector over the next 60 sessions by 1.9% (2018-2023) "
                      "and 3.2% (2024-2026), over 120 by 3.5% and 6.1% (95 stocks). A base rate, not a forecast; the action is unchanged.",
                "zh": "最新的 {form}（{date}）风险因素章节的改动超过 80% 的申报（与一年前相似度 {sim}，阈值 {cut}）。"
                      "95 只股票里，这类申报之后 60 个交易日平均跑输所在行业 1.9%（2018–2023）和 3.2%（2024–2026），"
                      "120 个交易日分别跑输 3.5% 和 6.1%。"
                      "这是同类申报的基准率，不是对这只股票的预测；操作建议不变。"},
    "filing": {"en": "Filing", "zh": "原文"},
}


def _load() -> Dict[str, Any]:
    try:
        mt = FILE.stat().st_mtime
    except OSError:
        return {}
    if _cache["mtime"] != mt:
        try:
            _cache.update(mtime=mt, data=json.loads(FILE.read_text()))
        except ValueError:
            return _cache["data"]
    return _cache["data"]


def _accepted_ny(item: Dict[str, Any]) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(str(item.get("accepted")))
    except ValueError:
        return None
    return (t if t.tzinfo else t.replace(tzinfo=ZoneInfo("UTC"))).astimezone(NY)


def get(symbol: str, now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """The symbol's flag if its latest filing is flagged and recent enough, else None."""
    it = (_load().get("items") or {}).get(symbol.upper())
    if not it or not it.get("flag"):
        return None
    acc = _accepted_ny(it)
    if acc is None or (now or datetime.now(NY)) - acc > timedelta(days=FLAG_DAYS):
        return None
    return it


def describe(it: Dict[str, Any], lang: str = "en") -> str:
    acc = _accepted_ny(it)
    return LABELS["explain"][lang].format(form=it.get("form"), date=acc.date().isoformat() if acc else it.get("filing_date"),
                                          sim=it.get("sim_1a"), cut=it.get("cut"))


def for_symbols(symbols: List[str], now: Optional[datetime] = None) -> Dict[str, Any]:
    items = {}
    for s in symbols:
        it = get(s, now)
        if it:
            acc = _accepted_ny(it)
            items[s] = {"form": it.get("form"), "date": acc.date().isoformat() if acc else it.get("filing_date"),
                        "direction": it.get("direction"), "sim_1a": it.get("sim_1a"), "cut": it.get("cut"), "url": it.get("url"),
                        "text": {"en": describe(it, "en"), "zh": describe(it, "zh")}}
    return {"items": items, "labels": LABELS, "generated_at": _load().get("generated_at")}


def _first_session(acc: datetime, days: List[str]) -> Optional[str]:
    """The first session the filing could be traded on: that day if accepted before the open, else the next."""
    d = acc.date().isoformat()
    before_open = (acc.hour, acc.minute) < (9, 30)
    for x in days:
        if x > d or (x == d and before_open):
            return x
    return None


def record_due(path: Optional[Path] = None, state: Optional[Dict[str, Any]] = None,
               chart_fn: Optional[Callable[..., Dict[str, Any]]] = None) -> List[str]:
    """Log each current flag of a held or watched stock once (hint_scores), on its first tradable session."""
    from . import hint_scores
    from . import portfolio as pf
    st = state if state is not None else pf.get_state(path=path, cached_quotes=True)
    held = {h["symbol"] for h in st.get("holdings") or []}
    followed = held | {w["symbol"] for w in st.get("watchlist") or []}
    chart_fn = chart_fn or pf.get_chart
    logged = []
    for s in sorted(followed):
        it = get(s)
        acc = _accepted_ny(it) if it else None
        if not acc:
            continue
        c = hint_scores._closes(chart_fn, s)
        day = _first_session(acc, c["days"])
        if not day:
            continue                                   # not traded yet: logged once the session has a bar
        price = c["close"][c["at"][day]]
        if hint_scores.record("filing_change", s, day, s in held, price, [it.get("direction") or "changed"],
                              {"form": it.get("form"), "accession": it.get("accession"), "sim_1a": it.get("sim_1a"),
                               "cut": it.get("cut"), "len_ratio": it.get("len_ratio")}, path):
            logged.append(s)
    return logged
