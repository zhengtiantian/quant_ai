"""The latest news-storyline judgment per followed stock (PROJECT_PLAN N.6 step 5, 2026-11-15).

quant_data's news_linkage DAG (research/news_linkage/live.py) groups each followed stock's news into
storylines every 30 minutes; when one bursts, opens on several sites at once, or reports a leadership
event, Claude (through the CLI) judges the combined effect on the stock and the judgment is stored in
quant_data.news_linkage_judgments. The holdings page shows the newest valid one of each stock from the last
SHOW_HOURS under the Action pill. It is a hint: daily_action and the advisor do not read it, and how often
it is right is still being measured (N.6 step 7).

Claude writes the summary in both languages and the chain / watch lines in English; the fixed words of the
hint come from here in both languages, so the page carries no Chinese of its own.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

COLLECTION = "news_linkage_judgments"
SHOW_HOURS = 72
HORIZONS = (1, 5, 20)          # trading days; scored in quant_data research/news_linkage/score.py
MIN_CALLS = 20                 # fewer scored 5-day calls than this: the record is shown as too thin to judge
CACHE_SECONDS = 60
_cache: Dict[str, Any] = {"at": 0.0, "out": None}
_mongo: Dict[str, Any] = {"client": None}

DIRECTION = {-2: ("Strongly bearish", "强利空"), -1: ("Bearish", "利空"), 0: ("No material effect", "无实质影响"),
             1: ("Bullish", "利好"), 2: ("Strongly bullish", "强利好")}
CONFIDENCE = {"low": ("low confidence", "把握低"), "medium": ("medium confidence", "把握中"), "high": ("high confidence", "把握高")}
NOVELTY = {"new": ("not yet priced in", "尚未反映"), "partly_priced_in": ("partly priced in", "部分已反映"),
           "priced_in": ("priced in", "已基本反映")}
REASON = {"burst": ("a burst of reports", "报道突然增多"), "new": ("a new story on several sites", "多家网站同时出现的新故事"),
          "leadership": ("a leadership event", "管理层事件")}
LABELS = {
    "news": {"en": "News", "zh": "新闻"},
    "chain": {"en": "Why", "zh": "推理"},
    "watch": {"en": "Watch for", "zh": "接下来关注"},
    "others": {"en": "Other companies", "zh": "牵连的公司"},
    "sources": {"en": "Sources", "zh": "引用的报道"},
    "storylines": {"en": "Storylines", "zh": "故事线"},
    "judged": {"en": "Judged", "zh": "判断于"},
    "after": {"en": "Since the call, vs SPY", "zh": "判断之后（相对 SPY）"},
    "days": {"en": "d", "zh": "天"},
    "hint": {"en": "A hint from the news alone: the action above is unchanged, and its accuracy is still being measured.",
             "zh": "仅是新闻提示：不改变上面的操作建议，准确率仍在统计中。"},
}


def _db():
    if _mongo["client"] is None:
        import pymongo
        from .pipeline_health import _mongo_uri
        uri = _mongo_uri()
        if not uri:
            raise RuntimeError("HOST_LOCAL_MONGO_URI not found in quant_data/.env")
        _mongo["client"] = pymongo.MongoClient(uri, serverSelectionTimeoutMS=4000, tz_aware=True)
    return _mongo["client"]["quant_data"]


def _pair(table: Dict, key, fallback: str = "") -> Dict[str, str]:
    en, zh = table.get(key, (fallback or str(key), fallback or str(key)))
    return {"en": en, "zh": zh}


def _reason(raw: str) -> Dict[str, str]:
    base, _, more = (raw or "").partition("+")
    en, zh = REASON.get(base, (base, base))
    if more == "escalation":
        en, zh = en + ", escalating", zh + "，事态升级"
    return {"en": en, "zh": zh}


def hint(doc: Dict[str, Any]) -> Dict[str, Any]:
    """One stored judgment as the page shows it."""
    a = doc.get("answer") or {}
    d = int(a.get("direction") or 0)
    direction, conf, nov = _pair(DIRECTION, d), _pair(CONFIDENCE, a.get("confidence")), _pair(NOVELTY, a.get("novelty"))
    at = doc["at"] if isinstance(doc["at"], datetime) else datetime.fromisoformat(str(doc["at"]))
    return {
        "symbol": doc["symbol"],
        "at": at.astimezone(timezone.utc).isoformat(timespec="minutes"),
        "direction": d,
        "label": {k: f"{LABELS['news'][k]}{': ' if k == 'en' else '：'}{direction[k]}" for k in ("en", "zh")},
        # how much is priced in means nothing when the news has no effect
        "detail": {k: conf[k] + (f" · {nov[k]}" if d else "") for k in ("en", "zh")},
        "summary": {"en": a.get("summary_en") or "", "zh": a.get("summary_zh") or a.get("summary_en") or ""},
        "chain": a.get("chain") or [],
        "watch": a.get("watch") or [],
        "others": [{"symbol": o.get("symbol"), "direction": o.get("direction"), "label": _pair(DIRECTION, o.get("direction")),
                    "why": o.get("why") or ""} for o in a.get("other_companies") or []],
        "sources": [{"title": c.get("title") or "", "url": c.get("url") or ""} for c in doc.get("cited") or []],
        "storylines": [{"head": s.get("head") or "", "reason": _reason(s.get("reason")), "n24": s.get("n24")}
                       for s in doc.get("storylines") or []],
        # what the stock did against SPY since the call, from the first price after it (step 7)
        "after": [{"h": h, "excess": sc[f"h{h}"]["excess"], "right": sc[f"h{h}"]["right"]}
                  for sc in [doc.get("score") or {}] for h in HORIZONS if f"h{h}" in sc],
    }


def track(docs) -> Dict[str, Any]:
    """Every scored live call: per horizon, directional calls, how many were right, the mean excess return in
    the called direction; and a line for the page."""
    out: Dict[str, Any] = {}
    for h in HORIZONS:
        k = f"h{h}"
        calls = [((d.get("answer") or {}).get("direction"), d["score"][k]) for d in docs
                 if (d.get("score") or {}).get(k) and (d.get("answer") or {}).get("direction")]
        out[k] = {"calls": len(calls), "right": sum(bool(s.get("right")) for _, s in calls),
                  "mean_called_excess": round(sum((1 if dr > 0 else -1) * s["excess"] for dr, s in calls) / len(calls), 2) if calls else None}
    parts_en, parts_zh = [], []
    for h in HORIZONS:
        t = out[f"h{h}"]
        if t["calls"]:
            parts_en.append(f"{h} day{'s' if h > 1 else ''}: {t['right']} of {t['calls']} right ({t['mean_called_excess']:+.1f}% vs SPY in the called direction)")
            parts_zh.append(f"{h} 天：{t['calls']} 次对 {t['right']} 次（按判断方向平均跑赢 SPY {t['mean_called_excess']:+.1f}%）")
    few = bool(parts_en) and out[f"h{HORIZONS[1]}"]["calls"] < MIN_CALLS
    out["text"] = {
        "en": "Live record: " + ("; ".join(parts_en) if parts_en else "no call scored yet") + (". Too few calls to judge yet." if few else "."),
        "zh": "实盘战绩：" + ("；".join(parts_zh) if parts_zh else "还没有可打分的判断") + ("。次数太少，还不能下结论。" if few else "。"),
    }
    return out


def latest(now: Optional[datetime] = None, db=None) -> Dict[str, Any]:
    """{hints: {symbol: hint}, labels, show_hours}: the newest valid judgment of each stock."""
    now = now or datetime.now(timezone.utc)
    coll = (db if db is not None else _db())[COLLECTION]
    hints: Dict[str, Any] = {}
    for doc in coll.find({"status": "ok", "at": {"$gte": now - timedelta(hours=SHOW_HOURS)}},
                         {"storylines.members": 0}).sort("at", -1):
        if doc["symbol"] not in hints:
            hints[doc["symbol"]] = hint(doc)
    scored = list(coll.find({"status": "ok", "score": {"$exists": True}}, {"answer.direction": 1, "score": 1}))
    return {"hints": hints, "labels": LABELS, "show_hours": SHOW_HOURS, "track": track(scored)}


def get(force: bool = False) -> Dict[str, Any]:
    if not force and _cache["out"] is not None and time.time() - _cache["at"] < CACHE_SECONDS:
        return _cache["out"]
    out = latest()
    _cache.update(at=time.time(), out=out)
    return out
