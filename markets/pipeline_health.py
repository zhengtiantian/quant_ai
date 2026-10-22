"""Freshness watchdog for the quant_data pipeline (PROJECT_PLAN Q.2 / J.4).

The daily pipeline runs under the host Airflow scheduler, whose metadata DB and data live in
Docker. In October 2026 Docker was off for a week: the scheduler crash-looped, nothing was
collected, and nobody noticed. Airflow's failure callback (quant_data/_host_common.py) only
fires when a task runs and fails; this catches the other case, when nothing runs at all.

Every CHECK_EVERY seconds quant_ai (always on) reads the newest date of each stage's output in
MongoDB and compares it with what a healthy pipeline would have by now. A problem is notified
once a day; the latest result is served at /api/markets/pipeline/health and carried in the
portfolio state for the desktop widget.
"""

import os
import subprocess
import json
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import budget_link

CHECK_EVERY = 30 * 60
QUANT_DATA_ENV = Path(os.getenv("QUANT_DATA_ENV", "/Users/xiz/Quant_trade/quant_data/.env"))

# stage -> (collection, how to read its newest date, allowed lag, what produces it)
#   lag in weekdays for daily outputs, hours for the intraday news
STAGES = [
    ("intraday news", "news_articles", "objectid", {"hours": 6}, "quant_intraday_news (every 30 min)"),
    ("company match", "news_articles_company_matched_v2", "objectid", {"days": 2}, "backfill_1 company_match (02:00 UTC)"),
    ("prices", "stock_prices_history", "price", {"weekdays": 2}, "price_history_backfill / daily_price"),
    ("macro", "macro_indicators", "date", {"weekdays": 2}, "daily_signal_pipeline macro"),
    ("features", "daily_symbol_features", "trade_date", {"weekdays": 2}, "daily_signal_pipeline daily_features"),
    ("signals", "daily_signals", "trade_date", {"weekdays": 2}, "daily_signal_pipeline score_signals"),
]

_state: Dict[str, Any] = {"checked_at": None, "ok": None, "stages": [], "problems": []}
_lock = threading.Lock()


def _mongo_uri() -> Optional[str]:
    if os.getenv("PIPELINE_MONGO_URI"):
        return os.environ["PIPELINE_MONGO_URI"]
    try:
        for line in QUANT_DATA_ENV.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() == "HOST_LOCAL_MONGO_URI":
                return v.strip()
    except OSError:
        return None
    return None


def weekdays_between(a: date, b: date) -> int:
    """Weekdays after a, up to and including b (0 when b <= a)."""
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def _newest(db, coll: str, how: str) -> Optional[datetime]:
    c = db[coll]
    if how == "objectid":
        doc = c.find_one(sort=[("_id", -1)], projection={"_id": 1})
        return doc["_id"].generation_time if doc else None
    if how == "price":            # indexed on (symbol, timestamp): ask one liquid symbol
        doc = c.find_one({"symbol": "AAPL"}, sort=[("timestamp", -1)], projection={"timestamp": 1})
        v = doc and doc.get("timestamp")
    else:
        doc = c.find_one(sort=[(how, -1)], projection={how: 1})
        v = doc and doc.get(how)
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v / 1000 if v > 1e11 else v, timezone.utc)
    return datetime.fromisoformat(str(v)[:10]).replace(tzinfo=timezone.utc)


def judge(newest: Optional[datetime], lag: Dict[str, int], now: datetime) -> Optional[str]:
    """None when fresh enough, else why not."""
    if newest is None:
        return "no data"
    if "hours" in lag:
        age = (now - newest).total_seconds() / 3600
        return None if age <= lag["hours"] else f"newest is {age:.0f} h old"
    if "days" in lag:
        age = (now - newest).total_seconds() / 86400
        return None if age <= lag["days"] else f"newest is {age:.1f} days old"
    behind = weekdays_between(newest.date(), now.date())
    return None if behind <= lag["weekdays"] else f"{behind} weekdays behind"


def check(now: Optional[datetime] = None, client=None) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    stages: List[Dict[str, Any]] = []
    problems: List[str] = []
    try:
        if client is None:
            import pymongo
            uri = _mongo_uri()
            if not uri:
                raise RuntimeError(f"HOST_LOCAL_MONGO_URI not found in {QUANT_DATA_ENV}")
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=4000)
        db = client["quant_data"]
        for name, coll, how, lag, producer in STAGES:
            newest = _newest(db, coll, how)
            why = judge(newest, lag, now)
            stages.append({"stage": name, "newest": newest.isoformat() if newest else None, "ok": why is None,
                           "problem": why, "producer": producer})
            if why:
                problems.append(f"{name}: {why} ({producer})")
    except Exception as e:  # noqa: BLE001 - unreachable Mongo is itself the finding
        problems = [f"MongoDB unreachable ({type(e).__name__}): is Docker running with core.sh, and Data4T mounted?"]
    result = {"checked_at": now.isoformat(), "ok": not problems, "stages": stages, "problems": problems}
    with _lock:
        _state.update(result)
    return result


def latest() -> Dict[str, Any]:
    with _lock:
        return dict(_state)


def _notify(title: str, text: str) -> None:
    script = f"display notification {json.dumps(text[:200])} with title {json.dumps(title)}"
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001
        pass


def check_and_alert(path=None) -> Dict[str, Any]:
    """Check, and notify at most once a day while anything is wrong."""
    result = check()
    if not result["ok"]:
        today = date.today().isoformat()
        if budget_link._meta(path, "pipeline_alerted") != today:
            budget_link._meta(path, "pipeline_alerted", today, write=True)
            _notify("Quant pipeline is stale", "; ".join(result["problems"])[:200])
    return result


_last = {"at": 0.0}


def run_due(path=None) -> None:
    """Called every minute by markets/background.py."""
    if time.time() - _last["at"] >= CHECK_EVERY:
        _last["at"] = time.time()
        check_and_alert(path)
