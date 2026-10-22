"""The one place the portfolio talks to job_agent's budget.

A push sends every line that should exist; job_agent creates, updates and removes lines to
match (budget.sync_external_entries), so repeating a push is harmless. When job_agent is not
running the push is marked pending and retry_pending() sends it again later.
"""

import json
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DB_PATH, JOB_AGENT_URL

SOURCE = "quant_ai.portfolio"
BUDGET_STOCK_ITEM = "股票"          # the user's own item name in the budget; referenced, never renamed
_FIELDS = ("key", "day", "amount", "company", "note")
_fx_cache: Dict[str, Any] = {"rate": None, "at": 0.0}


def _meta(path: Optional[Path], key: str, value: Optional[str] = None, write: bool = False) -> Optional[str]:
    conn = sqlite3.connect(str(path or DB_PATH), timeout=30)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        if write:
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
            conn.commit()
            return value
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _post(url: str, payload: Dict[str, Any], timeout: float = 10) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def push(lines: List[Dict[str, Any]], path: Optional[Path] = None) -> Dict[str, Any]:
    """{"entries": {key: entry id}} on success, {"error": message, "entries": None} otherwise."""
    payload = {"source": SOURCE, "item_name": BUDGET_STOCK_ITEM,
               "entries": [{k: ln[k] for k in _FIELDS} for ln in lines]}
    try:
        d = _post(f"{JOB_AGENT_URL}/api/budget/external/sync", payload)
        if d.get("status") != "success":
            raise RuntimeError(d.get("message") or "job_agent refused the budget lines")
    except Exception as e:  # noqa: BLE001 - kept for retry, never fails the save
        _meta(path, "budget_pending", str(time.time()), write=True)
        return {"error": f"Budget not updated yet ({type(e).__name__}: {e}); will retry", "entries": None}
    _meta(path, "budget_pending", None, write=True)
    return {"entries": d.get("entries") or {}}


def pending(path: Optional[Path] = None) -> bool:
    return bool(_meta(path, "budget_pending"))


def retry_pending(path: Optional[Path] = None) -> bool:
    """Called by the background loop; True if a pending push went through."""
    if not pending(path):
        return False
    from . import portfolio
    out = portfolio.sync_budget(path)
    return not any(v.get("pending") for v in out.values()) and not pending(path)


def usd_cny(max_age: float = 3600) -> Optional[float]:
    """CNY per USD at the budget's rate (market rate x the user's spread), cached for an hour."""
    if _fx_cache["rate"] is not None and time.time() - _fx_cache["at"] < max_age:
        return _fx_cache["rate"]
    try:
        with urllib.request.urlopen(f"{JOB_AGENT_URL}/api/budget/fx?currency=USD", timeout=3) as resp:
            rate = json.load(resp).get("rate")
        if rate:
            _fx_cache.update(rate=float(rate), at=time.time())
    except Exception:  # noqa: BLE001 - the last known rate (or None) is good enough for a display line
        pass
    return _fx_cache["rate"]
