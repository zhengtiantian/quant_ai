"""The one place the portfolio talks to job_agent's budget: the USD rate is read from it.

Sale profits are no longer pushed to the budget's 股票 item (stopped 2026-10-27 at the user's
request; the line pushed before stays in the budget). _meta() is the markets.db key-value table.
"""

import json
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from .config import DB_PATH, JOB_AGENT_URL

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
