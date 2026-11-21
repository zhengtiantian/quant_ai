"""Trading volume of the held and watched stocks (2026-11-21, the user's request): the latest full
session, the last month (21 sessions) and the last year (252 sessions), each as a total and as the
average per session. Today's volume so far comes with the quotes on the page; set against the
month's average it says whether a day is unusually busy.

Daily bars come from CNBC (Nasdaq when CNBC has none), one request per symbol, once per completed US
session or when a symbol is added. The result is kept in a JSON file and the page only reads that
file: calling get_chart for every holding inside a request locked markets.db on 2026-11-18.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import config
from . import portfolio as pf

MONTH_SESSIONS = 21
YEAR_SESSIONS = 252
BAR_DAYS = 400                      # calendar days fetched: a year of sessions with room to spare
SESSION_CLOSE_HOUR = 16             # ET: a session's bar is complete after the close
STATS_FILE = Path(os.getenv("VOLUME_STATS_FILE", str(config.DATA_DIR / "volume_stats.json")))
WORKERS = 3

_lock = threading.Lock()
_running = {"on": False}


def last_complete_session(now: Optional[datetime] = None) -> date:
    """The latest US session whose daily bar is final: today after 16:00 ET on a weekday, otherwise
    the weekday before (no holiday calendar: a holiday's missing bar only shifts the windows by one)."""
    t = (now or datetime.now(pf.US_EAST)).astimezone(pf.US_EAST)
    d = t.date()
    if t.weekday() < 5 and t.hour >= SESSION_CLOSE_HOUR:
        return d
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def tracked_symbols(path: Optional[Path] = None) -> Dict[str, List[str]]:
    """{"holdings": [...], "watchlist": [...]}: the open positions and the watched symbols not held."""
    pf.init_db(path)
    conn = pf._conn(path)
    try:
        txs = pf._transactions(conn)
        watch = [r["symbol"] for r in conn.execute("SELECT symbol FROM watchlist ORDER BY sort, symbol")]
    finally:
        conn.close()
    positions, _ = pf.compute_holdings(txs)
    held = sorted(s for s, p in positions.items() if p["quantity"] > pf.EPS)
    return {"holdings": held, "watchlist": [s for s in watch if s not in held]}


def summarize(bars: List[Dict[str, Any]], session: date) -> Optional[Dict[str, Any]]:
    """Volume figures from daily bars ({"time": "YYYY-MM-DD", "volume": n}); bars after `session`
    (today's unfinished one) are left out."""
    done = [b for b in sorted(bars, key=lambda b: b["time"]) if b["time"] <= session.isoformat() and b.get("volume")]
    if not done:
        return None

    def window(n: int) -> Dict[str, Any]:
        part = done[-n:]
        total = float(sum(b["volume"] for b in part))
        return {"total": total, "avg": round(total / len(part)), "sessions": len(part), "from": part[0]["time"]}

    last = done[-1]
    return {"last_date": last["time"], "last_volume": float(last["volume"]),
            "month": window(MONTH_SESSIONS), "year": window(YEAR_SESSIONS)}


def _bars(symbol: str, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    try:
        bars = pf._daily_cnbc(symbol, BAR_DAYS)
    except Exception:  # noqa: BLE001 - Nasdaq below
        bars = []
    if not bars and not pf.nasdaq_blocked():
        bars = pf._daily(symbol, pf._asset_class(symbol, path) or "stocks", BAR_DAYS)
    return bars


def refresh(path: Optional[Path] = None, bars_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
            now: Optional[datetime] = None, out: Optional[Path] = None) -> Dict[str, Any]:
    """Fetches the bars of every tracked symbol, writes the figures to STATS_FILE and returns them."""
    groups = tracked_symbols(path)
    session = last_complete_session(now)
    syms = groups["holdings"] + groups["watchlist"]
    fetch = bars_fn or (lambda s: _bars(s, path))

    def one(sym: str):
        try:
            return sym, summarize(fetch(sym), session), None
        except Exception as e:  # noqa: BLE001 - one symbol's failure leaves the others
            return sym, None, str(e)[:200]

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(one, syms))
    old = load(out).get("symbols", {})
    symbols, errors = {}, {}
    for sym, stats, err in results:
        if stats:
            symbols[sym] = stats
        else:
            errors[sym] = err or "no daily bars"
            if sym in old:                   # the last good figures stay rather than nothing
                symbols[sym] = {**old[sym], "stale": True}
    data = {"session": session.isoformat(), "computed_at": time.time(), "groups": groups,
            "symbols": symbols, "errors": errors,
            "windows": {"month": MONTH_SESSIONS, "year": YEAR_SESSIONS}}
    target = out or STATS_FILE
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(target)
    return data


def load(out: Optional[Path] = None) -> Dict[str, Any]:
    try:
        return json.loads((out or STATS_FILE).read_text())
    except (OSError, ValueError):
        return {}


def due(data: Dict[str, Any], groups: Dict[str, List[str]], now: Optional[datetime] = None) -> bool:
    """A new session has completed since the figures, or a tracked symbol has none yet."""
    if not data or data.get("session") != last_complete_session(now).isoformat():
        return True
    have = set(data.get("symbols", {})) | set(data.get("errors", {}))
    return any(s not in have for s in groups["holdings"] + groups["watchlist"])


def state(path: Optional[Path] = None, background: bool = True) -> Dict[str, Any]:
    """The stored figures at once; when they are out of date a refresh runs in a thread and
    "refreshing" tells the page to ask again shortly."""
    data = load()
    groups = tracked_symbols(path)
    refreshing = False
    if due(data, groups) and background:
        with _lock:
            start = not _running["on"]
            _running["on"] = True
        if start:
            def run():
                try:
                    refresh(path)
                finally:
                    with _lock:
                        _running["on"] = False
            threading.Thread(target=run, name="volume-stats", daemon=True).start()
        refreshing = True
    return {**data, "groups": groups, "refreshing": refreshing or _running["on"]}
