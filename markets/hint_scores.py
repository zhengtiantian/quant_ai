"""Every hint that stops a buy, scored (PROJECT_PLAN Q.17.0, 2026-11-17).

A blocking hint is one that refuses or argues against a buy: the buy gate refusing an add
(risk_rules.gate: not stabilised after a drop, too close to the cheapest lot, no cash or room under a
cap) and the devil's advocate before an add (devils_advocate.py). Blocked means nothing was done, so
scoring it against doing nothing would always give zero. Each one is scored instead against having
bought one unit at the close of the session it fired: a fall afterwards is a loss avoided, a rise a gain
missed. The two are reported apart and neither is called profit. Advisory hints (N.6's news judgments)
are scored where they are made (quant_data research/news_linkage/score.py).

After each US session (from 16:10 New York) the buy gate is probed for every held and watched stock, the
way daily_action probes its pilot: each refusal is one row per stock, session and kind (hint_log), with
the rules that refused it. A row is scored once enough sessions have passed: the stock's return from that
session's close over 5 / 20 / 60 sessions, and the same minus SPY's. Only rows are added; nothing here
changes an action or the advice.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from .config import DB_PATH

NY = ZoneInfo("America/New_York")
HORIZONS = (5, 20, 60)                 # sessions
PROBE_AFTER = (16, 10)                 # New York time; the close is in
SCORE_EVERY = 3600                     # seconds between scoring passes
CODES = ("falling", "spacing", "size")  # risk_rules.gate's refusals

SCHEMA = """
CREATE TABLE IF NOT EXISTS hint_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,            -- add_block | devils_advocate
    symbol TEXT NOT NULL,
    day TEXT NOT NULL,             -- the New York session it fired in
    held INTEGER NOT NULL,         -- 1 held, 0 watched
    ref_price REAL,                -- the price when it fired (shown; scoring uses the session's close)
    codes TEXT,                    -- JSON list: the gate's refusals, or the advocate's trigger
    detail TEXT,                   -- JSON
    created_at TEXT NOT NULL,
    r5 REAL, r20 REAL, r60 REAL,   -- % return from the session's close
    x5 REAL, x20 REAL, x60 REAL,   -- the same minus SPY's
    scored_at TEXT,
    UNIQUE (kind, symbol, day));
"""

_lock = threading.Lock()
_state: Dict[str, Any] = {"probed_day": None, "scored_at": 0.0}


def _conn(path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def record(kind: str, symbol: str, day: str, held: bool, ref_price: Optional[float], codes: List[str],
           detail: Optional[Dict[str, Any]] = None, path: Optional[Path] = None) -> bool:
    """One row per kind, stock and session; a second call the same session is ignored. True if added."""
    conn = _conn(path)
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO hint_log (kind, symbol, day, held, ref_price, codes, detail, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (kind, symbol.upper(), day, int(bool(held)), ref_price, json.dumps(codes), json.dumps(detail or {}, default=str),
             datetime.now(NY).isoformat(timespec="seconds")))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def blocked_on(symbol: str, day: str, kind: str = "add_block", path: Optional[Path] = None) -> bool:
    conn = _conn(path)
    try:
        return conn.execute("SELECT 1 FROM hint_log WHERE kind = ? AND symbol = ? AND day = ?", (kind, symbol.upper(), day)).fetchone() is not None
    finally:
        conn.close()


def previous_day(day: str, path: Optional[Path] = None) -> Optional[str]:
    """The latest session before `day` on which the gate was probed (any stock)."""
    conn = _conn(path)
    try:
        r = conn.execute("SELECT MAX(day) AS d FROM hint_log WHERE kind = 'add_block' AND day < ?", (day,)).fetchone()
        return r["d"] if r else None
    finally:
        conn.close()


# ------------------------------------------------------------------ probing the gate

def _session_day(chart_fn: Callable[..., Dict[str, Any]], now: datetime) -> Optional[str]:
    """Today's New York date if SPY traded today (so weekends and holidays probe nothing)."""
    today = now.date().isoformat()
    candles = (chart_fn("SPY", "1m", None) or {}).get("candles") or []
    return today if any(str(c.get("time")) == today for c in candles) else None


def probe(state: Dict[str, Any], path: Optional[Path] = None,
          analyze: Optional[Callable[..., Dict[str, Any]]] = None,
          stats_fn: Optional[Callable[..., Dict[str, Any]]] = None,
          catalysts_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None) -> List[Dict[str, Any]]:
    """The buy gate for one unit of every held and watched stock at its current price, as daily_action
    asks it for its pilot. Returns one entry per stock: symbol, held, price, allowed, codes, notes."""
    from . import lot_advisor, risk_rules
    analyze = analyze or lot_advisor.analyze_symbol
    stats_fn = stats_fn or lot_advisor._bar_stats
    catalysts_fn = catalysts_fn or lot_advisor._upcoming_7d_catalysts
    ex = risk_rules.exposure(state)
    out = []
    held = {h["symbol"]: h for h in state.get("holdings") or []}
    watched = [w for w in state.get("watchlist") or [] if w.get("symbol") not in held]
    for sym, h in held.items():
        price = float((h.get("quote") or {}).get("price") or 0.0)
        if price <= 0:
            continue
        try:
            advice = analyze(sym, path=path, state=state)
        except Exception as e:  # noqa: BLE001 - one stock's missing bars do not stop the others
            out.append({"symbol": sym, "held": True, "price": price, "allowed": None, "error": f"{type(e).__name__}: {e}"})
            continue
        stats = advice.get("stats") or {}
        events = [c for c in advice.get("catalysts_7d") or [] if c.get("impact") == "high"]
        g = risk_rules.gate({"symbol": sym, "side": "buy", "shares": 1, "limit": price}, ex, held=float(h.get("quantity") or 0.0),
                            atr=stats.get("atr14"), lowest_lot=(advice.get("dip_guardrail") or {}).get("anchor_cost"),
                            stabilized=stats.get("stabilized"), event_7d=bool(events))
        out.append({"symbol": sym, "held": True, "price": price, "allowed": g["allowed"], "codes": g["blocked_by"],
                    "notes": g["notes"], "notes_zh": g.get("notes_zh") or g["notes"], "stats": stats})
    for w in watched:
        sym = w["symbol"]
        price = float((w.get("quote") or {}).get("price") or 0.0)
        if price <= 0:
            continue
        try:
            stats = stats_fn(sym, price, path)
            events = [c for c in catalysts_fn(sym, path) or [] if c.get("impact") == "high"]
        except Exception as e:  # noqa: BLE001
            out.append({"symbol": sym, "held": False, "price": price, "allowed": None, "error": f"{type(e).__name__}: {e}"})
            continue
        g = risk_rules.gate({"symbol": sym, "side": "buy", "shares": 1, "limit": price}, ex, held=0.0, atr=stats.get("atr14"),
                            lowest_lot=None, stabilized=stats.get("stabilized"), event_7d=bool(events))
        out.append({"symbol": sym, "held": False, "price": price, "allowed": g["allowed"], "codes": g["blocked_by"],
                    "notes": g["notes"], "notes_zh": g.get("notes_zh") or g["notes"], "stats": stats})
    return out


def record_probe(day: str, results: List[Dict[str, Any]], path: Optional[Path] = None) -> List[str]:
    """Each refusal as a row; returns the held stocks the gate refuses today but did not refuse on the
    previous probed session (the devil's advocate's second trigger). The first probe ever has nothing to
    compare with and returns none."""
    prev = previous_day(day, path)
    newly = []
    for r in results:
        if r.get("allowed") is not False:
            continue
        stats = r.get("stats") or {}
        detail = {"notes": r.get("notes"), "notes_zh": r.get("notes_zh"),
                  "atr14": stats.get("atr14"), "drop_1d_atr": stats.get("drop_1d_atr"), "stabilized": stats.get("stabilized")}
        added = record("add_block", r["symbol"], day, r["held"], r["price"], r.get("codes") or [], detail, path)
        if added and r["held"] and prev is not None and not blocked_on(r["symbol"], prev, path=path):
            newly.append(r["symbol"])
    return newly


# ------------------------------------------------------------------ scoring

def _closes(chart_fn: Callable[..., Dict[str, Any]], symbol: str) -> Dict[str, Any]:
    candles = [c for c in (chart_fn(symbol, "1y", None) or {}).get("candles") or [] if not c.get("partial") and c.get("close")]
    days = [str(c["time"]) for c in candles]
    return {"days": days, "close": [float(c["close"]) for c in candles], "at": {d: i for i, d in enumerate(days)}}


def score_due(path: Optional[Path] = None, chart_fn: Optional[Callable[..., Dict[str, Any]]] = None) -> int:
    """Fill every horizon that has come due; returns how many values were written."""
    from . import portfolio as pf
    chart_fn = chart_fn or pf.get_chart
    conn = _conn(path)
    try:
        rows = conn.execute("SELECT * FROM hint_log WHERE r60 IS NULL ORDER BY day").fetchall()
        if not rows:
            return 0
        spy = _closes(chart_fn, "SPY")
        cache: Dict[str, Dict[str, Any]] = {}
        written = 0
        for row in rows:
            sym = row["symbol"]
            if sym not in cache:
                try:
                    cache[sym] = _closes(chart_fn, sym)
                except Exception:  # noqa: BLE001 - a stock without bars stays unscored
                    cache[sym] = {"days": [], "close": [], "at": {}}
            s = cache[sym]
            i, j = s["at"].get(row["day"]), spy["at"].get(row["day"])
            if i is None or j is None:
                continue
            sets = {}
            for h in HORIZONS:
                if row[f"r{h}"] is not None or i + h >= len(s["close"]) or j + h >= len(spy["close"]):
                    continue
                r = (s["close"][i + h] / s["close"][i] - 1) * 100
                m = (spy["close"][j + h] / spy["close"][j] - 1) * 100
                sets[f"r{h}"], sets[f"x{h}"] = round(r, 3), round(r - m, 3)
            if sets:
                conn.execute(f"UPDATE hint_log SET {', '.join(f'{k} = ?' for k in sets)}, scored_at = ? WHERE id = ?",
                             (*sets.values(), datetime.now(NY).isoformat(timespec="seconds"), row["id"]))
                written += len(sets)
        conn.commit()
        return written
    finally:
        conn.close()


def _side(values: List[float]) -> Dict[str, Any]:
    """Against having bought: falls are losses avoided, rises gains missed (both as positive %)."""
    falls = [-v for v in values if v < 0]
    rises = [v for v in values if v > 0]
    return {"n": len(values),
            "fell": len(falls), "loss_avoided": round(sum(falls) / len(falls), 2) if falls else None,
            "rose": len(rises), "gain_missed": round(sum(rises) / len(rises), 2) if rises else None,
            "net": round(-sum(values) / len(values), 2) if values else None}


def summary(path: Optional[Path] = None) -> Dict[str, Any]:
    """Per kind and refusal: how often the stock fell or rose afterwards and by how much on average."""
    conn = _conn(path)
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM hint_log").fetchall()]
    finally:
        conn.close()
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        codes = json.loads(r["codes"] or "[]")
        keys = [r["kind"]] + ([f"{r['kind']}:{c}" for c in codes] if r["kind"] == "add_block" else [])
        for k in keys:
            groups.setdefault(k, []).append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        out[k] = {"fired": len(rs), "first_day": min(r["day"] for r in rs), "last_day": max(r["day"] for r in rs),
                  "horizons": {str(h): {"own": _side([r[f"r{h}"] for r in rs if r[f"r{h}"] is not None]),
                                        "vs_spy": _side([r[f"x{h}"] for r in rs if r[f"x{h}"] is not None])}
                               for h in HORIZONS}}
    return {"groups": out, "labels": LABELS}


# The page's fixed words in both languages, so the front end carries no Chinese of its own.
LABELS = {
    "title": {"en": "Blocked buys, scored", "zh": "被拦下的买入：事后记分"},
    "explain": {"en": "Each block is compared with having bought one unit at that session's close. A fall afterwards is a loss avoided, "
                      "a rise a gain missed; neither is profit.",
                "zh": "每次拦截都对照「当天收盘买入一单位」：之后下跌记为少亏，上涨记为错过的涨幅，两者都不是盈利。"},
    "add_block": {"en": "Buy gate (all reasons)", "zh": "买入拦截（全部原因）"},
    "add_block:falling": {"en": "Not stabilised after a drop", "zh": "大跌后未企稳"},
    "add_block:spacing": {"en": "Too close to the cheapest lot", "zh": "离最便宜的一批太近"},
    "add_block:size": {"en": "No cash or room under a cap", "zh": "现金或仓位上限不够"},
    "devils_advocate": {"en": "Devil's advocate", "zh": "反方审讯"},
    "filing_change": {"en": "Risk factors rewritten (10-K / 10-Q)", "zh": "风险因素大改（10-K / 10-Q）"},
    "fired": {"en": "fired", "zh": "次"},
    "loss_avoided": {"en": "loss avoided", "zh": "少亏"},
    "gain_missed": {"en": "gain missed", "zh": "错过的涨幅"},
    "net": {"en": "net for the block", "zh": "拦截净效果"},
    "unscored": {"en": "not scored yet", "zh": "尚未到期"},
    "sessions": {"en": "sessions", "zh": "个交易日"},
}


# ------------------------------------------------------------------ the background tick

def run_due(now: Optional[datetime] = None, path: Optional[Path] = None,
            chart_fn: Optional[Callable[..., Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Called every minute by background.py: after the close, probe the gate once per session (and hand
    newly refused held stocks to the devil's advocate); score what has come due once an hour."""
    from . import portfolio as pf
    now = now or datetime.now(NY)
    chart_fn = chart_fn or pf.get_chart
    did: Dict[str, Any] = {}
    if not _lock.acquire(blocking=False):
        return did
    try:
        if now.weekday() < 5 and (now.hour, now.minute) >= PROBE_AFTER and _state["probed_day"] != now.date().isoformat():
            day = _session_day(chart_fn, now)
            _state["probed_day"] = now.date().isoformat()
            if day:
                st = pf.get_state(path=path, cached_quotes=True)
                newly = record_probe(day, probe(st, path), path)
                did["probed"], did["newly_blocked"] = day, newly
                if newly:
                    from . import devils_advocate
                    for sym in newly:
                        devils_advocate.request(sym, "add_block")
        if time.time() - _state["scored_at"] >= SCORE_EVERY:
            _state["scored_at"] = time.time()
            did["scored"] = score_due(path, chart_fn)
    finally:
        _lock.release()
    return did
