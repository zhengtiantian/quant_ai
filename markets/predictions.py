"""Checkable predictions from the briefs, and their scores.

A brief may end with predictions, one JSON object per line:

    {"symbol": "8035.T", "date": "2026-10-23", "op": "<=", "threshold": -3, "confidence": 0.7,
     "zh": "东京电子 10/23 收盘跌超 3%", "en": "Tokyo Electron closes down more than 3% on 10/23"}

which reads: the symbol's day change on that trading date (close vs the previous close, in %)
satisfies `op threshold`. op is one of >=, <=, abs>= (a move at least this big either way),
abs<= (a move at most this big). Only symbols the brief was shown are accepted, so each can be
scored from CNBC's daily bars without guessing what a name meant.

A call must be about a session that has not opened yet when the brief is written: a US close
predicted at 14:43 Dublin, with the stock already down 10% at the open, would be half known
and would flatter the score. Such calls are dropped.

Scoring runs after the target exchange has closed on that date. A prediction whose date had no
trading (a holiday) is voided after a few days. The scorecard reports the hit rate and the Brier
score of the stated confidence (0 is perfect, 0.25 is what always saying 50% earns), overall,
by brief kind, and by confidence bucket: a well-calibrated 70% should come true about 70% of
the time. That is the point: without the score, every confident-sounding call reads as right.
"""

import json
import re
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

from . import global_markets, portfolio as pf
from .config import DB_PATH

BARS_URL = "https://ts-api.cnbc.com/harmony/app/bars/{sym}/1D/{start}000000/{end}235959/adjusted/EST5EDT.json"
OPS = (">=", "<=", "abs>=", "abs<=")
SCORE_DELAY = 30 * 60          # seconds after the close before scoring (bars settle)
VOID_AFTER_DAYS = 4
MAX_PER_BRIEF = 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY, brief_id INTEGER NOT NULL, kind TEXT NOT NULL, symbol TEXT NOT NULL,
    name TEXT, exchange TEXT NOT NULL, target_date TEXT NOT NULL, op TEXT NOT NULL, threshold REAL NOT NULL,
    confidence REAL, text_zh TEXT, text_en TEXT, status TEXT NOT NULL DEFAULT 'pending',
    actual REAL, scored_at REAL, note TEXT, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status, target_date);
CREATE INDEX IF NOT EXISTS idx_pred_brief ON predictions(brief_id);
"""


def _conn(path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------- what may be predicted

US_INDEX_SYMBOLS = {".SPX": "S&P 500", ".IXIC": "Nasdaq Composite", ".DJI": "Dow Jones", ".SOX": "Philadelphia Semiconductor",
                    ".RUT": "Russell 2000", ".VIX": "VIX"}


def universe(us_symbols: Iterable[str] = ()) -> Dict[str, Tuple[str, str]]:
    """symbol -> (name, exchange code) for everything a prediction may name. Korean stocks are
    left out (CNBC has no bars for them); their indices stand in."""
    out = {sym: (name, ex) for sym, name, ex, _ in global_markets.GLOBAL_ITEMS if not sym.startswith(global_markets.NAVER_PREFIX)}
    out.update({s: (n, "US") for s, n in US_INDEX_SYMBOLS.items()})
    out.update({s: (s, "US") for s in us_symbols if re.fullmatch(r"[A-Z][A-Z.\-]{0,9}", s or "")})
    return out


def parse_block(text: str, allowed: Dict[str, Tuple[str, str]], issued: date,
                issued_ts: Optional[float] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Valid predictions and the reasons others were dropped. issued_ts: when the brief was
    written; a call on a session that had already opened by then is dropped."""
    good, dropped = [], []
    for line in (text or "").splitlines():
        line = line.strip().strip(",")
        if not line.startswith("{"):
            continue
        try:
            p = json.loads(line)
            sym = str(p["symbol"]).strip()
            target = date.fromisoformat(str(p["date"]))
            op = str(p["op"]).strip()
            threshold = float(p["threshold"])
            conf = p.get("confidence")
            conf = None if conf is None else min(1.0, max(0.0, float(conf)))
        except (ValueError, KeyError, TypeError) as e:
            dropped.append(f"unreadable: {line[:80]} ({e})")
            continue
        if sym not in allowed:
            dropped.append(f"{sym}: not a symbol the brief was given")
        elif op not in OPS:
            dropped.append(f"{sym}: unknown op {op!r}")
        elif op.startswith("abs") and threshold < 0:
            dropped.append(f"{sym}: abs threshold must be positive")
        elif not (issued <= target <= issued + timedelta(days=10)):
            dropped.append(f"{sym}: date {target} is not within 10 days")
        elif target.weekday() >= 5:
            dropped.append(f"{sym}: {target} is a weekend")
        elif issued_ts is not None and _open_ts(allowed[sym][1], target) <= issued_ts:
            dropped.append(f"{sym}: the {target} session had already opened")
        else:
            name, ex = allowed[sym]
            good.append({"symbol": sym, "name": name, "exchange": ex, "target_date": target.isoformat(), "op": op,
                         "threshold": threshold, "confidence": conf, "text_zh": str(p.get("zh") or "")[:200],
                         "text_en": str(p.get("en") or "")[:200]})
        if len(good) >= MAX_PER_BRIEF:
            break
    return good, dropped


def save(brief_id: int, kind: str, preds: List[Dict[str, Any]], path: Optional[Path] = None) -> None:
    """Replace the brief's predictions (a rewritten brief replaces its calls)."""
    conn = _conn(path)
    try:
        conn.execute("DELETE FROM predictions WHERE brief_id = ?", (brief_id,))
        now = time.time()
        conn.executemany(
            "INSERT INTO predictions(brief_id, kind, symbol, name, exchange, target_date, op, threshold, confidence, "
            "text_zh, text_en, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(brief_id, kind, p["symbol"], p["name"], p["exchange"], p["target_date"], p["op"], p["threshold"],
              p["confidence"], p["text_zh"], p["text_en"], now) for p in preds])
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------- scoring

def _session_ts(exchange: str, target: date, which: str) -> float:
    if exchange == "US":
        tz, hours = ZoneInfo("America/New_York"), ("09:30", "16:00")
    else:
        _, _, tzname, o, c = global_markets.EXCHANGES[exchange]
        tz, hours = ZoneInfo(tzname), (o, c)
    h, m = map(int, hours[0 if which == "open" else 1].split(":"))
    return datetime(target.year, target.month, target.day, h, m, tzinfo=tz).timestamp()


def _open_ts(exchange: str, target: date) -> float:
    return _session_ts(exchange, target, "open")


def _close_ts(exchange: str, target: date) -> float:
    return _session_ts(exchange, target, "close")


def void_already_open(path: Optional[Path] = None) -> int:
    """Void pending calls on sessions that had opened before their brief was written (for calls
    stored before parse_block checked it)."""
    conn = _conn(path)
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT p.id, p.exchange, p.target_date, b.started_at FROM predictions p JOIN briefs b ON b.id = p.brief_id "
            "WHERE p.status = 'pending'")]
        bad = [r["id"] for r in rows if r["started_at"] and _open_ts(r["exchange"], date.fromisoformat(r["target_date"])) <= r["started_at"]]
        conn.executemany("UPDATE predictions SET status = 'void', note = 'session had already opened when the brief was written' "
                         "WHERE id = ?", [(i,) for i in bad])
        conn.commit()
    finally:
        conn.close()
    return len(bad)


def day_change(symbol: str, target: date) -> Optional[float]:
    """Close-to-close change on that date in %, from CNBC daily bars; None if it did not trade."""
    start = (target - timedelta(days=12)).strftime("%Y%m%d")
    end = (target + timedelta(days=1)).strftime("%Y%m%d")
    d = pf._get_json(BARS_URL.format(sym=symbol, start=start, end=end))
    bars = ((d.get("barData") or {}).get("priceBars")) or []
    bars = sorted(bars, key=lambda b: b["tradeTime"])
    key = target.strftime("%Y%m%d")
    for i, b in enumerate(bars):
        if b["tradeTime"].startswith(key) and i > 0:
            prev, close = float(bars[i - 1]["close"]), float(b["close"])
            return (close / prev - 1) * 100 if prev else None
    return None


def holds(op: str, threshold: float, actual: float) -> bool:
    return {">=": actual >= threshold, "<=": actual <= threshold,
            "abs>=": abs(actual) >= threshold, "abs<=": abs(actual) <= threshold}[op]


def score_due(path: Optional[Path] = None, now: Optional[float] = None) -> int:
    now = now or time.time()
    conn = _conn(path)
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM predictions WHERE status = 'pending'")]
    finally:
        conn.close()
    done = 0
    for p in rows:
        target = date.fromisoformat(p["target_date"])
        if now < _close_ts(p["exchange"], target) + SCORE_DELAY:
            continue
        try:
            actual = day_change(p["symbol"], target)
        except Exception:  # noqa: BLE001 - retried on the next pass
            continue
        if actual is None:
            if now > _close_ts(p["exchange"], target) + VOID_AFTER_DAYS * 86400:
                status, note = "void", "no trading on that date"
            else:
                continue
        else:
            status, note = ("hit" if holds(p["op"], p["threshold"], actual) else "miss"), None
        conn = _conn(path)
        try:
            conn.execute("UPDATE predictions SET status = ?, actual = ?, scored_at = ?, note = ? WHERE id = ?",
                         (status, None if actual is None else round(actual, 3), now, note, p["id"]))
            conn.commit()
        finally:
            conn.close()
        done += 1
    return done


# ---------------------------------------------------------------- reading

def for_brief(brief_id: int, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    conn = _conn(path)
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM predictions WHERE brief_id = ? ORDER BY target_date, id", (brief_id,))]
    finally:
        conn.close()


def _stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in rows if r["status"] in ("hit", "miss")]
    hits = sum(r["status"] == "hit" for r in scored)
    with_conf = [r for r in scored if r["confidence"] is not None]
    brier = (sum((r["confidence"] - (r["status"] == "hit")) ** 2 for r in with_conf) / len(with_conf)) if with_conf else None
    return {"total": len(rows), "scored": len(scored), "hits": hits, "pending": sum(r["status"] == "pending" for r in rows),
            "hit_rate": hits / len(scored) if scored else None, "brier": brier,
            "mean_confidence": (sum(r["confidence"] for r in with_conf) / len(with_conf)) if with_conf else None}


def scorecard(path: Optional[Path] = None, recent: int = 40) -> Dict[str, Any]:
    conn = _conn(path)
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM predictions ORDER BY target_date DESC, id DESC")]
    finally:
        conn.close()
    buckets = []
    for lo in (0.5, 0.6, 0.7, 0.8, 0.9):
        b = [r for r in rows if r["confidence"] is not None and lo <= r["confidence"] < lo + 0.1 + (1e-9 if lo == 0.9 else 0)]
        s = _stats(b)
        buckets.append({"range": f"{int(lo * 100)}-{int(lo * 100) + 10}%", "scored": s["scored"], "hit_rate": s["hit_rate"]})
    return {"overall": _stats(rows),
            "by_kind": {k: _stats([r for r in rows if r["kind"] == k]) for k in sorted({r["kind"] for r in rows})},
            "calibration": buckets, "recent": rows[:recent]}
