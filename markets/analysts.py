"""Wall Street analysts on each held and watched stock: every firm's rating and price target (2026-11-11).

Source: Yahoo Finance through yfinance (the user's choice, 2026-11-11), the same library quant_data's
collectors use. Per stock it gives every rating or price-target change by firm since about 2012
(upgradeDowngradeHistory) and the consensus (price targets, buy / hold / sell counts). Yahoo answers
plain HTTP clients with 429; yfinance gets through by presenting a browser's TLS fingerprint. When
Yahoo blocks it anyway, this module does not work around that: the page keeps the stored data and
shows when it was fetched. A paid feed (Finnhub premium, FMP) is the replacement if that lasts.

Storage (markets.db): one row per change in `analyst_actions`, keyed (symbol, firm, graded_at), so a
refetch never duplicates; `analyst_summary` holds the consensus per stock. The page shows each firm's
newest row, so a new change replaces what is shown for that firm and the history stays underneath.
A headline from quant_data's news store that reports the change is attached to the firm's newest row.

Refresh (own thread, started by background.py): the stocks held or watched, every 30 minutes on US
weekdays from 10:30 to 15:00 Dublin (rating changes are mostly published before the US open), hourly
from 07:00 to 23:00 on weekdays, every 6 hours otherwise; a stock just added is fetched on its first
look. A rate-limit answer pauses the whole pass for an hour.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from .config import DB_PATH

DUBLIN = ZoneInfo("Europe/Dublin")
PAUSE = 1.5                      # seconds between two stocks in a pass
RATE_LIMIT_BACKOFF = 3600        # after a 429, no request for an hour
RECENT_DAYS = 7                  # a change this recent is flagged on the row
STALE_DAYS = 365                 # a firm whose newest change is older is shown greyed, after the others
NEWS_MAX_AGE_DAYS = 730          # headlines are looked up for changes up to two years old
NEWS_RETRY_DAYS = 4              # a change this recent with no headline yet is looked up again

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyst_actions (
    symbol TEXT NOT NULL, firm TEXT NOT NULL, graded_at TEXT NOT NULL,
    action TEXT, from_grade TEXT, to_grade TEXT, pt_action TEXT, pt_current REAL, pt_prior REAL,
    news TEXT, news_checked_at REAL, first_seen REAL NOT NULL,
    PRIMARY KEY (symbol, firm, graded_at));
CREATE TABLE IF NOT EXISTS analyst_summary (
    symbol TEXT PRIMARY KEY, data TEXT, fetched_at REAL, error TEXT, error_at REAL);
"""

_lock = threading.Lock()                 # one pass (or one first-look fetch) at a time
_started = threading.Event()
_paused_until = {"t": 0.0}


def _conn(path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# ------------------------------------------------------------------ ratings in plain words

_BUY = {"buy", "strong buy", "outperform", "overweight", "accumulate", "positive", "sector outperform",
        "market outperform", "add", "conviction buy", "top pick", "long-term buy", "moderate buy",
        "speculative buy", "outperformer", "buy (high risk)", "strong-buy", "industry outperform"}
_HOLD = {"hold", "neutral", "equal-weight", "equal weight", "market perform", "sector perform", "peer perform",
         "in-line", "inline", "perform", "sector weight", "mixed", "fair value", "market weight", "hold neutral",
         "sector neutral", "industry perform", "equalweight"}
_SELL = {"sell", "underperform", "underweight", "reduce", "strong sell", "negative", "sector underperform",
         "market underperform", "moderate sell", "underperformer", "industry underperform"}


def bucket(grade: Optional[str]) -> Optional[str]:
    """'Buy' / 'Hold' / 'Sell' for a firm's own wording (Overweight, Market Perform, ...), None if unknown."""
    g = " ".join((grade or "").lower().replace("_", " ").split())
    if g in _BUY:
        return "Buy"
    if g in _HOLD:
        return "Hold"
    if g in _SELL:
        return "Sell"
    return None


def direction(row: Dict[str, Any]) -> int:
    """+1 an upgrade or a raised target, -1 a downgrade or a cut target, 0 otherwise (the rating decides
    when the two disagree)."""
    a = (row.get("action") or "").lower()
    if a == "up":
        return 1
    if a == "down":
        return -1
    cur, prior = row.get("pt_current"), row.get("pt_prior")
    if cur and prior:
        return 1 if cur > prior else -1 if cur < prior else 0
    return 0


def _usd(v: Optional[float]) -> str:
    return f"${v:,.0f}" if v and v >= 100 else f"${v:,.2f}" if v else ""


def describe(row: Dict[str, Any]) -> str:
    """One English line: 'Downgraded to Neutral from Outperform; cut target to $200 from $500'."""
    a, to, frm = (row.get("action") or "").lower(), row.get("to_grade") or "", row.get("from_grade") or ""
    parts = []
    if a == "up":
        parts.append(f"Upgraded to {to}" + (f" from {frm}" if frm and frm != to else ""))
    elif a == "down":
        parts.append(f"Downgraded to {to}" + (f" from {frm}" if frm and frm != to else ""))
    elif a == "init":
        parts.append(f"Started coverage at {to}" if to else "Started coverage")
    elif a == "reit":
        parts.append(f"Reiterated {to}" if to else "Reiterated")
    elif to:
        parts.append(f"Kept {to}")
    cur, prior = row.get("pt_current"), row.get("pt_prior")
    pa = (row.get("pt_action") or "").lower()
    if cur:
        if prior and cur > prior:
            parts.append(f"raised target to {_usd(cur)} from {_usd(prior)}")
        elif prior and cur < prior:
            parts.append(f"cut target to {_usd(cur)} from {_usd(prior)}")
        elif pa == "announces" or not prior:
            parts.append(f"target {_usd(cur)}")
        else:
            parts.append(f"kept target {_usd(cur)}")
    text = "; ".join(parts)
    return text[:1].upper() + text[1:] if text else ""


def firm_key(firm: str) -> str:
    """Yahoo spells some firms two ways ('JP Morgan' / 'JPMorgan'): one key per firm."""
    return re.sub(r"[^a-z0-9]", "", (firm or "").lower().replace("&", "and"))


# ------------------------------------------------------------------ fetching

class RateLimited(Exception):
    pass


def fetch(symbol: str) -> Dict[str, Any]:
    """{'actions': [...], 'summary': {...}} from Yahoo. Raises RateLimited on a 429."""
    import yfinance as yf
    try:
        from yfinance.exceptions import YFRateLimitError
    except ImportError:                                       # older yfinance
        YFRateLimitError = ()  # type: ignore[assignment]
    t = yf.Ticker(symbol)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # yfinance's own pandas deprecation warnings
            ud = t.upgrades_downgrades
            targets = t.analyst_price_targets or {}
            rec = t.recommendations
    except YFRateLimitError as e:  # type: ignore[misc]
        raise RateLimited(str(e)) from e
    actions = []
    if ud is not None and len(ud):
        for ts, r in ud.iterrows():
            firm = str(r.get("Firm") or "").strip()
            if not firm:
                continue
            num = lambda v: float(v) if v is not None and v == v and float(v) > 0 else None  # noqa: E731
            actions.append({
                "firm": firm, "graded_at": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "action": str(r.get("Action") or "") or None,
                "from_grade": str(r.get("FromGrade") or "") or None, "to_grade": str(r.get("ToGrade") or "") or None,
                "pt_action": str(r.get("priceTargetAction") or "") or None,
                "pt_current": num(r.get("currentPriceTarget")), "pt_prior": num(r.get("priorPriceTarget")),
            })
    counts = {}
    if rec is not None and len(rec):
        now = rec[rec["period"] == "0m"] if "period" in rec else rec.head(1)
        if len(now):
            r = now.iloc[0]
            counts = {"buy": int(r.get("strongBuy", 0) + r.get("buy", 0)), "hold": int(r.get("hold", 0)),
                      "sell": int(r.get("sell", 0) + r.get("strongSell", 0))}
    summary = {
        "target_mean": targets.get("mean"), "target_median": targets.get("median"),
        "target_high": targets.get("high"), "target_low": targets.get("low"), "price": targets.get("current"),
        **counts, "analysts": sum(counts.values()) if counts else None,
    }
    return {"actions": actions, "summary": summary}


def store(symbol: str, data: Dict[str, Any], path: Optional[Path] = None) -> int:
    """Write a fetch; returns how many changes were not stored before."""
    now = time.time()
    conn = _conn(path)
    try:
        before = conn.execute("SELECT COUNT(*) FROM analyst_actions WHERE symbol = ?", (symbol,)).fetchone()[0]
        conn.executemany(
            "INSERT INTO analyst_actions(symbol, firm, graded_at, action, from_grade, to_grade, pt_action, pt_current, "
            "pt_prior, first_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(symbol, firm, graded_at) DO UPDATE SET action = excluded.action, from_grade = excluded.from_grade, "
            "to_grade = excluded.to_grade, pt_action = excluded.pt_action, pt_current = excluded.pt_current, "
            "pt_prior = excluded.pt_prior",
            [(symbol, a["firm"], a["graded_at"], a["action"], a["from_grade"], a["to_grade"], a["pt_action"],
              a["pt_current"], a["pt_prior"], now) for a in data["actions"]])
        conn.execute("INSERT INTO analyst_summary(symbol, data, fetched_at, error, error_at) VALUES (?, ?, ?, NULL, NULL) "
                     "ON CONFLICT(symbol) DO UPDATE SET data = excluded.data, fetched_at = excluded.fetched_at, "
                     "error = NULL, error_at = NULL", (symbol, json.dumps(data["summary"]), now))
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM analyst_actions WHERE symbol = ?", (symbol,)).fetchone()[0]
    finally:
        conn.close()
    return after - before


def _store_error(symbol: str, message: str, path: Optional[Path] = None) -> None:
    conn = _conn(path)
    try:
        conn.execute("INSERT INTO analyst_summary(symbol, error, error_at) VALUES (?, ?, ?) "
                     "ON CONFLICT(symbol) DO UPDATE SET error = excluded.error, error_at = excluded.error_at",
                     (symbol, message[:300], time.time()))
        conn.commit()
    finally:
        conn.close()


# ------------------------------------------------------------------ the headline behind a change

_RATING_WORDS = re.compile(
    r"target|upgrade|downgrade|rating|initiat|reiterat|coverage|overweight|underweight|outperform|underperform|"
    r"equal.weight|neutral|\bbuy\b|\bsell\b|\bhold\b|\bPT\b", re.I)
_ALIASES = {
    "jpmorgan": ["jp morgan", "jpmorgan", "j p morgan"], "bankofamerica": ["bank of america", "bofa"],
    "bofasecurities": ["bank of america", "bofa"], "goldmansachs": ["goldman"], "citigroup": ["citi", "citigroup"],
    "evercoreisigroup": ["evercore"], "rbccapital": ["rbc"], "bmocapital": ["bmo"], "tdcowen": ["td cowen", "cowen"],
    "keybanc": ["keybanc"], "wellsfargo": ["wells fargo"], "deutschebank": ["deutsche bank"],
}
_GENERIC = r"\b(capital|markets?|securities|research|group|partners|financial|advisors|equities|equity|llc|inc|co|& co)\b"


def _firm_pattern(firm: str) -> Optional[re.Pattern]:
    names = _ALIASES.get(firm_key(firm))
    if not names:
        core = " ".join(re.sub(_GENERIC, " ", firm.lower().replace(".", "")).split())
        names = [core] if len(core) >= 3 else []
    if not names:
        return None
    alts = [r"\s*".join(re.escape(w) for w in n.split()) for n in names]
    return re.compile(r"\b(" + "|".join(alts) + r")\b", re.I)


def _normal_title(t: str) -> str:
    return " ".join(re.sub(r"[.’']", "", t or "").replace("&", " & ").split())


def _news_for(symbol: str, firm: str, graded_at: str) -> Optional[Dict[str, Any]]:
    """The headline in quant_data's news that reports this firm's change, from a day before to four days after."""
    pat = _firm_pattern(firm)
    if pat is None:
        return None
    from bson import ObjectId
    from .advisor import _db
    db = _db()
    d = datetime.fromisoformat(graded_at)
    lo, hi = (d - timedelta(days=1)).strftime("%Y%m%d"), (d + timedelta(days=4)).strftime("%Y%m%d")
    proj = {"_id": 0, "title": 1, "url": 1, "date": 1, "source": 1}
    # the matched store has a (symbol, date) index; the raw one is read by insertion time for the last days,
    # which holds what the nightly match has not reached yet
    docs = list(db.news_articles_company_matched_v2.find({"symbol": symbol, "date": {"$gte": lo, "$lt": hi}}, proj).limit(800))
    if datetime.utcnow() - d < timedelta(days=NEWS_RETRY_DAYS + 1):
        since = ObjectId.from_datetime(d - timedelta(days=1))
        docs += list(db.news_articles.find({"_id": {"$gte": since}, "symbol": symbol}, proj).limit(3000))
    best, best_gap = None, None
    for doc in docs:
        title = _normal_title(doc.get("title") or "")
        if not (pat.search(title) and _RATING_WORDS.search(title)):
            continue
        try:
            when = datetime.strptime(str(doc.get("date") or "")[:8], "%Y%m%d")
        except ValueError:
            continue
        gap = abs((when.date() - d.date()).days)
        if best_gap is None or gap < best_gap:
            best, best_gap = doc, gap
    if not best:
        return None
    src = best.get("source") or {}
    source = src if isinstance(src, str) else src.get("name") or src.get("platform")      # some collectors store a string
    return {"title": best.get("title"), "url": best.get("url"), "date": str(best.get("date") or "")[:8], "source": source}


def attach_news(symbol: str, path: Optional[Path] = None, limit: int = 80) -> int:
    """Look up headlines for each firm's newest change that has none yet; returns how many were found."""
    rows = latest_by_firm(symbol, path)
    now = time.time()
    found = 0
    todo = []
    for r in rows[:limit]:
        age = (datetime.utcnow() - datetime.fromisoformat(r["graded_at"])).days
        if r["news"] or age > NEWS_MAX_AGE_DAYS:
            continue
        if r["news_checked_at"] and age > NEWS_RETRY_DAYS:
            continue
        todo.append(r)
    if not todo:
        return 0
    updates = []
    for r in todo:
        try:
            n = _news_for(symbol, r["firm"], r["graded_at"])
        except Exception as e:  # noqa: BLE001 - the news store is optional for this page
            print(f"[analysts] news lookup failed for {symbol}: {type(e).__name__}: {e}")
            return found
        found += n is not None
        updates.append((json.dumps(n) if n else None, now, symbol, r["firm"], r["graded_at"]))
    conn = _conn(path)
    try:
        conn.executemany("UPDATE analyst_actions SET news = ?, news_checked_at = ? WHERE symbol = ? AND firm = ? AND graded_at = ?",
                         updates)
        conn.commit()
    finally:
        conn.close()
    return found


# ------------------------------------------------------------------ what the page reads

def _row(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    d["news"] = json.loads(d["news"]) if d.get("news") else None
    return d


def latest_by_firm(symbol: str, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Each firm's newest change, newest first."""
    conn = _conn(path)
    try:
        rows = conn.execute("SELECT * FROM analyst_actions WHERE symbol = ? ORDER BY graded_at DESC", (symbol,)).fetchall()
    finally:
        conn.close()
    seen, out = set(), []
    for r in rows:
        k = firm_key(r["firm"])
        if k in seen:
            continue
        seen.add(k)
        out.append(_row(r))
    return out


def detail(symbol: str, path: Optional[Path] = None) -> Dict[str, Any]:
    """Everything the expanded row shows: the consensus and one line per firm (current coverage first)."""
    symbol = symbol.upper()
    now = datetime.utcnow()
    firms = []
    for r in latest_by_firm(symbol, path):
        age = (now - datetime.fromisoformat(r["graded_at"])).days
        firms.append({
            "firm": r["firm"], "graded_at": r["graded_at"], "age_days": age, "stale": age > STALE_DAYS,
            "rating": r["to_grade"], "rating_bucket": bucket(r["to_grade"]), "from_rating": r["from_grade"],
            "action": r["action"], "target": r["pt_current"], "prior_target": r["pt_prior"],
            "direction": direction(r), "text": describe(r), "news": r["news"],
        })
    firms.sort(key=lambda f: (f["stale"], f["age_days"]))
    s = summaries([symbol], path).get(symbol) or {}
    return {"symbol": symbol, "summary": s, "firms": firms}


def summaries(symbols: List[str], path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """The column on the holdings and watchlist rows: consensus, counts, and the newest change if recent."""
    if not symbols:
        return {}
    conn = _conn(path)
    try:
        qs = ",".join("?" * len(symbols))
        summ = {r["symbol"]: dict(r) for r in conn.execute(f"SELECT * FROM analyst_summary WHERE symbol IN ({qs})", symbols)}
        since = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
        recent: Dict[str, List[Dict[str, Any]]] = {}
        for r in conn.execute(f"SELECT * FROM analyst_actions WHERE symbol IN ({qs}) AND graded_at >= ? ORDER BY graded_at DESC",
                              [*symbols, since]):
            recent.setdefault(r["symbol"], []).append(dict(r))
    finally:
        conn.close()
    out = {}
    now = datetime.utcnow()
    for s in symbols:
        row = summ.get(s)
        if not row:
            continue
        d = json.loads(row["data"]) if row.get("data") else {}
        rs = recent.get(s, [])
        dirs = [direction(r) for r in rs]
        newest = rs[0] if rs else None
        latest = None
        if newest and (now - datetime.fromisoformat(newest["graded_at"])).days < RECENT_DAYS:
            latest = {"firm": newest["firm"], "graded_at": newest["graded_at"], "direction": direction(newest),
                      "text": describe(newest)}
        out[s] = {**d, "fetched_at": row.get("fetched_at"), "error": row.get("error"), "error_at": row.get("error_at"),
                  "up_30d": dirs.count(1), "down_30d": dirs.count(-1), "latest": latest}
    return out


# ------------------------------------------------------------------ schedule

def interval(now: Optional[datetime] = None) -> int:
    t = (now or datetime.now(DUBLIN)).astimezone(DUBLIN)
    if t.weekday() >= 5:
        return 6 * 3600
    m = t.hour * 60 + t.minute
    if 10 * 60 + 30 <= m < 15 * 60:
        return 1800
    if 7 * 60 <= m < 23 * 60:
        return 3600
    return 6 * 3600


def followed_symbols(path: Optional[Path] = None) -> List[str]:
    from . import portfolio as pf
    conn = pf._conn(path)
    try:
        txs = pf._transactions(conn)
        watch = [r["symbol"] for r in conn.execute("SELECT symbol FROM watchlist ORDER BY sort, symbol")]
    finally:
        conn.close()
    positions, _ = pf.compute_holdings(txs)
    held = sorted(s for s, p in positions.items() if p["quantity"] > pf.EPS)
    return held + [w for w in watch if w not in held]


def refresh(symbol: str, path: Optional[Path] = None) -> Dict[str, Any]:
    """Fetch one stock now and attach headlines; the outcome as a dict (never raises)."""
    symbol = symbol.upper()
    if time.time() < _paused_until["t"]:
        return {"symbol": symbol, "status": "paused"}
    try:
        new = store(symbol, fetch(symbol), path)
    except RateLimited as e:
        _paused_until["t"] = time.time() + RATE_LIMIT_BACKOFF
        _store_error(symbol, f"rate limited: {e}", path)
        return {"symbol": symbol, "status": "rate_limited"}
    except Exception as e:  # noqa: BLE001 - one stock failing must not stop the pass
        _store_error(symbol, f"{type(e).__name__}: {e}", path)
        return {"symbol": symbol, "status": "error", "error": str(e)}
    found = attach_news(symbol, path)
    return {"symbol": symbol, "status": "ok", "new_changes": new, "headlines": found}


def run_due(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Fetch every followed stock whose data is older than the interval for this hour."""
    if not _lock.acquire(blocking=False):
        return []
    try:
        symbols = followed_symbols(path)
        if not symbols:
            return []
        conn = _conn(path)
        try:
            last = {r["symbol"]: max(r["fetched_at"] or 0, r["error_at"] or 0)
                    for r in conn.execute("SELECT symbol, fetched_at, error_at FROM analyst_summary")}
        finally:
            conn.close()
        every, now, done = interval(), time.time(), []
        for s in symbols:
            if now - last.get(s, 0) < every:
                continue
            r = refresh(s, path)
            done.append(r)
            if r["status"] in ("rate_limited", "paused"):
                break
            time.sleep(PAUSE)
        return done
    finally:
        _lock.release()


def ensure(symbol: str, path: Optional[Path] = None) -> None:
    """A stock opened before its first fetch (just added to the watchlist): fetch it now."""
    conn = _conn(path)
    try:
        have = conn.execute("SELECT 1 FROM analyst_summary WHERE symbol = ?", (symbol.upper(),)).fetchone()
    finally:
        conn.close()
    if not have:
        with _lock:
            refresh(symbol, path)


def _loop() -> None:
    while True:
        try:
            done = run_due()
            if done:
                ok = sum(r["status"] == "ok" for r in done)
                new = sum(r.get("new_changes", 0) for r in done)
                print(f"[analysts] refreshed {ok}/{len(done)} stocks, {new} new rating changes")
        except Exception as e:  # noqa: BLE001
            print(f"[analysts] pass failed: {type(e).__name__}: {e}")
        time.sleep(60)


def start() -> None:
    if _started.is_set():
        return
    _started.set()
    threading.Thread(target=_loop, name="markets-analysts", daemon=True).start()
