"""Holdings advisor (PROJECT_PLAN Q.5, 2026-10-25): event-driven advice on the user's own positions.

The user's priority is profit: get the losing positions back to cost and grow. When something worth
reacting to happens to a holding, a report says what to do, at what limit, how much and until when,
and what happens if it is wrong. Advice only: nothing here places an order.

This module is the code half; briefs.py runs Claude on what it builds.
  build()           the snapshot: each position with break-even, support / resistance, ATR, its own
                    move threshold, earnings and the options-implied move, a week of news, the quant
                    model's rank, and candidate limit prices computed here so the model quotes them
                    rather than inventing numbers; buy candidates from the model's LONG list; cash.
  check_triggers()  every 15 minutes, no model: a move beyond the stock's threshold, a recommended
                    limit reached, earnings tomorrow or just out, important news.
  parse_orders()    the report's ORDERS block, validated (held or candidate symbol, limit near the
                    price, at most 30 days).
  score_due()       from daily bars: was the limit reached, and at expiry did acting beat not acting.
"""

import hashlib
import json
import math
import re
import sqlite3
import statistics
import threading
import time
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from . import events, predictions, strategy
from . import portfolio as pf
from .config import DB_PATH

NEW_YORK = ZoneInfo("America/New_York")
DUBLIN = ZoneInfo("Europe/Dublin")

DUST_VALUE = 5.0               # USD; a position worth less is not followed (WDC's 0.00001 share left)
# A move counts when it exceeds twice the stock's median absolute daily move over six months. A fixed
# 3% was measured first (2026-10-25): INTC, SK hynix, Seagate and WDC moved more than 3% on 50-57%
# of days. At 2x median (INTC/SKHY/STX ~6%, TSM ~3%) each name triggers about once a week.
THRESHOLD_MULT = 2.0
THRESHOLD_FLOOR = 1.5          # % - a very quiet stock still needs a real move
URGENT_MULT = 1.5              # a move beyond 1.5x the threshold is urgent
NEWS_DAYS = 7
NEWS_PER_SYMBOL = 8
EARNINGS_DAYS = 10             # trading days of earnings calendar in the snapshot
TRIGGER_EVERY = 15 * 60
COOLDOWN = 2 * 3600            # per holding and kind of event
DAILY_CAP = 6                  # focused updates per Dublin day; urgent ones may exceed it
ORDER_MAX_DAYS = 30
ORDER_MAX_DEVIATION = 0.35     # a limit further than 35% from the price is not a plan, it is a typo
SCORE_DELAY = 30 * 60          # after the close, before a day's bar is used
SIDES = {"sell", "buy"}
TAGS = {"trim", "rebuy", "add", "new", "stop", "exit"}
LIVE = ("open", "filled", "waiting")   # waiting: a buy-back whose sale has not filled yet

# Headlines that should make the advisor look again. Keyword rules first; a local model only if
# these prove too coarse (PROJECT_PLAN Q.5). MAJOR ones are urgent.
HIGH_IMPACT = re.compile(
    r"\b(earnings|results|revenue|guidance|outlook|forecast|downgrad|upgrad|price target|rating|acqui|merger|"
    r"takeover|buyout|buyback|repurchase|dividend|lawsuit|sued|probe|investigat|antitrust|tariff|export (?:ban|control|curb)|"
    r"sanction|ceo|cfo|resign|layoff|job cuts|recall|halt|bankrupt|delist|stake|offering|contract|order)", re.I)
MAJOR = re.compile(
    r"(acqui|merger|takeover|buyout|profit warning|cuts? (?:its |full-year |annual )?(?:guidance|forecast|outlook)|"
    r"investigat|probe|charges|antitrust|export (?:ban|control|curb)|bankrupt|delist|(?:ceo|cfo) (?:steps down|resigns|ousted|exits))",
    re.I)

FOUR_WEEK_RULE = ("Ireland, CGT: when shares are sold at a loss and the same shares are bought back within four "
                  "weeks, the loss can only be set against a gain on those re-bought shares, not against other gains. "
                  "Trimming a losing position and re-buying it soon is affected; check with Revenue or an adviser.")

SCHEMA = """
CREATE TABLE IF NOT EXISTS advice_orders (
    id INTEGER PRIMARY KEY, brief_id INTEGER, symbol TEXT NOT NULL, side TEXT NOT NULL, tag TEXT,
    pct REAL, shares REAL, amount_usd REAL, limit_price REAL NOT NULL,
    issued_ts REAL NOT NULL, issued_price REAL, valid_until TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open', filled_date TEXT, eval_date TEXT, eval_close REAL, edge_pct REAL,
    zh TEXT, en TEXT, note TEXT, parent_id INTEGER, plan TEXT);
CREATE INDEX IF NOT EXISTS idx_advice_status ON advice_orders(status);
CREATE TABLE IF NOT EXISTS advisor_triggers (key TEXT PRIMARY KEY, at REAL NOT NULL);
"""
_lock = threading.Lock()
_threshold_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
_mongo = {"client": None}


def _conn(path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(advice_orders)")}
    for col in ("parent_id INTEGER", "plan TEXT"):          # added with the strategy engine, 2026-10-25
        if col.split()[0] not in have:
            conn.execute(f"ALTER TABLE advice_orders ADD COLUMN {col}")
    return conn


def _us_today(now: Optional[datetime] = None) -> date:
    return (now or datetime.now(NEW_YORK)).astimezone(NEW_YORK).date()


# ---------------------------------------------------------------- price statistics and levels

def price_stats(symbol: str, path: Optional[Path] = None) -> Dict[str, Any]:
    """Six months of daily bars: the stock's move threshold, ATR, recent range and trend."""
    candles = pf.get_chart(symbol, "6m", path).get("candles") or []
    done = [c for c in candles if not c.get("partial")]
    if len(done) < 25:
        raise ValueError(f"{symbol}: only {len(done)} daily bars")
    moves = [abs(done[i]["close"] / done[i - 1]["close"] - 1) * 100 for i in range(1, len(done)) if done[i - 1]["close"]]
    median_move = statistics.median(moves)
    true_ranges = [max(c["high"] - c["low"], abs(c["high"] - p["close"]), abs(c["low"] - p["close"]))
                   for p, c in zip(done[-15:-1], done[-14:])]
    last = candles[-1]["close"]
    atr = sum(true_ranges) / len(true_ranges)
    closes = [c["close"] for c in done]
    window = lambda n: done[-n:]
    return {
        "bars": len(done), "last_close": round(done[-1]["close"], 4),
        "median_abs_daily_move_pct": round(median_move, 2),
        "move_threshold_pct": round(max(THRESHOLD_FLOOR, THRESHOLD_MULT * median_move), 1),
        "days_beyond_threshold_pct": round(sum(m >= THRESHOLD_MULT * median_move for m in moves) / len(moves) * 100, 1),
        "atr14": round(atr, 4), "atr14_pct": round(atr / last * 100, 2) if last else None,
        "high_20d": max(c["high"] for c in window(20)), "low_20d": min(c["low"] for c in window(20)),
        "high_60d": max(c["high"] for c in window(60)), "low_60d": min(c["low"] for c in window(60)),
        "sma20": round(sum(closes[-20:]) / 20, 4), "sma50": round(sum(closes[-50:]) / 50, 4) if len(closes) >= 50 else None,
        "return_1m_pct": round((closes[-1] / closes[-22] - 1) * 100, 1) if len(closes) > 22 else None,
        "return_3m_pct": round((closes[-1] / closes[-64] - 1) * 100, 1) if len(closes) > 64 else None,
    }


def threshold(symbol: str, path: Optional[Path] = None) -> Optional[float]:
    """The stock's move threshold in %, recomputed once a US day."""
    key = (symbol, _us_today().isoformat())
    with _lock:
        hit = _threshold_cache.get(key)
    if hit is None:
        try:
            hit = price_stats(symbol, path)
        except Exception:  # noqa: BLE001 - no history, no move trigger for it today
            hit = {}
        with _lock:
            _threshold_cache[key] = hit
    return hit.get("move_threshold_pct")


def levels(price: float, avg_cost: Optional[float], stats: Dict[str, Any], implied_move_pct: Optional[float]) -> Dict[str, Any]:
    """Candidate limit prices. Around earnings the options-implied move sets the band; otherwise twice
    the 14-day ATR, about two days of typical range. The model picks among these and the 20-day range."""
    if implied_move_pct:
        basis, m = "options-implied move to the earnings expiry", float(implied_move_pct)
    else:
        basis, m = "2 x ATR(14)", 2 * float(stats.get("atr14_pct") or 0)
    r = lambda x: round(x, 2)
    return {
        "basis": basis, "band_pct": round(m, 2),
        "sell_limits": [r(price * (1 + m / 200)), r(price * (1 + m / 100))],
        "buy_limits": [r(price * (1 - m / 200)), r(price * (1 - m / 100))],
        "resistance_20d": stats.get("high_20d"), "support_20d": stats.get("low_20d"),
        "resistance_60d": stats.get("high_60d"), "support_60d": stats.get("low_60d"),
        "break_even": r(avg_cost) if avg_cost else None,
        "to_break_even_pct": round((avg_cost / price - 1) * 100, 1) if avg_cost and price else None,
    }


# ---------------------------------------------------------------- news and the quant model (quant_data)

def _db():
    if _mongo["client"] is None:
        import pymongo
        from .pipeline_health import _mongo_uri
        uri = _mongo_uri()
        if not uri:
            raise RuntimeError("HOST_LOCAL_MONGO_URI not found in quant_data/.env")
        _mongo["client"] = pymongo.MongoClient(uri, serverSelectionTimeoutMS=4000)
    return _mongo["client"]["quant_data"]


def impact(title: str) -> Optional[str]:
    if MAJOR.search(title or ""):
        return "major"
    if HIGH_IMPACT.search(title or ""):
        return "high"
    return None


def news(symbols: List[str], days: int = NEWS_DAYS, per_symbol: int = NEWS_PER_SYMBOL) -> Dict[str, List[Dict[str, Any]]]:
    """Recent articles per symbol, newest first, one per headline. Read from the raw store, which the
    collectors fill every 30 minutes, not only from the matched one, which the nightly company match
    fills: a 14:00 headline must reach the 15-minute check the same afternoon. Sentiment and event
    labels are added where the nightly enrichment has already run."""
    since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y%m%d")
    q = {"symbol": {"$in": symbols}, "date": {"$gte": since}}
    proj = {"_id": 0, "symbol": 1, "date": 1, "title": 1, "source": 1, "url": 1}
    labelled = {d["url"]: d for d in _db()["news_articles_company_matched_v2"].find(
        q, {**proj, "llm_sentiment_final": 1, "llm_event_type_a": 1}).sort("date", -1).limit(60 * len(symbols)) if d.get("url")}
    raw = list(_db()["news_articles"].find(q, proj).sort("date", -1).limit(60 * len(symbols)))
    rows = sorted([labelled.get(r.get("url"), r) for r in raw] + [d for u, d in labelled.items() if u not in {r.get("url") for r in raw}],
                  key=lambda r: str(r.get("date") or ""), reverse=True)
    out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in symbols}
    seen = set()
    for r in rows:
        title = (r.get("title") or "").strip()
        key = (r["symbol"], title.lower())
        if not title or key in seen or len(out[r["symbol"]]) >= per_symbol:
            continue
        seen.add(key)
        src = r.get("source")
        if isinstance(src, dict):
            src = src.get("name") or src.get("platform")
        d = str(r.get("date") or "")
        out[r["symbol"]].append({
            "time_utc": f"{d[:4]}-{d[4:6]}-{d[6:8]}" + (f" {d[8:10]}:{d[10:12]}" if len(d) >= 12 else ""),
            "title": title, "source": src, "url": r.get("url"), "impact": impact(title),
            "sentiment": r.get("llm_sentiment_final"), "event": r.get("llm_event_type_a")})
    return out


def quant_signals(held: List[str], n_candidates: int = 10) -> Dict[str, Any]:
    """The quant model's latest ranks: the holdings' own, and its LONG list as buy candidates."""
    coll = _db()["daily_signals"]
    last = coll.find_one(sort=[("trade_date", -1)], projection={"trade_date": 1})
    if not last:
        return {"trade_date": None, "holdings": {}, "candidates": []}
    rows = list(coll.find({"trade_date": last["trade_date"]}, {"_id": 0}).sort("signal_rank", 1))
    pick = lambda r: {"rank": r.get("signal_rank"), "signal": r.get("signal_type"),
                      "score": round(r["composite_score"], 3) if r.get("composite_score") is not None else None}
    return {
        "trade_date": last["trade_date"],
        "regime": next((r.get("regime_label") for r in rows if r.get("regime_label")), None),
        "universe_note": "The model ranks a 100-stock US universe daily; a holding outside it has no rank.",
        "holdings": {r["symbol"]: pick(r) for r in rows if r["symbol"] in held},
        "candidates": [{"symbol": r["symbol"], **pick(r),
                        "avg_sentiment_5d": r.get("avg_sentiment_5d"), "earnings_beat_signal": r.get("earnings_beat_signal")}
                       for r in rows if r.get("signal_type") == "LONG" and r["symbol"] not in held][:n_candidates],
    }


# ---------------------------------------------------------------- the snapshot

def reactions(symbol: str) -> List[Dict[str, Any]]:
    """Past earnings reactions with their daily bars (quant_data research/features/earnings_reactions.py)."""
    return list(_db()["earnings_reactions"].find({"symbol": symbol}, {"_id": 0, "computed_at": 0}).sort("report_time", 1))


def strategies(symbol: str, holding: Dict[str, Any], price: float, stats: Dict[str, Any], earnings: Optional[Dict[str, Any]],
               cash: float, path: Optional[Path] = None) -> Dict[str, Any]:
    """The holding's candidate plans, each tested on its own history (strategy.py)."""
    try:
        past = reactions(symbol)
    except Exception:  # noqa: BLE001 - quant_data down: plans are tested on the rolling windows only
        past = []
    candles = pf.get_chart(symbol, "1y", path).get("candles") or []
    return strategy.evaluate_holding(symbol, holding["quantity"], price, holding.get("avg_cost"), cash, candles,
                                     stats.get("atr14_pct"), earnings, past, _us_today())


def followed(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [h for h in state.get("holdings") or [] if (h.get("market_value") or 0) >= DUST_VALUE]


def build(path: Optional[Path] = None, focus: Optional[str] = None) -> Dict[str, Any]:
    st = pf.get_state(path=path)
    held = followed(st)
    syms = [h["symbol"] for h in held]
    prices = {h["symbol"]: (h.get("quote") or {}).get("price") for h in held}
    gaps: List[str] = []
    try:
        cal = events.upcoming(syms, days=EARNINGS_DAYS, path=path, prices=prices)
        earnings = {e["symbol"]: e for e in cal.get("earnings") or []}
    except Exception as e:  # noqa: BLE001
        earnings, gaps = {}, gaps + [f"earnings calendar: {e}"]
    try:
        sig = quant_signals(syms)
    except Exception as e:  # noqa: BLE001
        sig, gaps = {"holdings": {}, "candidates": []}, gaps + [f"quant signals: {e}"]
    try:
        nw = news(syms)
    except Exception as e:  # noqa: BLE001
        nw, gaps = {}, gaps + [f"news: {e}"]
    positions = []
    for h in held:
        s = h["symbol"]
        q = h.get("quote") or {}
        price = q.get("price")
        try:
            stats = price_stats(s, path)
        except Exception as e:  # noqa: BLE001
            stats, gaps = {}, gaps + [f"{s} price history: {e}"]
        e = earnings.get(s)
        implied = ((e or {}).get("implied") or {}).get("implied_move_pct")
        try:
            strat = strategies(s, h, price, stats, e, pf.get_cash(path), path) if price and stats else None
        except Exception as ex:  # noqa: BLE001
            strat, gaps = None, gaps + [f"{s} strategies: {type(ex).__name__}: {ex}"]
        positions.append({
            "symbol": s, "name": q.get("name"), "quantity": round(h["quantity"], 6), "avg_cost": round(h["avg_cost"], 4),
            "price": price, "us_session": q.get("session"), "regular_close": q.get("regular_price") or q.get("prev_close"),
            "day_change_pct": round(h["day_change_pct"], 2) if h.get("day_change_pct") is not None else None,
            "market_value": round(h.get("market_value") or 0, 2), "cost": round(h.get("cost") or 0, 2),
            "unrealized": round(h.get("unrealized") or 0, 2),
            "unrealized_pct": round(h["unrealized_pct"], 2) if h.get("unrealized_pct") is not None else None,
            "weight_pct": round(h["weight"], 1) if h.get("weight") is not None else None,
            "price_stats": stats, "earnings": e,
            "levels": levels(price, h.get("avg_cost"), stats, implied) if price and stats else None,
            "news_7d": nw.get(s, []), "quant_model": sig.get("holdings", {}).get(s), "strategies": strat,
        })
    cand = sig.get("candidates") or []
    if cand:
        try:
            qs = pf.get_quotes([c["symbol"] for c in cand], path=path)
            for c in cand:
                c["price"] = (qs.get(c["symbol"]) or {}).get("price")
        except Exception:  # noqa: BLE001
            pass
    t = st.get("totals") or {}
    weights = sorted(((p["weight_pct"] or 0, p["symbol"]) for p in positions), reverse=True)
    return {
        "cash_usd": pf.get_cash(path),
        "totals": {k: round(t[k], 2) for k in ("market_value", "cost", "unrealized", "unrealized_pct", "day_pnl") if t.get(k) is not None},
        "concentration": {"largest": weights[0][1] if weights else None, "largest_weight_pct": weights[0][0] if weights else None,
                          "positions": len(positions)},
        "positions": positions,
        "buy_candidates": cand, "quant_trade_date": sig.get("trade_date"), "market_regime": sig.get("regime"),
        "focus": focus, "rules": {"irish_cgt_four_week": FOUR_WEEK_RULE}, "data_gaps": gaps,
    }


# ---------------------------------------------------------------- triggers (every 15 minutes, no model)

def _fired(conn: sqlite3.Connection, key: str) -> Optional[float]:
    row = conn.execute("SELECT at FROM advisor_triggers WHERE key = ?", (key,)).fetchone()
    return row["at"] if row else None


def _cooling(conn: sqlite3.Connection, symbol: str, kind: str, now: float) -> bool:
    at = _fired(conn, f"cool|{symbol}|{kind}")
    return at is not None and now - at < COOLDOWN


def _mark(conn: sqlite3.Connection, keys: List[str], now: float) -> None:
    conn.executemany("INSERT OR REPLACE INTO advisor_triggers(key, at) VALUES (?, ?)", [(k, now) for k in keys])


def _next_trading_day(d: date) -> date:
    n = d + timedelta(days=1)
    while n.weekday() >= 5:
        n += timedelta(days=1)
    return n


def _prev_trading_day(d: date) -> date:
    p = d - timedelta(days=1)
    while p.weekday() >= 5:
        p -= timedelta(days=1)
    return p


def check_triggers(path: Optional[Path] = None, now: Optional[float] = None,
                   news_fn=None, earnings_fn=None) -> List[Dict[str, Any]]:
    """Holdings that need a fresh look now, each with its reasons and whether it is urgent. Every
    reason fires once (and each kind at most every COOLDOWN per holding), so a stock that stays down
    8% does not produce a report every 15 minutes."""
    now = now or time.time()
    if datetime.fromtimestamp(now, DUBLIN).weekday() >= 5:
        return []
    st = pf.get_state(path=path)
    held = followed(st)
    if not held:
        return []
    syms = [h["symbol"] for h in held]
    us_today = _us_today(datetime.fromtimestamp(now, NEW_YORK))
    found: Dict[str, Dict[str, Any]] = {}

    def add(symbol: str, kind: str, key: str, text: str, urgent: bool) -> None:
        f = found.setdefault(symbol, {"symbol": symbol, "reasons": [], "urgent": False, "keys": []})
        f["reasons"].append({"kind": kind, "text": text, "urgent": urgent})
        f["urgent"] = f["urgent"] or urgent
        f["keys"] += [key, f"cool|{symbol}|{kind}"]

    conn = _conn(path)
    try:
        open_orders = [dict(r) for r in conn.execute("SELECT * FROM advice_orders WHERE status IN ('open', 'filled')")]
        for h in held:
            s, q = h["symbol"], h.get("quote") or {}
            price, session = q.get("price"), q.get("session")
            # 1. a move beyond the stock's own threshold
            move = q.get("session_change_pct") if session in ("pre", "post") else h.get("day_change_pct")
            thr = threshold(s, path)
            if session in ("pre", "regular", "post") and move is not None and thr and abs(move) >= thr:
                key = f"{s}|move|{us_today}|{session}"
                if not _fired(conn, key) and not _cooling(conn, s, "move", now):
                    add(s, "move", key, f"{s} {move:+.1f}% in the {session} session (its threshold is {thr}%)",
                        abs(move) >= URGENT_MULT * thr)
            # 2. a recommended limit reached
            for o in open_orders:
                if o["symbol"] != s or o["status"] != "open" or not price or str(o.get("plan") or "").startswith("pulse:"):
                    continue
                hit = (o["side"] == "buy" and price <= o["limit_price"]) or (o["side"] == "sell" and o["tag"] == "stop" and price <= o["limit_price"]) \
                    or (o["side"] == "sell" and o["tag"] != "stop" and price >= o["limit_price"])
                key = f"order|{o['id']}"
                if hit and not _fired(conn, key):
                    add(s, "limit", key, f"{s} at {price} reached the advised {o['side']} limit {o['limit_price']} "
                                         f"(order {o['id']}, valid until {o['valid_until']})", True)
        # 3. earnings: tomorrow (prepare) or just out (react)
        try:
            # From the previous trading day: "just out" covers last night's after-the-close reports.
            cal = (earnings_fn or (lambda ss: events.upcoming(ss, days=3, path=path, start=_prev_trading_day(us_today))
                                   .get("earnings") or []))(syms)
        except Exception:  # noqa: BLE001
            cal = []
        sessions = {h["symbol"]: (h.get("quote") or {}).get("session") for h in held}
        for e in cal:
            s, when, d = e.get("symbol"), (e.get("when") or "").lower(), e.get("date")
            if s not in syms or not d:
                continue
            ed = date.fromisoformat(str(d)[:10])
            if ed == _next_trading_day(us_today):
                key = f"{s}|earnings_prep|{ed}"
                if not _fired(conn, key):
                    add(s, "earnings_prep", key, f"{s} reports {ed} {e.get('when') or ''}: prepare the plan", False)
            out = ((ed == us_today and "after" in when and sessions.get(s) in ("post", "closed"))
                   or (ed == us_today and "after" not in when and sessions.get(s) in ("pre", "regular"))
                   or (ed == _prev_trading_day(us_today) and "after" in when and sessions.get(s) in ("pre", "regular")))
            key = f"{s}|earnings_out|{ed}"
            if out and not _fired(conn, key):
                add(s, "earnings_out", key, f"{s} has reported ({ed}, {e.get('when') or 'time n/a'}): read it and adjust", True)
        # 4. important news since the last check
        try:
            nw = (news_fn or (lambda ss: news(ss, days=1)))(syms)
        except Exception:  # noqa: BLE001 - quant_data down: the other triggers still work
            nw = {}
        for s, items in nw.items():
            hot = [n for n in items if n.get("impact")]
            fresh = [n for n in hot if not _fired(conn, "news|" + hashlib.sha1(f"{s}|{n['title'].lower()}".encode()).hexdigest()[:16])]
            if not fresh:
                continue
            news_keys = ["news|" + hashlib.sha1(f"{s}|{n['title'].lower()}".encode()).hexdigest()[:16] for n in fresh]
            if _cooling(conn, s, "news", now):
                _mark(conn, news_keys, now)      # seen while cooling: not raised again later
                continue
            urgent = any(n["impact"] == "major" for n in fresh)
            add(s, "news", news_keys[0], "; ".join(f"[{n['source']}] {n['title']}" for n in fresh[:3]), urgent)
            found[s]["keys"] += news_keys[1:]
        # daily cap on focused updates (urgent ones still go through)
        today_dub = datetime.fromtimestamp(now, DUBLIN).date().isoformat()
        written = conn.execute("SELECT COUNT(*) FROM briefs WHERE kind = 'advisor' AND topic != '' AND session_date = ?",
                               (today_dub,)).fetchone()[0] if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'briefs'").fetchone() else 0
        out_list = []
        for f in sorted(found.values(), key=lambda f: not f["urgent"]):
            if not f["urgent"] and written + len(out_list) >= DAILY_CAP:
                continue
            _mark(conn, f["keys"], now)
            out_list.append({k: f[k] for k in ("symbol", "reasons", "urgent")})
        conn.commit()
        return out_list
    finally:
        conn.close()


# ---------------------------------------------------------------- orders

ORDER_RULES = """Orders for the holdings come only from the plans the platform computed and tested: each position's
strategies.candidates, in whole shares, each replayed on that stock's own past (around its past earnings
when it reports within the horizon, else on rolling windows of the last year: "history", gain against
holding in % of the position) and simulated for this time ("outlook": how likely each limit fills, and the
regret rate, sold and never bought back below the sale). "hold" is the baseline; its history is what holding
alone returned. Only candidates with "eligible": true may be chosen: they beat holding on the stock's own
history and never sell below the average cost without a buy-back. Choose at most one plan per holding, or
none, and say in the report why, quoting the history (n, mean, win rate) and the outlook fill rates; when
no plan is eligible, say so and what the history shows, rather than inventing trades. Then the PLANS block,
one JSON object per line (empty when every holding is best left alone):
{{"symbol": "<a holding>", "plan": "<candidate id>", "zh": "<the plan and why, in Chinese>", "en": "<the same in English>"}}
The platform turns a plan into its limit orders (buy-backs become active once their sale fills) valid
until strategies.valid_until.

New buys go in the ORDERS block, one JSON object per line, only for symbols that are not holdings:
{{"symbol": "<a buy candidate>", "side": "buy", "tag": "new", "amount_usd": <dollars to spend>, "limit": <price>,
 "valid_until": "YYYY-MM-DD", "zh": "<the order and why, in Chinese>", "en": "<the same in English>"}}
They are rounded down to whole shares; keep the limit within 35% of the price and the validity within
{max_days} days. Cash is snapshot.advisor.cash_usd: a buy larger than the cash must name the plan that pays
for it."""


def parse_orders(text: str, prices: Dict[str, Optional[float]], session_date: date,
                 held: Optional[set] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Keep the orders that make sense; return (orders, reasons for each dropped one)."""
    orders, dropped = [], []
    for line in (text or "").splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
            sym = str(o.get("symbol", "")).upper().strip()
            side = str(o.get("side", "")).lower()
            tag = str(o.get("tag", "")).lower() or ("trim" if side == "sell" else "add")
            limit = float(o["limit"])
            until = date.fromisoformat(str(o["valid_until"])[:10])
        except (ValueError, KeyError, TypeError) as e:
            dropped.append(f"unreadable: {line[:80]} ({e})")
            continue
        price = prices.get(sym)
        why = None
        if sym in (held or ()):
            why = "a holding: its orders come from the PLANS block"
        elif sym not in prices:
            why = "not a holding or a buy candidate"
        elif side not in SIDES or tag not in TAGS:
            why = f"side/tag {side}/{tag}"
        elif not price or limit <= 0 or abs(limit / price - 1) > ORDER_MAX_DEVIATION:
            why = f"limit {limit} too far from the price {price}"
        elif not session_date <= until <= session_date + timedelta(days=ORDER_MAX_DAYS):
            why = f"valid until {until}"
        if why:
            dropped.append(f"{sym}: {why}")
            continue
        num = lambda k: float(o[k]) if o.get(k) not in (None, "") else None
        shares = math.floor(num("shares") or ((num("amount_usd") or 0) / limit))       # brokers fill whole shares
        if shares < 1:
            dropped.append(f"{sym}: less than one share at {limit}")
            continue
        orders.append({"symbol": sym, "side": side, "tag": tag, "pct": None, "shares": shares,
                       "amount_usd": round(shares * limit, 2), "limit_price": round(limit, 4), "issued_price": price,
                       "valid_until": until.isoformat(), "zh": str(o.get("zh", ""))[:500], "en": str(o.get("en", ""))[:500]})
    return orders, dropped


def parse_plans(text: str, positions: List[Dict[str, Any]], session_date: date) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The chosen plans as orders: each leg one order, a buy-back pointing at its sale ("parent": index
    in the returned list). Only eligible candidates of a holding are accepted, one plan per holding."""
    by_sym = {p["symbol"]: p for p in positions}
    orders, dropped, done = [], [], set()
    for line in (text or "").splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
            sym, pid = str(o.get("symbol", "")).upper().strip(), str(o.get("plan", "")).strip()
        except (ValueError, TypeError) as e:
            dropped.append(f"unreadable: {line[:80]} ({e})")
            continue
        st = (by_sym.get(sym) or {}).get("strategies") or {}
        plan = next((c for c in st.get("candidates") or [] if c["id"] == pid), None)
        if sym in done:
            why = "a second plan for the same holding"
        elif not plan:
            why = f"no candidate {pid!r}"
        elif not plan.get("eligible"):
            why = f"{pid} is not eligible: {plan.get('why')}"
        else:
            why = None
        if why:
            dropped.append(f"{sym}: {why}")
            continue
        done.add(sym)
        if not plan["legs"]:
            continue
        until = min(date.fromisoformat(st["valid_until"]), session_date + timedelta(days=ORDER_MAX_DAYS))
        h = plan["history"]
        note = (f"{pid}: {plan['label']}. History, {h['n']} windows: {h['mean_pct']:+.2f}% of the position against holding, "
                f"beat it in {h['win_rate']:.0%}.")
        base = len(orders)
        for leg in plan["legs"]:
            orders.append({"symbol": sym, "side": leg["side"], "tag": "rebuy" if leg["after"] is not None else
                           ("trim" if leg["side"] == "sell" else "add"), "pct": None, "shares": leg["shares"],
                           "amount_usd": round(leg["shares"] * leg["limit"], 2), "limit_price": leg["limit"],
                           "issued_price": st.get("price"), "valid_until": until.isoformat(),
                           "zh": str(o.get("zh", ""))[:500], "en": str(o.get("en", ""))[:500], "note": note,
                           "plan": pid, "parent": base + leg["after"] if leg["after"] is not None else None})
    return orders, dropped


def save_orders(brief_id: int, orders: List[Dict[str, Any]], issued_ts: float, path: Optional[Path] = None) -> None:
    """A buy-back is saved as 'waiting' with parent_id set to its sale's row."""
    conn = _conn(path)
    try:
        conn.execute("DELETE FROM advice_orders WHERE brief_id = ? AND status IN ('open', 'waiting')", (brief_id,))
        ids: List[int] = []
        for o in orders:
            parent = ids[o["parent"]] if o.get("parent") is not None else None
            cur = conn.execute(
                "INSERT INTO advice_orders(brief_id, symbol, side, tag, pct, shares, amount_usd, limit_price, issued_ts, "
                "issued_price, valid_until, zh, en, note, parent_id, plan, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (brief_id, o["symbol"], o["side"], o["tag"], o["pct"], o["shares"], o["amount_usd"], o["limit_price"], issued_ts,
                 o["issued_price"], o["valid_until"], o["zh"], o["en"], o.get("note"), parent, o.get("plan"),
                 "waiting" if parent else "open"))
            ids.append(cur.lastrowid)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------- scoring (daily bars)

def _bars(symbol: str, start: date, end: date) -> List[Dict[str, Any]]:
    d = pf._get_json(predictions.BARS_URL.format(sym=symbol, start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d")))
    bars = ((d.get("barData") or {}).get("priceBars")) or []
    return sorted(({"date": datetime.strptime(b["tradeTime"][:8], "%Y%m%d").date(), "high": float(b["high"]),
                    "low": float(b["low"]), "close": float(b["close"])} for b in bars), key=lambda b: b["date"])


def filled(order: Dict[str, Any], bar: Dict[str, Any]) -> bool:
    lim = order["limit_price"]
    if order["side"] == "buy" or order["tag"] == "stop":
        return bar["low"] <= lim
    return bar["high"] >= lim


def edge_pct(order: Dict[str, Any], close: float) -> float:
    """Acting against not acting, at the evaluation close: a sale gains what the price later lost
    below the limit; a purchase gains what the price later rose above it."""
    lim = order["limit_price"]
    return (lim / close - 1) * 100 if order["side"] == "sell" else (close / lim - 1) * 100


def score_due(path: Optional[Path] = None, now: Optional[float] = None, bars_fn=None) -> int:
    now = now or time.time()
    conn = _conn(path)
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM advice_orders WHERE status IN (?, ?, ?) ORDER BY id", LIVE)]
    finally:
        conn.close()
    changed = 0
    cache: Dict[str, List[Dict[str, Any]]] = {}
    # Earnings-pulse orders live for minutes in extended hours: daily bars cannot score them (pulse.py).
    rows = [o for o in rows if not str(o.get("plan") or "").startswith("pulse:")]
    for o in rows:
        issued = datetime.fromtimestamp(o["issued_ts"], NEW_YORK)
        until = date.fromisoformat(o["valid_until"])
        first = issued.date() if issued.time() < dtime(16, 0) else issued.date() + timedelta(days=1)
        if o.get("parent_id"):
            # A buy-back counts from the session after its sale filled; if the sale never fills, it is void.
            conn = _conn(path)
            try:
                parent = conn.execute("SELECT status, filled_date FROM advice_orders WHERE id = ?", (o["parent_id"],)).fetchone()
                if o["status"] == "waiting":
                    new = "open" if parent and parent["filled_date"] else "void" if not parent or parent["status"] not in LIVE else None
                    if new:
                        conn.execute("UPDATE advice_orders SET status = ? WHERE id = ?", (new, o["id"]))
                        conn.commit()
                        changed += 1
                        o["status"] = new
            finally:
                conn.close()
            if o["status"] in ("waiting", "void"):
                continue
            first = max(first, date.fromisoformat(parent["filled_date"]) + timedelta(days=1))
        last_usable = _us_today(datetime.fromtimestamp(now - SCORE_DELAY, NEW_YORK))
        if datetime.fromtimestamp(now - SCORE_DELAY, NEW_YORK).time() < dtime(16, 0):
            last_usable -= timedelta(days=1)
        if first > until and last_usable >= until:
            conn = _conn(path)                     # a buy-back whose sale filled on the last valid day
            try:
                conn.execute("UPDATE advice_orders SET status = 'expired' WHERE id = ?", (o["id"],))
                conn.commit()
            finally:
                conn.close()
            changed += 1
            continue
        if last_usable < first:
            continue
        key = o["symbol"]
        try:
            if key not in cache:
                cache[key] = (bars_fn or _bars)(key, first - timedelta(days=3), max(last_usable, first))
            bars = [b for b in cache[key] if first <= b["date"] <= min(until, last_usable)]
        except Exception:  # noqa: BLE001 - tried again on the next pass
            continue
        upd: Dict[str, Any] = {}
        if o["status"] == "open":
            hit = next((b for b in bars if filled(o, b)), None)
            if hit:
                upd.update(status="filled", filled_date=hit["date"].isoformat())
        if last_usable >= until and bars:
            final = bars[-1]
            if (upd.get("status") or o["status"]) == "filled":
                upd.update(status="scored", eval_date=final["date"].isoformat(), eval_close=final["close"],
                           edge_pct=round(edge_pct(o, final["close"]), 2))
            else:
                upd.update(status="expired", eval_date=final["date"].isoformat(), eval_close=final["close"])
        if upd:
            conn = _conn(path)
            try:
                conn.execute(f"UPDATE advice_orders SET {', '.join(f'{k} = ?' for k in upd)} WHERE id = ?", (*upd.values(), o["id"]))
                conn.commit()
            finally:
                conn.close()
            changed += 1
    return changed


# ---------------------------------------------------------------- reading

def orders_for_brief(brief_id: int, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    conn = _conn(path)
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM advice_orders WHERE brief_id = ? ORDER BY id", (brief_id,))]
    finally:
        conn.close()


def active_orders(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Open, waiting and filled orders still inside their validity, newest first."""
    conn = _conn(path)
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM advice_orders WHERE status IN (?, ?, ?) ORDER BY issued_ts DESC, id", LIVE)]
    finally:
        conn.close()


def scorecard(path: Optional[Path] = None) -> Dict[str, Any]:
    conn = _conn(path)
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM advice_orders ORDER BY issued_ts DESC")]
    finally:
        conn.close()
    scored = [r for r in rows if r["status"] == "scored"]
    finished = [r for r in rows if r["status"] in ("scored", "expired")]
    edges = [r["edge_pct"] for r in scored if r["edge_pct"] is not None]
    return {
        "orders": len(rows), "open": sum(r["status"] == "open" for r in rows), "filled_live": sum(r["status"] == "filled" for r in rows),
        "finished": len(finished), "fill_rate": round(len(scored) / len(finished), 3) if finished else None,
        "mean_edge_pct": round(sum(edges) / len(edges), 2) if edges else None,
        "win_rate": round(sum(e > 0 for e in edges) / len(edges), 3) if edges else None,
        "recent": rows[:40],
    }
