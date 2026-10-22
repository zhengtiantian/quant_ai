"""Upcoming events and sudden moves, for the forward-looking briefs.

- Earnings calendar and economic calendar (api.nasdaq.com, no key), cached per day.
- Implied move around an earnings date from the at-the-money straddle of the first option
  expiry on or after it (Nasdaq option chain): roughly the move the options market prices in,
  size only, never direction.
- Bellwethers: the companies whose results move whole sectors (Lam Research's guidance on
  2026-10-21 took Tokyo Electron -7%, SK hynix -5% and ASML -3% the next day). A move of
  FLASH_THRESHOLD % or more in pre-market or after hours triggers a flash brief.
"""

import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from . import portfolio as pf

NEW_YORK = ZoneInfo("America/New_York")
EARNINGS_URL = "https://api.nasdaq.com/api/calendar/earnings?date={d}"
ECON_URL = "https://api.nasdaq.com/api/calendar/economicevents?date={d}"
OPTIONS_URL = ("https://api.nasdaq.com/api/quote/{sym}/option-chain?assetclass=stocks&limit=200&fromdate={start}"
               "&todate={end}&excode=oprac&callput=callput&money=at&type=all")
CALENDAR_TTL = 6 * 3600

# symbol -> (name, group): the group tells the brief which chain a move travels along.
BELLWETHERS = {
    "LRCX": ("Lam Research", "chip equipment"), "AMAT": ("Applied Materials", "chip equipment"),
    "KLAC": ("KLA", "chip equipment"), "ASML": ("ASML (ADR)", "chip equipment"),
    "TSM": ("TSMC (ADR)", "foundry"), "NVDA": ("Nvidia", "AI chips"), "AMD": ("AMD", "AI chips"),
    "AVGO": ("Broadcom", "AI chips"), "MU": ("Micron", "memory"), "INTC": ("Intel", "chips"),
    "SMCI": ("Super Micro", "AI servers"), "MSFT": ("Microsoft", "AI capex / cloud"),
    "GOOGL": ("Alphabet", "AI capex / cloud"), "AMZN": ("Amazon", "AI capex / cloud"),
    "META": ("Meta", "AI capex"), "AAPL": ("Apple", "consumer hardware"), "TSLA": ("Tesla", "EV / AI capex"),
}
FLASH_THRESHOLD = 5.0          # % move vs the regular close, in pre-market or after hours
FLASH_DAILY_CAP = 4            # flash briefs per US date, to bound the quota spent on a wild night
FLASH_CHECK_EVERY = 300        # seconds between checks during pre-market / after hours

# Economic releases worth a line (the calendar lists ~60 a day, most of them minor).
ECON_COUNTRIES = {"United States", "Euro Zone", "China", "Japan", "United Kingdom", "Germany", "South Korea", "Taiwan"}
ECON_KEYWORDS = re.compile(
    r"\b(CPI|PPI|PCE|GDP|Nonfarm|Payrolls?|Unemployment|Jobless|Retail Sales|ISM|PMI|Fed|FOMC|Powell|ECB|Lagarde|"
    r"BoJ|BOJ|Ueda|BoE|BOE|Bailey|Interest Rate|Rate Decision|Durable Goods|Consumer Confidence|Trade Balance|"
    r"Industrial Production|Inflation|Exports)\b", re.I)


def _cache(path: Optional[Path], key: str, fetch, ttl: float = CALENDAR_TTL) -> Any:
    pf.init_db(path)
    conn = pf._conn(path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS calendar_cache (key TEXT PRIMARY KEY, data TEXT NOT NULL, fetched_at REAL NOT NULL)")
        row = conn.execute("SELECT data, fetched_at FROM calendar_cache WHERE key = ?", (key,)).fetchone()
    finally:
        conn.close()
    if row and time.time() - row["fetched_at"] < ttl:
        return json.loads(row["data"])
    try:
        data = fetch()
    except Exception:  # noqa: BLE001 - an old copy beats nothing
        if row:
            return json.loads(row["data"])
        raise
    conn = pf._conn(path)
    try:
        conn.execute("INSERT OR REPLACE INTO calendar_cache(key, data, fetched_at) VALUES (?, ?, ?)", (key, json.dumps(data), time.time()))
        conn.commit()
    finally:
        conn.close()
    return data


def trading_days(start: date, n: int) -> List[date]:
    """The next n weekdays from start (inclusive); no holiday calendar."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def earnings_on(d: date, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    def fetch():
        rows = ((pf._get_json(EARNINGS_URL.format(d=d.isoformat())).get("data") or {}).get("rows")) or []
        return [{"symbol": r.get("symbol"), "name": r.get("name"),
                 "when": {"time-pre-market": "before the open", "time-after-hours": "after the close"}.get(r.get("time"), "time not given"),
                 "eps_forecast": pf._money(r.get("epsForecast")), "estimates": r.get("noOfEsts"),
                 "eps_last_year": pf._money(r.get("lastYearEPS")), "fiscal_quarter": r.get("fiscalQuarterEnding"),
                 "market_cap": r.get("marketCap")} for r in rows]
    return _cache(path, f"earnings:{d.isoformat()}", fetch)


def econ_on(d: date, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    def fetch():
        rows = ((pf._get_json(ECON_URL.format(d=d.isoformat())).get("data") or {}).get("rows")) or []
        clean = lambda v: (str(v or "").replace("&nbsp;", "").strip() or None)
        return [{"gmt": r.get("gmt"), "country": r.get("country"), "event": r.get("eventName"),
                 "consensus": clean(r.get("consensus")), "previous": clean(r.get("previous")), "actual": clean(r.get("actual"))}
                for r in rows]
    return _cache(path, f"econ:{d.isoformat()}", fetch)


def implied_move(symbol: str, on: date, price: Optional[float]) -> Optional[Dict[str, Any]]:
    """At-the-money straddle of the first expiry on or after `on`, as a % of the price."""
    if not price:
        return None
    d = pf._get_json(OPTIONS_URL.format(sym=symbol, start=on.isoformat(), end=(on + timedelta(days=14)).isoformat()))
    rows = ((d.get("data") or {}).get("table") or {}).get("rows") or []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    current = None
    for r in rows:
        if r.get("expirygroup"):
            current = r["expirygroup"]
            continue
        if current and r.get("strike"):
            groups.setdefault(current, []).append(r)
    for expiry, strikes in groups.items():           # in date order, first one on/after `on`
        best = min(strikes, key=lambda r: abs((pf._money(r["strike"]) or 0) - price))
        def mid(prefix: str) -> Optional[float]:
            bid, ask, last = (pf._money(best.get(f"{prefix}_{k}")) for k in ("Bid", "Ask", "Last"))
            return (bid + ask) / 2 if bid and ask and ask >= bid else last
        c, p = mid("c"), mid("p")
        if c and p:
            return {"expiry": expiry, "strike": pf._money(best["strike"]), "straddle": round(c + p, 2),
                    "implied_move_pct": round((c + p) / price * 100, 1)}
    return None


def upcoming(symbols: List[str], days: int = 5, path: Optional[Path] = None,
             prices: Optional[Dict[str, float]] = None, max_options: int = 8) -> Dict[str, Any]:
    """Earnings of these symbols and the main economic releases over the next `days` trading days."""
    today = datetime.now(NEW_YORK).date()
    wanted = set(symbols)
    earnings, econ, options_left = [], [], max_options
    for d in trading_days(today, days):
        for e in earnings_on(d, path):
            if e["symbol"] in wanted:
                item = {"date": d.isoformat(), **e,
                        "bellwether": BELLWETHERS.get(e["symbol"], (None, None))[1]}
                price = (prices or {}).get(e["symbol"])
                if price and options_left > 0:
                    options_left -= 1
                    try:
                        item["implied"] = implied_move(e["symbol"], d, price)
                    except Exception:  # noqa: BLE001 - the brief can still search for it
                        item["implied"] = None
                earnings.append(item)
        for r in econ_on(d, path):
            if r["country"] in ECON_COUNTRIES and r["event"] and ECON_KEYWORDS.search(r["event"]):
                econ.append({"date": d.isoformat(), **r})
    return {"from": today.isoformat(), "trading_days": days, "earnings": earnings, "economic": econ[:60]}


# ---------------------------------------------------------------- flash triggers

def big_moves(path: Optional[Path] = None, threshold: float = FLASH_THRESHOLD) -> List[Dict[str, Any]]:
    """Bellwethers moving `threshold` % or more against the regular close in pre-market or
    after hours, from fresh Nasdaq quotes."""
    quotes = pf.get_quotes(list(BELLWETHERS), path=path, ttl_by_symbol={s: FLASH_CHECK_EVERY - 30 for s in BELLWETHERS})
    out = []
    for sym, q in quotes.items():
        move = q.get("session_change_pct")
        if q.get("session") in ("pre", "post") and move is not None and not q.get("stale") and abs(move) >= threshold:
            out.append({"symbol": sym, "name": BELLWETHERS[sym][0], "group": BELLWETHERS[sym][1], "session": q["session"],
                        "price": q.get("price"), "regular_close": q.get("regular_price"), "move_pct": round(move, 2)})
    return sorted(out, key=lambda m: -abs(m["move_pct"]))
