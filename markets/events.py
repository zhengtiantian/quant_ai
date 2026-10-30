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
import os
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


# Nasdaq's calendar misses some listings: SK hynix's ADR (SKHY, listed 2026-07-10) never appears,
# its results being announced in Seoul. Finnhub has them (under 000660.KS for SKHY), so a symbol
# Nasdaq does not list in the window is looked up there, one call per symbol and day (2026-10-25).
FINNHUB_EARNINGS_URL = "https://finnhub.io/api/v1/calendar/earnings?symbol={sym}&from={start}&to={end}&token={key}"
FINNHUB_ENV = Path(os.getenv("FINNHUB_ENV", "/Users/xiz/Quant_trade/quant_data/collectors/news/finnhub/.env"))


def _finnhub_key() -> Optional[str]:
    if os.getenv("FINNHUB_API_KEY"):
        return os.environ["FINNHUB_API_KEY"]
    try:
        for line in FINNHUB_ENV.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() == "FINNHUB_API_KEY":
                return v.strip().strip('"')
    except OSError:
        return None
    return None


def finnhub_earnings(symbol: str, start: date, end: date, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    key = _finnhub_key()
    if not key:
        return []

    def fetch():
        d = pf._get_json(FINNHUB_EARNINGS_URL.format(sym=symbol, start=start.isoformat(), end=end.isoformat(), key=key))
        out = []
        for r in d.get("earningsCalendar") or []:
            listed = str(r.get("symbol") or "")
            when = {"amc": "after the close", "bmo": "before the open"}.get(r.get("hour") or "")
            if not when:
                # A Seoul announcement before the Korean open is the previous evening in New York.
                when = "after the close (Seoul morning)" if listed.endswith(".KS") else "time not given"
            out.append({"symbol": symbol, "listed_as": listed, "date": r.get("date"), "when": when,
                        "eps_forecast": r.get("epsEstimate"), "revenue_forecast": r.get("revenueEstimate"),
                        # per share of the listing Finnhub found: won for 000660.KS, TWD for 2330.TW, EUR for ASML.AS
                        "forecast_currency": _listing_currency(listed),
                        "fiscal_quarter": f"Q{r.get('quarter')} {r.get('year')}", "source": "finnhub"})
        return out
    return _cache(path, f"finnhub_earnings:{symbol}:{start.isoformat()}:{end.isoformat()}", fetch)


def upcoming(symbols: List[str], days: int = 5, path: Optional[Path] = None,
             prices: Optional[Dict[str, float]] = None, max_options: int = 8,
             start: Optional[date] = None) -> Dict[str, Any]:
    """Earnings of these symbols and the main economic releases over `days` trading days from
    `start` (today in New York by default)."""
    today = start or datetime.now(NEW_YORK).date()
    wanted = set(symbols)
    earnings, econ, options_left = [], [], max_options
    window = trading_days(today, days)
    for d in window:
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
    found = {e["symbol"] for e in earnings}
    for sym in sorted(wanted - found):
        try:
            rows = finnhub_earnings(sym, window[0], window[-1], path) if window else []
        except Exception:  # noqa: BLE001 - one missing date must not take the calendar down
            rows = []
        for e in rows:
            item = {**e, "bellwether": BELLWETHERS.get(sym, (None, None))[1]}
            price = (prices or {}).get(sym)
            if price and options_left > 0 and e.get("date"):
                options_left -= 1
                try:
                    item["implied"] = implied_move(sym, date.fromisoformat(e["date"]), price)
                except Exception:  # noqa: BLE001
                    item["implied"] = None
            earnings.append(item)
    earnings.sort(key=lambda e: e.get("date") or "")
    return {"from": today.isoformat(), "trading_days": days, "earnings": earnings, "economic": econ[:60]}



DIVIDENDS_URL = "https://api.nasdaq.com/api/quote/{sym}/dividends?assetclass={cls}"
SPLITS_URL = "https://api.nasdaq.com/api/calendar/splits?date={d}"
DIVIDENDS_TTL = 12 * 3600


def _listing_currency(listed: str) -> str:
    u = (listed or "").upper()
    for suffix, cur in ((".KS", "KRW"), (".KQ", "KRW"), (".TW", "TWD"), (".TWO", "TWD"),
                        (".AS", "EUR"), (".PA", "EUR"), (".DE", "EUR"), (".T", "JPY"),
                        (".L", "GBP"), (".HK", "HKD")):
        if u.endswith(suffix):
            return cur
    return "USD"


def _parse_mdy(val: Any) -> Optional[str]:
    s = str(val or "").strip()
    if not s or s.upper() == "N/A":
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:10], fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _known_asset_class(symbol: str, path: Optional[Path] = None) -> Optional[str]:
    pf.init_db(path)
    conn = pf._conn(path)
    try:
        row = conn.execute("SELECT asset_class FROM quotes WHERE symbol = ?", (symbol.upper(),)).fetchone()
        return row["asset_class"] if row and row["asset_class"] else None
    except Exception:  # noqa: BLE001
        return None
    finally:
        conn.close()


def dividends_for(symbol: str, asset_class: Optional[str] = None, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Dividend declarations / history for `symbol` from Nasdaq, cached for 12 hours."""
    sym = symbol.strip().upper()
    cls_hint = asset_class or _known_asset_class(sym, path)
    classes = [cls_hint] if cls_hint else ["stocks", "etf"]

    def fetch():
        for cls in classes:
            try:
                resp = pf._get_json(DIVIDENDS_URL.format(sym=sym, cls=cls))
            except Exception:  # noqa: BLE001
                continue
            data = resp.get("data") or {}
            rows = ((data.get("dividends") or {}).get("rows")) or []
            if not rows:
                continue
            out = []
            for r in rows[:8]:
                ex_d = _parse_mdy(r.get("exOrEffDate"))
                pay_d = _parse_mdy(r.get("paymentDate"))
                amt = pf._money(r.get("amount"))
                if not (ex_d or pay_d) or amt is None:
                    continue
                out.append({
                    "symbol": sym,
                    "ex_date": ex_d,
                    "pay_date": pay_d,
                    "record_date": _parse_mdy(r.get("recordDate")),
                    "declaration_date": _parse_mdy(r.get("declarationDate")),
                    "amount": amt,
                    "currency": str(r.get("currency") or "USD").strip() or "USD",
                    "div_type": str(r.get("type") or "Cash").strip(),
                })
            return out
        return []

    return _cache(path, f"dividends:{sym}", fetch, ttl=DIVIDENDS_TTL)


def splits_on(d: date, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Stock splits scheduled on `d` from Nasdaq, cached per day."""
    def fetch():
        rows = ((pf._get_json(SPLITS_URL.format(d=d.isoformat())).get("data") or {}).get("rows")) or []
        out = []
        for r in rows:
            sym = str(r.get("symbol") or "").strip().upper()
            if not sym:
                continue
            out.append({
                "symbol": sym,
                "name": r.get("name"),
                "ratio": r.get("ratio"),
                "execution_date": _parse_mdy(r.get("executionDate")) or d.isoformat(),
            })
        return out

    return _cache(path, f"splits:{d.isoformat()}", fetch, ttl=DIVIDENDS_TTL)


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
