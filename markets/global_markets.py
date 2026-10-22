"""Global markets: Asian and European exchanges and the macro board (oil, gold, US indices,
VIX, Treasury yields, the dollar), for the portfolio page, the desktop widget and the briefs.
"""

import json
import threading
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from . import portfolio as pf

# Asian and European markets, grouped region -> country -> exchange, each exchange with its
# benchmark indices and a few representative stocks. Quotes come from CNBC's quote service (one
# request for everything, no key); Korean stocks, which CNBC does not carry, from Naver Finance.
# Nasdaq covers only US listings and Yahoo answered 429 here.
# Each exchange has its own local trading hours: an open one refreshes every GLOBAL_TTL seconds,
# a closed one is fetched once after its close for the final price and then left alone.
GLOBAL_TTL = 60
CNBC_URL = ("https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol?symbols={syms}"
            "&requestMethod=itv&noform=1&partnerId=2&fund=1&exthrs=1&output=json")
NAVER_URL = "https://polling.finance.naver.com/api/realtime/domestic/stock/{codes}"
NAVER_PREFIX = "KRX:"
# country code: (name, flag, region)
COUNTRIES = {
    "CN": ("China", "🇨🇳", "Asia"), "HK": ("Hong Kong", "🇭🇰", "Asia"), "JP": ("Japan", "🇯🇵", "Asia"),
    "KR": ("South Korea", "🇰🇷", "Asia"), "TW": ("Taiwan", "🇹🇼", "Asia"),
    "GB": ("United Kingdom", "🇬🇧", "Europe"), "DE": ("Germany", "🇩🇪", "Europe"), "FR": ("France", "🇫🇷", "Europe"),
    "NL": ("Netherlands", "🇳🇱", "Europe"), "CH": ("Switzerland", "🇨🇭", "Europe"), "IE": ("Ireland", "🇮🇪", "Europe"),
    "IT": ("Italy", "🇮🇹", "Europe"), "ES": ("Spain", "🇪🇸", "Europe"), "EU": ("Eurozone", "🇪🇺", "Europe"),
}
# exchange code: (name, country, time zone, open HH:MM, close HH:MM) - weekdays, no holiday calendar
EXCHANGES = {
    "SSE": ("Shanghai Stock Exchange", "CN", "Asia/Shanghai", "09:30", "15:00"),
    "SZSE": ("Shenzhen Stock Exchange", "CN", "Asia/Shanghai", "09:30", "15:00"),
    "HKEX": ("Hong Kong Exchange", "HK", "Asia/Hong_Kong", "09:30", "16:10"),
    "TSE": ("Tokyo Stock Exchange", "JP", "Asia/Tokyo", "09:00", "15:30"),
    "KOSPI": ("Korea Exchange · KOSPI", "KR", "Asia/Seoul", "09:00", "15:30"),
    "KOSDAQ": ("Korea Exchange · KOSDAQ", "KR", "Asia/Seoul", "09:00", "15:30"),
    "TWSE": ("Taiwan Stock Exchange", "TW", "Asia/Taipei", "09:00", "13:30"),
    "LSE": ("London Stock Exchange", "GB", "Europe/London", "08:00", "16:35"),
    "XETRA": ("Xetra (Frankfurt)", "DE", "Europe/Berlin", "09:00", "17:35"),
    "ENXPA": ("Euronext Paris", "FR", "Europe/Paris", "09:00", "17:35"),
    "ENXAM": ("Euronext Amsterdam", "NL", "Europe/Amsterdam", "09:00", "17:35"),
    "SIX": ("SIX Swiss Exchange", "CH", "Europe/Zurich", "09:00", "17:30"),
    "ENXDU": ("Euronext Dublin", "IE", "Europe/Dublin", "08:00", "16:30"),
    "MIL": ("Borsa Italiana (Milan)", "IT", "Europe/Rome", "09:00", "17:35"),
    "BME": ("Bolsa de Madrid", "ES", "Europe/Madrid", "09:00", "17:35"),
    "STOXX": ("STOXX (pan-eurozone)", "EU", "Europe/Berlin", "09:00", "17:35"),
}
# (symbol, display name, exchange, kind), in display order within each exchange
GLOBAL_ITEMS = [
    (".SSEC", "Shanghai Composite", "SSE", "index"), (".STAR50", "STAR 50", "SSE", "index"),
    ("600519-CN", "Kweichow Moutai", "SSE", "stock"), ("601398-CN", "ICBC", "SSE", "stock"),
    ("600036-CN", "China Merchants Bank", "SSE", "stock"), ("601318-CN", "Ping An", "SSE", "stock"),
    (".SZI", "Shenzhen Component", "SZSE", "index"),
    ("300750-CN", "CATL", "SZSE", "stock"), ("002594-CN", "BYD", "SZSE", "stock"),
    ("000858-CN", "Wuliangye", "SZSE", "stock"), ("000333-CN", "Midea", "SZSE", "stock"),
    (".HSI", "Hang Seng", "HKEX", "index"), (".HSTECH", "Hang Seng Tech", "HKEX", "index"),
    ("0700.HK", "Tencent", "HKEX", "stock"), ("9988.HK", "Alibaba", "HKEX", "stock"),
    ("3690.HK", "Meituan", "HKEX", "stock"), ("1299.HK", "AIA", "HKEX", "stock"), ("0005.HK", "HSBC", "HKEX", "stock"),
    (".N225", "Nikkei 225", "TSE", "index"), (".TOPX", "TOPIX", "TSE", "index"),
    ("7203.T", "Toyota", "TSE", "stock"), ("6758.T", "Sony", "TSE", "stock"), ("9984.T", "SoftBank Group", "TSE", "stock"),
    ("8306.T", "MUFG", "TSE", "stock"), ("8035.T", "Tokyo Electron", "TSE", "stock"),
    (".KS11", "KOSPI", "KOSPI", "index"),
    ("KRX:005930", "Samsung Electronics", "KOSPI", "stock"), ("KRX:000660", "SK hynix", "KOSPI", "stock"),
    ("KRX:005380", "Hyundai Motor", "KOSPI", "stock"), ("KRX:373220", "LG Energy Solution", "KOSPI", "stock"),
    (".KQ11", "KOSDAQ", "KOSDAQ", "index"),
    ("KRX:196170", "Alteogen", "KOSDAQ", "stock"), ("KRX:247540", "EcoPro BM", "KOSDAQ", "stock"),
    ("KRX:028300", "HLB", "KOSDAQ", "stock"),
    (".TWII", "TAIEX", "TWSE", "index"),
    ("2330-TW", "TSMC", "TWSE", "stock"), ("2317-TW", "Hon Hai (Foxconn)", "TWSE", "stock"),
    ("2454-TW", "MediaTek", "TWSE", "stock"), ("2308-TW", "Delta Electronics", "TWSE", "stock"),
    (".FTSE", "FTSE 100", "LSE", "index"), (".FTMC", "FTSE 250", "LSE", "index"),
    ("SHEL-GB", "Shell", "LSE", "stock"), ("AZN-GB", "AstraZeneca", "LSE", "stock"),
    ("HSBA-GB", "HSBC", "LSE", "stock"), ("ULVR-GB", "Unilever", "LSE", "stock"),
    (".GDAXI", "DAX", "XETRA", "index"), (".MDAXI", "MDAX", "XETRA", "index"),
    ("SAP.DE", "SAP", "XETRA", "stock"), ("SIE-DE", "Siemens", "XETRA", "stock"),
    ("ALV-DE", "Allianz", "XETRA", "stock"), ("DTE-DE", "Deutsche Telekom", "XETRA", "stock"),
    (".FCHI", "CAC 40", "ENXPA", "index"),
    ("MC-FR", "LVMH", "ENXPA", "stock"), ("RMS-FR", "Hermès", "ENXPA", "stock"),
    ("TTE-FR", "TotalEnergies", "ENXPA", "stock"), ("AIR-FR", "Airbus", "ENXPA", "stock"),
    (".AEX", "AEX", "ENXAM", "index"),
    ("ASML.AS", "ASML", "ENXAM", "stock"), ("PRX-NL", "Prosus", "ENXAM", "stock"),
    ("INGA-NL", "ING", "ENXAM", "stock"), ("ADYEN-NL", "Adyen", "ENXAM", "stock"),
    (".SSMI", "SMI", "SIX", "index"),
    ("NESN-CH", "Nestlé", "SIX", "stock"), ("ROG.VX", "Roche", "SIX", "stock"),
    ("NOVN-CH", "Novartis", "SIX", "stock"), ("UBSG-CH", "UBS", "SIX", "stock"),
    (".ISEQ", "ISEQ All-Share", "ENXDU", "index"),
    ("RYA-IE", "Ryanair", "ENXDU", "stock"), ("KRZ-IE", "Kerry Group", "ENXDU", "stock"),
    ("A5G-IE", "AIB", "ENXDU", "stock"), ("BIRG-IE", "Bank of Ireland", "ENXDU", "stock"),
    (".FTMIB", "FTSE MIB", "MIL", "index"),
    ("RACE-IT", "Ferrari", "MIL", "stock"), ("ENEL-IT", "Enel", "MIL", "stock"), ("ISP-IT", "Intesa Sanpaolo", "MIL", "stock"),
    (".IBEX", "IBEX 35", "BME", "index"),
    ("ITX-ES", "Inditex", "BME", "stock"), ("SAN-ES", "Santander", "BME", "stock"), ("IBE-ES", "Iberdrola", "BME", "stock"),
    (".STOXX50E", "Euro Stoxx 50", "STOXX", "index"),
]
_global_lock = threading.Lock()
_global_state = {"running": False}


def market_status(exchange: str, now: Optional[datetime] = None) -> Dict[str, Any]:
    """open?, today's close as a UTC timestamp (None on weekends), and the local clock."""
    _, _, tz, o, c = EXCHANGES[exchange]
    t = (now or datetime.now(ZoneInfo("UTC"))).astimezone(ZoneInfo(tz))
    if t.weekday() >= 5:
        return {"open": False, "close_ts": None, "local_time": t.strftime("%H:%M"), "tz": tz}
    oh, om = map(int, o.split(":")); ch, cm = map(int, c.split(":"))
    opened = t.replace(hour=oh, minute=om, second=0, microsecond=0)
    closed = t.replace(hour=ch, minute=cm, second=0, microsecond=0)
    return {"open": opened <= t < closed, "close_ts": closed.timestamp(), "open_ts": opened.timestamp(),
            "local_time": t.strftime("%H:%M"), "tz": tz}


def _parse_cnbc(q: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    last = pf._money(q.get("last"))
    if last is None:
        return None
    return {"price": last, "change": pf._money(q.get("change")), "change_pct": pf._money(q.get("change_pct")),
            "prev_close": pf._money(q.get("previous_day_closing")), "currency": q.get("currencyCode"),
            "trade_time": q.get("last_time"), "full_name": q.get("name")}


def _parse_naver(q: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # closePrice is the regular-session price; the after-market one (overMarketPriceInfo) is left out.
    last = pf._money(q.get("closePrice"))
    if last is None:
        return None
    change = pf._money(q.get("compareToPreviousClosePrice"))
    return {"price": last, "change": change, "change_pct": pf._money(q.get("fluctuationsRatio")),
            "prev_close": last - change if change is not None else None, "currency": "KRW",
            "trade_time": q.get("localTradedAt"), "full_name": q.get("stockName")}


def _global_due(items: List[tuple], stored: Dict[str, Dict[str, Any]], now: float) -> List[str]:
    """Symbols worth a request now: never fetched, open and older than GLOBAL_TTL, or closed
    but not fetched since today's close (one last fetch for the closing price)."""
    due = []
    for sym, _, ex, _ in items:
        row = stored.get(sym)
        st = market_status(ex)
        if not row:
            due.append(sym)
        elif st["open"] and now - row["fetched_at"] >= GLOBAL_TTL:
            due.append(sym)
        elif not st["open"] and st["close_ts"] and now >= st["close_ts"] and row["fetched_at"] < st["close_ts"]:
            due.append(sym)
    return due


def _fetch_global(due: List[str]) -> Dict[str, Dict[str, Any]]:
    """symbol -> parsed quote; one request per source, and a failing source leaves the other's."""
    out: Dict[str, Dict[str, Any]] = {}
    naver = [s for s in due if s.startswith(NAVER_PREFIX)]
    cnbc = [s for s in due if not s.startswith(NAVER_PREFIX)]
    if cnbc:
        try:
            d = pf._get_json(CNBC_URL.format(syms=urllib.parse.quote("|".join(cnbc))))
            for q in ((d.get("FormattedQuoteResult") or {}).get("FormattedQuote")) or []:
                parsed = _parse_cnbc(q)
                if parsed and q.get("symbol") in cnbc:
                    out[q["symbol"]] = parsed
        except Exception:  # noqa: BLE001 - stored quotes stay as they are
            pass
    if naver:
        try:
            d = pf._get_json(NAVER_URL.format(codes=",".join(s[len(NAVER_PREFIX):] for s in naver)))
            for q in d.get("datas") or []:
                sym, parsed = NAVER_PREFIX + str(q.get("itemCode")), _parse_naver(q)
                if parsed and sym in naver:
                    out[sym] = parsed
        except Exception:  # noqa: BLE001
            pass
    return out


def refresh_global(path: Optional[Path] = None) -> int:
    pf.init_db(path)
    conn = pf._conn(path)
    try:
        stored = {r["symbol"]: dict(r) for r in conn.execute("SELECT * FROM global_quotes")}
    finally:
        conn.close()
    due = _global_due(GLOBAL_ITEMS, stored, time.time())
    if not due:
        return 0
    quotes = _fetch_global(due)
    now = time.time()
    conn = pf._conn(path)
    try:
        for sym, parsed in quotes.items():
            conn.execute("INSERT INTO global_quotes(symbol, data, fetched_at) VALUES (?, ?, ?) ON CONFLICT(symbol) "
                         "DO UPDATE SET data=excluded.data, fetched_at=excluded.fetched_at",
                         (sym, json.dumps(parsed), now))
        conn.commit()
    finally:
        conn.close()
    return len(quotes)


def refresh_global_in_background(path: Optional[Path] = None) -> bool:
    with _global_lock:
        if _global_state["running"]:
            return True
        _global_state["running"] = True

    def run():
        try:
            refresh_global(path)
        except Exception:  # noqa: BLE001 - stored quotes stay as they are
            pass
        finally:
            with _global_lock:
                _global_state["running"] = False

    threading.Thread(target=run, name="global-quotes", daemon=True).start()
    return True


def global_state(path: Optional[Path] = None, background: bool = True) -> Dict[str, Any]:
    """regions -> countries -> exchanges -> {indices, stocks}, in display order."""
    pf.init_db(path)
    conn = pf._conn(path)
    try:
        stored = {r["symbol"]: dict(r) for r in conn.execute("SELECT * FROM global_quotes")}
    finally:
        conn.close()
    refreshing = False
    if background and _global_due(GLOBAL_ITEMS, stored, time.time()):
        refreshing = refresh_global_in_background(path)
    exchanges: Dict[str, Dict[str, Any]] = {}
    for code, (name, country, tz, o, c) in EXCHANGES.items():
        st = market_status(code)
        exchanges[code] = {"code": code, "name": name, "open": st["open"], "local_time": st["local_time"],
                           "hours": f"{o}–{c}", "tz": tz, "indices": [], "stocks": []}
    for sym, name, ex, kind in GLOBAL_ITEMS:
        row = stored.get(sym)
        exchanges[ex]["indices" if kind == "index" else "stocks"].append(
            {"symbol": sym, "name": name, "fetched_at": row["fetched_at"] if row else None,
             **(json.loads(row["data"]) if row else {})})
    regions: List[Dict[str, Any]] = []
    for ccode, (cname, flag, region) in COUNTRIES.items():
        exs = [exchanges[c] for c, v in EXCHANGES.items() if v[1] == ccode and (exchanges[c]["indices"] or exchanges[c]["stocks"])]
        if not exs:
            continue
        if not regions or regions[-1]["name"] != region:
            regions.append({"name": region, "countries": []})
        regions[-1]["countries"].append({"code": ccode, "name": cname, "flag": flag,
                                         "open": any(e["open"] for e in exs), "exchanges": exs})
    any_open = any(e["open"] for e in exchanges.values())
    return {"regions": regions, "any_open": any_open, "refreshing": refreshing,
            "refresh_seconds": GLOBAL_TTL if any_open else None}


# ---------------------------------------------------------------- state


# ---------------------------------------------------------------- macro board
# Fetched on demand (the briefs, the page) and kept for MACRO_TTL seconds; these trade nearly
# around the clock, so there is no market-hours schedule to follow.
MACRO_TTL = 60
# (CNBC symbol, display name, group)
MACRO_ITEMS = [
    (".SPX", "S&P 500", "US indices"), (".IXIC", "Nasdaq Composite", "US indices"), (".DJI", "Dow Jones", "US indices"),
    (".RUT", "Russell 2000", "US indices"), (".SOX", "Philadelphia Semiconductor", "US indices"),
    ("@SP.1", "S&P 500 futures", "US futures"), ("@ND.1", "Nasdaq 100 futures", "US futures"),
    (".VIX", "VIX", "Volatility"),
    ("US2Y", "US 2Y yield", "Rates"), ("US10Y", "US 10Y yield", "Rates"), ("US30Y", "US 30Y yield", "Rates"),
    ("DE10Y", "German 10Y yield", "Rates"), ("JP10Y", "Japan 10Y yield", "Rates"),
    (".DXY", "US dollar index", "FX"), ("EUR=", "EUR/USD", "FX"), ("JPY=", "USD/JPY", "FX"),
    ("CNY=", "USD/CNY", "FX"), ("GBP=", "GBP/USD", "FX"),
    ("@LCO.1", "Brent crude", "Commodities"), ("@CL.1", "WTI crude", "Commodities"), ("@GC.1", "Gold", "Commodities"),
    ("@SI.1", "Silver", "Commodities"), ("@HG.1", "Copper", "Commodities"), ("@NG.1", "Natural gas", "Commodities"),
    ("BTC.CB=", "Bitcoin", "Crypto"),
]


def fetch_cnbc(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """One CNBC request for any list of symbols; symbol -> parsed quote (missing ones left out)."""
    if not symbols:
        return {}
    d = pf._get_json(CNBC_URL.format(syms=urllib.parse.quote("|".join(symbols))))
    out = {}
    for q in ((d.get("FormattedQuoteResult") or {}).get("FormattedQuote")) or []:
        parsed = _parse_cnbc(q)
        if parsed and q.get("symbol") in symbols:
            out[q["symbol"]] = parsed
    return out


def macro_state(path: Optional[Path] = None, max_age: float = MACRO_TTL) -> Dict[str, Any]:
    pf.init_db(path)
    syms = [m[0] for m in MACRO_ITEMS]
    conn = pf._conn(path)
    try:
        stored = {r["symbol"]: dict(r) for r in conn.execute(
            f"SELECT * FROM global_quotes WHERE symbol IN ({','.join('?' * len(syms))})", syms)}
    finally:
        conn.close()
    now = time.time()
    due = [s for s in syms if s not in stored or now - stored[s]["fetched_at"] >= max_age]
    if due:
        try:
            fresh = fetch_cnbc(due)
        except Exception:  # noqa: BLE001 - stored values stay
            fresh = {}
        if fresh:
            conn = pf._conn(path)
            try:
                for sym, q in fresh.items():
                    conn.execute("INSERT INTO global_quotes(symbol, data, fetched_at) VALUES (?, ?, ?) ON CONFLICT(symbol) "
                                 "DO UPDATE SET data=excluded.data, fetched_at=excluded.fetched_at", (sym, json.dumps(q), now))
                    stored[sym] = {"symbol": sym, "data": json.dumps(q), "fetched_at": now}
                conn.commit()
            finally:
                conn.close()
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for sym, name, group in MACRO_ITEMS:
        row = stored.get(sym)
        groups.setdefault(group, []).append({"symbol": sym, "name": name, "fetched_at": row["fetched_at"] if row else None,
                                             **(json.loads(row["data"]) if row else {})})
    return {"groups": [{"name": g, "items": items} for g, items in groups.items()]}
