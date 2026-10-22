"""
The user's own US stock portfolio (moved here from job_agent's dashboard on 2026-10-22).

Trades are entered one by one (Revolut has no API for personal trading accounts); holdings,
average cost and P&L are derived from them, and prices come live from Nasdaq, including the
pre-market and after-hours sessions.

Cost methods, both always reported: average cost (a sale realises (price - average cost) x
quantity minus its fee and leaves the average of the remaining shares unchanged) and FIFO
(a sale takes the oldest lots first - the Irish CGT basis). Fractional shares are fine.

Quotes: api.nasdaq.com (no key; unofficial, so every failure falls back to the last quote
stored in the database rather than failing the page). Free Finnhub, used by quant_api, has no
extended-hours prices; Yahoo answered 429 when this was built (2026-10-20).

Storage is SQLite on the host (markets/config.py), so none of this needs Docker running.
"""

import json
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from datetime import time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from . import budget_link, global_markets
from .config import DB_PATH

US_EAST = ZoneInfo("America/New_York")
EPS = 1e-9

TX_TYPES = {"buy": "Buy", "sell": "Sell", "dividend": "Dividend", "fee": "Fee"}
SESSIONS = {"pre": "Pre-market", "regular": "Market open", "post": "After hours", "closed": "Closed"}
# Seconds a quote stays fresh, by session. The page polls every REFRESH_SECONDS[session];
# while the market is closed it does not poll at all and wakes up when pre-market opens.
# Held symbols refresh fast during the regular session; the watchlist is a once-a-minute glance.
QUOTE_TTL = {"regular": 5, "pre": 30, "post": 30, "closed": 3600}             # holdings
WATCH_TTL = {"regular": 5, "pre": 30, "post": 30, "closed": 3600}             # watchlist
REFRESH_SECONDS = {"regular": 5, "pre": 30, "post": 30, "closed": None}
# The user's waking hours (Dublin): every 5 s in any trading session, pre-market and after hours
# included (asked 2026-10-22). Outside it the tables above apply. US pre-market opens 09:00 Dublin
# and after hours ends 01:00, so on a weekday the window is all trading time.
FAST_ZONE = ZoneInfo("Europe/Dublin")
FAST_WINDOW = (dtime(10, 0), dtime(22, 0))
FAST_SECONDS = 5


def in_fast_window(now: Optional[datetime] = None) -> bool:
    t = (now or datetime.now(FAST_ZONE)).astimezone(FAST_ZONE).time()
    return FAST_WINDOW[0] <= t < FAST_WINDOW[1]


def refresh_interval(session: str, table: Dict[str, Any], now: Optional[datetime] = None) -> Any:
    """Seconds for this session: FAST_SECONDS inside the window while any session trades,
    otherwise the session's entry in `table` (None = do not poll)."""
    if session in ("pre", "regular", "post") and in_fast_window(now):
        return FAST_SECONDS
    return table.get(session)

NASDAQ = "https://api.nasdaq.com/api/quote/{sym}/{endpoint}?assetclass={cls}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
           "Accept": "application/json"}
_STATUS_TO_SESSION = {"pre-market": "pre", "open": "regular", "after-hours": "post", "closed": "closed"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY, trade_date TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
    quantity REAL NOT NULL DEFAULT 0, price REAL NOT NULL DEFAULT 0, fee REAL NOT NULL DEFAULT 0,
    currency TEXT NOT NULL DEFAULT 'USD', note TEXT, created_at TEXT);
CREATE INDEX IF NOT EXISTS idx_tx_symbol ON transactions(symbol, trade_date);
CREATE TABLE IF NOT EXISTS watchlist (
    symbol TEXT PRIMARY KEY, note TEXT, sort INTEGER NOT NULL DEFAULT 0, created_at TEXT);
CREATE TABLE IF NOT EXISTS charts (
    symbol TEXT NOT NULL, range TEXT NOT NULL, data TEXT NOT NULL, fetched_at REAL NOT NULL,
    PRIMARY KEY (symbol, range));
CREATE TABLE IF NOT EXISTS global_quotes (
    symbol TEXT PRIMARY KEY, data TEXT NOT NULL, fetched_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS quotes (
    symbol TEXT PRIMARY KEY, asset_class TEXT, data TEXT NOT NULL, fetched_at REAL NOT NULL);
"""


def _conn(path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Optional[Path] = None) -> None:
    conn = _conn(path)
    try:
        conn.executescript(SCHEMA)
        # request_id: the idempotency key of the save that created the row (added 2026-10-20).
        if "request_id" not in {r["name"] for r in conn.execute("PRAGMA table_info(transactions)")}:
            conn.execute("ALTER TABLE transactions ADD COLUMN request_id TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tx_request ON transactions(request_id)")
        # budget_sync / budget_entry_id: a sale's profit is written to the budget's 股票 item (added 2026-10-20).
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(transactions)")}
        if "budget_sync" not in cols:
            conn.execute("ALTER TABLE transactions ADD COLUMN budget_sync INTEGER NOT NULL DEFAULT 0")
        if "budget_entry_id" not in cols:
            conn.execute("ALTER TABLE transactions ADD COLUMN budget_entry_id INTEGER")
        conn.commit()
    finally:
        conn.close()


def _write_conn(path: Optional[Path] = None) -> sqlite3.Connection:
    """A connection whose transaction starts with BEGIN IMMEDIATE: the write lock is taken
    before the books are read, so two saves cannot both pass the "do you hold enough shares"
    check against the same state. A second writer waits (busy timeout 30 s) instead."""
    conn = _conn(path)
    conn.isolation_level = None
    conn.execute("BEGIN IMMEDIATE")
    return conn


# ---------------------------------------------------------------- transactions

def _num(v, name: str, minimum: float = 0.0) -> float:
    try:
        x = float(str(v if v is not None else "").replace(",", "").replace("$", "").strip() or 0)
    except ValueError:
        raise ValueError(f"{name} is not a number")
    if x < minimum:
        raise ValueError(f"{name} cannot be less than {minimum:g}")
    return x


def save_transaction(data: Dict[str, Any], path: Optional[Path] = None) -> int:
    side = data.get("side") or "buy"
    if side not in TX_TYPES:
        raise ValueError(f"Unknown type {side!r}")
    symbol = re.sub(r"\s+", "", str(data.get("symbol") or "")).upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-/]{0,9}", symbol):
        raise ValueError("Invalid ticker (e.g. NVDA, VOO, BRK.B)")
    try:
        trade_date = date.fromisoformat(str(data.get("trade_date") or "")).isoformat()
    except ValueError:
        raise ValueError("Date must be YYYY-MM-DD")
    quantity = _num(data.get("quantity"), "Quantity")
    price = _num(data.get("price"), "Price" if side in ("buy", "sell") else "Amount")
    # "Bought $100 worth at $8": the share count follows from the amount (Revolut buys by amount).
    if side in ("buy", "sell") and data.get("entry_mode") == "amount":
        amount = _num(data.get("amount"), "Amount")
        if amount <= 0 or price <= 0:
            raise ValueError("When entering by amount, both amount and price must be greater than 0")
        quantity = round(amount / price, 8)
    fee = _num(data.get("fee"), "Fee")
    if side in ("buy", "sell") and (quantity <= 0 or price <= 0):
        raise ValueError("Buy / sell quantity and price must both be greater than 0")
    if side in ("dividend", "fee"):
        quantity = 0.0
        if price <= 0:
            raise ValueError("Dividend / fee amount must be greater than 0")
    tid = int(data["id"]) if data.get("id") not in (None, "") else None
    request_id = str(data.get("request_id") or "").strip()[:64] or None
    # Sales record their profit in the budget unless the form unticks it; other types never do.
    budget_sync = 1 if side == "sell" and str(data.get("budget_sync", "1")).lower() not in ("0", "false", "", "off") else 0
    fields = (trade_date, symbol, side, quantity, price, fee, "USD", str(data.get("note") or "").strip())
    conn = _write_conn(path)
    try:
        if tid:
            if not conn.execute("UPDATE transactions SET trade_date=?, symbol=?, side=?, quantity=?, price=?, fee=?, "
                                "currency=?, note=?, budget_sync=? WHERE id=?", (*fields, budget_sync, tid)).rowcount:
                raise ValueError("This transaction no longer exists (it may have been deleted elsewhere). Refresh and try again")
        else:
            # The same form submitted twice (double click, retry, two tabs) carries the same key.
            if request_id:
                row = conn.execute("SELECT id FROM transactions WHERE request_id = ?", (request_id,)).fetchone()
                if row:
                    conn.execute("ROLLBACK")
                    return row["id"]
            tid = conn.execute("INSERT INTO transactions (trade_date, symbol, side, quantity, price, fee, currency, "
                               "note, created_at, request_id, budget_sync) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (*fields, datetime.now().isoformat(sep=" ", timespec="seconds"), request_id,
                                budget_sync)).lastrowid
        # A sale larger than the position at that date would make the books meaningless.
        _, problems = compute_holdings(_transactions(conn))
        if problems:
            raise ValueError(problems[0])
        conn.execute("COMMIT")
        return tid
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def delete_transaction(tid: int, path: Optional[Path] = None) -> None:
    """Deleting a row that is already gone is a no-op, so a repeated click is harmless. A sale's
    profit line in the budget goes with it on the next sync_budget (the caller runs one)."""
    conn = _write_conn(path)
    try:
        conn.execute("DELETE FROM transactions WHERE id = ?", (tid,))
        _, problems = compute_holdings(_transactions(conn))
        if problems:
            raise ValueError(f"After deleting it: {problems[0]}")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def add_watch(symbol: str, note: str = "", path: Optional[Path] = None) -> str:
    """Watch a symbol without holding it. No network here (a Nasdaq lookup takes seconds): the
    page fetches the price right after, and a symbol Nasdaq does not know shows as not found."""
    sym = re.sub(r"\s+", "", str(symbol or "")).upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-/]{0,9}", sym):
        raise ValueError("Invalid ticker (e.g. NVDA, VOO)")
    conn = _conn(path)
    try:
        sort = conn.execute("SELECT COALESCE(MAX(sort), 0) + 1 FROM watchlist").fetchone()[0]
        conn.execute("INSERT INTO watchlist(symbol, note, sort, created_at) VALUES (?, ?, ?, ?) "
                     "ON CONFLICT(symbol) DO UPDATE SET note = excluded.note",
                     (sym, str(note or "").strip(), sort, datetime.now().isoformat(sep=" ", timespec="seconds")))
        conn.commit()
    finally:
        conn.close()
    return sym


def remove_watch(symbol: str, path: Optional[Path] = None) -> None:
    conn = _conn(path)
    try:
        conn.execute("DELETE FROM watchlist WHERE symbol = ?", (str(symbol or "").upper(),))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------- budget link
# A sale's profit (average-cost rule: shares x (price - average cost) - fee) becomes one line of
# the budget's 股票 income item in job_agent, the itemised list the user keeps by hand. The line
# follows the sale: edited with it, deleted with it, and recomputed when an earlier buy changes
# the average. job_agent owns the budget, so this side only says which lines should exist
# (markets/budget_link.py); when the dashboard is down the push waits and is retried.

def budget_lines(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every budget line the linked sales call for, keyed by transaction id."""
    conn = _conn(path)
    try:
        txs = _transactions(conn)
    finally:
        conn.close()
    positions, _ = compute_holdings(txs)
    sales = {tid: s for p in positions.values() for tid, s in p["sales"].items()}
    lines = []
    for t in txs:
        sale = sales.get(t["id"])
        if t["side"] == "sell" and t.get("budget_sync") and sale:
            lines.append({"key": f"tx:{t['id']}", "tx_id": t["id"], "day": t["trade_date"],
                          "amount": round(sale["profit"], 2), "company": t["symbol"], "profit": sale["profit"],
                          "avg_cost": sale["avg_cost"],
                          "note": f"Auto from portfolio: sold {t['quantity']:g} shares @ ${t['price']:g}, "
                                  f"avg cost ${sale['avg_cost']:.2f}"})
    return lines


def sync_budget(path: Optional[Path] = None) -> Dict[int, Dict[str, Any]]:
    """Push the linked sales' lines to job_agent. Returns {transaction id: info}; when job_agent
    cannot be reached the push is marked pending (budget_link retries it) and info says so."""
    lines = budget_lines(path)
    result = budget_link.push(lines, path)
    conn = _conn(path)
    try:
        if result.get("entries") is not None:
            ids = {int(k.split(":", 1)[1]): v for k, v in result["entries"].items()}
            conn.execute("UPDATE transactions SET budget_entry_id = NULL")
            conn.executemany("UPDATE transactions SET budget_entry_id = ? WHERE id = ?", [(v, k) for k, v in ids.items()])
            conn.commit()
    finally:
        conn.close()
    out = {}
    for ln in lines:
        info = {"profit": ln["profit"], "avg_cost": ln["avg_cost"]}
        if result.get("error"):
            info["pending"] = True
            info["error"] = result["error"]
        else:
            info["entry_id"] = result["entries"].get(ln["key"])
        out[ln["tx_id"]] = info
    return out


def _transactions(conn) -> List[Dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM transactions ORDER BY trade_date, id")]


def compute_holdings(txs: List[Dict[str, Any]]):
    """(positions by symbol, problems), in trade-date order, under two cost rules at once:

    * average cost (the user's rule, the headline): a sale leaves the average unchanged and
      realises against it; only a buy changes the average.
    * FIFO (fifo_*): a sale consumes the oldest lots first, so the average of what remains
      moves. This is how disposals of shares are matched for Irish CGT.
    Buy fees are part of a lot's cost; sell fees reduce that sale's realised gain."""
    pos: Dict[str, Dict[str, Any]] = {}
    problems: List[str] = []
    for t in sorted(txs, key=lambda x: (x["trade_date"], x["id"])):
        p = pos.setdefault(t["symbol"], {"symbol": t["symbol"], "quantity": 0.0, "cost": 0.0, "realized": 0.0,
                                         "dividends": 0.0, "fees": 0.0, "first_date": t["trade_date"], "trades": 0,
                                         "lots": [], "fifo_realized": 0.0, "sales": {}})
        p["trades"] += 1
        p["fees"] += t["fee"]
        if t["side"] == "buy":
            p["quantity"] += t["quantity"]
            p["cost"] += t["quantity"] * t["price"] + t["fee"]
            p["lots"].append([t["quantity"], (t["quantity"] * t["price"] + t["fee"]) / t["quantity"]])
        elif t["side"] == "sell":
            if t["quantity"] > p["quantity"] + EPS:
                problems.append(f"{t['trade_date']}: selling {t['quantity']:g} {t['symbol']} shares, but only {p['quantity']:g} were held then")
                continue
            avg = p["cost"] / p["quantity"] if p["quantity"] > EPS else 0.0
            p["realized"] += t["quantity"] * (t["price"] - avg) - t["fee"]
            left, fifo_cost = t["quantity"], 0.0
            while left > EPS and p["lots"]:
                lot = p["lots"][0]
                take = min(lot[0], left)
                fifo_cost += take * lot[1]
                lot[0] -= take
                left -= take
                if lot[0] < EPS:
                    p["lots"].pop(0)
            p["fifo_realized"] += t["quantity"] * t["price"] - t["fee"] - fifo_cost
            p["sales"][t["id"]] = {"avg_cost": avg, "profit": t["quantity"] * (t["price"] - avg) - t["fee"],
                                   "fifo_profit": t["quantity"] * t["price"] - t["fee"] - fifo_cost}
            p["cost"] -= avg * t["quantity"]
            p["quantity"] -= t["quantity"]
            if p["quantity"] < EPS:
                p["quantity"], p["cost"], p["lots"] = 0.0, 0.0, []
        elif t["side"] == "dividend":
            p["dividends"] += t["price"] - t["fee"]
        else:   # fee
            p["realized"] -= t["price"] + t["fee"]
            p["fifo_realized"] -= t["price"] + t["fee"]
            p["fees"] += t["price"]
    for p in pos.values():
        p["avg_cost"] = p["cost"] / p["quantity"] if p["quantity"] > EPS else None
        p["fifo_cost"] = sum(q * c for q, c in p["lots"])
        p["fifo_avg_cost"] = p["fifo_cost"] / p["quantity"] if p["quantity"] > EPS else None
        p["lots"] = [{"quantity": q, "unit_cost": c} for q, c in p["lots"]]
    return pos, problems


# ---------------------------------------------------------------- quotes

def _money(s) -> Optional[float]:
    if s in (None, "", "N/A"):
        return None
    try:
        return float(str(s).replace("$", "").replace(",", "").replace("+", "").replace("%", "").strip())
    except ValueError:
        return None


def _get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def _nasdaq(symbol: str, endpoint: str, cls: str) -> Optional[Dict[str, Any]]:
    d = _get_json(NASDAQ.format(sym=urllib.parse.quote(symbol.replace(".", "/")), endpoint=endpoint, cls=cls))
    return d.get("data") if (d.get("status") or {}).get("rCode") == 200 else None


def session_now(now: Optional[datetime] = None) -> str:
    """US equity session by the clock (no holiday calendar); Nasdaq's own status wins when known."""
    t = (now or datetime.now(US_EAST)).astimezone(US_EAST)
    if t.weekday() >= 5:
        return "closed"
    hm = t.hour * 60 + t.minute
    if 4 * 60 <= hm < 9 * 60 + 30:
        return "pre"
    if 9 * 60 + 30 <= hm < 16 * 60:
        return "regular"
    if 16 * 60 <= hm < 20 * 60:
        return "post"
    return "closed"


def seconds_until_next_session(now: Optional[datetime] = None) -> int:
    """Seconds until the next pre-market open (4:00 ET on a weekday). No holiday calendar:
    on a holiday the page wakes up, sees Nasdaq report "Closed" and goes back to sleep."""
    from datetime import timedelta
    t = (now or datetime.now(US_EAST)).astimezone(US_EAST)
    nxt = t.replace(hour=4, minute=0, second=0, microsecond=0)
    if nxt <= t:
        nxt += timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return max(60, int((nxt - t).total_seconds()))


def fetch_quote(symbol: str, asset_class: Optional[str] = None, known_prev_close: Optional[float] = None) -> Dict[str, Any]:
    """One live quote from Nasdaq. Raises on failure (the caller falls back to the stored one).

    Each Nasdaq call takes 1-3 s (measured 2026-10-20), so the calls run in parallel rather than
    one after another: the price and the previous close together, and when the asset class is
    not known yet, the stock and ETF variants together (sequentially an ETF took 6.5 s).
    The previous close changes once a day; a caller that already has today's passes it in."""
    classes = [asset_class] if asset_class else ["stocks", "etf"]
    calls = [(cls, "info") for cls in classes]
    if known_prev_close is None:
        calls += [(cls, "summary") for cls in classes]

    def call(item):
        cls, ep = item
        reply = None
        for _ in range(STALE_RETRIES if ep == "info" else 1):
            try:
                reply = _nasdaq(symbol, ep, cls)
            except Exception:  # noqa: BLE001 - a failed variant is simply not the answer
                reply = None
            if ep != "info" or not _frozen_reply(reply):
                break
        return item, reply

    with ThreadPoolExecutor(max_workers=len(calls)) as ex:
        replies = dict(ex.map(call, calls))
    for cls in classes + [c for c in ("stocks", "etf") if c not in classes]:
        info = replies.get((cls, "info"))
        if (cls, "info") not in replies and asset_class:
            info = call((cls, "info"))[1]          # the remembered class stopped answering
        if not info or not info.get("primaryData"):
            continue
        primary, secondary = info["primaryData"], info.get("secondaryData") or {}
        status = str(info.get("marketStatus") or "").strip().lower()
        session = _STATUS_TO_SESSION.get(status, session_now())
        prev_close = known_prev_close
        if prev_close is None:
            summary = replies.get((cls, "summary")) or {}
            prev_close = _money(((summary.get("summaryData") or {}).get("PreviousClose") or {}).get("value"))
        if _frozen_reply(info):
            raise StaleQuote(f"{symbol}: Nasdaq kept answering from a stale copy")
        price = _money(primary.get("lastSalePrice"))
        if price is None:
            raise ValueError(f"{symbol}: no price in Nasdaq reply")
        price, price_basis = _sane_extended_price(price, session, _money(primary.get("bidPrice")), _money(primary.get("askPrice")))
        return {
            "symbol": symbol, "asset_class": cls, "name": info.get("companyName") or symbol,
            "price": price, "session": session, "market_status": info.get("marketStatus"),
            "trade_time": primary.get("lastTradeTimestamp"),
            "session_change": _money(primary.get("netChange")), "session_change_pct": _money(primary.get("percentageChange")),
            # During pre-market / after-hours Nasdaq puts the regular session's last price here.
            "regular_price": _money(secondary.get("lastSalePrice")), "regular_time": secondary.get("lastTradeTimestamp"),
            "prev_close": prev_close, "prev_close_date": datetime.now(US_EAST).date().isoformat(), "source": "nasdaq",
            "price_basis": price_basis, "volume": _money(primary.get("volume")),
        }
    raise ValueError(f"Nasdaq: symbol {symbol} not found")


# Nasdaq answers from several servers and one of them can be frozen: on 2026-10-21 at 9:50 ET it
# kept returning INTC $91.215 with bid/ask "N/A" and volume stuck at 10,644,128 while live replies
# had a bid/ask and a volume of 13.7M, and the lastTradeTimestamp looked current either way. During
# the regular session a live reply always carries a quote, so one without it is retried, and a batch
# row whose volume went backwards is fetched again on its own.
STALE_RETRIES = 3


class StaleQuote(ValueError):
    pass


def _frozen_reply(info: Optional[Dict[str, Any]]) -> bool:
    if not info or not info.get("primaryData"):
        return False
    status = str(info.get("marketStatus") or "").strip().lower()
    if _STATUS_TO_SESSION.get(status) != "regular":
        return False
    p = info["primaryData"]
    return _money(p.get("bidPrice")) is None and _money(p.get("askPrice")) is None


# Pre-market / after-hours trading is thin and Nasdaq reports odd-lot and out-of-sequence prints as
# the last sale: TSM after hours on 2026-10-20 went $430.50, $430.40, then $399.55 (-6%) while the
# quote stayed $430.00 / $430.50. Outside the regular session the last sale is therefore checked
# against the quote and replaced by the bid/ask midpoint when it falls outside it.
EXT_QUOTE_TOLERANCE = 0.005      # a print up to 0.5% outside the bid/ask still counts
EXT_MAX_SPREAD = 0.02            # a quote wider than 2% of its midpoint is too thin to judge by


def _sane_extended_price(price: float, session: str, bid: Optional[float], ask: Optional[float]):
    if session not in ("pre", "post") or not bid or not ask or bid <= 0 or ask < bid:
        return price, "last"
    mid = (bid + ask) / 2
    if (ask - bid) / mid > EXT_MAX_SPREAD:
        return price, "last"
    if bid * (1 - EXT_QUOTE_TOLERANCE) <= price <= ask * (1 + EXT_QUOTE_TOLERANCE):
        return price, "last"
    return round(mid, 4), "mid"


# Regular session: one request for many symbols. Nasdaq's site watchlist endpoint answered 20 of
# 22 symbols in 1.7 s (2026-10-20) where per-symbol calls took 5-8 s for the same set. Whether it
# carries pre-market / after-hours prices is not verified yet, so those sessions stay per-symbol.
BATCH_SIZE = 25
_BATCH_MISSES: set = set()        # symbols the batch reply left out; fetched per symbol in parallel
_BATCH_STATUS = {"market open": "regular", "pre market": "pre", "pre-market": "pre", "after hours": "post",
                 "after-hours": "post", "market closed": "closed", "closed": "closed"}


EXT_JUMP = 0.03                  # an unconfirmed extended-hours move larger than this waits one refresh


def _hold_back_jump(new: Dict[str, Any], old: Dict[str, Any]) -> Dict[str, Any]:
    """Outside the regular session, when no bid/ask vouched for the price, a jump of more than
    EXT_JUMP from the previous price is shown only once the next refresh confirms it."""
    if new.get("session") not in ("pre", "post") or new.get("price_basis") == "mid" or not old.get("price"):
        return new
    prev, cand = old["price"], new["price"]
    if abs(cand - prev) / prev <= EXT_JUMP:
        return {**new, "pending_jump": None}
    pending = old.get("pending_jump")
    if pending and abs(cand - pending) / pending <= 0.01:      # second sighting of the same level: real
        return {**new, "pending_jump": None}
    return {**new, "price": prev, "pending_jump": cand, "price_basis": "held"}


def fetch_quotes_batch(items: List[tuple]) -> Dict[str, Dict[str, Any]]:
    """{symbol: quote} for [(symbol, asset_class or None)], via the watchlist endpoint.
    Symbols missing from the reply are simply absent; the caller fetches them one by one."""
    def chunk_fetch(chunk):
        qs = "&".join("symbol=" + urllib.parse.quote(f"{s.lower().replace('.', '/')}|{cls or 'stocks'}") for s, cls in chunk)
        d = _get_json("https://api.nasdaq.com/api/quote/watchlist?" + qs)
        return d.get("data") or [] if (d.get("status") or {}).get("rCode") == 200 else []

    chunks = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(chunks))) as ex:
        for part in ex.map(chunk_fetch, chunks):
            rows += part
    today = datetime.now(US_EAST).date().isoformat()
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sym = str(r.get("symbol") or "").upper().replace("/", ".")
        price = _money(r.get("lastSalePrice"))
        if not sym or price is None:
            continue
        status = str(r.get("marketStatus") or "").strip().lower()
        prev = r.get("previousClosePrice")
        out[sym] = {
            "symbol": sym, "asset_class": "etf" if str(r.get("assetClass") or "").upper() == "ETF" else "stocks",
            "name": r.get("companyName") or sym, "price": price,
            "session": _BATCH_STATUS.get(status, session_now()), "market_status": r.get("marketStatus"),
            "trade_time": r.get("lastTradeTimestamp"),
            "session_change": _money(r.get("netChange")), "session_change_pct": _money(r.get("percentageChange")),
            "regular_price": None, "regular_time": None,
            "prev_close": _money(prev) if prev is not None else None, "prev_close_date": today, "source": "nasdaq-batch",
            "volume": _money(r.get("volume")),
        }
    return out


def get_quotes(symbols: List[str], force: bool = False, path: Optional[Path] = None,
               cached_only: bool = False, ttl_by_symbol: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, Any]]:
    """Quotes for the symbols, refreshed when older than their session's TTL; on any failure
    the last stored quote is served with stale=True and the error attached.

    cached_only: no network at all. Stored quotes are served as they are and a symbol never
    quoted comes back with pending=True. Used after a save, which must answer in milliseconds;
    the page fetches live prices right after."""
    if not symbols:
        return {}
    conn = _conn(path)
    try:
        stored = {r["symbol"]: dict(r) for r in conn.execute(
            f"SELECT * FROM quotes WHERE symbol IN ({','.join('?' * len(symbols))})", symbols)}
    finally:
        conn.close()
    now, clock_session = time.time(), session_now()
    out: Dict[str, Dict[str, Any]] = {}
    todo = []
    for s in symbols:
        row = stored.get(s)
        if row:
            q = json.loads(row["data"])
            ttl = (ttl_by_symbol or {}).get(s, refresh_interval(clock_session, QUOTE_TTL) or 60)
            if cached_only or (not force and now - row["fetched_at"] < ttl):
                out[s] = {**q, "fetched_at": row["fetched_at"], "stale": False}
                continue
        known = None
        if row:
            q = json.loads(row["data"])
            if q.get("prev_close_date") == datetime.now(US_EAST).date().isoformat():
                known = q.get("prev_close")
        if cached_only:
            out[s] = {"symbol": s, "price": None, "stale": False, "pending": True}
            continue
        todo.append((s, row["asset_class"] if row else None, known))

    def work(item):
        s, cls, known = item
        try:
            return s, fetch_quote(s, cls, known), None
        except Exception as e:  # noqa: BLE001
            return s, None, f"{type(e).__name__}: {e}"

    results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        if todo and clock_session == "regular":
            # Symbols the batch endpoint has left out before (SPY, WDC on 2026-10-20) start their
            # per-symbol fetch alongside the batch instead of after it.
            known_missing = [t for t in todo if t[0] in _BATCH_MISSES]
            early = [ex.submit(work, t) for t in known_missing]
            asked = [t for t in todo if t[0] not in _BATCH_MISSES]
            try:
                batch = fetch_quotes_batch([(s, cls) for s, cls, _ in asked]) if asked else {}
            except Exception:  # noqa: BLE001 - the per-symbol path below covers everything then
                batch = {}
            # A row whose volume has not grown since the stored quote came from the frozen copy
            # (a live regular session trades within seconds; a thin stock just costs one extra call).
            def older(s):
                prev = json.loads(stored[s]["data"]).get("volume") if stored.get(s) else None
                new = batch[s].get("volume")
                return bool(prev and new and new <= prev)
            behind = {s for s in batch if older(s)}
            results = [(s, batch[s], None) for s, _, _ in asked if s in batch and s not in behind]
            late = [t for t in asked if t[0] not in batch or t[0] in behind]
            if batch:                                   # an empty reply is an outage, not a verdict
                _BATCH_MISSES.update(t[0] for t in late if t[0] not in batch)
            results += [f.result() for f in early] + list(ex.map(work, late))
        else:
            results = list(ex.map(work, todo))
    conn = _conn(path)
    try:
        for s, q, err in results:
            if q and stored.get(s):
                q = _hold_back_jump(q, json.loads(stored[s]["data"]))
            if q:
                conn.execute("INSERT INTO quotes(symbol, asset_class, data, fetched_at) VALUES (?, ?, ?, ?) "
                             "ON CONFLICT(symbol) DO UPDATE SET asset_class=excluded.asset_class, data=excluded.data, "
                             "fetched_at=excluded.fetched_at", (s, q["asset_class"], json.dumps(q), now))
                out[s] = {**q, "fetched_at": now, "stale": False}
            elif stored.get(s):
                out[s] = {**json.loads(stored[s]["data"]), "fetched_at": stored[s]["fetched_at"], "stale": True, "error": err}
            else:
                out[s] = {"symbol": s, "price": None, "stale": True, "error": err}
        conn.commit()
    finally:
        conn.close()
    return out


# ---------------------------------------------------------------- K-line charts
# 1D: today's minute prices from Nasdaq (pre-market from 4:00 ET to after-hours) folded into
# 5-minute candles. Longer ranges: daily OHLC from Nasdaq's historical endpoint, plus today's
# candle built from the minute data, since the historical rows stop at yesterday.
CHART_RANGES = {"1d": None, "1m": 31, "3m": 92, "6m": 183, "1y": 366, "5y": 1830}
CHART_TTL = {"1d": 60, "daily": 600}
INTRADAY_BUCKET_MIN = 5


def _asset_class(symbol: str, path: Optional[Path] = None) -> Optional[str]:
    conn = _conn(path)
    try:
        row = conn.execute("SELECT asset_class FROM quotes WHERE symbol = ?", (symbol,)).fetchone()
    finally:
        conn.close()
    return row["asset_class"] if row else None


def _intraday(symbol: str, cls: str) -> Dict[str, Any]:
    data = _nasdaq(symbol, "chart", cls + "&charttype=rs") or {}
    pts = [(p["x"] // 1000, float(p["y"]), float(p.get("w") or 0)) for p in data.get("chart") or [] if p.get("y") is not None]
    pts = _drop_spikes(pts)
    candles: List[Dict[str, Any]] = []
    step = INTRADAY_BUCKET_MIN * 60
    for ts, price, vol in pts:
        b = ts - ts % step
        if candles and candles[-1]["time"] == b:
            c = candles[-1]
            c["high"], c["low"], c["close"] = max(c["high"], price), min(c["low"], price), price
            c["volume"] += vol
        else:
            candles.append({"time": b, "open": price, "high": price, "low": price, "close": price, "volume": vol})
    # Nasdaq's "x" is US-Eastern wall-clock time encoded as if it were UTC (the 4:00 AM point is
    # 04:00 UTC). Read it as UTC to get the ET clock; the chart shows it as ET the same way.
    from datetime import timezone
    for c in candles:
        t = datetime.fromtimestamp(c["time"], timezone.utc)
        hm = t.hour * 60 + t.minute
        c["session"] = "pre" if hm < 570 else "post" if hm >= 960 else "regular"
    prev = _money(data.get("chartPrevClose")) if data.get("chartPrevClose") is not None else None
    return {"candles": candles, "prev_close": prev}


SPIKE = 0.02


def _drop_spikes(pts: List[tuple]) -> List[tuple]:
    """Remove isolated minute prints (the $411.51 among $429s at 4:55 PM): a point is dropped only
    when the minutes before it and the minutes after it agree with each other and it is more than
    SPIKE away from both. A real move changes the level on one side, so it is kept, and the first
    and last points (one side only) are never judged."""
    from statistics import median
    out = []
    for i, p in enumerate(pts):
        left = [q[1] for q in pts[max(0, i - 2):i]]
        right = [q[1] for q in pts[i + 1:i + 3]]
        if left and right:
            lm, rm = median(left), median(right)
            ref = (lm + rm) / 2
            if ref and abs(lm - rm) / ref <= SPIKE and abs(p[1] - lm) / lm > SPIKE and abs(p[1] - rm) / rm > SPIKE:
                continue
        out.append(p)
    return out


def _daily(symbol: str, cls: str, days: int) -> List[Dict[str, Any]]:
    today = datetime.now(US_EAST).date()
    frm = date.fromordinal(today.toordinal() - days)
    data = _nasdaq(symbol, f"historical", f"{cls}&fromdate={frm.isoformat()}&todate={today.isoformat()}&limit=9999") or {}
    rows = ((data.get("tradesTable") or {}).get("rows")) or []
    out = []
    for r in rows:
        try:
            d = datetime.strptime(r["date"], "%m/%d/%Y").date().isoformat()
        except (KeyError, ValueError):
            continue
        o, h, l, c = (_money(r.get(k)) for k in ("open", "high", "low", "close"))
        if None in (o, h, l, c):
            continue
        out.append({"time": d, "open": o, "high": h, "low": l, "close": c, "volume": _money(r.get("volume")) or 0})
    return sorted(out, key=lambda x: x["time"])


def get_chart(symbol: str, rng: str = "1d", path: Optional[Path] = None) -> Dict[str, Any]:
    symbol = re.sub(r"\s+", "", str(symbol or "")).upper()
    if rng not in CHART_RANGES:
        raise ValueError(f"unknown range {rng!r}")
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-/]{0,9}", symbol):
        raise ValueError("bad symbol")
    ttl = CHART_TTL["1d" if rng == "1d" else "daily"]
    conn = _conn(path)
    try:
        row = conn.execute("SELECT data, fetched_at FROM charts WHERE symbol = ? AND range = ?", (symbol, rng)).fetchone()
    finally:
        conn.close()
    if row and time.time() - row["fetched_at"] < ttl:
        return {**json.loads(row["data"]), "cached": True}
    known = _asset_class(symbol, path)
    last_err = None
    for cls in ([known] if known else []) + [c for c in ("stocks", "etf") if c != known]:
        try:
            intraday = _intraday(symbol, cls)
            if rng == "1d":
                if not intraday["candles"]:
                    continue
                result = {"symbol": symbol, "range": rng, "interval": f"{INTRADAY_BUCKET_MIN}m", **intraday}
            else:
                candles = _daily(symbol, cls, CHART_RANGES[rng])
                if not candles:
                    continue
                reg = [c for c in intraday["candles"] if c["session"] == "regular"]
                today = datetime.now(US_EAST).date().isoformat()
                if reg and candles[-1]["time"] < today:
                    candles.append({"time": today, "open": reg[0]["open"], "high": max(c["high"] for c in reg),
                                    "low": min(c["low"] for c in reg), "close": reg[-1]["close"],
                                    "volume": sum(c["volume"] for c in reg), "partial": True})
                result = {"symbol": symbol, "range": rng, "interval": "1d", "candles": candles}
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    else:
        if row:
            return {**json.loads(row["data"]), "cached": True, "stale": True}
        raise ValueError(f"No chart data for {symbol}{f': {last_err}' if last_err else ''}")
    conn = _conn(path)
    try:
        conn.execute("INSERT INTO charts(symbol, range, data, fetched_at) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, range) "
                     "DO UPDATE SET data=excluded.data, fetched_at=excluded.fetched_at",
                     (symbol, rng, json.dumps(result), time.time()))
        conn.commit()
    finally:
        conn.close()
    return {**result, "cached": False}


def _usd_cny() -> Optional[float]:
    """USD -> CNY at the budget's rate (job_agent), so the two apps agree; None if unknown."""
    return budget_link.usd_cny()


# One background refresh at a time. The page never waits on Nasdaq (20 symbols took ~8 s
# synchronously on 2026-10-20): it gets the stored quotes at once, this thread brings them up
# to date, and the page polls again a couple of seconds later to pick the new prices up.
_refresh_lock = threading.Lock()
_refresh_state = {"running": False, "finished_at": 0.0}


def refresh_in_background(symbols: List[str], force: bool = False, path: Optional[Path] = None,
                          ttl_by_symbol: Optional[Dict[str, float]] = None) -> bool:
    """Start refreshing these symbols' quotes unless a refresh is already running. True if one is."""
    with _refresh_lock:
        if _refresh_state["running"]:
            return True
        if not symbols:
            return False
        _refresh_state["running"] = True

    def run():
        try:
            get_quotes(symbols, force=force, path=path, ttl_by_symbol=ttl_by_symbol)
        except Exception:  # noqa: BLE001 - a failed round leaves the stored quotes in place
            pass
        finally:
            with _refresh_lock:
                _refresh_state["running"] = False
                _refresh_state["finished_at"] = time.time()

    threading.Thread(target=run, name="portfolio-quotes", daemon=True).start()
    return True


def get_state(force_quotes: bool = False, path: Optional[Path] = None, cached_quotes: bool = False,
              background_refresh: bool = False) -> Dict[str, Any]:
    """background_refresh: answer from stored quotes at once and, if any is out of date (or
    force_quotes), refresh them in a background thread; "refreshing" tells the page to poll soon."""
    if background_refresh:
        cached_quotes = True
    conn = _conn(path)
    try:
        txs = _transactions(conn)
        watch_rows = [dict(r) for r in conn.execute("SELECT * FROM watchlist ORDER BY sort, symbol")]
    finally:
        conn.close()
    positions, problems = compute_holdings(txs)
    open_syms = sorted(s for s, p in positions.items() if p["quantity"] > EPS)
    # A watched symbol that is now held shows in the holdings table only.
    watch_rows = [w for w in watch_rows if w["symbol"] not in open_syms]
    all_syms = open_syms + [w["symbol"] for w in watch_rows]
    quotes = get_quotes(all_syms, force=force_quotes and not cached_quotes, path=path, cached_only=cached_quotes)
    refreshing = False
    if background_refresh:
        clock = session_now()
        ttl_by_symbol = {s: refresh_interval(clock, QUOTE_TTL) or 60 for s in open_syms}
        ttl_by_symbol.update({w["symbol"]: refresh_interval(clock, WATCH_TTL) or 60 for w in watch_rows})
        out_of_date = any(q.get("pending") or time.time() - (q.get("fetched_at") or 0) >= ttl_by_symbol.get(s, 60)
                          for s, q in quotes.items())
        if force_quotes or out_of_date:
            refreshing = refresh_in_background(all_syms, force=force_quotes, path=path, ttl_by_symbol=ttl_by_symbol)
    watchlist = []
    for w in watch_rows:
        q = quotes.get(w["symbol"]) or {}
        pc, price = q.get("prev_close"), q.get("price")
        watchlist.append({**w, "quote": q,
                          "day_change": price - pc if pc and price is not None else None,
                          "day_change_pct": (price - pc) / pc * 100 if pc and price is not None else None})

    holdings, closed = [], []
    tot = {"market_value": 0.0, "cost": 0.0, "day_pnl": 0.0, "realized": 0.0, "dividends": 0.0, "priced": 0,
           "fifo_cost": 0.0, "fifo_realized": 0.0}
    for s, p in positions.items():
        tot["realized"] += p["realized"]
        tot["fifo_realized"] += p["fifo_realized"]
        tot["dividends"] += p["dividends"]
        if p["quantity"] <= EPS:
            closed.append(p)
            continue
        q = quotes.get(s) or {}
        price = q.get("price")
        h = {**p, "quote": q}
        if price is not None:
            h["market_value"] = price * p["quantity"]
            h["unrealized"] = h["market_value"] - p["cost"]
            h["unrealized_pct"] = h["unrealized"] / p["cost"] * 100 if p["cost"] > EPS else None
            h["fifo_unrealized"] = h["market_value"] - p["fifo_cost"]
            tot["fifo_cost"] += p["fifo_cost"]
            pc = q.get("prev_close")
            h["day_change"] = price - pc if pc else None
            h["day_change_pct"] = (price - pc) / pc * 100 if pc else None
            h["day_pnl"] = h["day_change"] * p["quantity"] if pc else None
            tot["market_value"] += h["market_value"]
            tot["cost"] += p["cost"]
            tot["day_pnl"] += h["day_pnl"] or 0.0
            tot["priced"] += 1
        holdings.append(h)
    for h in holdings:
        h["weight"] = h.get("market_value", 0) / tot["market_value"] * 100 if tot["market_value"] > EPS else None
    holdings.sort(key=lambda h: -(h.get("market_value") or 0))
    closed.sort(key=lambda p: p["symbol"])
    tot["unrealized"] = tot["market_value"] - tot["cost"]
    tot["fifo_unrealized"] = tot["market_value"] - tot["fifo_cost"]
    tot["unrealized_pct"] = tot["unrealized"] / tot["cost"] * 100 if tot["cost"] > EPS else None
    prev_value = tot["market_value"] - tot["day_pnl"]
    tot["day_pnl_pct"] = tot["day_pnl"] / prev_value * 100 if prev_value > EPS else None

    sessions = [q.get("session") for q in quotes.values() if q.get("session") and not q.get("stale")]
    session = max(set(sessions), key=sessions.count) if sessions else session_now()
    return {
        "transactions": sorted(txs, key=lambda t: (t["trade_date"], t["id"]), reverse=True),
        "holdings": holdings, "closed": closed, "watchlist": watchlist, "totals": tot, "problems": problems,
        "session": session, "session_label": SESSIONS.get(session, session),
        "refresh_seconds": refresh_interval(session, REFRESH_SECONDS),
        "watch_refresh_seconds": refresh_interval(session, WATCH_TTL) if refresh_interval(session, REFRESH_SECONDS) else None,
        "next_session_in": seconds_until_next_session() if refresh_interval(session, REFRESH_SECONDS) is None else None,
        "us_time": datetime.now(US_EAST).strftime("%H:%M"), "usd_cny": _usd_cny(),
        "tx_types": TX_TYPES, "server_time": time.time(), "refreshing": refreshing,
        "global": global_markets.global_state(path, background=background_refresh),
    }
