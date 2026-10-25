"""Live earnings pulse for the holdings (PROJECT_PLAN Q.5 v4, 2026-10-25).

Right after an after-the-close release a stock often spikes for minutes and fades before or during
the call. quant_data's research/models/earnings_pulse.py learned from 2,490 past reports when to
put a three-step sell ladder into that spike and how to buy the shares back (walk-forward
2019-2026, strict costs: +0.64% per acted report, 59% wins; INTC +1.65% on 13 of 30). This module
runs it on report days:

  15:55 ET    a holding reports after the close today: the watch starts; the pre-release context
              (run-up, volatility, typical report move, news) comes from quant_data
  every 5 s   the quote; the release minute is the first after 16:00 with a 0.5% move
  placements  1/3/5/10/20/30 min after the release, 5 min before the call: the model scores every
              step it could put in now (12 levels x 7 buy-back rules) for gain and downside and
              composes the plan (how many steps, which levels, what weights); the first plan that
              clears its margin, on a stock whose out-of-sample record is positive, becomes orders
  orders      whole shares (PULSE_FRACTION of the position, split by the plan's weights), limits,
              saved with the advisor's orders (tag pulse, the buy-backs waiting on their sales) and
              announced by an urgent notification: the move lasts minutes, no report is written first

Advice only: nothing is traded. The broker must accept extended-hours limit orders.
"""

import json
import math
import os
import subprocess
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from . import advisor, events
from . import portfolio as pf
from .config import DB_PATH

NY = ZoneInfo("America/New_York")
DUBLIN = ZoneInfo("Europe/Dublin")
QUANT_DATA = Path(os.getenv("QUANT_DATA_DIR", "/Users/xiz/Quant_trade/quant_data"))
PREDICT = [str(QUANT_DATA / ".venv311" / "bin" / "python"), str(QUANT_DATA / "research" / "models" / "earnings_pulse.py"), "predict"]
POLL = 5
RELEASE_MOVE = 0.5                     # % from the close that marks the release minute (as in training)
AFTER_RELEASE = (1, 3, 5, 10, 20, 30)
DEFAULT_CALL = (17, 0)
FRACTION = float(os.getenv("PULSE_FRACTION", "0.3"))     # of the whole shares, over the three steps
WINDOW_END = (18, 0)
LOG_DIR = Path(os.getenv("PULSE_LOG_DIR", str(Path(DB_PATH).parent / "pulse")))
BUYBACK_TEXT = {"c+5": "5 minutes into the call", "c+30": "30 minutes into the call", "ah_end": "at 19:55 ET, before after-hours ends",
                "next_open": "at the next regular open, 09:30 ET"}
_state = {"day": None, "thread": None}


def _predict(req: Dict[str, Any]) -> Dict[str, Any]:
    r = subprocess.run(PREDICT, input=json.dumps(req), capture_output=True, text=True, timeout=120, cwd=str(QUANT_DATA))
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-300:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def reporting_today(held: List[Dict[str, Any]], today: date, cal_fn=None) -> List[Dict[str, Any]]:
    syms = [h["symbol"] for h in held]
    cal = (cal_fn or (lambda ss: events.upcoming(ss, days=1, start=today).get("earnings") or []))(syms)
    # A US release after the close only: SK hynix's "after the close (Seoul morning)" comes out around
    # 19:00-20:00 ET, outside what the model learned.
    return [e for e in cal if e.get("symbol") in syms and str(e.get("date"))[:10] == today.isoformat()
            and "after" in str(e.get("when") or "").lower() and "seoul" not in str(e.get("when") or "").lower()]


def shares_for(quantity: float, weights: List[float], fraction: float = FRACTION) -> List[int]:
    """Whole shares per step: `fraction` of the whole position split by the policy's weights (largest
    remainders first), at least one share per step while the budget allows; steps that get none are
    dropped by the caller."""
    whole = int(math.floor(quantity))
    if whole < 1 or not weights:
        return []
    budget = max(1, min(whole, int(whole * fraction)))
    w = [x / sum(weights) for x in weights]
    raw = [budget * x for x in w]
    out = [int(math.floor(x)) for x in raw]
    for i in sorted(range(len(w)), key=lambda i: raw[i] - out[i], reverse=True)[:budget - sum(out)]:
        out[i] += 1
    if 0 in out and budget >= len(out):                 # every step gets a share before any gets two
        for i in [i for i, n in enumerate(out) if n == 0]:
            j = max(range(len(out)), key=lambda j: out[j])
            out[j] -= 1
            out[i] += 1
    return out


class Watch:
    """One holding's report evening. step() is fed (time, price); it decides when to ask the model and
    returns the orders when a plan clears. Feed and clock are injected, so a past evening replays."""

    def __init__(self, symbol: str, quantity: float, avg_cost: Optional[float], close: float, day: date,
                 context: Dict[str, Any], predict: Callable[[Dict[str, Any]], Dict[str, Any]] = _predict,
                 call: tuple = DEFAULT_CALL):
        self.symbol, self.quantity, self.avg_cost, self.close, self.day = symbol, quantity, avg_cost, close, day
        self.context, self.predict = context, predict
        self.ticks: List[tuple] = []
        self.release: Optional[datetime] = None
        self.done: set = set()
        self.decisions: List[Dict[str, Any]] = []
        self.orders: Optional[Dict[str, Any]] = None
        self.call = datetime(day.year, day.month, day.day, *call, tzinfo=NY)

    def _at(self, h: int, m: int) -> datetime:
        return datetime(self.day.year, self.day.month, self.day.day, h, m, tzinfo=NY)

    def placements(self) -> Dict[str, datetime]:
        out: Dict[str, datetime] = {}                  # the policy puts nothing in before the release (2026-10-26)
        if self.release:
            out.update({f"r+{k}": self.release + timedelta(minutes=k) for k in AFTER_RELEASE})
        out["c-5"] = self.call - timedelta(minutes=5)
        return out

    def step(self, now: datetime, price: Optional[float], regular_close: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if regular_close:
            self.close = regular_close                 # the official 16:00 close once the quote has it
        if price:
            if now < self._at(16, 0):
                self.close = price                     # until the bell, the close is the last price
            self.ticks.append((now, price))
            if self.release is None and now >= self._at(16, 0) and abs(price / self.close - 1) * 100 >= RELEASE_MOVE:
                self.release = now.replace(second=0, microsecond=0)
        if self.orders:
            return None
        for name, when in sorted(self.placements().items(), key=lambda kv: kv[1]):
            if name in self.done or now < when:
                continue
            self.done.add(name)
            if name != "pre" and name != "c-5" and self.release is None:
                continue
            after = [p for t, p in self.ticks if t >= self._at(16, 0)] or [self.close]
            last = self.ticks[-1][1] if self.ticks else self.close
            req = {"symbol": self.symbol, "place": name, "move_at": (last / self.close - 1) * 100,
                   "maxmove_at": (max(after) / self.close - 1) * 100 if name != "pre" else 0.0,
                   "minmove_at": (min(after) / self.close - 1) * 100 if name != "pre" else 0.0,
                   "release_min": self.release.hour * 60 + self.release.minute if self.release and name != "pre" else None,
                   **{k: self.context.get(k) for k in ("run5", "run20", "vol20", "past_move_med", "news_pre_mean")}}
            try:
                ans = self.predict(req)
            except Exception as e:  # noqa: BLE001 - the next placement asks again
                ans = {"error": f"{type(e).__name__}: {e}"}
            self.decisions.append({"at": now.isoformat(), "request": req, "answer": ans})
            if ans.get("act"):
                self.orders = self.make_orders(ans, last, now)
                return self.orders
        return None

    def make_orders(self, ans: Dict[str, Any], p0: float, now: datetime) -> Dict[str, Any]:
        best = ans["best"]
        sizes = shares_for(self.quantity, best.get("weights") or [1.0] * len(best["ladder"]))
        sells = [{"shares": n, "limit": round(p0 * (1 + lev / 100), 2), "step_pct": lev}
                 for n, lev in zip(sizes, best["ladder"]) if n > 0]
        bb = best["buyback"]
        if bb.startswith("lim"):
            b = int(bb[3:])
            rebuy = [{"shares": s["shares"], "limit": round(s["limit"] * (1 - b / 100), 2), "after_sale": i} for i, s in enumerate(sells)]
            how = f"each filled step bought back {b}% below its sale; anything not bought back by then at the next open"
        else:
            rebuy = [{"shares": s["shares"], "limit": s["limit"], "after_sale": i} for i, s in enumerate(sells)]
            when = self.call + timedelta(minutes=5 if bb == "c+5" else 30) if bb.startswith("c+") else None
            how = f"buy back what filled {BUYBACK_TEXT[bb]}" + (f" ({when:%H:%M} ET, {when.astimezone(DUBLIN):%H:%M} Dublin)" if when else "") + \
                  ", at market"
        rec = ans.get("stock_record") or {}
        return {"symbol": self.symbol, "at": now.isoformat(), "price": p0, "plan": best["plan"], "pred_pct": best["pred_pct"],
                "sells": sells, "rebuys": rebuy, "buyback": bb, "how": how, "record": rec, "model_record": ans.get("model_record")}


# ---------------------------------------------------------------- orders, notification, log

def save(orders: Dict[str, Any], day: date, path: Optional[Path] = None) -> None:
    until = events_next_session(day).isoformat()
    rec = orders["record"]
    note = (f"pulse: {orders['plan']}; predicted {orders['pred_pct']:+.2f}% of the price on the ladder; {orders['symbol']} "
            f"out of sample: {rec.get('acted')} of {rec.get('reports')} reports acted, {rec.get('mean_when_acting', 0):+.2f}%, "
            f"{(rec.get('win') or 0):.0%} wins")
    legs: List[Dict[str, Any]] = []
    for s in orders["sells"]:
        legs.append({"symbol": orders["symbol"], "side": "sell", "tag": "pulse", "pct": None, "shares": s["shares"],
                     "amount_usd": round(s["shares"] * s["limit"], 2), "limit_price": s["limit"], "issued_price": orders["price"],
                     "valid_until": until, "zh": f"盘后限价卖出 {s['shares']} 股 @ {s['limit']}（当前价 +{s['step_pct']}%）",
                     "en": f"Extended-hours limit sell {s['shares']} @ {s['limit']} (+{s['step_pct']}% on {orders['price']:.2f})",
                     "note": note, "plan": "pulse:" + orders["plan"], "parent": None})
    for r in orders["rebuys"]:
        legs.append({"symbol": orders["symbol"], "side": "buy", "tag": "rebuy", "pct": None, "shares": r["shares"],
                     "amount_usd": round(r["shares"] * r["limit"], 2), "limit_price": r["limit"], "issued_price": orders["price"],
                     "valid_until": until, "zh": f"卖出成交后买回 {r['shares']} 股：{orders['how']}",
                     "en": f"After that sale fills, buy back {r['shares']}: {orders['how']}", "note": note,
                     "plan": "pulse:" + orders["plan"], "parent": r["after_sale"]})
    advisor.save_orders(None, legs, datetime.fromisoformat(orders["at"]).timestamp(), path)


def events_next_session(d: date) -> date:
    n = d + timedelta(days=1)
    while n.weekday() >= 5:
        n += timedelta(days=1)
    return n


def notify(orders: Dict[str, Any]) -> None:
    from .briefs import _notify
    s = orders["sells"]
    _notify(f"{orders['symbol']} earnings pulse: orders now",
            "Sell " + ", ".join(f"{x['shares']}@{x['limit']}" for x in s) + f" (ext. hours). Then: {orders['how']}")


def write_log(w: Watch) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{w.day}_{w.symbol}.json").write_text(json.dumps(
        {"symbol": w.symbol, "close": w.close, "release": w.release.isoformat() if w.release else None,
         "ticks": [(t.isoformat(), p) for t, p in w.ticks], "decisions": w.decisions, "orders": w.orders}, default=str))


# ---------------------------------------------------------------- scoring, the next morning (1-minute bars)

ALPACA_BARS = "https://data.alpaca.markets/v2/stocks/{sym}/bars"
SCORE_AFTER = (10, 45)                 # ET on the next session: the latest buy-back (the open) is in the bars


def _alpaca_headers() -> Dict[str, str]:
    from .pipeline_health import QUANT_DATA_ENV
    env = dict(line.strip().split("=", 1) for line in QUANT_DATA_ENV.read_text().splitlines() if "=" in line and not line.startswith("#"))
    return {"APCA-API-KEY-ID": env["ALPACA_API_KEY_ID"].strip(), "APCA-API-SECRET-KEY": env["ALPACA_API_SECRET_KEY"].strip()}


def minute_bars(symbol: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
    """1-minute bars incl. extended hours (SIP: free for anything older than 15 minutes)."""
    import requests
    out, token = [], None
    while True:
        params = {"timeframe": "1Min", "start": start.astimezone(ZoneInfo("UTC")).isoformat(), "limit": 10000,
                  "end": end.astimezone(ZoneInfo("UTC")).isoformat(), "feed": "sip", "adjustment": "raw"}
        if token:
            params["page_token"] = token
        r = requests.get(ALPACA_BARS.format(sym=symbol), headers=_alpaca_headers(), params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        out += [{"t": datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(NY), "o": b["o"], "h": b["h"], "l": b["l"],
                 "c": b["c"]} for b in j.get("bars") or []]
        token = j.get("next_page_token")
        if not token:
            return out


def replay_leg(sell_limit: float, p0: float, issued: datetime, buyback: str, bars: List[Dict[str, Any]],
               call: datetime) -> Dict[str, Any]:
    """One ladder step on the evening's 1-minute bars: when (if) the sale filled, when and at what the
    shares came back, and the gain against holding in % of the price when the orders were given."""
    after = [b for b in bars if b["t"] >= issued.replace(second=0, microsecond=0) + timedelta(minutes=1)]
    sale = next((b for b in after if b["h"] >= sell_limit), None)
    if sale is None:
        return {"filled": False}
    sell = max(sell_limit, sale["o"])
    later = [b for b in after if b["t"] > sale["t"]]
    nxt_open = next((b for b in later if b["t"].date() > issued.date() and b["t"].hour * 60 + b["t"].minute >= 570), None)
    buy_bar = None
    if buyback.startswith("lim"):
        back = sell * (1 - int(buyback[3:]) / 100)
        buy_bar = next((b for b in later if (nxt_open is None or b["t"] <= nxt_open["t"]) and b["l"] <= back), None)
        buy = min(back, buy_bar["o"]) if buy_bar else None
    else:
        when = {"c+5": call + timedelta(minutes=5), "c+30": call + timedelta(minutes=30),
                "ah_end": issued.replace(hour=19, minute=55, second=0, microsecond=0)}.get(buyback)
        buy_bar = next((b for b in later if when and b["t"] >= when and (nxt_open is None or b["t"] < nxt_open["t"])), None)
        buy = buy_bar["o"] if buy_bar else None
    if buy is None:
        if nxt_open is None:
            return {"filled": True, "sold_at": sale["t"], "sell": sell, "pending": True}
        buy_bar, buy = nxt_open, nxt_open["o"]
    return {"filled": True, "sold_at": sale["t"], "sell": sell, "bought_at": buy_bar["t"], "buy": buy,
            "gain_pct": round((sell - buy) / p0 * 100, 3)}


def score_due(path: Optional[Path] = None, now: Optional[datetime] = None, bars_fn=None) -> int:
    """The morning after a report evening: replay each pulse order on that evening's 1-minute bars
    and store the result with the order (sale filled or expired; buy-back scored or void)."""
    now = now or datetime.now(NY)
    conn = advisor._conn(path)
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM advice_orders WHERE plan LIKE 'pulse:%' AND status IN ('open', 'waiting')")]
    finally:
        conn.close()
    sells = [r for r in rows if r["side"] == "sell"]
    kids = {r["parent_id"]: r for r in rows if r["parent_id"]}
    done = 0
    for s in sells:
        until = date.fromisoformat(s["valid_until"])
        if now < datetime(until.year, until.month, until.day, *SCORE_AFTER, tzinfo=NY):
            continue
        issued = datetime.fromtimestamp(s["issued_ts"], NY)
        try:
            bars = (bars_fn or minute_bars)(s["symbol"], issued - timedelta(minutes=5),
                                            datetime(until.year, until.month, until.day, 10, 30, tzinfo=NY))
        except Exception as e:  # noqa: BLE001 - tried again on the next tick
            print(f"[pulse] scoring bars for {s['symbol']} failed: {type(e).__name__}: {e}")
            continue
        buyback = s["plan"].split("|")[-1].strip()
        call = issued.replace(hour=DEFAULT_CALL[0], minute=DEFAULT_CALL[1], second=0, microsecond=0)
        r = replay_leg(s["limit_price"], s["issued_price"] or s["limit_price"], issued, buyback, bars, call)
        if r.get("pending"):
            continue
        kid = kids.get(s["id"])
        conn = advisor._conn(path)
        try:
            if not r["filled"]:
                conn.execute("UPDATE advice_orders SET status = 'expired', eval_date = ? WHERE id = ?", (until.isoformat(), s["id"]))
                if kid:
                    conn.execute("UPDATE advice_orders SET status = 'void' WHERE id = ?", (kid["id"],))
            else:
                conn.execute("UPDATE advice_orders SET status = 'scored', filled_date = ?, eval_date = ?, eval_close = ?, edge_pct = ? "
                             "WHERE id = ?", (r["sold_at"].isoformat(), r["bought_at"].isoformat(), r["buy"], r["gain_pct"], s["id"]))
                if kid:
                    conn.execute("UPDATE advice_orders SET status = 'scored', filled_date = ?, eval_close = ? WHERE id = ?",
                                 (r["bought_at"].isoformat(), r["buy"], kid["id"]))
            conn.commit()
        finally:
            conn.close()
        done += 1
    return done


# ---------------------------------------------------------------- the evening loop

def _run_evening(day: date, watches: List[Watch], path: Optional[Path] = None) -> None:
    end = datetime(day.year, day.month, day.day, *WINDOW_END, tzinfo=NY)
    while datetime.now(NY) < end and not all(w.orders for w in watches):
        now = datetime.now(NY)
        try:
            quotes = pf.get_quotes([w.symbol for w in watches], path=path, force=True)
        except Exception as e:  # noqa: BLE001
            print(f"[pulse] quotes failed: {type(e).__name__}: {e}")
            quotes = {}
        for w in watches:
            q = quotes.get(w.symbol) or {}
            price = q.get("price")
            try:
                out = w.step(now, price, q.get("regular_price") if q.get("session") in ("post", "closed") else None)
            except Exception as e:  # noqa: BLE001 - keep watching the others
                print(f"[pulse] {w.symbol} step failed: {type(e).__name__}: {e}")
                out = None
            if out:
                save(out, day, path)
                notify(out)
                print(f"[pulse] {w.symbol}: {out['plan']} -> {out['sells']}")
            write_log(w)
        time.sleep(POLL)
    for w in watches:
        write_log(w)


def run_due(path: Optional[Path] = None, now: Optional[datetime] = None) -> bool:
    """Called every minute by background.py: on a report day, start the evening watch at 15:55 ET."""
    now = now or datetime.now(NY)
    day = now.date()
    if now.weekday() >= 5 or _state["day"] == day or not (dtime_between(now, (15, 55), WINDOW_END)):
        return False
    _state["day"] = day
    st = pf.get_state(path=path)
    held = advisor.followed(st)
    try:
        todays = reporting_today(held, day)
    except Exception as e:  # noqa: BLE001
        print(f"[pulse] earnings calendar failed: {type(e).__name__}: {e}")
        return False
    if not todays:
        return False
    watches = []
    for e in todays:
        h = next(x for x in held if x["symbol"] == e["symbol"])
        q = h.get("quote") or {}
        try:
            ctx = _predict({"symbol": e["symbol"], "context_only": True})
        except Exception as ex:  # noqa: BLE001
            print(f"[pulse] {e['symbol']} context failed: {ex}")
            ctx = {}
        close = q.get("regular_price") or q.get("price") or ctx.get("close")
        watches.append(Watch(e["symbol"], h["quantity"], h.get("avg_cost"), float(close), day, ctx))
    print(f"[pulse] watching {[w.symbol for w in watches]} this evening")
    t = threading.Thread(target=_run_evening, args=(day, watches, path), name="earnings-pulse", daemon=True)
    _state["thread"] = t
    t.start()
    return True


def dtime_between(now: datetime, start: tuple, end: tuple) -> bool:
    m = now.hour * 60 + now.minute
    return start[0] * 60 + start[1] <= m < end[0] * 60 + end[1]
