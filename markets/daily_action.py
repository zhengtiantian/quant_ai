"""Today's action for a holding (PROJECT_PLAN Q.13, 2026-10-31): the three layers in one line.

Layer 1 says whether there is a tested edge (the earnings pulse on report nights; the advisor's
plans only when they pass its significance gate), layer 2 (risk_rules) what the portfolio can take,
layer 3 the limits. Most days the answer is Hold: an action appears only when a rule or a tested
signal calls for one, and each says which.

  earnings  the stock reports tonight or before tomorrow's open: the pulse watch runs, no buys
  trim      the holding is above the symbol cap: sell the excess into a rebound, in two limits at
            +1.5 and +2 x ATR above the price, good for 15 sessions. How often such limits were reached
            after a recent drop of >= 2 x ATR depends on its cause (REBOUND): 39% / 28% after an
            earnings drop of >= 4 x ATR, 65% / 54% after a drop with no report. Selling there rather
            than at today's low is execution, not a forecast that the stock will rise
  add       a buy every rule of the gate allows (cash, caps, spacing, stabilised)
  hold      nothing to do; what blocks a buy is listed

A pilot: only DAILY_ACTION_SYMBOLS (INTC) until the user has judged the result. Each answer is kept
(daily_actions) so it can be scored against doing nothing.
"""

import json
import math
import os
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from . import lot_advisor, risk_rules
from . import portfolio as pf
from .config import DB_PATH

NY = ZoneInfo("America/New_York")
SYMBOLS = [s.strip().upper() for s in os.getenv("DAILY_ACTION_SYMBOLS", "INTC").split(",") if s.strip()]
TRIM_STEPS_ATR = (1.5, 2.0)
# % of >= 2 x ATR drops whose high reached +1.5 / +2 x ATR within 15 sessions, by what caused the drop,
# and the close 15 sessions later against the next open when +1.5 was not reached (102 stocks,
# 2016-2026, 2,455 drops; quant_data research/models/drop_rebound.py, 2026-11-02). Until then one
# pooled 61% / 50% was quoted for every drop. Split by cause, the rates predicted better out of
# sample than the pooled ones at every level (walk-forward 2019-2026); a LightGBM / logistic model
# on 17 features did not, and neither did picking the limit per drop by model.
REBOUND = {
    "earnings_big": {"label": "an earnings drop of >= 4 x ATR", "n": 193, 1.5: 39, 2.0: 28, "unfilled": -4.6},
    "earnings": {"label": "an earnings drop", "n": 509, 1.5: 50, 2.0: 36, "unfilled": -4.3},
    "other": {"label": "a drop without an earnings report", "n": 1753, 1.5: 65, 2.0: 54, "unfilled": -6.3},
}
VALID_SESSIONS = 15


def drop_cause(drop: Optional[Dict[str, Any]]) -> Optional[str]:
    if not drop or drop.get("earnings") is None:      # no recent large drop, or no calendar to tell its cause
        return None
    if drop.get("earnings"):
        return "earnings_big" if (drop.get("drop_atr") or 0) >= 4 else "earnings"
    return "other"
CACHE_SECONDS = 600
_cache: Dict[str, Any] = {"at": 0.0, "out": None}


def decide(sym: str, holding: Dict[str, Any], ex: Dict[str, Any], advice: Dict[str, Any]) -> Dict[str, Any]:
    """One holding's action from its position, the portfolio's exposure and the Q.10 analysis."""
    stats = advice.get("stats") or {}
    price = float((holding.get("quote") or {}).get("price") or 0.0) or None
    atr = float(stats.get("atr14") or 0.0) or None
    qty = float(holding.get("quantity") or 0.0)
    cats = advice.get("catalysts_7d") or []
    events = [c for c in cats if c.get("impact") == "high"]
    reports = [c for c in cats if c.get("bucket") == "direct" and c.get("category") == "earnings" and c.get("days_away", 9) <= 1]
    weight = ex["weights"].get(sym, 0.0)
    out: Dict[str, Any] = {"symbol": sym, "price": price, "atr": atr, "weight_pct": weight, "cluster_pct": ex["cluster_pct"],
                           "orders": [], "reasons": [], "evidence": "", "blocks": []}

    # What a buy would meet, listed whatever the action (the user asks "can I average down?").
    if price:
        lowest = (advice.get("dip_guardrail") or {}).get("anchor_cost")
        probe = risk_rules.gate({"symbol": sym, "side": "buy", "shares": 1, "limit": price}, ex, held=qty, atr=atr,
                                lowest_lot=lowest, stabilized=stats.get("stabilized"), event_7d=bool(events))
        out["blocks"] = probe["notes"]
        out["buy_allowed"] = probe["allowed"]

    if reports:
        out.update(action="earnings", headline="Reports tonight: the earnings pulse watch runs; no new buys",
                   evidence="tested (earnings pulse champion)")
        out["reasons"].append(reports[0].get("title", "Earnings"))
        return out

    n = risk_rules.trim_shares(qty, ex["mv"].get(sym, 0.0), ex["total"])
    if n > 0 and price and atr:
        cause = drop_cause(stats.get("big_drop"))
        rates = REBOUND.get(cause or "", {})
        steps = [math.ceil(n / 2), n // 2] if n > 1 else [n]
        orders = []
        for k, sh in zip(TRIM_STEPS_ATR, steps):
            if sh <= 0:
                continue
            o = risk_rules.gate({"symbol": sym, "side": "sell", "shares": sh, "limit": round(price + k * atr, 2)}, ex, held=qty)
            if o["allowed"]:
                orders.append({**o, "atr_multiple": k, "hit_pct": rates.get(k), "valid_sessions": VALID_SESSIONS})
        out.update(action="trim", orders=orders, drop_cause=cause,
                   headline=f"Trim {sum(o['shares'] for o in orders)} shares into a rebound: "
                            + ", ".join(f"{o['shares']} @ ${o['limit']:.2f}" for o in orders),
                   evidence="rule (symbol cap) + reach rates by drop cause" if cause else "rule (symbol cap)")
        out["reasons"].append(f"{sym} is {weight:.1f}% of the portfolio; the cap is {risk_rules.MAX_SYMBOL_PCT:g}% "
                              f"({n} shares over at today's price)")
        if cause:
            out["reasons"].append(
                f"Limits at +1.5 and +2 x ATR: after {rates['label']} (102 stocks, 2016-2026, {rates['n']} cases) the high "
                f"reached them within 15 sessions in {rates[1.5]}% and {rates[2.0]}% of cases. When +1.5 was not reached, "
                f"the close 15 sessions later averaged {rates['unfilled']}% against selling at the next open")
        elif stats.get("big_drop"):
            out["reasons"].append("Limits at +1.5 and +2 x ATR. The earnings calendar could not say whether the last large "
                                  "drop was a report, so no reach rate is quoted")
        else:
            out["reasons"].append("Limits at +1.5 and +2 x ATR. No drop of >= 2 x ATR in the last 5 sessions, so no measured "
                                  "reach rate applies")
        out["after"] = after = _after_sales(sym, orders, ex, price)
        out["reasons"].append(f"If both fill: cash {after['cash_pct']:.1f}% (target {risk_rules.MIN_CASH_PCT:g}%), "
                              f"chips / memory {after['cluster_pct']:.1f}% (cap {risk_rules.MAX_CLUSTER_PCT:g}%)"
                              + (f"; ${after['cluster_excess']:,.0f} more of chip names to sell to reach the cap"
                                 if after["cluster_excess"] > 0 else ""))
        return out

    if out.get("buy_allowed"):
        out.update(action="add", headline="A buy fits every rule; no tested signal sizes it", evidence="rule")
        return out
    out.update(action="hold", headline="Hold: no rule or tested signal calls for an order", evidence="default")
    return out


def reported_into(sym: str, drop: Dict[str, Any], path: Optional[Path] = None) -> Optional[bool]:
    """Whether the drop session carried an earnings report: after the previous close or before its
    open (Nasdaq's calendar; a report with no time given counts on either day). None if unknown."""
    from . import events
    try:
        before = {r["symbol"]: r["when"] for r in events.earnings_on(date.fromisoformat(drop["prev_date"]), path)}
        same = {r["symbol"]: r["when"] for r in events.earnings_on(date.fromisoformat(drop["date"]), path)}
    except Exception:  # noqa: BLE001 - no calendar: the cause stays unknown
        return None
    return before.get(sym, "before the open") != "before the open" or same.get(sym, "after the close") != "after the close"


def _after_sales(sym: str, orders: List[Dict[str, Any]], ex: Dict[str, Any], price: float) -> Dict[str, Any]:
    """The portfolio once the trim limits fill, at today's prices (shares become cash; the total stays)."""
    sold = sum(o["shares"] for o in orders) * price
    total, cash = ex["total"], ex["cash"] + sold
    cluster = ex["cluster_mv"] - (sold if sym in lot_advisor.SEMI_MEMORY_CLUSTER else 0.0)
    pct = lambda v: round(v / total * 100, 1) if total > 0 else 0.0
    return {"cash_pct": pct(cash), "cluster_pct": pct(cluster),
            "cluster_excess": round(max(0.0, cluster - risk_rules.MAX_CLUSTER_PCT / 100 * total), 2)}


def compute(path: Optional[Path] = None, state: Optional[Dict[str, Any]] = None,
            analyze: Callable[..., Dict[str, Any]] = lot_advisor.analyze_symbol) -> Dict[str, Any]:
    st = state if state is not None else pf.get_state(path=path, cached_quotes=True)
    ex = risk_rules.exposure(st)
    actions = []
    for sym in SYMBOLS:
        h = next((x for x in st.get("holdings") or [] if x["symbol"] == sym), None)
        if h is None:
            actions.append({"symbol": sym, "action": "watch", "headline": "Not held", "evidence": "default", "orders": [], "reasons": []})
            continue
        try:
            advice = analyze(sym, path=path, state=st)
        except Exception as e:  # noqa: BLE001 - one stock's missing bars do not hide the others
            actions.append({"symbol": sym, "action": "hold", "headline": f"No analysis: {type(e).__name__}: {e}",
                            "evidence": "default", "orders": [], "reasons": []})
            continue
        drop = (advice.get("stats") or {}).get("big_drop")
        if drop and "earnings" not in drop:
            drop["earnings"] = reported_into(sym, drop, path)
        actions.append(decide(sym, h, ex, advice))
    out = {"generated_at": datetime.now(NY).isoformat(timespec="seconds"), "symbols": SYMBOLS, "actions": actions,
           "portfolio": {"total": round(ex["total"], 2), "cash": ex["cash"], "cash_pct": ex["cash_pct"], "cluster_pct": ex["cluster_pct"],
                         "max_symbol_pct": risk_rules.MAX_SYMBOL_PCT, "max_cluster_pct": risk_rules.MAX_CLUSTER_PCT,
                         "min_cash_pct": risk_rules.MIN_CASH_PCT,
                         "breaches": risk_rules.summary(ex)}}
    _keep(out, path)
    return out


def get(path: Optional[Path] = None, force: bool = False) -> Dict[str, Any]:
    if not force and _cache["out"] and time.time() - _cache["at"] < CACHE_SECONDS and path is None:
        return _cache["out"]
    out = compute(path)
    if path is None:
        _cache.update(at=time.time(), out=out)
    return out


def _keep(out: Dict[str, Any], path: Optional[Path]) -> None:
    """The day's latest answer per symbol, for scoring against doing nothing later."""
    conn = sqlite3.connect(str(path or DB_PATH), timeout=30)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS daily_actions (day TEXT, symbol TEXT, action TEXT, payload TEXT, "
                     "created_at TEXT, PRIMARY KEY (day, symbol))")
        day = datetime.now(NY).date().isoformat()
        conn.executemany("INSERT OR REPLACE INTO daily_actions VALUES (?, ?, ?, ?, ?)",
                         [(day, a["symbol"], a["action"], json.dumps(a, default=str), out["generated_at"]) for a in out["actions"]])
        conn.commit()
    finally:
        conn.close()
