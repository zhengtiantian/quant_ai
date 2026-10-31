"""Today's action for a holding (PROJECT_PLAN Q.13, 2026-10-31): the three layers in one line.

Layer 1 says whether there is a tested edge (the earnings pulse on report nights; the advisor's
plans only when they pass its significance gate), layer 2 (risk_rules) what the portfolio can take,
layer 3 the limits. Most days the answer is Hold: an action appears only when a rule or a tested
signal calls for one, and each says which.

  earnings  the stock reports tonight or before tomorrow's open: the pulse watch runs, no buys
  trim      the holding is above the symbol cap: sell the excess into a rebound, in two limits at
            +1.5 and +2 x ATR above the price, good for 15 sessions. After the 899 drops of >= 2 x ATR
            in 2023-2026 (102 stocks, daily bars) the high reached +1.5 x ATR within 15 sessions in
            61% and +2 x ATR in 50%; selling there rather than at today's low is execution, not a
            forecast that the stock will rise
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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from . import lot_advisor, risk_rules
from . import portfolio as pf
from .config import DB_PATH

NY = ZoneInfo("America/New_York")
SYMBOLS = [s.strip().upper() for s in os.getenv("DAILY_ACTION_SYMBOLS", "INTC").split(",") if s.strip()]
TRIM_STEPS_ATR = (1.5, 2.0)
REBOUND_HIT = {1.5: 61, 2.0: 50}       # % of >= 2 x ATR drops reaching the level within 15 sessions (2023-2026)
VALID_SESSIONS = 15
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
        steps = [math.ceil(n / 2), n // 2] if n > 1 else [n]
        orders = []
        for k, sh in zip(TRIM_STEPS_ATR, steps):
            if sh <= 0:
                continue
            o = risk_rules.gate({"symbol": sym, "side": "sell", "shares": sh, "limit": round(price + k * atr, 2)}, ex, held=qty)
            if o["allowed"]:
                orders.append({**o, "atr_multiple": k, "hit_pct": REBOUND_HIT.get(k), "valid_sessions": VALID_SESSIONS})
        out.update(action="trim", orders=orders,
                   headline=f"Trim {sum(o['shares'] for o in orders)} shares into a rebound: "
                            + ", ".join(f"{o['shares']} @ ${o['limit']:.2f}" for o in orders),
                   evidence="rule (symbol cap) + execution statistics")
        out["reasons"].append(f"{sym} is {weight:.1f}% of the portfolio; the cap is {risk_rules.MAX_SYMBOL_PCT:g}% "
                              f"({n} shares over at today's price)")
        out["reasons"].append("Limits at +1.5 and +2 x ATR: after large drops (2023-2026) the high reached them within "
                              f"15 sessions in {REBOUND_HIT[1.5]}% and {REBOUND_HIT[2.0]}% of cases; selling at today's low is not needed")
        avg = holding.get("avg_cost")
        if avg and orders and max(o["limit"] for o in orders) < avg:
            out["reasons"].append(f"Below your average cost (${avg:.2f}): a loss for tax; buying {sym} back within 4 weeks "
                                  "restricts that loss under Irish CGT")
        if ex["cluster_pct"] > risk_rules.MAX_CLUSTER_PCT:
            out["reasons"].append(f"Chips / memory are {ex['cluster_pct']:.1f}% of the portfolio (cap {risk_rules.MAX_CLUSTER_PCT:g}%)")
        return out

    if out.get("buy_allowed"):
        out.update(action="add", headline="A buy fits every rule; no tested signal sizes it", evidence="rule")
        return out
    out.update(action="hold", headline="Hold: no rule or tested signal calls for an order", evidence="default")
    return out


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
        actions.append(decide(sym, h, ex, advice))
    out = {"generated_at": datetime.now(NY).isoformat(timespec="seconds"), "symbols": SYMBOLS, "actions": actions,
           "portfolio": {"total": round(ex["total"], 2), "cash": ex["cash"], "cluster_pct": ex["cluster_pct"],
                         "max_symbol_pct": risk_rules.MAX_SYMBOL_PCT, "max_cluster_pct": risk_rules.MAX_CLUSTER_PCT,
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
