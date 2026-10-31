"""Layer 2 of the advice stack: the portfolio risk gate (PROJECT_PLAN Q.12, 2026-10-31).

Signals (layer 1: the earnings pulse, the advisor's tested plans) say whether there is an edge;
execution (layer 3) says at what limit. This layer says whether the portfolio can take the order at
all, and how much of it. Its rules are fixed, not learned: with a few stocks and four reports a
year there is not the data to learn them, and a cap is a choice, not a prediction.

  shares    a sale cannot exceed the whole shares held
  cash      a buy needs the cash for its whole shares
  symbol    a buy may not take one stock above MAX_SYMBOL_PCT of the portfolio (holdings + cash);
            a holding above it is trimmed back to it (trim_shares)
  cluster   the same for the semiconductor / memory / storage names together, MAX_CLUSTER_PCT
  falling   no buy while the stock has not stabilised after a >= 2 x ATR drop (in 2023-2026, 47%
            of such drops fell another 1 x ATR within three sessions)
  spacing   a buy sits at least MIN_SPACING_ATR x ATR below the cheapest open lot, EVENT_SPACING_ATR
            when a high-impact event is within 7 days (Q.10)

The caps are placeholders until the user sets them (env RISK_MAX_SYMBOL_PCT / RISK_MAX_CLUSTER_PCT).
"""

import math
import os
from typing import Any, Dict, List, Optional

from .lot_advisor import SEMI_MEMORY_CLUSTER

MAX_SYMBOL_PCT = float(os.getenv("RISK_MAX_SYMBOL_PCT", "30"))
MAX_CLUSTER_PCT = float(os.getenv("RISK_MAX_CLUSTER_PCT", "70"))
MIN_SPACING_ATR, EVENT_SPACING_ATR = 1.5, 2.0


def exposure(state: Dict[str, Any]) -> Dict[str, Any]:
    """Market values by symbol, the portfolio total (holdings + cash) and the cluster's share."""
    mv = {h["symbol"]: float(h.get("market_value") or 0.0) for h in state.get("holdings") or []}
    cash = float(state.get("cash") or 0.0)
    total = sum(mv.values()) + cash
    cluster = sum(v for s, v in mv.items() if s in SEMI_MEMORY_CLUSTER)
    pct = lambda v: round(v / total * 100, 1) if total > 0 else 0.0
    return {"mv": mv, "cash": cash, "total": total, "weights": {s: pct(v) for s, v in mv.items()},
            "cluster_mv": cluster, "cluster_pct": pct(cluster), "max_symbol_pct": MAX_SYMBOL_PCT, "max_cluster_pct": MAX_CLUSTER_PCT}


def trim_shares(quantity: float, mv: float, total: float, cap_pct: float = MAX_SYMBOL_PCT) -> int:
    """Whole shares to sell so the holding is back at the cap at today's price (0 when within it)."""
    if total <= 0 or mv <= cap_pct / 100 * total or quantity <= 0:
        return 0
    return min(int(math.floor(quantity)), int(math.ceil(quantity * (1 - cap_pct / 100 * total / mv))))


def gate(order: Dict[str, Any], ex: Dict[str, Any], held: float = 0.0, atr: Optional[float] = None,
         lowest_lot: Optional[float] = None, stabilized: Optional[bool] = None, event_7d: bool = False) -> Dict[str, Any]:
    """order {symbol, side, shares, limit}. Returns the order as the portfolio can take it: allowed,
    shares (possibly fewer), blocked_by (rule codes) and notes (one line each, shown to the user)."""
    sym, side, limit = order["symbol"], order["side"], float(order["limit"])
    shares, blocked, notes = int(order["shares"]), [], []
    if side == "sell":
        if shares > math.floor(held):
            shares = int(math.floor(held))
            notes.append(f"Only {shares} whole shares held")
        if shares <= 0:
            blocked.append("shares")
        return {**order, "shares": shares, "allowed": not blocked, "blocked_by": blocked, "notes": notes}

    if stabilized is False:
        blocked.append("falling")
        notes.append("Not stabilised after a large drop: no buys yet")
    if atr and lowest_lot:
        need = EVENT_SPACING_ATR if event_7d else MIN_SPACING_ATR
        ceiling = round(lowest_lot - need * atr, 2)
        if limit > ceiling:
            blocked.append("spacing")
            notes.append(f"Too close to your cheapest lot (${lowest_lot:.2f}): buy at ${ceiling:.2f} or lower ({need:g} x ATR)")
    total, mv = ex["total"], ex["mv"].get(sym, 0.0)
    fits = shares
    if limit > 0:
        fits = min(fits, int(ex["cash"] // limit))
        if fits < shares:
            notes.append(f"Cash ${ex['cash']:,.2f} buys {max(fits, 0)} shares at ${limit:.2f}")
        room = MAX_SYMBOL_PCT / 100 * total - mv
        if int(room // limit) < fits:
            fits = max(0, int(room // limit))
            notes.append(f"{sym} is {ex['weights'].get(sym, 0):.1f}% of the portfolio (cap {MAX_SYMBOL_PCT:g}%)")
        if sym in SEMI_MEMORY_CLUSTER:
            room_c = MAX_CLUSTER_PCT / 100 * total - ex["cluster_mv"]
            if int(room_c // limit) < fits:
                fits = max(0, int(room_c // limit))
                notes.append(f"Chips / memory are {ex['cluster_pct']:.1f}% of the portfolio (cap {MAX_CLUSTER_PCT:g}%)")
    if fits <= 0:
        blocked.append("size")
    return {**order, "shares": max(fits, 0), "allowed": not blocked, "blocked_by": blocked, "notes": notes}


def summary(ex: Dict[str, Any]) -> List[str]:
    """The portfolio-level breaches, one line each."""
    out = [f"{s} is {w:.1f}% of the portfolio (cap {MAX_SYMBOL_PCT:g}%)" for s, w in sorted(ex["weights"].items(), key=lambda kv: -kv[1])
           if w > MAX_SYMBOL_PCT]
    if ex["cluster_pct"] > MAX_CLUSTER_PCT:
        out.append(f"Chips / memory are {ex['cluster_pct']:.1f}% of the portfolio (cap {MAX_CLUSTER_PCT:g}%)")
    return out
