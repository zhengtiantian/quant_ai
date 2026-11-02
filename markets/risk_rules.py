"""Layer 2 of the advice stack: the portfolio risk gate (PROJECT_PLAN Q.12, 2026-10-31).

Signals (layer 1: the earnings pulse, the advisor's tested plans) say whether there is an edge;
execution (layer 3) says at what limit. This layer says whether the portfolio can take the order at
all, and how much of it. Its rules are fixed, not learned: with a few stocks and four reports a
year there is not the data to learn them, and a cap is a choice, not a prediction.

  shares    a sale cannot exceed the whole shares held
  cash      a buy needs the cash for its whole shares, and must leave MIN_CASH_PCT of the portfolio in
            cash (with no cash the dip rules can never act)
  symbol    a buy may not take one stock above MAX_SYMBOL_PCT of the portfolio (holdings + cash);
            a holding above it is trimmed back to it (trim_shares)
  cluster   the same for the semiconductor / memory / storage names together, MAX_CLUSTER_PCT
  falling   no buy while the stock has not stabilised after a >= 2 x ATR drop (in 2023-2026, 47%
            of such drops fell another 1 x ATR within three sessions)
  spacing   a buy sits at least MIN_SPACING_ATR x ATR below the cheapest open lot, EVENT_SPACING_ATR
            when a high-impact event is within 7 days (Q.10)

Targets agreed with the user on 2026-10-31: 10% cash, chips / memory below 70%; the single-stock cap
30% (env RISK_MAX_SYMBOL_PCT / RISK_MAX_CLUSTER_PCT / RISK_MIN_CASH_PCT).
"""

import math
import os
from typing import Any, Dict, List, Optional

from .lot_advisor import SEMI_MEMORY_CLUSTER

MAX_SYMBOL_PCT = float(os.getenv("RISK_MAX_SYMBOL_PCT", "30"))
MAX_CLUSTER_PCT = float(os.getenv("RISK_MAX_CLUSTER_PCT", "70"))
MIN_CASH_PCT = float(os.getenv("RISK_MIN_CASH_PCT", "10"))
MIN_SPACING_ATR, EVENT_SPACING_ATR = 1.5, 2.0


def exposure(state: Dict[str, Any]) -> Dict[str, Any]:
    """Market values by symbol, the portfolio total (holdings + cash) and the cluster's share."""
    mv = {h["symbol"]: float(h.get("market_value") or 0.0) for h in state.get("holdings") or []}
    cash = float(state.get("cash") or 0.0)
    total = sum(mv.values()) + cash
    cluster = sum(v for s, v in mv.items() if s in SEMI_MEMORY_CLUSTER)
    pct = lambda v: round(v / total * 100, 1) if total > 0 else 0.0
    return {"mv": mv, "cash": cash, "total": total, "weights": {s: pct(v) for s, v in mv.items()}, "cash_pct": pct(cash),
            "cluster_mv": cluster, "cluster_pct": pct(cluster), "max_symbol_pct": MAX_SYMBOL_PCT, "max_cluster_pct": MAX_CLUSTER_PCT,
            "min_cash_pct": MIN_CASH_PCT}


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
    shares, blocked, notes, notes_zh = int(order["shares"]), [], [], []   # notes_zh: the same lines in Chinese
    if side == "sell":
        if shares > math.floor(held):
            shares = int(math.floor(held))
            notes.append(f"Only {shares} whole shares held")
            notes_zh.append(f"只持有 {shares} 整股")
        if shares <= 0:
            blocked.append("shares")
        return {**order, "shares": shares, "allowed": not blocked, "blocked_by": blocked, "notes": notes, "notes_zh": notes_zh}

    if stabilized is False:
        blocked.append("falling")
        notes.append("Not stabilised after a large drop: no buys yet")
        notes_zh.append("大跌后尚未企稳，暂不买入")
    if atr and lowest_lot:
        need = EVENT_SPACING_ATR if event_7d else MIN_SPACING_ATR
        ceiling = round(lowest_lot - need * atr, 2)
        if limit > ceiling:
            blocked.append("spacing")
            notes.append(f"Too close to your cheapest lot (${lowest_lot:.2f}): buy at ${ceiling:.2f} or lower ({need:g} x ATR)")
            notes_zh.append(f"离你最便宜的一批（${lowest_lot:.2f}）太近：${ceiling:.2f} 或更低才买（{need:g} 倍 ATR）")
    total, mv = ex["total"], ex["mv"].get(sym, 0.0)
    fits = shares
    if limit > 0:
        spendable = ex["cash"] - MIN_CASH_PCT / 100 * total
        fits = min(fits, max(0, int(spendable // limit)))
        if fits < shares:
            notes.append(f"Cash ${ex['cash']:,.2f} buys {fits} shares at ${limit:.2f} (keeping {MIN_CASH_PCT:g}% in cash)")
            notes_zh.append(f"现金 ${ex['cash']:,.2f}，在 ${limit:.2f} 只能买 {fits} 股（保留 {MIN_CASH_PCT:g}% 现金）")
        room = MAX_SYMBOL_PCT / 100 * total - mv
        if int(room // limit) < fits:
            fits = max(0, int(room // limit))
            notes.append(f"{sym} is {ex['weights'].get(sym, 0):.1f}% of the portfolio (cap {MAX_SYMBOL_PCT:g}%)")
            notes_zh.append(f"{sym} 占组合 {ex['weights'].get(sym, 0):.1f}%（上限 {MAX_SYMBOL_PCT:g}%）")
        if sym in SEMI_MEMORY_CLUSTER:
            room_c = MAX_CLUSTER_PCT / 100 * total - ex["cluster_mv"]
            if int(room_c // limit) < fits:
                fits = max(0, int(room_c // limit))
                notes.append(f"Chips / memory are {ex['cluster_pct']:.1f}% of the portfolio (cap {MAX_CLUSTER_PCT:g}%)")
                notes_zh.append(f"芯片 / 存储类占组合 {ex['cluster_pct']:.1f}%（上限 {MAX_CLUSTER_PCT:g}%）")
    if fits <= 0:
        blocked.append("size")
    return {**order, "shares": max(fits, 0), "allowed": not blocked, "blocked_by": blocked, "notes": notes, "notes_zh": notes_zh}


def summary(ex: Dict[str, Any]) -> List[str]:
    """The portfolio-level breaches, one line each."""
    out = [f"{s} is {w:.1f}% of the portfolio (cap {MAX_SYMBOL_PCT:g}%)" for s, w in sorted(ex["weights"].items(), key=lambda kv: -kv[1])
           if w > MAX_SYMBOL_PCT]
    if ex["cluster_pct"] > MAX_CLUSTER_PCT:
        out.append(f"Chips / memory are {ex['cluster_pct']:.1f}% of the portfolio (cap {MAX_CLUSTER_PCT:g}%)")
    if ex.get("cash_pct", 0.0) < MIN_CASH_PCT:
        out.append(f"Cash is {ex.get('cash_pct', 0.0):.1f}% of the portfolio (target {MIN_CASH_PCT:g}%)")
    return out
