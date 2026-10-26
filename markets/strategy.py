"""Order strategies for a holding, generated and tested in code (PROJECT_PLAN Q.5, 2026-10-25).

The first advisor let the model invent its limit orders. It echoed the user's own idea (trim before
INTC's earnings, buy back lower), used fractional shares a broker cannot fill and had no answer to
"is this better than holding?". Here the code builds the candidates, in whole shares, and tests
each on this stock's own history; the model only chooses among them and explains why.

Candidates (all as limit orders, sized in whole shares of the current position):
  hold          do nothing
  t1            sell s at +a, buy the same s back at -b once the sale filled
  t2            two such pairs, a nearer and a farther one
  breakeven     sell s at the average cost, when the cost is within reach
  trim          sell s at +a, no buy-back
  add           buy k at -b with available cash
where a and b are fractions of the stock's move scale m: the options-implied move when earnings
fall inside the horizon, else twice the 14-day ATR.

Each candidate gets
  history  replayed on the stock's real daily bars: around each of its past earnings (16 events
           for INTC, STX, TSM; SK hynix from its Seoul listing) when earnings are coming, else on
           rolling 15-session windows of the last year. Orders go in four sessions before the
           reaction day; a sale fills when the high reaches the limit (at the open when it gaps
           through), a buy-back only on a later session when the low reaches its limit. Result
           = what the trades earned against holding, in % of the position's value.
  outlook  this time: 3,000 simulated paths, daily volatility from 60 sessions of returns and, on
           the reaction day, a past reaction drawn at random and scaled to today's implied move.
           Gives the chance each limit fills and the chance of the regret case: sold, no
           buy-back, and the stock ends above the sale price.

A random walk has no edge for any limit-order plan, so the simulation alone would rank them all
near zero; the replay on real paths is where an edge, if the stock has one (say it drifts the
same way after reports), shows up. Both are reported, with their sample sizes.
"""

from __future__ import annotations

import math
import random
import statistics
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

ENTRY_OFFSET = 4            # sessions before the reaction day when the orders go in
HOLD_AFTER = 10             # sessions after the reaction day a buy-back may still fill
WINDOW = 15                 # sessions per non-earnings replay window
MIN_REPLAYS = 8             # fewer past windows than this: not enough to call an edge
MIN_EDGE_PCT = 0.05         # mean gain against holding, % of the position, for a plan to be offered
# The edge must also be significant (2026-10-26): TSM's two-step ladder was offered report after
# report on +0.36% over 47 windows that overlap three times over (15 sessions every 5), about 16
# independent ones, t well under 2. The daily policies of Q.7 (48,645 decision points) found no
# edge over holding out of sample either. A plan now needs t >= MIN_T on the effective sample.
MIN_T = 2.0
PATHS = 3000
SEED = 7


# ---------------------------------------------------------------- candidates

def _shares(position: float, frac: float) -> int:
    return max(1, int(math.floor(position * frac))) if position >= 1 else 0


def candidates(position: float, price: float, avg_cost: Optional[float], m_pct: float, cash: float) -> List[Dict[str, Any]]:
    """Plans as lists of legs: {side, shares, limit_pct (vs entry price), after (leg index) | None}."""
    n = int(math.floor(position))
    out: List[Dict[str, Any]] = [{"id": "hold", "label": "Hold", "legs": []}]
    if n < 1 or not m_pct:
        return out
    s15, s25 = _shares(position, 0.15), _shares(position, 0.25)
    for a, b, s in ((0.5, 0.5, s25), (0.75, 0.75, s25), (1.0, 1.0, s25)):
        out.append({"id": f"t1_{int(a * 100)}_{int(b * 100)}",
                    "label": f"T: sell {s} at +{a:g}m, buy back at -{b:g}m",
                    "legs": [{"side": "sell", "shares": s, "pct": a * m_pct, "after": None},
                             {"side": "buy", "shares": s, "pct": -b * m_pct, "after": 0}]})
    out.append({"id": "t2", "label": f"Ladder: sell {s15} at +0.5m and {s15} at +1m, buy each back 0.5m / 1m lower",
                "legs": [{"side": "sell", "shares": s15, "pct": 0.5 * m_pct, "after": None},
                         {"side": "sell", "shares": s15, "pct": 1.0 * m_pct, "after": None},
                         {"side": "buy", "shares": s15, "pct": -0.5 * m_pct, "after": 0},
                         {"side": "buy", "shares": s15, "pct": -1.0 * m_pct, "after": 1}]})
    for a in (0.75, 1.25):
        out.append({"id": f"trim_{int(a * 100)}", "label": f"Trim: sell {s25} at +{a:g}m, no buy-back",
                    "legs": [{"side": "sell", "shares": s25, "pct": a * m_pct, "after": None}]})
    if avg_cost and price and 0 < (avg_cost / price - 1) * 100 <= 1.5 * m_pct:
        out.append({"id": "breakeven", "label": f"Sell {s25} at the average cost {avg_cost:.2f}",
                    "legs": [{"side": "sell", "shares": s25, "pct": (avg_cost / price - 1) * 100, "after": None}]})
    if cash and price:
        k = int(cash * 0.5 // (price * (1 - m_pct / 100)))
        if k >= 1:
            out.append({"id": "add", "label": f"Add {k} at -1m with half the cash",
                        "legs": [{"side": "buy", "shares": k, "pct": -m_pct, "after": None}]})
    return out


# ---------------------------------------------------------------- one replay on one price path

def replay(legs: List[Dict[str, Any]], entry: float, bars: List[Dict[str, float]]) -> Dict[str, Any]:
    """Run the legs on bars (each {open, high, low, close}) after the entry price; return the gain
    against holding as % of the entry value of one share-unit of the position (caller scales)."""
    fills: Dict[int, Tuple[int, float]] = {}            # leg index -> (bar index, fill price)
    for bi, b in enumerate(bars):
        for li, leg in enumerate(legs):
            if li in fills:
                continue
            if leg["after"] is not None and (leg["after"] not in fills or fills[leg["after"]][0] >= bi):
                continue                                  # its sale has not filled on an earlier session
            limit = entry * (1 + leg["pct"] / 100)
            if leg["side"] == "sell" and b["high"] >= limit:
                fills[li] = (bi, max(limit, b["open"]))
            elif leg["side"] == "buy" and b["low"] <= limit:
                fills[li] = (bi, min(limit, b["open"]))
    final = bars[-1]["close"]
    gain = 0.0
    for li, (bi, px) in fills.items():
        sh = legs[li]["shares"]
        gain += sh * (px - final) if legs[li]["side"] == "sell" else sh * (final - px)
    sold = [li for li, leg in enumerate(legs) if leg["side"] == "sell" and li in fills]
    rebought = {legs[li]["after"] for li in fills if legs[li]["side"] == "buy" and legs[li]["after"] is not None}
    regret = any(li not in rebought and final > fills[li][1] for li in sold)
    return {"gain": gain, "fills": sorted(fills), "regret": regret, "hold_pct": (final / entry - 1) * 100}


def _pctile(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def _summary(gains: List[float], fill_counts: List[int], regrets: int, value: float, overlap: float = 1.0) -> Dict[str, Any]:
    """Gains are in $ per $ of entry price; value is the position in shares: the result is in % of
    the position's value. Win / loss = beat / trailed holding by more than 0.05%. overlap: how many
    windows share each session (rolling windows of 15 every 5: 3), so the effective sample is n /
    overlap and t = mean / (sd / sqrt(n_eff))."""
    if not gains:
        return {"n": 0}
    pct = [g / value * 100 for g in gains]
    sd = statistics.stdev(pct) if len(pct) > 1 else 0.0
    n_eff = len(pct) / max(overlap, 1.0)
    t = statistics.mean(pct) / (sd / math.sqrt(n_eff)) if sd > 0 and n_eff > 0 else 0.0
    return {"n": len(pct), "n_eff": round(n_eff, 1), "sd_pct": round(sd, 2), "t": round(t, 2),
            "mean_pct": round(statistics.mean(pct), 2), "median_pct": round(statistics.median(pct), 2),
            "win_rate": round(sum(p > 0.05 for p in pct) / len(pct), 2), "loss_rate": round(sum(p < -0.05 for p in pct) / len(pct), 2),
            "p10_pct": round(_pctile(pct, 0.1), 2), "p90_pct": round(_pctile(pct, 0.9), 2),
            "worst_pct": round(min(pct), 2), "best_pct": round(max(pct), 2),
            "leg_fill_rate": [round(c / len(pct), 2) for c in fill_counts], "regret_rate": round(regrets / len(pct), 2)}


def _hold(moves: List[float]) -> Dict[str, Any]:
    """What holding alone did over the same windows, % of the position."""
    if not moves:
        return {"n": 0}
    return {"n": len(moves), "mean_pct": round(statistics.mean(moves), 2), "median_pct": round(statistics.median(moves), 2),
            "up_rate": round(sum(m > 0 for m in moves) / len(moves), 2),
            "p10_pct": round(_pctile(moves, 0.1), 2), "p90_pct": round(_pctile(moves, 0.9), 2)}


def _scaled(legs: List[Dict[str, Any]], k: float) -> List[Dict[str, Any]]:
    return [{**leg, "pct": leg["pct"] * k} for leg in legs]


def history_test(plan: Dict[str, Any], position: float, windows: List[Tuple[float, List[Dict[str, float]]]],
                 scale: float = 1.0, overlap: float = 1.0) -> Dict[str, Any]:
    """scale converts today's limits to the past: a limit at +0.5 of today's move scale is replayed at
    +0.5 of the past typical move (SK hynix: ADR implied 9.3% today, the Seoul listing's typical 2.5%)."""
    if not plan["legs"]:
        return _hold([replay([], e, b)["hold_pct"] for e, b in windows])
    legs = _scaled(plan["legs"], scale)
    gains, regrets, counts = [], 0, [0] * len(plan["legs"])
    for entry, bars in windows:
        r = replay(legs, entry, bars)
        gains.append(r["gain"] / entry)                   # per $ of entry price per share
        regrets += r["regret"]
        for li in r["fills"]:
            counts[li] += 1
    return _summary(gains, counts, regrets, position, overlap)


# ---------------------------------------------------------------- the simulated outlook

def simulate(plan: Dict[str, Any], position: float, price: float, daily_vol_pct: float, sessions: int,
             reaction_session: Optional[int], jumps_pct: List[float], paths: int = PATHS) -> Dict[str, Any]:
    rng = random.Random(SEED)
    sd = daily_vol_pct / 100
    gains, regrets, counts, holds = [], 0, [0] * len(plan["legs"]), []
    for _ in range(paths):
        p, bars = price, []
        for s in range(sessions):
            if reaction_session is not None and s == reaction_session and jumps_pct:
                j = rng.choice(jumps_pct) / 100
                o = p * (1 + j)
                c = o * math.exp(rng.gauss(0, sd))
            else:
                o = p * math.exp(rng.gauss(0, sd * 0.3))
                c = p * math.exp(rng.gauss(0, sd))
            spread = abs(rng.gauss(0, sd * 0.6))
            bars.append({"open": o, "high": max(o, c) * (1 + spread), "low": min(o, c) * (1 - spread), "close": c})
            p = c
        r = replay(plan["legs"], price, bars)
        if not plan["legs"]:
            holds.append(r["hold_pct"])
            continue
        gains.append(r["gain"] / price)
        regrets += r["regret"]
        for li in r["fills"]:
            counts[li] += 1
    return _hold(holds) if not plan["legs"] else _summary(gains, counts, regrets, position)


# ---------------------------------------------------------------- inputs from the price history

def earnings_windows(reactions: List[Dict[str, Any]], lead: int = ENTRY_OFFSET) -> List[Tuple[float, List[Dict[str, float]]]]:
    """(entry close, bars after it) for each past report, the entry `lead` sessions before the reaction
    day, as today is before the coming one."""
    out = []
    for ev in reactions:
        bars, k = ev.get("bars") or [], ev.get("reaction_offset", 5)
        start = k - min(lead, k)
        if start < 0 or len(bars) < k + 2:
            continue
        out.append((bars[start]["close"], bars[start + 1:k + HOLD_AFTER + 1]))
    return out


def rolling_windows(candles: List[Dict[str, float]], step: int = 5) -> List[Tuple[float, List[Dict[str, float]]]]:
    done = [c for c in candles if not c.get("partial")]
    return [(done[i]["close"], done[i + 1:i + 1 + WINDOW]) for i in range(0, len(done) - WINDOW - 1, step)]


def daily_vol_pct(candles: List[Dict[str, float]], exclude: set = frozenset()) -> float:
    done = [c for c in candles if not c.get("partial")][-61:]
    rets = [math.log(b["close"] / a["close"]) for a, b in zip(done, done[1:]) if a["close"] and b.get("time") not in exclude]
    return statistics.pstdev(rets) * 100 if len(rets) > 5 else 2.0


def scaled_jumps(reactions: List[Dict[str, Any]], implied_pct: Optional[float]) -> List[float]:
    moves = [ev["move_pct"] for ev in reactions if ev.get("move_pct") is not None]
    if not moves:
        return []
    # Centred: the simulation answers "how likely is each limit to fill", not "which way"; the
    # direction evidence comes from the replay on the real paths.
    mean = statistics.mean(moves)
    moves = [m - mean for m in moves]
    med = statistics.median(abs(m) for m in moves) or 1.0
    k = (implied_pct / med) if implied_pct else 1.0
    return [m * k for m in moves]


# ---------------------------------------------------------------- what may be recommended

def eligible(plan: Dict[str, Any]) -> Tuple[bool, str]:
    """Profit first (the user, 2026-10-25): a plan is offered only when it beat holding on this stock's
    own history, and never sells below the average cost without buying the shares back lower."""
    if not plan["legs"]:
        return True, "the baseline"
    h = plan["history"]
    if h.get("n", 0) < MIN_REPLAYS:
        return False, f"only {h.get('n', 0)} past windows"
    children = {leg["after"] for leg in plan["legs"] if leg["after"] is not None}
    if any(leg["below_cost"] and i not in children for i, leg in enumerate(plan["legs"]) if leg["side"] == "sell"):
        return False, "sells below the average cost with no buy-back"
    if h["mean_pct"] < MIN_EDGE_PCT:
        return False, f"trailed or matched holding ({h['mean_pct']:+.2f}% of the position on average)"
    if h["win_rate"] < h["loss_rate"]:
        return False, f"beat holding in {h['win_rate']:.0%} of windows, trailed in {h['loss_rate']:.0%}"
    if h.get("t", 0.0) < MIN_T:
        return False, (f"edge not significant: {h['mean_pct']:+.2f}% with t {h.get('t', 0):.1f} on about "
                       f"{h.get('n_eff', h['n']):.0f} independent windows (needs t >= {MIN_T:g})")
    return True, (f"beat holding by {h['mean_pct']:+.2f}% on average, in {h['win_rate']:.0%} of {h['n']} windows "
                  f"(t {h['t']:.1f} on about {h['n_eff']:.0f} independent)")


def _add_sessions(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        n -= d.weekday() < 5
    return d


# ---------------------------------------------------------------- one holding

def evaluate_holding(symbol: str, position: float, price: float, avg_cost: Optional[float], cash: float,
                     candles_1y: List[Dict[str, float]], atr_pct: Optional[float], earnings: Optional[Dict[str, Any]],
                     reactions: List[Dict[str, Any]], today: date) -> Dict[str, Any]:
    implied = ((earnings or {}).get("implied") or {}).get("implied_move_pct")
    hist_moves = [abs(ev["move_pct"]) for ev in reactions if ev.get("move_pct") is not None]
    reaction_session = None
    if earnings and earnings.get("date"):
        d = date.fromisoformat(str(earnings["date"])[:10])
        sessions_to = sum(1 for i in range(1, (d - today).days + 1) if (today + timedelta(days=i)).weekday() < 5)
        # SK hynix reports in the Seoul morning, i.e. after the US close: the ADR reacts next session.
        after = "after" in str(earnings.get("when") or "")
        reaction_session = max(0, sessions_to - (0 if after else 1)) if sessions_to <= 10 else None
    scale = 1.0
    if reaction_session is not None:
        m = implied or (statistics.median(hist_moves) if hist_moves else 2 * (atr_pct or 2))
        basis = "implied move" if implied else "median past earnings move"
        windows = earnings_windows(reactions, lead=reaction_session + 1)
        overlap = 1.0                                     # one window per past report: independent
        sessions = reaction_session + 1 + HOLD_AFTER
        if hist_moves and m:
            scale = statistics.median(hist_moves) / m
    else:
        m, basis = 2 * (atr_pct or 2), "2 x ATR(14)"
        windows = rolling_windows(candles_1y)
        overlap = WINDOW / 5                              # 15-session windows every 5 sessions
        sessions = WINDOW
    vol = daily_vol_pct(candles_1y)
    jumps = scaled_jumps(reactions, implied) if reaction_session is not None else []
    plans = []
    for c in candidates(position, price, avg_cost, m, cash):
        hist = history_test(c, position, windows, scale, overlap)
        sim = simulate(c, position, price, vol, sessions, reaction_session, jumps)
        legs = [{**leg, "limit": round(price * (1 + leg["pct"] / 100), 2)} for leg in c["legs"]]
        for leg in legs:
            leg["below_cost"] = bool(avg_cost and leg["side"] == "sell" and leg["limit"] < avg_cost)
        plan = {"id": c["id"], "label": c["label"], "legs": legs, "history": hist, "outlook": sim,
                "history_gain_usd": round(hist["mean_pct"] / 100 * position * price, 2) if legs and hist.get("n") else None}
        plan["eligible"], plan["why"] = eligible(plan)
        plans.append(plan)
    return {"symbol": symbol, "price": price, "position_shares": round(position, 6), "whole_shares": int(math.floor(position)),
            "move_scale_pct": round(m, 2), "move_scale_basis": basis, "history_scale": round(scale, 3),
            "earnings_in_horizon": reaction_session is not None, "sessions_to_reaction": reaction_session,
            "past_earnings": {"n": len(hist_moves), "median_abs_move_pct": round(statistics.median(hist_moves), 2) if hist_moves else None,
                              "up_share": round(sum(ev["move_pct"] > 0 for ev in reactions) / len(reactions), 2) if reactions else None,
                              "source": (reactions[0].get("history_source") if reactions else None)},
            "replay_windows": len(windows), "daily_vol_pct": round(vol, 2),
            "valid_until": _add_sessions(today, sessions).isoformat(), "candidates": plans}
