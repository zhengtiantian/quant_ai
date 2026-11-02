"""Which holdings to sell to meet the portfolio caps, by risk (PROJECT_PLAN Q.13 / Q.14, 2026-11-02).

The risk gate says how far the portfolio is over its caps (one stock 30%, the chip / memory cluster
70%, cash at least 10%) but not which names to sell beyond each stock's own excess. No return
forecast has earned a say: the factor score ranked the user's holdings no better than chance
(quant_data research/backtest/factor_lab.py, Q.14 item 2), so the choice is made on risk alone,
as a desk's optimiser does with a zero alpha:

  1. each stock's excess over the single-stock cap is sold (the rule, as before);
  2. what the cluster is still over its cap, and what cash is still short of its floor, is sold in
     steps of STEP_USD, each step from the holding whose sale lowers the portfolio's variance most
     (largest marginal contribution, 2 * Sigma w, per dollar), within the cluster for the cluster.

The covariance is that of the last ~6 months' daily returns, shrunk 20% toward its diagonal.
Selling the least needed is fixed by the caps; the optimiser only decides where it comes from.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import portfolio as pf
from . import risk_budget
from .lot_advisor import SEMI_MEMORY_CLUSTER
from .risk_rules import MAX_CLUSTER_PCT, MAX_SYMBOL_PCT, MIN_CASH_PCT

STEP_USD = 50.0
SHRINK = 0.2
MIN_BARS = 60


def covariance(symbols: List[str], chart_fn: Optional[Callable] = None, path=None) -> Optional[np.ndarray]:
    """Annualised covariance of daily returns over the dates every symbol has, or None if too few."""
    closes: Dict[str, Dict[str, float]] = {}
    for s in symbols:
        candles = (chart_fn or pf.get_chart)(s, "6m", path).get("candles") or []
        closes[s] = {c["time"]: float(c["close"]) for c in candles if not c.get("partial") and c.get("close")}
    common = sorted(set.intersection(*(set(v) for v in closes.values()))) if closes else []
    if len(common) < MIN_BARS:
        return None
    px = np.array([[closes[s][d] for s in symbols] for d in common])
    rets = np.diff(np.log(px), axis=0)
    cov = np.cov(rets, rowvar=False) * 252
    cov = np.atleast_2d(cov)
    return (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov))


def plan(ex: Dict[str, Any], prices: Dict[str, float], cov: Optional[np.ndarray],
         day_cov: Optional[np.ndarray] = None, budget_usd: Optional[float] = None) -> Dict[str, Any]:
    """Dollars (and whole shares) to sell per holding so every cap holds, chosen by risk; then, with a
    daily covariance and a budget, until a 1-in-20 bad day loses no more than the budget."""
    syms = list(ex["mv"])
    v = np.array([ex["mv"][s] for s in syms], float)
    total, cash = float(ex["total"]), float(ex["cash"])
    sell = np.zeros(len(syms))
    in_cluster = np.array([s in SEMI_MEMORY_CLUSTER for s in syms])
    sym_cap = MAX_SYMBOL_PCT / 100 * total
    sell += np.maximum(v - sym_cap, 0.0)                                       # 1. each stock's own excess

    def variance(held):
        return float(held @ cov @ held) if cov is not None else 0.0

    def step_from(mask):
        """The holding (within mask, with something left) whose next STEP_USD sold cuts variance most."""
        held = v - sell
        left = mask & (held > 1e-6)
        if not left.any():
            return None
        if cov is None:                                                          # no covariance: the largest first
            return int(np.argmax(np.where(left, held, -1)))
        grad = 2 * cov @ held
        return int(np.argmax(np.where(left, grad, -np.inf)))

    def need_cluster():
        return max(0.0, float(((v - sell) * in_cluster).sum()) - MAX_CLUSTER_PCT / 100 * total)

    def need_cash():
        return max(0.0, MIN_CASH_PCT / 100 * total - cash - float(sell.sum()))

    for need, mask in ((need_cluster, in_cluster), (need_cash, np.ones(len(syms), bool))):
        while need() > 1e-6:
            i = step_from(mask)
            if i is None:
                break
            sell[i] += min(STEP_USD, need(), v[i] - sell[i])
    loss_before = risk_budget.day_loss(v, day_cov)
    if day_cov is not None and budget_usd:
        # 3. the loss budget: each step from the holding whose sale cuts the bad-day loss most
        while (risk_budget.day_loss(v - sell, day_cov) or 0.0) > budget_usd:
            held = v - sell
            left = held > 1e-6
            if not left.any():
                break
            i = int(np.argmax(np.where(left, day_cov @ held, -np.inf)))
            sell[i] += min(STEP_USD, held[i])

    held_after = v - sell
    vol = lambda h: math.sqrt(variance(h)) / total * 100 if cov is not None and total > 0 else None
    rows = {}
    for i, s in enumerate(syms):
        if sell[i] > 0.5:
            px = prices.get(s) or 0.0
            whole = sell[i] >= v[i] - 1.0                      # the whole holding: its exact (fractional) quantity
            shares = (round(v[i] / px, 4) if whole else min(int(math.ceil(sell[i] / px)), int(math.floor(v[i] / px + 1e-9)))) if px > 0 else None
            rows[s] = {"usd": round(float(sell[i]), 2), "shares": shares, "all": bool(whole),
                       "weight_before": round(v[i] / total * 100, 1), "weight_after": round(held_after[i] / total * 100, 1)}
    contrib = {}
    if cov is not None and variance(v) > 0:
        rc = v * (cov @ v) / variance(v) * 100
        contrib = {s: round(float(rc[i]), 1) for i, s in enumerate(syms)}
    loss_after = risk_budget.day_loss(held_after, day_cov)
    return {"sell": rows, "total_usd": round(float(sell.sum()), 2), "risk_model": cov is not None,
            "budget_usd": None if not budget_usd else round(budget_usd, 2),
            "bad_day_loss_before": None if loss_before is None else round(loss_before, 2),
            "bad_day_loss_after": None if loss_after is None else round(loss_after, 2),
            "vol_before_pct": None if vol(v) is None else round(vol(v), 1),
            "vol_after_pct": None if vol(held_after) is None else round(vol(held_after), 1),
            "risk_share_before": contrib,
            "cluster_after_pct": round(float((held_after * in_cluster).sum()) / total * 100, 1) if total > 0 else 0.0,
            "cash_after_pct": round((cash + float(sell.sum())) / total * 100, 1) if total > 0 else 0.0}


def build(state: Dict[str, Any], ex: Dict[str, Any], chart_fn: Optional[Callable] = None, path=None) -> Dict[str, Any]:
    prices = {h["symbol"]: float((h.get("quote") or {}).get("price") or 0.0) for h in state.get("holdings") or []}
    syms = [s for s in ex["mv"] if ex["mv"][s] > 0]
    try:
        cov = covariance(syms, chart_fn, path) if syms else None
        day_cov = risk_budget.ewma_cov(syms, chart_fn, path) if syms else None
    except Exception:  # noqa: BLE001 - no bars: the plan falls back to selling the largest first
        cov = day_cov = None
    sub = {**ex, "mv": {s: ex["mv"][s] for s in syms}}
    return plan(sub, prices, cov, day_cov, risk_budget.BUDGET_PCT / 100 * ex["total"])
