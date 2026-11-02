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
import re
import math
import os
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from . import lot_advisor, model_advice, optimizer, risk_budget, risk_rules
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
    "earnings_big": {"label": "an earnings drop of >= 4 x ATR", "label_zh": "≥ 4 倍 ATR 的财报下跌", "n": 193, 1.5: 39, 2.0: 28, "unfilled": -4.6},
    "earnings": {"label": "an earnings drop", "label_zh": "财报下跌", "n": 509, 1.5: 50, 2.0: 36, "unfilled": -4.3},
    "other": {"label": "a drop without an earnings report", "label_zh": "非财报下跌", "n": 1753, 1.5: 65, 2.0: 54, "unfilled": -6.3},
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


EVIDENCE_ZH = {"rule (symbol cap)": "规则（单只上限）", "reach rates by drop cause": "按下跌原因的成交率", "risk plan": "风险计划",
               "loss budget (report night)": "风险预算（财报夜）", "tested (earnings pulse champion)": "已检验（财报脉冲冠军模型）",
               "rule": "规则", "default": "默认"}


# The page's fixed words, in both languages, so the front end carries no Chinese of its own.
ACTION_LABELS = {"hold": {"en": "Hold", "zh": "持有"}, "trim": {"en": "Trim", "zh": "减仓"}, "add": {"en": "Add", "zh": "加仓"},
                 "earnings": {"en": "Earnings", "zh": "财报"}, "watch": {"en": "Watch", "zh": "观察"}}
LABELS = {"blocked": {"en": "Buying is blocked", "zh": "买入被拦截"}, "basis": {"en": "Basis", "zh": "依据"},
          "more": {"en": "More", "zh": "更多细节"}, "less": {"en": "Less", "zh": "收起"}}


def _evidence_zh(en: str) -> str:
    return " + ".join(EVIDENCE_ZH.get(p.strip(), p.strip()) for p in en.split(" + "))


def decide(sym: str, holding: Dict[str, Any], ex: Dict[str, Any], advice: Dict[str, Any],
           plan: Optional[Dict[str, Any]] = None, market: Optional[Dict[str, Any]] = None,
           implied_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """One holding's action from its position, the portfolio's exposure and the Q.10 analysis.
    Every line is written in English and Chinese ({en, zh}); `key` lines are shown first, the rest
    under "more". Every answer ends with the loss budget, the next report and the market state."""
    stats = advice.get("stats") or {}
    price = float((holding.get("quote") or {}).get("price") or 0.0) or None
    atr = float(stats.get("atr14") or 0.0) or None
    qty = float(holding.get("quantity") or 0.0)
    cats = advice.get("catalysts_7d") or []
    events = [c for c in cats if c.get("impact") == "high"]
    reports = [c for c in cats if c.get("bucket") == "direct" and c.get("category") == "earnings" and c.get("days_away", 9) <= 1]
    weight = ex["weights"].get(sym, 0.0)
    out: Dict[str, Any] = {"symbol": sym, "price": price, "atr": atr, "weight_pct": weight, "cluster_pct": ex["cluster_pct"],
                           "orders": [], "notes": [], "evidence": "", "blocks": [], "blocks_zh": []}

    def note(en: str, zh: str, key: bool = True) -> None:
        out["notes"].append({"en": en, "zh": zh, "key": key})

    # What a buy would meet, listed whatever the action (the user asks "can I average down?").
    if price:
        lowest = (advice.get("dip_guardrail") or {}).get("anchor_cost")
        probe = risk_rules.gate({"symbol": sym, "side": "buy", "shares": 1, "limit": price}, ex, held=qty, atr=atr,
                                lowest_lot=lowest, stabilized=stats.get("stabilized"), event_7d=bool(events))
        out["blocks"], out["blocks_zh"] = probe["notes"], probe.get("notes_zh") or probe["notes"]
        out["buy_allowed"] = probe["allowed"]

    budget = risk_budget.BUDGET_PCT / 100 * ex["total"]
    value = ex["mv"].get(sym, 0.0)
    upcoming = sorted((c for c in cats if c.get("bucket") == "direct" and c.get("category") == "earnings"),
                      key=lambda c: c.get("days_away", 9))
    night = None
    if upcoming and price:
        move = risk_budget.report_move(sym, _day(upcoming[0].get("date")), price, (atr / price * 100) if atr else None, implied_fn)
        pct = risk_budget.night_loss_pct(move["typical_pct"])
        if pct:
            night = {**move, "days_away": upcoming[0].get("days_away"), "loss_pct": pct, "loss_usd": round(value * pct / 100, 2),
                     "keep_usd": round(budget / (pct / 100), 2)}
    out["event"] = night

    def finish(o):
        _context(sym, note, ex, plan, market, night, budget, cats)
        o["reasons"] = [n["en"] for n in o["notes"]]               # the English lines, as stored before 2026-11-02
        o["action_label"] = ACTION_LABELS.get(o.get("action", ""), {"en": o.get("action", ""), "zh": o.get("action", "")})
        o["labels"] = LABELS
        o["evidence_zh"] = _evidence_zh(o["evidence"])
        for od in o["orders"]:
            reach = od.get("hit_pct")
            od["text"] = {"en": f"Sell {od['shares']} @ ${od['limit']:.2f}" + (f" · reached {reach}% of the time" if reach is not None else "")
                          + (" · before the report" if od.get("before_report") else ""),
                          "zh": f"卖 {od['shares']} 股 @ ${od['limit']:.2f}" + (f" · 历史成交率 {reach}%" if reach is not None else "")
                          + (" · 财报前" if od.get("before_report") else "")}
        return o

    if reports:
        out.update(action="earnings", headline="Reports tonight: the earnings pulse watch runs; no new buys",
                   headline_zh="今晚发财报：财报脉冲模型盯盘，不新买", evidence="tested (earnings pulse champion)")
        if night and price and night["loss_usd"] > budget:
            n = min(int(math.floor(qty)), int(math.ceil((value - night["keep_usd"]) / price)))
            o = risk_rules.gate({"symbol": sym, "side": "sell", "shares": n, "limit": round(price, 2)}, ex, held=qty)
            if o["allowed"] and o["shares"] > 0:
                out["orders"] = [{**o, "atr_multiple": 0.0, "hit_pct": None, "valid_sessions": 1, "before_report": True}]
                out["headline"] = (f"Reports tonight: a bad night could lose ${night['loss_usd']:,.0f}, over the "
                                   f"${budget:,.0f} budget. Sell {o['shares']} at ${price:.2f} before the report")
                out["headline_zh"] = (f"今晚发财报：坏的一晚可能亏 ${night['loss_usd']:,.0f}，超出 ${budget:,.0f} 预算。"
                                      f"财报前以 ${price:.2f} 卖 {o['shares']} 股")
                out["evidence"] = "loss budget (report night) + tested (earnings pulse champion)"
        return finish(out)

    n_cap = risk_rules.trim_shares(qty, ex["mv"].get(sym, 0.0), ex["total"])
    planned = ((plan or {}).get("sell") or {}).get(sym) or {}
    n = max(n_cap, min(int(math.floor(qty)), planned.get("shares") or 0))   # the risk plan may sell more (cluster, cash, budget)
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
        total_sh = sum(o["shares"] for o in orders)
        out.update(action="trim", orders=orders, drop_cause=cause,
                   headline=f"Trim {total_sh} shares into a rebound: " + ", ".join(f"{o['shares']} @ ${o['limit']:.2f}" for o in orders),
                   headline_zh=f"等反弹减仓 {total_sh} 股：" + "、".join(f"{o['shares']} 股 @ ${o['limit']:.2f}" for o in orders),
                   evidence="rule (symbol cap) + reach rates by drop cause" if cause else "rule (symbol cap)")
        note(f"{sym} is {weight:.1f}% of the portfolio, cap {risk_rules.MAX_SYMBOL_PCT:g}% ({n_cap} shares over)",
             f"{sym} 占组合 {weight:.1f}%，上限 {risk_rules.MAX_SYMBOL_PCT:g}%（超出 {n_cap} 股）")
        if plan and plan.get("sell"):
            items = sorted(plan["sell"].items(), key=lambda kv: -kv[1]["usd"])
            parts = ", ".join(f"{s} ${r['usd']:,.0f}" for s, r in items)
            parts_zh = "、".join(f"{s} ${r['usd']:,.0f}" for s, r in items)
            vol = plan.get("vol_before_pct") is not None
            note(f"Risk plan: sell ${plan['total_usd']:,.0f} ({parts}) so chips stay under {risk_rules.MAX_CLUSTER_PCT:g}%, cash over "
                 f"{risk_rules.MIN_CASH_PCT:g}% and a bad day within the budget"
                 + (f"; volatility {plan['vol_before_pct']}% -> {plan['vol_after_pct']}% a year" if vol else ""),
                 f"风险计划：共卖 ${plan['total_usd']:,.0f}（{parts_zh}），让芯片类低于 {risk_rules.MAX_CLUSTER_PCT:g}%、"
                 f"现金高于 {risk_rules.MIN_CASH_PCT:g}%、坏日子亏损在预算内"
                 + (f"；年化波动 {plan['vol_before_pct']}% → {plan['vol_after_pct']}%" if vol else ""))
            note("Beyond each stock's own excess, every dollar comes from the holding whose sale lowers the risk most "
                 "(6-month covariance). No return forecast is used",
                 "超出上限的部分之外，每一美元都从'卖掉后风险下降最多'的那只里出（近 6 个月协方差），不使用涨跌预测", key=False)
            out["evidence"] += " + risk plan"
        if cause:
            note(f"Limits at +1.5 / +2 x ATR: after {rates['label']} they were reached within 15 sessions "
                 f"{rates[1.5]}% / {rates[2.0]}% of the time",
                 f"挂单在 +1.5 / +2 倍 ATR：历史上 {rates['label_zh']}之后，15 天内成交率 {rates[1.5]}% / {rates[2.0]}%")
            note(f"Unfilled, the close 15 sessions later averaged {rates['unfilled']}% against selling at the next open "
                 f"({rates['n']} cases, 102 stocks, 2016-2026)",
                 f"没成交时，15 天后的收盘价平均比次日开盘卖出低 {abs(rates['unfilled'])}%（{rates['n']} 例，102 只股票，2016–2026）", key=False)
        elif stats.get("big_drop"):
            note("Limits at +1.5 / +2 x ATR. The calendar could not tell whether the last large drop was a report: no reach rate quoted",
                 "挂单在 +1.5 / +2 倍 ATR。无法判断最近的大跌是否由财报引起，不给成交率", key=False)
        else:
            note("Limits at +1.5 / +2 x ATR. No drop of >= 2 x ATR in the last 5 sessions: no measured reach rate applies",
                 "挂单在 +1.5 / +2 倍 ATR。最近 5 天没有 ≥ 2 倍 ATR 的下跌，没有可用的成交率", key=False)
        out["after"] = after = _after_sales(sym, orders, ex, price)
        extra = after["cluster_excess"] > 0
        note(f"If both fill: cash {after['cash_pct']:.1f}%, chips {after['cluster_pct']:.1f}%"
             + (f"; ${after['cluster_excess']:,.0f} more of chip names to sell for the cap" if extra else ""),
             f"两笔都成交后：现金 {after['cash_pct']:.1f}%，芯片类 {after['cluster_pct']:.1f}%"
             + (f"；还要再卖 ${after['cluster_excess']:,.0f} 芯片股才到上限" if extra else ""), key=False)
        return finish(out)

    if out.get("buy_allowed"):
        out.update(action="add", headline="A buy fits every rule; no tested signal sizes it",
                   headline_zh="规则允许买入，但没有经过检验的信号来定数量", evidence="rule")
        return finish(out)
    out.update(action="hold", headline="Hold: no rule or tested signal calls for an order",
               headline_zh="持有：没有规则或经过检验的信号要求下单", evidence="default")
    return finish(out)


def _day(s: Optional[str]) -> date:
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return datetime.now(NY).date()


def _context(sym: str, note: Callable, ex: Dict[str, Any], plan: Optional[Dict[str, Any]], market: Optional[Dict[str, Any]],
             night: Optional[Dict[str, Any]], budget: float, cats: List[Dict[str, Any]]) -> None:
    """The loss budget, the next report and the market, under every action."""
    bp = risk_budget.BUDGET_PCT
    if plan and plan.get("bad_day_loss_before") is not None:
        b = plan["bad_day_loss_before"]
        over = b > budget and plan.get("bad_day_loss_after") is not None
        note(f"Loss budget {bp:g}% (${budget:,.0f}): a 1-in-20 bad day would lose ${b:,.0f} ({b / ex['total'] * 100:.1f}%)"
             + (f"; after the plan's sales ${plan['bad_day_loss_after']:,.0f}" if over else ""),
             f"风险预算 {bp:g}%（${budget:,.0f}）：坏的一天（20 天一遇）会亏 ${b:,.0f}（{b / ex['total'] * 100:.1f}%）"
             + (f"，按计划卖出后 ${plan['bad_day_loss_after']:,.0f}" if over else ""))
        note("Checked on 2017-2026: 5.0% of days lost more than this estimate",
             "2017–2026 年检验：只有 5.0% 的日子亏得比这个估计多", key=False)
    if night:
        days = night.get("days_away") or 0
        when, when_zh = ("tonight", "今晚") if days <= 0 else (f"in {days} day(s)", f"{days} 天后")
        src = f"options ±{night['straddle_pct']}%" if night["source"] == "options" else "estimated from the ATR"
        src_zh = f"期权 ±{night['straddle_pct']}%" if night["source"] == "options" else "按 ATR 估计"
        note(f"{sym} reports {when}: a bad night is -{night['loss_pct']}% ({src}), ${night['loss_usd']:,.0f}; "
             f"the budget allows ${night['keep_usd']:,.0f}",
             f"{sym} {when_zh}发财报：坏的一晚 −{night['loss_pct']}%（{src_zh}），约 ${night['loss_usd']:,.0f}；"
             f"预算内最多持有 ${night['keep_usd']:,.0f}")
    if market:
        vix = f"VIX {market['vix']}" if market.get("vix") is not None else "VIX unknown"
        spy = f", SPY {market['spy_5d_pct']:+.2f}% in 5 days" if market.get("spy_5d_pct") is not None else ""
        spy_zh = f"，SPY 5 天 {market['spy_5d_pct']:+.2f}%" if market.get("spy_5d_pct") is not None else ""
        note(f"Market: {market['label']} ({vix}{spy})", f"市场：{market.get('label_zh', market['label'])}（{vix}{spy_zh}）")
        note(f"After {market['n']} such weeks (2016-2026) SPY was up 5 days later {market['up5']}% of the time, mean "
             f"{market['mean5']:+.2f}%, worst tenth {market['p10_5']}%. Context, not a forecast",
             f"历史上 {market['n']} 个类似的周（2016–2026）：5 天后上涨 {market['up5']}%，平均 {market['mean5']:+.2f}%，"
             f"最差 10% 为 {market['p10_5']}%。仅供参考，不是预测", key=False)
    macro = [c for c in cats if c.get("bucket") == "macro" and c.get("impact") == "high" and c.get("days_away", 9) <= 1]
    if macro:
        names = [re.sub(r"^[\U0001F1E6-\U0001F1FF]+\s*", "", c.get("title", "")) for c in macro]   # drop a leading flag
        en = "; ".join(dict.fromkeys([t for t in names if t.isascii()] or names))
        zh = "；".join(dict.fromkeys([t for t in names if not t.isascii()] or names))
        election = any("lection" in t or "选举" in t for t in names)
        note(f"Within a day: {en}", f"一天内的市场事件：{zh}")
        if election:
            note(risk_budget.ELECTIONS, risk_budget.ELECTIONS_ZH, key=False)


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
            analyze: Callable[..., Dict[str, Any]] = lot_advisor.analyze_symbol,
            plan_fn: Optional[Callable[..., Dict[str, Any]]] = None,
            market_fn: Optional[Callable[..., Dict[str, Any]]] = None,
            model_fn: Optional[Callable[..., Optional[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    plan_fn = plan_fn or optimizer.build
    model_fn = model_fn or model_advice.render
    market_fn = market_fn or risk_budget.market_state
    st = state if state is not None else pf.get_state(path=path, cached_quotes=True)
    ex = risk_rules.exposure(st)
    try:
        plan = plan_fn(st, ex, path=path)
    except Exception:  # noqa: BLE001 - without a plan the symbol cap alone sizes the trim
        plan = None
    try:
        market = market_fn(path)
    except Exception:  # noqa: BLE001
        market = None
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
        rule = decide(sym, h, ex, advice, plan, market)
        # What the user reads is the models' advice (2026-11-02); the rule-based action is kept with it.
        try:
            m = model_fn(sym, h, advice.get("stats") or {}, (advice.get("dip_guardrail") or {}).get("anchor_cost"), float(st.get("cash") or 0.0),
                         total=ex["total"], budget_pct=risk_budget.BUDGET_PCT)
        except Exception:  # noqa: BLE001 - no forecast: the rule-based action is shown
            m = None
        if m:
            actions.append({**rule, **m, "reasons": [n["en"] for n in m["notes"]], "blocks": [], "blocks_zh": [],
                            "rule_action": {k: rule.get(k) for k in ("action", "headline", "orders", "evidence", "reasons")}})
        else:
            actions.append(rule)
    out = {"generated_at": datetime.now(NY).isoformat(timespec="seconds"), "symbols": SYMBOLS, "actions": actions,
           "portfolio": {"total": round(ex["total"], 2), "cash": ex["cash"], "cash_pct": ex["cash_pct"], "cluster_pct": ex["cluster_pct"],
                         "max_symbol_pct": risk_rules.MAX_SYMBOL_PCT, "max_cluster_pct": risk_rules.MAX_CLUSTER_PCT,
                         "min_cash_pct": risk_rules.MIN_CASH_PCT,
                         "breaches": risk_rules.summary(ex), "plan": plan, "market": market,
                         "loss_budget_pct": risk_budget.BUDGET_PCT}}
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
