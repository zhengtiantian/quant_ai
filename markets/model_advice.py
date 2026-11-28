"""The daily action as the models see it (PROJECT_PLAN Q.16, 2026-11-02).

quant_data's research/models/daily_advice.py runs three times each weekday: for each holding it reads
the causes of recent large drops from the news store (summarised by the local model from the
headlines), and forecasts the next 5, 20 and 60 sessions (LightGBM trained walk-forward on 102 stocks
2016-2026, with an MLP challenger), calibrated so that P(up) and the 80% range come true as often as
they say. It writes ADVICE_FILE; this turns it into the action the user reads, in English and Chinese.
The rule-based action is still computed and kept beside it, not shown.

The direction call (reduce / hold / add) is shown only if it beat holding in the years its thresholds
did not see (2023-2026). On 2026-11-02 it did not (days called "reduce" went on to do better than
"hold"), so the action says hold and says why. The median forecast is not shown: its rank IC was ~0.

One rule is kept on top of the models, by the user's wish (2026-11-02, "以后就完全规避这个补仓过密的
行为"): an add is shown only when the stock has stopped falling and the price is at least 2 x ATR under
the cheapest lot held; otherwise the next price that would qualify is shown instead.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ADVICE_FILE = Path(os.getenv("ADVICE_FILE", "/Volumes/Data4T/daily_advice/latest.json"))
ADD_SPACING_ATR = 2.0
ADD_SPACING_STRESS_ATR = 3.0
REDUCE_SHARE = 1 / 3            # a "reduce" call sells a third of the position
ADD_SHARE = 0.10                # an "add" buys 10% of the shares held
STALE_HOURS = 36
SECTION_ORDER = {"market": 0, "why": 1, "recover": 2, "events": 3, "odds": 4, "add": 5, "quality": 6}

CALLS = {"reduce": ("Reduce", "减仓"), "hold": ("Hold", "持有"), "add": ("Add", "加仓"), "recover": ("Recover", "解套")}
HZ = {"5": ("next 5 days", "未来 5 天"), "20": ("next 20 days", "未来 20 天"), "60": ("next 60 days", "未来 60 天")}
LABELS = {"market": {"en": "Market & macro", "zh": "大盘与宏观"}, "why": {"en": "Why it fell", "zh": "为什么跌"},
          "recover": {"en": "Getting out", "zh": "解套"}, "events": {"en": "Coming up", "zh": "近期事件"},
          "odds": {"en": "Outlook", "zh": "预测"}, "add": {"en": "Adding", "zh": "补仓"},
          "quality": {"en": "How good is the model", "zh": "模型可信度"},
          "more": {"en": "More", "zh": "更多细节"}, "less": {"en": "Less", "zh": "收起"},
          "basis": {"en": "Basis", "zh": "依据"}, "blocked": {"en": "Buying is blocked", "zh": "买入被拦截"}}


def load(path: Path = ADVICE_FILE) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _hours_old(ts: str) -> float:
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 3600
    except ValueError:
        return 1e9


def render(sym: str, holding: Dict[str, Any], stats: Dict[str, Any], lowest_lot: Optional[float],
           cash: float, data: Optional[Dict[str, Any]] = None, total: Optional[float] = None,
           budget_pct: Optional[float] = None, market: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The model's action for one holding, or None when there is no forecast for it."""
    data = data if data is not None else load()
    m = (data or {}).get("symbols", {}).get(sym)
    if not m:
        return None
    price = float((holding.get("quote") or {}).get("price") or m["close"])
    qty = float(holding.get("quantity") or 0.0)
    atr = float(stats.get("atr14") or 0.0) or None
    reliable = bool(m.get("direction_reliable"))
    notes: List[Dict[str, Any]] = []

    def note(section, en, zh, key=True):
        notes.append({"section": section, "en": en, "zh": zh, "key": key})

    tighten = bool((market or {}).get("tighten_buys"))
    spacing_atr = ADD_SPACING_STRESS_ATR if tighten else ADD_SPACING_ATR

    # Market panic / chip stress / macro headwind context (always visible under stress, never dropped by model override)
    if market and market.get("severity") in ("critical", "warn"):
        m_title = market.get("title") or {}
        m_detail = market.get("detail") or {}
        pills_txt = " · ".join(f"{p['label']} {p['value']}" for p in (market.get("pills") or []) if p.get("stress"))
        en_head = m_title.get("en") or market.get("label") or "Market stress"
        zh_head = m_title.get("zh") or market.get("label_zh") or "市场承压"
        if pills_txt:
            en_head = f"{en_head} ({pills_txt})"
            zh_head = f"{zh_head}（{pills_txt}）"
        note("market", en_head, zh_head, key=True)
        if m_detail.get("en") and m_detail.get("zh"):
            note("market", m_detail["en"], m_detail["zh"], key=False)

    # the call, and the user's rule against adding too close to the last lot or into a fall / market panic
    call = m["call"] if reliable else "hold"
    blocked_en, blocked_zh = [], []
    next_add = round(lowest_lot - spacing_atr * atr, 2) if (lowest_lot and atr) else None
    if call == "add":
        if tighten:
            blocked_en.append("market panic / chip sector selloff (buy guardrail paused)")
            blocked_zh.append("市场恐慌 / 半导体板块急跌，暂停补仓")
        if stats.get("stabilized") is False:
            blocked_en.append("still falling")
            blocked_zh.append("还在急跌，没企稳")
        if next_add is not None and price > next_add:
            blocked_en.append(f"within {spacing_atr:g} x ATR of your cheapest lot (${lowest_lot:.2f})")
            blocked_zh.append(f"离你最便宜的一批（${lowest_lot:.2f}）不到 {spacing_atr:g} 倍 ATR")
        if blocked_en:
            call = "hold"
    label_en, label_zh = CALLS[call]

    orders = []
    if call == "reduce":
        n = max(1, int(math.floor(qty * REDUCE_SHARE)))
        orders.append({"symbol": sym, "side": "sell", "shares": n, "limit": round(price, 2),
                       "text": {"en": f"Sell {n} @ ${price:.2f} (a third)", "zh": f"卖 {n} 股 @ ${price:.2f}（三分之一）"}})
    elif call == "add":
        n = max(1, int(round(qty * ADD_SHARE)))
        orders.append({"symbol": sym, "side": "buy", "shares": n, "limit": round(price, 2),
                       "text": {"en": f"Buy {n} @ ${price:.2f} (10% more)", "zh": f"买 {n} 股 @ ${price:.2f}（加 10%）"}})

    # why it fell: the news behind the recent large drops
    driver_tags = {
        "company_specific": ("[Company] ", "【公司事件】"),
        "peer_spillover": ("[Peer spillover] ", "【同行传导】"),
        "sector_macro": ("[Sector/Macro] ", "【板块共振】"),
    }
    events = m.get("events") or []
    for e in reversed(events):
        en_tag, zh_tag = driver_tags.get(e.get("driver_type"), ("", ""))
        note("why", f"{e['date'][5:]} {e['ret_pct']:+.1f}%: {en_tag}{e['cause_en']}",
             f"{e['date'][5:]} {e['ret_pct']:+.1f}%：{zh_tag}{e['cause_zh']}")
        if e.get("sources"):
            note("why", "Sources: " + " | ".join(x["title"] for x in e["sources"][:2]),
                 "来源：" + " | ".join(x["title"] for x in e["sources"][:2]), key=False)
    if not events:
        note("why", "No session down 5% or more in the last 30", "最近 30 个交易日没有单日跌 5% 以上")

    # getting out (Q.16.7): the barriers, what similar situations did held or managed, and the loss budget
    rec_orders = _recover(sym, m, price, qty, cost=float(holding.get("avg_cost") or 0.0), total=total, budget_pct=budget_pct,
                          study=(data or {}).get("news_study") or {}, note=note)
    if rec_orders and call == "hold":
        orders, call = rec_orders, "recover"
        label_en, label_zh = "Recover", "解套"

    _events_ahead(sym, m, (data or {}).get("calendar_study") or {}, note)

    # the outlook: calibrated odds and the 80% range per horizon
    if not reliable:
        note("odds", "Direction: no reliable signal. The models' up / down calls did not beat holding in 2023-2026, so the advice is hold",
             "方向：没有可靠信号。模型的涨跌判断在 2023–2026 年没有跑赢持有，所以建议持有")
    h = m["horizons"]
    for k in ("5", "20", "60"):
        x = h.get(k) or h.get(int(k))
        if not x:
            continue
        en, zh = HZ[k]
        lo, hi = price * (1 + x["q10"] / 100), price * (1 + x["q90"] / 100)
        note("odds", f"{en.capitalize()}: up in {x['p_up']:.0f}% of such cases; 8 in 10 between {x['q10']:+.0f}% and {x['q90']:+.0f}% "
             f"(${lo:.2f}-${hi:.2f})",
             f"{zh}：类似情况上涨的概率 {x['p_up']:.0f}%；十有八九在 {x['q10']:+.0f}% ~ {x['q90']:+.0f}% 之间（${lo:.2f}–${hi:.2f}）",
             key=(k != "60"))

    # adding
    tranche = max(1, int(round(qty * ADD_SHARE)))
    if blocked_en:
        note("add", f"The model leans to adding, but not here: {'; '.join(blocked_en)}. The next add worth considering is "
             + (f"${next_add:.2f} or lower" if next_add else "after it stops falling"),
             f"模型偏向加仓，但现在不补：{'；'.join(blocked_zh)}。下一次可考虑补仓价："
             + (f"${next_add:.2f} 或更低" if next_add else "企稳之后"))
    elif call != "add" and next_add is not None:
        stress_en = " — widened during market/sector stress" if tighten else ""
        stress_zh = "，市场/板块承压期已放宽间距" if tighten else ""
        note("add", f"No add now. If you add later: ${next_add:.2f} or lower ({spacing_atr:g} x ATR under your cheapest lot ${lowest_lot:.2f}{stress_en}), "
             f"at most {tranche} shares at a time",
             f"现在不补仓。以后要补：${next_add:.2f} 或更低（比最便宜的一批 ${lowest_lot:.2f} 低 {spacing_atr:g} 倍 ATR{stress_zh}），每次最多 {tranche} 股")
    if call == "add" and orders and cash < orders[0]["shares"] * price:
        note("add", f"Cash ${cash:,.0f}: the add needs money from a sale first", f"现金 ${cash:,.0f}：要补仓得先卖出别的换现金")

    # how good the model is
    q = (data or {}).get("model_quality", {}).get("20") or {}
    dt = ((data or {}).get("decision_test") or {}).get("2023-2026 (held out for thresholds)") or {}
    if q:
        note("quality", f"Out of sample 2019-2026, the 20-day forecast ranked stocks with IC {q.get('ic')} (t {q.get('ic_t')}): no real "
             "skill at picking direction. The odds and ranges above are calibrated to what actually happened",
             f"样本外（2019–2026）：20 天预测的排序相关性 IC {q.get('ic')}（t {q.get('ic_t')}），对方向基本没有判断力。"
             "上面的概率和区间已按历史实际结果校准", key=False)
    if dt.get("reduce") and dt.get("hold"):
        note("quality", f"2023-2026: days it called reduce went {dt['reduce']['mean20']:+.2f}% over 20 days, hold {dt['hold']['mean20']:+.2f}% "
             f"(t {dt.get('reduce_minus_hold_t')}): the calls are not used",
             f"2023–2026 年：模型说'减仓'的日子之后 20 天平均 {dt['reduce']['mean20']:+.2f}%，'持有' {dt['hold']['mean20']:+.2f}%"
             f"（t = {dt.get('reduce_minus_hold_t')}），所以不采用它的涨跌判断", key=False)
    age = _hours_old(data.get("generated_at", ""))
    stamp = data.get("generated_at", "?")[:16].replace("T", " ")
    note("quality", f"Forecast made {stamp} UTC from prices to {m['as_of']}" + (" — out of date" if age > STALE_HOURS else ""),
         f"预测生成于 {stamp} UTC，行情截至 {m['as_of']}" + ("（已过时）" if age > STALE_HOURS else ""), key=False)

    notes.sort(key=lambda n: SECTION_ORDER.get(n["section"], 9))
    x20 = h.get("20") or h.get(20)
    no_sig_en, no_sig_zh = ("" if reliable else " (no reliable direction signal)"), ("" if reliable else "（没有可靠的方向信号）")
    if call == "recover":
        head_en = "Getting out: " + "; ".join(o["text"]["en"] for o in orders)
        head_zh = "解套：" + "；".join(o["text"]["zh"] for o in orders)
    else:
        head_en = f"Model: {label_en.lower()}{no_sig_en}" + (f" — {orders[0]['text']['en']}" if orders else "") + \
            f". 20 days: 8 in 10 between {x20['q10']:+.0f}% and {x20['q90']:+.0f}%"
        head_zh = f"模型建议：{label_zh}{no_sig_zh}" + (f"——{orders[0]['text']['zh']}" if orders else "") + \
            f"。未来 20 天十有八九在 {x20['q10']:+.0f}% ~ {x20['q90']:+.0f}% 之间"
    return {"action": call, "action_label": {"en": label_en, "zh": label_zh}, "headline": head_en, "headline_zh": head_zh,
            "notes": notes, "orders": orders, "evidence": "models (LightGBM walk-forward 2019-2026) + news causes (local model)",
            "evidence_zh": "模型（LightGBM，2019–2026 年滚动样本外验证）+ 新闻原因（本地大模型）", "model": m, "stale": age > STALE_HOURS,
            "labels": LABELS}


def _recover(sym: str, m: Dict[str, Any], price: float, qty: float, cost: float, total: Optional[float], budget_pct: Optional[float],
             study: Dict[str, Any], note) -> List[Dict[str, Any]]:
    """For a holding under water: the 5-day barrier odds (the one barrier model that beat the base rate,
    calibrated), what similar situations did over 20 sessions held to the end or managed with a sell at
    +1.5 x ATR and a stop at -2 x ATR, and the news drift. No orders and no loss-budget sizing: the user
    asked for the models' and the data's view only. The barrier outcomes come from the walk-forward days of 102 stocks (2019-2026; today's large
    caps, so holding looks better there than it would across all stocks)."""
    b = m.get("barrier")
    if not b or qty <= 0 or price >= cost:
        return []
    s5, s20 = b["sets"]["5"], b["sets"]["20"]
    m5 = b["model_5"]
    value = qty * price
    sit_en = {"deep_badnews": "30%+ under its 1-year high after bad news / a guidance cut", "deep": "30%+ under its 1-year high",
              "normal_badnews": "after bad news / a guidance cut", "normal": "no special situation"}[b["situation"]]
    sit_zh = {"deep_badnews": "距一年高点 30% 以上，且近期有坏消息 / 指引下调", "deep": "距一年高点 30% 以上",
              "normal_badnews": "近期有坏消息 / 指引下调", "normal": "无特殊情况"}[b["situation"]]
    # The model foresees the size of a move, not its side (Q.17 double sort, 2026-11-18: within each fifth of
    # 1 - P(none), the share of P(lower) added nothing): the chance of a move leads, the split is marked weak.
    moved = m5["p_upper"] + m5["p_lower"]
    note("recover", f"Next 5 days (model, calibrated): {moved:.0f}% the price reaches ${s5['sell_at']:.2f} or ${s5['stop_at']:.2f} "
         f"(a move of about 1-1.5 x ATR), {m5['p_none']:.0f}% neither. Which comes first is weak evidence: up {m5['p_upper']:.0f}% / "
         f"down {m5['p_lower']:.0f}% (the direction part failed its test: the model foresees the size of a move, not its side)",
         f"未来 5 天（模型，已校准）：碰到 ${s5['sell_at']:.2f} 或 ${s5['stop_at']:.2f}（约 1–1.5 倍 ATR 的波动）的概率 {moved:.0f}%，"
         f"都碰不到 {m5['p_none']:.0f}%。先碰哪一边只能参考：先涨 {m5['p_upper']:.0f}% / 先跌 {m5['p_lower']:.0f}%"
         f"（方向部分没通过检验：模型能预示波动大小，不能预示方向）")
    hold_loss, bar_loss = value * -s20["hold_worst5"] / 100, value * -s20["barrier_worst5"] / 100
    note("recover", f"20 days in this situation ({sit_en}; {s20['n']:,} past cases): holding averaged {s20['hold_mean']:+.1f}%, worst 5% "
         f"{s20['hold_worst5']:.0f}% (${hold_loss:,.0f} on this position); selling at ${s20['sell_at']:.2f} or stopping at ${s20['stop_at']:.2f} "
         f"averaged {s20['barrier_mean']:+.1f}%, worst 5% {s20['barrier_worst5']:.0f}% (${bar_loss:,.0f})",
         f"20 天，同类情况（{sit_zh}；历史 {s20['n']:,} 例）：一直拿着平均 {s20['hold_mean']:+.1f}%，最坏 5% {s20['hold_worst5']:.0f}%"
         f"（这个仓位约 ${hold_loss:,.0f}）；挂 ${s20['sell_at']:.2f} 止盈、${s20['stop_at']:.2f} 止损，平均 {s20['barrier_mean']:+.1f}%，"
         f"最坏 5% {s20['barrier_worst5']:.0f}%（约 ${bar_loss:,.0f}）")
    gc = study.get("guidance cut") or {}
    if b.get("news_20", {}).get("guide_cut_20", 0) > 0 and gc.get("60d_vs_all"):
        x = gc["60d_vs_all"]
        note("recover", f"{int(b['news_20']['guide_cut_20'])} guidance-cut articles in 20 sessions: after guidance cuts stocks trailed the market "
             f"by {abs(x['mean']):.1f}% more over 60 sessions (t {x['t']}, {gc.get('days', 0):,} cases). Do not add",
             f"近 20 天有 {int(b['news_20']['guide_cut_20'])} 篇指引下调的报道：历史上指引下调后 60 天平均再跑输大盘 {abs(x['mean']):.1f}%"
             f"（t {x['t']}，{gc.get('days', 0):,} 例）。不要补仓")
    note("recover", "Holding averaged more because these 102 stocks are today's large caps (survivors); the stop is what caps the worst case",
         "一直拿着平均更好，部分是因为这 102 只都是今天的大公司（幸存者偏差）；挂止损的作用是限制最坏情况", key=False)
    return []          # no sizing by the loss budget here: the user asked for the models' view only (2026-11-02)


def _events_ahead(sym: str, m: Dict[str, Any], study: Dict[str, Any], note) -> None:
    """Scheduled events ahead and what such sessions did in 2016-2026 (quant_data calendar_events.py):
    small effects, shown as context; the 5-day barrier model already takes them in."""
    cal = (m.get("barrier") or {}).get("calendar")
    if not cal:
        return
    near = 10
    fa, nf = study.get("fomc_ahead_5d") or {}, study.get("no_fomc_5d") or {}
    if cal["to_fomc"] <= 5 and fa:
        note("events", f"Fed decision in {cal['to_fomc']} sessions: with one within 5 sessions SPY averaged {fa['mean']:+.2f}% over 5 days, "
             f"{nf.get('mean', 0):+.2f}% otherwise", f"{cal['to_fomc']} 个交易日后美联储议息：历史上议息前 5 天 SPY 平均 {fa['mean']:+.2f}%，"
             f"没有议息时 {nf.get('mean', 0):+.2f}%")
    else:
        note("events", f"Next Fed decision in {cal['to_fomc']} sessions", f"下次美联储议息在 {cal['to_fomc']} 个交易日后", key=cal["to_fomc"] <= near)
    ox = study.get("opex_week") or {}
    if cal["to_opex"] <= 4 and ox:
        note("events", f"Options expiration in {cal['to_opex']} sessions: in expiration weeks SPY averaged {ox['mean']:+.2f}% a day against "
             f"other days (t {ox['t']})", f"{cal['to_opex']} 个交易日后期权到期：历史上期权到期周 SPY 每天比平常 {ox['mean']:+.2f}%（t {ox['t']}）")
    else:
        note("events", f"Options expiration in {cal['to_opex']} sessions", f"期权到期在 {cal['to_opex']} 个交易日后", key=False)
    if cal["to_nfp"] <= 5:
        note("events", f"Jobs report in {cal['to_nfp']} sessions (no reliable effect on the day in 2016-2026)",
             f"{cal['to_nfp']} 个交易日后公布非农（2016–2026 年发布当天没有可靠的涨跌效应）", key=False)
    if cal.get("pre_holiday"):
        ph = study.get("pre_holiday") or {}
        note("events", f"The session before a market holiday: {ph.get('mean', 0):+.2f}% a day against others (t {ph.get('t')}, not significant)",
             f"明天休市前的交易日：历史上比平常 {ph.get('mean', 0):+.2f}%（t {ph.get('t')}，不显著）", key=False)
    rd = cal.get("to_report_days")
    br = study.get("day_before_report") or {}
    if rd is not None:
        if rd <= 2 and br:
            note("events", f"{sym} reports in {rd} day(s): the session before a report beat the market by {br['mean']:+.2f}% on average (t {br['t']})",
                 f"{sym} {rd} 天后发财报：历史上财报前一天平均比大盘多涨 {br['mean']:+.2f}%（t {br['t']}）")
        else:
            note("events", f"{sym} reports in {rd} days", f"{sym} 的下次财报在 {rd} 天后", key=rd <= 14)
