"""Short-term swing lot exit & dip-entry spacing advisor (PROJECT_PLAN Q.10, 2026-10-30).

Bridges the gap between single-stock price models and the user's actual short-term tranche workflow
(`Q.9` per-stock buy/sell lots + `Q.8` 3-layer upcoming binary catalysts):

1. Empirical OOS calibration (102 stocks, 2016-2026 daily bars, 703 out-of-sample >=2xATR drop events):
   - After a >=2.0xATR daily drop, stocks fall at least another 1.0xATR within 3 sessions 42.8% of the
     time, and another 1.5xATR 24.9% of the time -> consecutive dip buys spaced <1.0xATR apart double
     exposure into unfinished post-event drift.
   - Spacing the next limit buy >=1.5xATR below the prior tranche anchor lifts +2.0xATR 15-session
     take-profit hit rate to 54.2% and mean realized swing gain to +2.50% (vs +0.97% chasing Day-1).
   - For open swing tranches, +1.5xATR hits 54.8% within 15 sessions (preferred pre-catalyst harvest
     target when binary macro/earnings risk sits inside 7 days); +2.0xATR is the standard swing target.
2. Inventory-aware guardrails:
   - Evaluates portfolio concentration (single-stock weight & semiconductor/memory/storage cluster weight),
     available cash, PEAD Day-1..3 falling-knife stabilization, and 7-day binary catalysts (`Q.8`).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from . import portfolio as pf
from .config import DB_PATH

NY = ZoneInfo("America/New_York")

SEMI_MEMORY_CLUSTER = {
    "INTC", "TSM", "SKHY", "STX", "WDC", "MU", "NVDA", "AMD", "AVGO",
    "ASML", "AMAT", "LRCX", "KLAC", "QCOM", "MRVL", "ARM", "SMCI", "TXN", "ON", "NXPI",
}

EMPIRICAL_OOS = {
    "universe_stocks": 102,
    "oos_years": "2023-2026",
    "oos_2atr_drop_events": 703,
    "follow_through_1atr_3d_pct": 42.8,
    "follow_through_15atr_3d_pct": 24.9,
    "immediate_tp15_hit_pct": 54.8,
    "immediate_tp15_mean_pct": 0.97,
    "spaced_15atr_fill_pct": 33.6,
    "spaced_15atr_tp20_hit_pct": 54.2,
    "spaced_15atr_mean_pct": 2.50,
}


def _compute_rsi14(closes: List[float]) -> Optional[float]:
    if len(closes) < 15:
        return None
    gains, losses = [], []
    for prev, cur in zip(closes[-15:-1], closes[-14:]):
        diff = cur - prev
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_g = sum(gains) / 14.0
    avg_l = sum(losses) / 14.0
    if avg_l <= 1e-9:
        return 100.0
    rs = avg_g / avg_l
    return round(100.0 - 100.0 / (1.0 + rs), 1)


def _bar_stats(
    symbol: str,
    live_price: Optional[float],
    path: Optional[Path] = None,
    chart_fn: Optional[Callable[[str, str, Optional[Path]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    raw = (chart_fn or pf.get_chart)(symbol, "6m", path)
    candles = [c for c in (raw.get("candles") or []) if c.get("close") is not None and c.get("high") is not None and c.get("low") is not None]
    done = [c for c in candles if not c.get("partial")]
    if len(done) < 16:
        px = float(live_price or (candles[-1]["close"] if candles else 0.0) or 0.0)
        fallback_atr = round(px * 0.035, 4) if px > 0 else 1.0
        return {
            "bars": len(done),
            "atr14": fallback_atr,
            "atr14_pct": 3.5 if px > 0 else None,
            "rsi14": None,
            "sma20": None,
            "drop_1d_pct": 0.0,
            "drop_1d_atr": 0.0,
            "drop_3d_pct": 0.0,
            "drop_3d_atr": 0.0,
            "stabilized": True,
        }

    trs = [
        max(c["high"] - c["low"], abs(c["high"] - p["close"]), abs(c["low"] - p["close"]))
        for p, c in zip(done[-15:-1], done[-14:])
    ]
    atr14 = max(sum(trs) / len(trs), 1e-4)
    closes = [float(c["close"]) for c in done]
    ref_px = float(live_price) if (live_price and live_price > 0) else closes[-1]

    # Compare current live price (if fresher/intraday) or latest done close vs 1-day and 3-day prior closes
    last_bar = candles[-1]
    same_bar = abs(ref_px - float(done[-1]["close"])) / max(ref_px, 1e-6) < 0.002
    c_prev1 = closes[-2] if same_bar and len(closes) >= 2 else closes[-1]
    c_prev3 = closes[-4] if same_bar and len(closes) >= 4 else closes[max(0, len(closes) - 3)]

    chg_1d_pct = (ref_px / c_prev1 - 1.0) * 100.0 if c_prev1 > 0 else 0.0
    chg_3d_pct = (ref_px / c_prev3 - 1.0) * 100.0 if c_prev3 > 0 else 0.0
    drop_1d_atr = max(0.0, (c_prev1 - ref_px) / atr14)
    drop_3d_atr = max(0.0, (c_prev3 - ref_px) / atr14)

    # Stabilization check after a sharp selloff: green body and holding above prior low
    prev_bar = done[-2] if len(done) >= 2 else done[-1]
    cur_open = float(last_bar.get("open") or ref_px)
    cur_low = min(float(last_bar.get("low") or ref_px), ref_px)
    stabilized = bool(drop_3d_atr < 1.8 or (ref_px >= cur_open and cur_low >= float(prev_bar["low"]) * 0.998 and chg_1d_pct > -0.5))

    return {
        "bars": len(done),
        "atr14": round(atr14, 4),
        "atr14_pct": round(atr14 / ref_px * 100.0, 2) if ref_px > 0 else None,
        "rsi14": _compute_rsi14(closes[:-1] + [ref_px] if same_bar else closes + [ref_px]),
        "sma20": round(sum(closes[-20:]) / min(len(closes), 20), 2),
        "drop_1d_pct": round(chg_1d_pct, 2),
        "drop_1d_atr": round(drop_1d_atr, 2),
        "drop_3d_pct": round(chg_3d_pct, 2),
        "drop_3d_atr": round(drop_3d_atr, 2),
        "stabilized": stabilized,
    }


def _upcoming_7d_catalysts(
    symbol: str,
    path: Optional[Path] = None,
    catalysts_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    if catalysts_fn is not None:
        return catalysts_fn(symbol)
    try:
        from . import catalysts
        feed = catalysts.upcoming_feed(scope="all", days=8, symbols=[symbol], include_bellwethers=True, force=False, path=path)
        stock_block = (feed.get("per_stock") or {}).get(symbol) or {}
        out: List[Dict[str, Any]] = []
        for bucket, items in (("direct", stock_block.get("direct") or []),
                              ("spillover", stock_block.get("spillover") or []),
                              ("macro", stock_block.get("macro") or [])):
            for ev in items:
                imp = str(ev.get("impact_level") or ev.get("impact") or "").lower()
                da = ev.get("days_away")
                if bucket == "macro" and da == 0:
                    continue
                if bucket != "direct" and imp != "high":
                    continue
                out.append({
                    "date": ev.get("date"),
                    "days_away": da,
                    "title": ev.get("title_zh") or ev.get("title") or "",
                    "category": ev.get("kind") or ev.get("category") or bucket,
                    "impact": imp or "medium",
                    "bucket": bucket,
                })
        out.sort(key=lambda x: (x.get("days_away") if x.get("days_away") is not None else 99, 0 if x["bucket"] == "direct" else 1))
        return out[:6]
    except Exception:  # noqa: BLE001
        return []


def analyze_symbol(
    symbol: str,
    proposed_buy_price: Optional[float] = None,
    path: Optional[Path] = None,
    state: Optional[Dict[str, Any]] = None,
    chart_fn: Optional[Callable[[str, str, Optional[Path]], Dict[str, Any]]] = None,
    catalysts_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Compute empirically calibrated dip-entry guardrails & per-lot swing exit plans for `symbol`."""
    sym = symbol.strip().upper()
    if not sym:
        raise ValueError("symbol is required")
    db_path = path or DB_PATH
    st = state if state is not None else pf.get_state(path=db_path, cached_quotes=True)
    holdings = st.get("holdings") or []
    holding = next((h for h in holdings if h.get("symbol") == sym), None)
    if holding is None:
        holding = next((c for c in (st.get("closed") or []) if c.get("symbol") == sym), None)
    if holding is None:
        raise ValueError(f"no holding or history for {sym}")

    quote = holding.get("quote") or {}
    live_price = float(quote.get("price") or 0.0) or None
    total_mv = float((st.get("totals") or {}).get("market_value") or 0.0)
    cash = float(st.get("cash") or 0.0)

    sym_mv = float(holding.get("market_value") or 0.0)
    symbol_weight_pct = round(sym_mv / total_mv * 100.0, 1) if total_mv > 0 else 0.0
    semi_mv = sum(float(h.get("market_value") or 0.0) for h in holdings if h.get("symbol") in SEMI_MEMORY_CLUSTER)
    semi_cluster_weight_pct = round(semi_mv / total_mv * 100.0, 1) if total_mv > 0 else 0.0

    stats = _bar_stats(sym, live_price, db_path, chart_fn=chart_fn)
    atr = float(stats["atr14"])
    cats_7d = _upcoming_7d_catalysts(sym, db_path, catalysts_fn=catalysts_fn)
    has_direct_7d = any(c["bucket"] == "direct" for c in cats_7d)
    has_binary_7d = len(cats_7d) > 0

    buy_orders = [b for b in (holding.get("buy_orders") or []) if b.get("status") != "closed" and float(b.get("open_qty") or 0) > 1e-8]
    lowest_open = min(buy_orders, key=lambda b: float(b["unit_cost"])) if buy_orders else None
    latest_open = max(buy_orders, key=lambda b: (str(b.get("trade_date") or ""), int(b.get("id") or 0))) if buy_orders else None

    # Check if the user's two most recent active buy lots were spaced too tightly (< 1.0xATR)
    recent_pair_tight: Optional[Dict[str, Any]] = None
    if len(buy_orders) >= 2 and atr > 0:
        by_recency = sorted(buy_orders, key=lambda b: (str(b.get("trade_date") or ""), int(b.get("id") or 0)), reverse=True)
        b1, b2 = by_recency[0], by_recency[1]
        gap_usd = abs(float(b1["unit_cost"]) - float(b2["unit_cost"]))
        gap_atr = round(gap_usd / atr, 2)
        if gap_atr < 1.0:
            recent_pair_tight = {
                "lot_a_id": b1["id"],
                "lot_a_cost": round(float(b1["unit_cost"]), 2),
                "lot_b_id": b2["id"],
                "lot_b_cost": round(float(b2["unit_cost"]), 2),
                "gap_usd": round(gap_usd, 2),
                "gap_atr": gap_atr,
            }

    # Required spacing escalates from 1.5xATR to 2.0xATR when binary risk, unstabilized plunge, or heavy concentration is present
    elevated_risk = has_binary_7d or not stats["stabilized"] or semi_cluster_weight_pct >= 75.0 or symbol_weight_pct >= 25.0
    min_spacing_atr = 2.0 if elevated_risk else 1.5
    anchor_cost = float(lowest_open["unit_cost"]) if lowest_open else (live_price or 0.0)
    raw_limit = anchor_cost - min_spacing_atr * atr if anchor_cost > 0 else 0.0
    if live_price and live_price > 0 and raw_limit > live_price:
        suggested_buy_limit = round(max(0.01, live_price - 0.5 * atr), 2)
    else:
        suggested_buy_limit = round(max(0.01, raw_limit), 2) if raw_limit > 0 else None

    eval_px = float(proposed_buy_price) if (proposed_buy_price and proposed_buy_price > 0) else live_price
    spacing_vs_lowest_atr = round((anchor_cost - eval_px) / atr, 2) if (lowest_open and eval_px and atr > 0) else None

    reasons_zh: List[str] = []
    verdict = "green"

    if lowest_open and spacing_vs_lowest_atr is not None:
        if spacing_vs_lowest_atr < 1.0:
            verdict = "red"
            reasons_zh.append(
                f"距最低持仓批次 #{lowest_open['id']} (${lowest_open['unit_cost']:.2f}) 仅拉开 {spacing_vs_lowest_atr:+.2f}×ATR，"
                f"低于 {min_spacing_atr:.1f}×ATR 安全分批间距（建议挂单 ≤ ${suggested_buy_limit:.2f}）"
            )
        elif spacing_vs_lowest_atr < min_spacing_atr:
            verdict = "amber"
            reasons_zh.append(
                f"距最低持仓批次 #{lowest_open['id']} (${lowest_open['unit_cost']:.2f}) 已拉开 {spacing_vs_lowest_atr:.2f}×ATR，"
                f"但当前风险环境下建议拉宽至 {min_spacing_atr:.1f}×ATR（≤ ${suggested_buy_limit:.2f}）"
            )
        else:
            reasons_zh.append(
                f"距最低持仓批次 #{lowest_open['id']} (${lowest_open['unit_cost']:.2f}) 已拉开 {spacing_vs_lowest_atr:.2f}×ATR（满足 ≥{min_spacing_atr:.1f}×ATR 间距）"
            )

    if recent_pair_tight:
        reasons_zh.append(
            f"历史复盘：最近两笔活跃批次 #{recent_pair_tight['lot_b_id']} (${recent_pair_tight['lot_b_cost']:.2f}) 与 "
            f"#{recent_pair_tight['lot_a_id']} (${recent_pair_tight['lot_a_cost']:.2f}) 间距仅 ${recent_pair_tight['gap_usd']:.2f} "
            f"({recent_pair_tight['gap_atr']:.2f}×ATR)，下一笔务必拉开档位避免重蹈覆辙"
        )

    if not stats["stabilized"] or stats["drop_1d_atr"] >= 2.0 or stats["drop_3d_atr"] >= 2.2:
        if verdict == "green":
            verdict = "amber"
        reasons_zh.append(
            f"急跌未企稳（近1日 {stats['drop_1d_pct']:+.1f}% / 近3日 {stats['drop_3d_pct']:+.1f}%）："
            f"102股样本外统计显示单日跌超 2×ATR 后，3日内有 {EMPIRICAL_OOS['follow_through_1atr_3d_pct']}% 概率再探 1×ATR，宜等缩量阳线或挂深档限价单"
        )

    if has_binary_7d:
        if has_direct_7d:
            verdict = "red" if (spacing_vs_lowest_atr is not None and spacing_vs_lowest_atr < 1.5) else ("amber" if verdict == "green" else verdict)
        elif verdict == "green":
            verdict = "amber"
        ev_titles = "、".join(f"{c['date'][5:]} {c['title'][:18]}" for c in cats_7d[:3])
        reasons_zh.append(f"未来 7 天有 {len(cats_7d)} 个高冲击催化剂（{ev_titles}），事件前宜控制单次加仓比例")

    if sym in SEMI_MEMORY_CLUSTER and semi_cluster_weight_pct >= 75.0:
        if verdict == "green" and symbol_weight_pct >= 25.0:
            verdict = "amber"
        reasons_zh.append(
            f"组合集中度提示：半导体/存储板块已占总市值 {semi_cluster_weight_pct:.1f}%（{sym} 单股占 {symbol_weight_pct:.1f}%）"
        )

    headline_map = {
        "green": f"🟢 档位与波动率达标：可按 ≤ ${suggested_buy_limit:.2f} 分批挂单" if suggested_buy_limit else "🟢 当前波动率与仓位处于可控区间",
        "amber": f"🟡 建议等企稳或拉宽挂单至 ≤ ${suggested_buy_limit:.2f}（≥{min_spacing_atr:.1f}×ATR）" if suggested_buy_limit else "🟡 建议等待缩量企稳信号后再分批介入",
        "red": f"🔴 暂缓追单 / 档位过密：下一安全加仓区 ≤ ${suggested_buy_limit:.2f}（≥{min_spacing_atr:.1f}×ATR）" if suggested_buy_limit else "🔴 风险集中，暂缓在当前价位密集加仓",
    }

    # Per-lot swing exit targets
    lot_exit_plans: List[Dict[str, Any]] = []
    for b in sorted(buy_orders, key=lambda x: float(x["unit_cost"])):
        uc = float(b["unit_cost"])
        oq = float(b["open_qty"])
        tp1 = round(uc + 1.5 * atr, 2)
        tp2 = round(uc + 2.0 * atr, 2)
        pnl_atr = round((live_price - uc) / atr, 2) if (live_price and atr > 0) else None
        trail = round(live_price - 1.0 * atr, 2) if (live_price and pnl_atr is not None and pnl_atr >= 1.2) else None
        rec_px = tp1 if has_binary_7d else tp2

        if pnl_atr is not None and pnl_atr >= 2.0:
            code = "harvest_now"
            adv = f"🟢 已达 +2.0×ATR 目标区（+{pnl_atr:.1f}×ATR）：建议止盈落袋或设 ${trail:.2f} (1×ATR) 移动止盈"
        elif pnl_atr is not None and pnl_atr >= 1.2:
            if has_binary_7d:
                code = "harvest_now"
                adv = f"🟢 已浮盈 +{pnl_atr:.1f}×ATR 且临近7日内二元事件：建议在 ${tp1:.2f}~${live_price:.2f} 优先锁利回收现金"
            else:
                code = "trail_protect"
                adv = f"🟢 浮盈 +{pnl_atr:.1f}×ATR：可挂 ${tp2:.2f} (+2×ATR) 止盈，并用 ${trail:.2f} 作保护止损"
        elif pnl_atr is not None and pnl_atr > 0.05:
            code = "in_profit"
            adv = (
                f"🟢 小幅浮盈（+{pnl_atr:.2f}×ATR）：事件前稳健目标 ${tp1:.2f}（+1.5×ATR），标准波段目标 ${tp2:.2f}（+2×ATR）"
            )
        elif pnl_atr is not None and pnl_atr >= -1.0:
            code = "near_cost"
            adv = f"🟡 距批次成本 {pnl_atr:+.2f}×ATR：反弹至 ${rec_px:.2f} 可做波段减仓周转"
        else:
            code = "underwater_swing"
            adv = f"🔵 暂套 {pnl_atr:+.2f}×ATR：无需等总均价（${float(holding.get('avg_cost') or uc):.2f}），反弹至 ${tp1:.2f} 即完成本批盈利周转" if pnl_atr is not None else f"波段参考止盈价 ${rec_px:.2f}"

        lot_exit_plans.append({
            "buy_id": int(b["id"]),
            "trade_date": b["trade_date"],
            "unit_cost": round(uc, 4),
            "open_qty": oq,
            "open_pnl": b.get("open_pnl"),
            "open_pnl_pct": b.get("open_pnl_pct"),
            "pnl_atr": pnl_atr,
            "tp1_price": tp1,
            "tp1_gain_usd": round((tp1 - uc) * oq, 2),
            "tp2_price": tp2,
            "tp2_gain_usd": round((tp2 - uc) * oq, 2),
            "trail_stop_price": trail,
            "recommended_price": rec_px,
            "status_code": code,
            "advice_zh": adv,
        })

    return {
        "symbol": sym,
        "generated_at": datetime.now(NY).isoformat(),
        "live_price": live_price,
        "cash": round(cash, 2),
        "symbol_weight_pct": symbol_weight_pct,
        "semi_cluster_weight_pct": semi_cluster_weight_pct,
        "is_semi_cluster": sym in SEMI_MEMORY_CLUSTER,
        "stats": stats,
        "empirical_oos": EMPIRICAL_OOS,
        "catalysts_7d": cats_7d,
        "has_binary_7d": has_binary_7d,
        "dip_guardrail": {
            "verdict": verdict,
            "headline_zh": headline_map[verdict],
            "reasons_zh": reasons_zh,
            "min_spacing_atr": min_spacing_atr,
            "anchor_lot_id": int(lowest_open["id"]) if lowest_open else None,
            "anchor_cost": round(anchor_cost, 4) if anchor_cost else None,
            "latest_lot_id": int(latest_open["id"]) if latest_open else None,
            "latest_cost": round(float(latest_open["unit_cost"]), 4) if latest_open else None,
            "evaluated_buy_price": round(eval_px, 4) if eval_px else None,
            "spacing_vs_lowest_atr": spacing_vs_lowest_atr,
            "suggested_buy_limit": suggested_buy_limit,
            "recent_pair_tight": recent_pair_tight,
        },
        "lot_exit_plans": lot_exit_plans,
    }
