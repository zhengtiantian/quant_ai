"""Portfolio review agent (F.17 / F13) — a second opinion on rule & real-portfolio actions.

Division of labour:
  - Code gathers context, picks decisions to review, fetches evidence, and runs
    all arithmetic portfolio checks (weight caps, cluster concentration, cash floor,
    regime exposure, market weather panic).
  - LLM evaluates qualitative news/catalysts around a move (`agree` vs `flag`).
  - Three explicit gates (now backed by `grounding.py` + `injection_guard.py`) validate
    every verdict:
      1. Gate 1 (Schema): JSON structure & value bounds + 1-shot repair round-trip
      2. Gate 2 (Grounding): per-tool citation attribution + tolerance + pct/decimal match
      3. Gate 3 (Business rules): covered symbol validation & demotion of ungrounded flags

Supports:
  - `mode="real"`: reviews user's actual holdings (`get_my_holdings`), `get_daily_action`
    recommendations (trim/add/drawdowns/overweight), active advisor limit orders, and
    real risk limits (`30%` single-stock cap / `70%` chip cluster / `10%` cash floor / VIX).
  - `mode="paper"`: classic paper-portfolio review (`track_positions` exits + 5%/40% rules).
  - `mode="both"`: combines both views in one report.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
import uuid
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

import grounding
import injection_guard
import mcp_client
from agent import _chat as _agent_chat, _content

_chat = _agent_chat
THINK_PREFILL = "<think>\n\n</think>\n\n"

LOCAL_MONGO_URI = os.getenv(
    "LOCAL_MONGO_URI", "mongodb://root:root@localhost:37018/?authSource=admin"
).strip()

# How many rule decisions to review in one run.
MAX_DECISIONS = int(os.getenv("PORTFOLIO_MAX_DECISIONS", "5"))

# Paper portfolio limits (kept intact for backward compatibility & paper tests).
MAX_POSITION_WEIGHT_PCT = float(os.getenv("PORTFOLIO_MAX_WEIGHT_PCT", "5"))
CONCENTRATION_WARN_PCT = float(os.getenv("PORTFOLIO_CONCENTRATION_PCT", "40"))

# Real personal portfolio limits (mirrors quant_ai/markets/risk_rules.py).
REAL_MAX_POSITION_WEIGHT_PCT = float(os.getenv("REAL_PORTFOLIO_MAX_WEIGHT_PCT", "30"))
REAL_CONCENTRATION_WARN_PCT = float(os.getenv("REAL_PORTFOLIO_CONCENTRATION_PCT", "40"))
REAL_MIN_CASH_PCT = float(os.getenv("REAL_PORTFOLIO_MIN_CASH_PCT", "10"))

VERDICTS = ("agree", "flag")

REVIEW_PROMPT = (
    "You are reviewing one portfolio decision or rule action. You are not second-guessing "
    "deterministic rules for their own sake: they are usually right, and agreeing is expected. "
    "Flag only when evidence in the observations contradicts the rule — e.g., a sector-wide "
    "whipsaw already reversing with no company-specific negative news, or an imminent high-impact "
    "catalyst / negative news headline that contradicts an add.\n\n"
    "Reply with JSON only, no prose around it:\n"
    "{\n"
    '  "verdict": "agree" | "flag",\n'
    '  "confidence": 0.0-1.0,\n'
    '  "reasoning": "one or two sentences",\n'
    '  "evidence": [{"tool": "...", "field": "...", "value": <number or string>}]\n'
    "}\n\n"
    "Every entry in `evidence` must come strictly from the named `tool`'s observations shown to you. "
    "Do not invent numbers or symbols.\n"
    + injection_guard.UNTRUSTED_DATA_RULE
)

_HISTORY_COLL = None


def _build_system_prompt(lang: str = "en") -> str:
    if (lang or "en").strip().lower().startswith("zh"):
        return (
            REVIEW_PROMPT
            + "\nWrite `reasoning` in concise Simplified Chinese (简体中文), while keeping JSON keys "
            "and `verdict` strictly in English (`agree` or `flag`)."
        )
    return REVIEW_PROMPT


# =====================================================
# Tool access
# =====================================================

def _call(name: str, args: dict | None = None) -> Any:
    """Invoke an MCP tool and parse its JSON. Returns None on any failure."""
    try:
        raw = mcp_client.get_client().call_tool(name, args or {})
        return json.loads(raw)
    except Exception:  # noqa: BLE001 - absence of evidence is handled by the caller
        return None


# =====================================================
# Grounding & Schema helpers (delegating to grounding.py)
# =====================================================

def _observed_values(observations: Iterable[Any]) -> set[str]:
    return grounding.flatten_observed_values(observations)


def _is_grounded(value: Any, observed: set[str]) -> bool:
    return grounding.is_value_grounded(value, observed)


def _parse_json(text: str) -> dict | None:
    return grounding.extract_json(text)


def _schema_error(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return "response must be a JSON object"
    verdict = obj.get("verdict")
    if verdict not in VERDICTS:
        return f"`verdict` must be one of {list(VERDICTS)}, got {verdict!r}"
    confidence = obj.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return "`confidence` must be a number between 0 and 1"
    if not 0.0 <= float(confidence) <= 1.0:
        return f"`confidence` must be between 0 and 1, got {confidence}"
    if not isinstance(obj.get("reasoning"), str) or not obj["reasoning"].strip():
        return "`reasoning` must be a non-empty string"
    evidence = obj.get("evidence")
    if not isinstance(evidence, list):
        return "`evidence` must be a list"
    for i, e in enumerate(evidence):
        if not isinstance(e, dict):
            return f"evidence[{i}] must be an object with tool/field/value"
        if "value" not in e:
            return f"evidence[{i}] is missing `value`"
    return None


# =====================================================
# LLM Second-Opinion Review with All Three Gates
# =====================================================

def _review_one(
    decision: dict,
    evidence: list[dict],
    observed: set[str] | None,
    *,
    allowed_symbols: Optional[set[str]] = None,
    lang: str = "en",
) -> dict:
    """Review a single decision through F09 sanitisation + Schema + Grounding + Business gates."""
    clean_evidence, inj_flagged = grounding.sanitise_evidence(evidence)
    payload = {"decision": decision, "observations": clean_evidence}
    messages = [
        {"role": "system", "content": _build_system_prompt(lang)},
        {"role": "user", "content": json.dumps(payload, indent=2, default=str, ensure_ascii=False)},
    ]

    prefill = THINK_PREFILL if _chat is _agent_chat else None
    try:
        parsed, error = grounding.run_structured_gate(
            messages, _chat, _content, _schema_error, max_repairs=1, assistant_prefill=prefill
        )
    except Exception as e:  # noqa: BLE001
        return _skipped(decision, f"model call failed: {e}")

    if parsed is None:
        return _skipped(decision, f"schema gate: {error}")

    # Gate 2 — Grounding. Respect explicit empty `observed=set()` (unit test invariant).
    if observed is not None and len(observed) == 0:
        idx = grounding.GroundingIndex()
    else:
        idx = grounding.build_grounding_index(clean_evidence)
        if observed:
            idx.global_strings.update(observed)

    grounded, dropped_items = grounding.verify_evidence(parsed.get("evidence", []), idx)
    notes: list[str] = []
    if dropped_items:
        notes.append(f"{len(dropped_items)} unverifiable citation(s) dropped")

    # Gate 3 — Business rules.
    verdict, biz_notes, rejected = grounding.check_business_rules(
        decision.get("symbol"),
        parsed,
        grounded,
        allowed_symbols=allowed_symbols,
        observed_symbols=idx.symbols_seen,
    )
    notes.extend(biz_notes)
    if rejected:
        return _skipped(decision, "; ".join(notes))

    reasoning = parsed["reasoning"].strip()
    ans_problems = [
        p for p in injection_guard.check_answer(reasoning, n_sources=0, cited=[]) if p != "no_citations"
    ]
    if inj_flagged or ans_problems:
        notes.append(
            f"security: flagged={inj_flagged}, answer_problems={ans_problems}"
        )

    return {
        "symbol": decision.get("symbol"),
        "action": decision.get("action"),
        "ruleReason": decision.get("reason"),
        "sourceMode": decision.get("sourceMode", "paper"),
        "verdict": verdict,
        "confidence": round(float(parsed["confidence"]), 3),
        "reasoning": reasoning,
        "evidence": grounded,
        "notes": notes,
        "refPrice": decision.get("refPrice"),
        "held": bool(decision.get("held", False)),
    }


def _skipped(decision: dict, why: str) -> dict:
    return {
        "symbol": decision.get("symbol"),
        "action": decision.get("action"),
        "ruleReason": decision.get("reason"),
        "sourceMode": decision.get("sourceMode", "paper"),
        "verdict": "skipped",
        "confidence": 0.0,
        "reasoning": "",
        "evidence": [],
        "notes": [why],
        "refPrice": decision.get("refPrice"),
        "held": bool(decision.get("held", False)),
    }


# =====================================================
# Deterministic Portfolio Checks — Zero LLM
# =====================================================

def _portfolio_checks(
    holdings: dict | None,
    signals: Any,
    *,
    mode: str = "paper",
    daily_action: Any = None,
    weather: Any = None,
) -> list[dict]:
    """Arithmetic portfolio risk checks. Code never hands arithmetic to an LLM."""
    findings: list[dict] = []
    if not isinstance(holdings, dict):
        return findings
    rows = holdings.get("holdings") or []
    totals = holdings.get("totals") or {}
    if not rows:
        return findings

    is_real = mode == "real"
    max_w = REAL_MAX_POSITION_WEIGHT_PCT if is_real else MAX_POSITION_WEIGHT_PCT
    warn_w = REAL_CONCENTRATION_WARN_PCT if is_real else CONCENTRATION_WARN_PCT

    for h in rows:
        weight = float(h.get("weightPct") or 0)
        if weight > max_w:
            findings.append({
                "check": "position_weight",
                "severity": "high" if weight > warn_w else "medium",
                "symbol": h.get("symbol"),
                "detail": (f"{h.get('symbol')} is {weight:.1f}% of total capital, over the "
                           f"{max_w:.0f}% per-position limit"),
                "value": round(weight, 2),
                "limit": max_w,
            })

    if is_real:
        # Mirror daily_action.portfolio deterministic checks (breaches, bad_day_loss, cash floor).
        da_port = daily_action.get("portfolio") if isinstance(daily_action, dict) else None
        if isinstance(da_port, dict):
            cash_pct = da_port.get("cash_pct")
            if isinstance(cash_pct, (int, float)) and cash_pct < REAL_MIN_CASH_PCT:
                findings.append({
                    "check": "cash_floor",
                    "severity": "high" if cash_pct <= 2.0 else "medium",
                    "symbol": None,
                    "detail": f"Cash is {cash_pct:.1f}% of portfolio (floor {REAL_MIN_CASH_PCT:.0f}%)",
                    "value": round(float(cash_pct), 2),
                    "limit": REAL_MIN_CASH_PCT,
                })
            cluster_pct = da_port.get("cluster_pct")
            max_cluster = float(da_port.get("max_cluster_pct") or 70.0)
            if isinstance(cluster_pct, (int, float)) and cluster_pct > max_cluster:
                findings.append({
                    "check": "cluster_concentration",
                    "severity": "high" if cluster_pct >= 85.0 else "medium",
                    "symbol": None,
                    "detail": f"Semiconductor/chip cluster is {cluster_pct:.1f}% (cap {max_cluster:.0f}%)",
                    "value": round(float(cluster_pct), 2),
                    "limit": max_cluster,
                })
            bad_day = da_port.get("bad_day_loss")
            budget = da_port.get("loss_budget_usd")
            if isinstance(bad_day, (int, float)) and isinstance(budget, (int, float)) and budget > 0 and bad_day > budget:
                findings.append({
                    "check": "bad_day_loss_budget",
                    "severity": "high" if bad_day > budget * 1.5 else "medium",
                    "symbol": None,
                    "detail": f"1-in-20 bad day estimated loss ${bad_day:.2f} exceeds 5% budget ${budget:.2f}",
                    "value": round(float(bad_day), 2),
                    "limit": round(float(budget), 2),
                })

        # Market weather panic check
        if isinstance(weather, dict):
            mkt = weather.get("market") if isinstance(weather.get("market"), dict) else weather
            if isinstance(mkt, dict) and mkt.get("tighten_buys"):
                reasons = mkt.get("tighten_reasons") or mkt.get("headline") or "market volatility elevated"
                findings.append({
                    "check": "market_weather_panic",
                    "severity": "high",
                    "symbol": None,
                    "detail": f"Buy tightening active from market weather: {reasons}",
                    "value": mkt.get("vix_level"),
                    "limit": 22.0,
                })
    else:
        invested = float(totals.get("holdingsValue") or 0)
        total = float(totals.get("totalValue") or 0)
        if total > 0:
            exposure = invested / total * 100
            regime = _regime_of(signals)
            if regime in ("RISK_OFF", "STRESSED") and exposure > 60:
                findings.append({
                    "check": "regime_exposure",
                    "severity": "medium",
                    "symbol": None,
                    "detail": (f"{exposure:.1f}% of capital is invested while the regime is "
                               f"{regime}; the signal engine is already scaling conviction down"),
                    "value": round(exposure, 2),
                    "limit": 60.0,
                })

    held = {str(h.get("symbol")) for h in rows if h.get("symbol")}
    ranked = {str(s.get("symbol")) for s in _signal_rows(signals)[:10] if s.get("symbol")}
    if ranked and held and not (held & ranked):
        findings.append({
            "check": "signal_alignment",
            "severity": "medium",
            "symbol": None,
            "detail": ("no holding appears in today's top 10 ranked signals — the "
                       "portfolio and the model currently disagree about what to own"),
            "value": 0,
            "limit": 1,
        })
    return findings


def _signal_rows(signals: Any) -> list[dict]:
    if isinstance(signals, list):
        return [s for s in signals if isinstance(s, dict)]
    if isinstance(signals, dict):
        for key in ("signals", "data", "items"):
            if isinstance(signals.get(key), list):
                return [s for s in signals[key] if isinstance(s, dict)]
    return []


def _regime_of(signals: Any) -> str:
    rows = _signal_rows(signals)
    return str(rows[0].get("regimeLabel") or rows[0].get("regime_label") or "UNKNOWN") if rows else "UNKNOWN"


# =====================================================
# Choosing Decisions (Paper & Real)
# =====================================================

def _recent_exits(positions: Any, limit: int) -> list[dict]:
    rows = positions if isinstance(positions, list) else []
    closed = [p for p in rows if isinstance(p, dict) and p.get("status") == "closed"]
    closed.sort(key=lambda p: str(p.get("exitDate") or p.get("exit_date") or ""), reverse=True)
    out = []
    for p in closed[:limit]:
        ret = p.get("exitReturn", p.get("exit_return"))
        out.append({
            "symbol": p.get("symbol"),
            "action": "exit",
            "sourceMode": "paper",
            "reason": p.get("exitTrigger") or p.get("exit_trigger"),
            "entryDate": p.get("entryDate") or p.get("entry_date"),
            "exitDate": p.get("exitDate") or p.get("exit_date"),
            "daysHeld": p.get("daysHeld", p.get("days_held")),
            "realisedReturnPct": round(float(ret) * 100, 2) if isinstance(ret, (int, float)) else None,
        })
    return out


def _real_decisions(
    holdings: Any,
    daily_action: Any,
    advisor_orders: list[dict],
    limit: int,
) -> list[dict]:
    """Selects real-portfolio actions, drawdowns, overweight holdings, or advisor orders to review."""
    rows = (holdings or {}).get("holdings") or [] if isinstance(holdings, dict) else []
    by_sym = {str(h.get("symbol") or "").upper(): h for h in rows if h.get("symbol")}

    actions_list = (daily_action or {}).get("actions") or [] if isinstance(daily_action, dict) else []
    da_by_sym = {str(a.get("symbol") or "").upper(): a for a in actions_list if a.get("symbol")}

    candidates: list[tuple[int, dict]] = []
    seen_syms: set[str] = set()

    # Priority 1: concrete daily actions (`trim` / `add`) or sharp moves / weight breaches
    for sym, h in by_sym.items():
        da = da_by_sym.get(sym) or {}
        act = str(da.get("action") or "").lower()
        day_chg = float(h.get("dayChangePct") or 0)
        unr_pct = float(h.get("unrealisedPnlPct") or 0)
        weight = float(h.get("weightPct") or 0)
        price = h.get("price") or h.get("regularClose")

        if act in ("trim", "add"):
            candidates.append((1, {
                "symbol": sym,
                "action": act,
                "sourceMode": "real",
                "held": True,
                "refPrice": price,
                "reason": da.get("headline") or ", ".join(da.get("reasons") or []) or act,
                "weightPct": weight,
                "unrealisedPnlPct": unr_pct,
                "dayChangePct": day_chg,
                "orders": da.get("orders") or [],
            }))
            seen_syms.add(sym)
        elif day_chg <= -3.0 or unr_pct <= -8.0 or weight > REAL_MAX_POSITION_WEIGHT_PCT:
            reasons = []
            if weight > REAL_MAX_POSITION_WEIGHT_PCT:
                reasons.append(f"overweight {weight:.1f}%>{REAL_MAX_POSITION_WEIGHT_PCT:.0f}%")
            if day_chg <= -3.0:
                reasons.append(f"day drop {day_chg:.2f}%")
            if unr_pct <= -8.0:
                reasons.append(f"drawdown {unr_pct:.2f}%")
            candidates.append((2, {
                "symbol": sym,
                "action": act or "hold_alert",
                "sourceMode": "real",
                "held": True,
                "refPrice": price,
                "reason": "; ".join(reasons),
                "weightPct": weight,
                "unrealisedPnlPct": unr_pct,
                "dayChangePct": day_chg,
            }))
            seen_syms.add(sym)

    # Priority 2: active advisor limit orders
    for ord_row in advisor_orders or []:
        sym = str(ord_row.get("symbol") or "").upper()
        if not sym or sym in seen_syms:
            continue
        held_row = by_sym.get(sym)
        candidates.append((3, {
            "symbol": sym,
            "action": f"advisor_{str(ord_row.get('side') or 'limit').lower()}",
            "sourceMode": "real",
            "held": sym in by_sym,
            "refPrice": (held_row or {}).get("price") or ord_row.get("limit_price"),
            "reason": f"Advisor limit @ {ord_row.get('limit_price')} ({ord_row.get('status', 'open')})",
        }))
        seen_syms.add(sym)

    # Fallback: largest holdings by weight so review is always actionable
    remaining = sorted(
        (h for sym, h in by_sym.items() if sym not in seen_syms),
        key=lambda h: float(h.get("weightPct") or 0),
        reverse=True,
    )
    for h in remaining:
        sym = str(h.get("symbol") or "").upper()
        da = da_by_sym.get(sym) or {}
        candidates.append((4, {
            "symbol": sym,
            "action": str(da.get("action") or "hold"),
            "sourceMode": "real",
            "held": True,
            "refPrice": h.get("price") or h.get("regularClose"),
            "reason": da.get("headline") or f"Top holding ({h.get('weightPct', 0):.1f}% weight)",
            "weightPct": h.get("weightPct"),
            "unrealisedPnlPct": h.get("unrealisedPnlPct"),
            "dayChangePct": h.get("dayChangePct"),
        }))

    candidates.sort(key=lambda x: (x[0], -float(x[1].get("weightPct") or 0)))
    return [c[1] for c in candidates[:limit]]


def _evidence_for(decision: dict) -> list[dict]:
    symbol = decision.get("symbol")
    if not symbol:
        return []
    evidence = []
    exit_date = str(decision.get("exitDate") or "").replace("-", "")
    news = _call("search_news", {
        "symbol": symbol,
        "from_date": str(decision.get("entryDate") or "").replace("-", ""),
        "to_date": exit_date,
        "limit": 5,
    })
    if news:
        evidence.append({"tool": "search_news", "result": news})
    history = _call("get_feature_history", {
        "symbol": symbol,
        "fields": "avg_sentiment_5d,sentiment_shift_5d,past_ret_20d,macro_vix",
        "days": 60,
    })
    if history:
        evidence.append({"tool": "get_feature_history",
                         "result": {k: history.get(k) for k in
                                    ("symbol", "from", "to", "rows", "summary")}})
    return evidence


def _evidence_for_real(decision: dict, weather: Any = None, catalysts: Any = None) -> list[dict]:
    symbol = decision.get("symbol")
    if not symbol:
        return []
    today = dt.datetime.now(dt.timezone.utc).date()
    from_d = (today - dt.timedelta(days=14)).strftime("%Y%m%d")
    to_d = today.strftime("%Y%m%d")

    evidence: list[dict] = []
    news = _call("search_news", {"symbol": symbol, "from_date": from_d, "to_date": to_d, "limit": 5})
    if isinstance(news, dict):
        articles = news.get("articles") or news.get("results") or []
        compact_articles = [
            {
                "title": a.get("title"),
                "date": a.get("date"),
                "sentiment": a.get("sentiment") or a.get("llm_sentiment_final"),
                "snippet": str(a.get("excerpt") or a.get("summary") or "")[:180],
            }
            for a in articles[:5]
            if isinstance(a, dict)
        ]
        evidence.append({"tool": "search_news", "result": {"symbol": symbol, "articles": compact_articles or news}})
    elif news:
        evidence.append({"tool": "search_news", "result": news})

    history = _call("get_feature_history", {
        "symbol": symbol,
        "fields": "close,avg_sentiment_5d,past_ret_20d,macro_vix",
        "days": 30,
    })
    if history:
        evidence.append({
            "tool": "get_feature_history",
            "result": {k: history.get(k) for k in ("symbol", "from", "to", "rows", "summary")},
        })

    lots = _call("get_lot_advice", {"symbol": symbol})
    if lots and not lots.get("error"):
        evidence.append({
            "tool": "get_lot_advice",
            "result": {
                k: lots.get(k)
                for k in ("symbol", "current_price", "advice", "open_lots_count", "avg_cost", "notes")
                if k in lots
            } or lots,
        })

    if isinstance(catalysts, dict) and not catalysts.get("error"):
        events = [
            ev for ev in (catalysts.get("events") or [])
            if isinstance(ev, dict) and (ev.get("symbol") == symbol or symbol in (ev.get("affected_symbols") or []))
        ][:5]
        if events:
            evidence.append({"tool": "get_catalysts", "result": {"symbol": symbol, "events": events}})

    if isinstance(weather, dict) and not weather.get("error"):
        mkt = weather.get("market") if isinstance(weather.get("market"), dict) else {}
        if mkt:
            evidence.append({
                "tool": "get_market_weather",
                "result": {
                    k: mkt.get(k)
                    for k in ("vix_level", "tighten_buys", "tighten_reasons", "headline", "sox_5d_pct")
                    if k in mkt
                },
            })
    return evidence


def _active_advisor_orders() -> list[dict]:
    try:
        from markets import advisor
        return advisor.active_orders() or []
    except Exception:
        return []


def _record_hint_flags(reviews: list[dict], run_id: str) -> int:
    recorded = 0
    flagged = [r for r in reviews if r.get("verdict") == "flag" and r.get("symbol")]
    if not flagged:
        return 0
    try:
        from markets import hint_scores
        day_str = dt.datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        for rev in flagged:
            ok = hint_scores.record(
                "portfolio_review_flag",
                str(rev["symbol"]),
                day_str,
                bool(rev.get("held")),
                rev.get("refPrice"),
                [str(rev.get("action") or "review"), str(rev.get("ruleReason") or "")[:80]],
                detail={
                    "run_id": run_id,
                    "confidence": rev.get("confidence"),
                    "reasoning": rev.get("reasoning"),
                    "evidence_count": len(rev.get("evidence") or []),
                },
            )
            if ok:
                recorded += 1
    except Exception:
        return recorded
    return recorded


# =====================================================
# Mongo History Helpers (`portfolio_review_runs`)
# =====================================================

def _history_collection():
    global _HISTORY_COLL
    if _HISTORY_COLL is not None:
        return _HISTORY_COLL
    try:
        from pymongo import MongoClient

        client = MongoClient(LOCAL_MONGO_URI, serverSelectionTimeoutMS=2500)
        _HISTORY_COLL = client["quant_data"]["portfolio_review_runs"]
        return _HISTORY_COLL
    except Exception:
        return None


def save_portfolio_review(doc: dict[str, Any]) -> Optional[str]:
    coll = _history_collection()
    if coll is None:
        return None
    try:
        coll.insert_one(dict(doc))
        return str(doc.get("run_id") or "")
    except Exception:
        return None


def list_portfolio_reviews(mode: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
    coll = _history_collection()
    if coll is None:
        return []
    try:
        query: dict[str, Any] = {}
        if mode and mode.strip():
            query["mode"] = mode.strip().lower()
        cur = coll.find(
            query,
            {
                "_id": 0,
                "run_id": 1,
                "mode": 1,
                "regime": 1,
                "decisionsReviewed": 1,
                "verdicts": 1,
                "findingsCount": 1,
                "elapsed_s": 1,
                "asOf": 1,
                "hintsRecorded": 1,
            },
        ).sort("asOf", -1).limit(max(1, min(limit, 50)))
        return list(cur)
    except Exception:
        return []


def get_portfolio_review(run_id: str) -> Optional[dict[str, Any]]:
    coll = _history_collection()
    if coll is None:
        return None
    try:
        return coll.find_one({"run_id": run_id}, {"_id": 0})
    except Exception:
        return None


# =====================================================
# Entry point
# =====================================================

def run_portfolio_review(
    max_decisions: int = MAX_DECISIONS,
    *,
    mode: str = "paper",
    lang: str = "en",
    save_history: bool = False,
    record_hints: bool = False,
) -> dict:
    """Review portfolio decisions (`paper`, `real`, or `both`) plus deterministic checks."""
    started = time.time()
    norm_mode = (mode or "paper").strip().lower()
    if norm_mode not in ("paper", "real", "both"):
        norm_mode = "paper"

    holdings = _call("get_my_holdings")
    signals = _call("get_latest_signals", {"limit": 10})
    positions = _call("get_positions") if norm_mode in ("paper", "both") else []
    daily_action = _call("get_daily_action") if norm_mode in ("real", "both") else None
    weather = _call("get_market_weather") if norm_mode in ("real", "both") else None
    catalysts = _call("get_catalysts", {"scope": "held", "days": 14}) if norm_mode in ("real", "both") else None
    advisor_orders = _active_advisor_orders() if norm_mode in ("real", "both") else []

    held_syms = {
        str(h.get("symbol") or "").upper()
        for h in ((holdings or {}).get("holdings") or [])
        if isinstance(h, dict) and h.get("symbol")
    }
    watch_syms = {
        str(w.get("symbol") or "").upper()
        for w in ((holdings or {}).get("watchlist") or [])
        if isinstance(w, dict) and w.get("symbol")
    }
    paper_syms = {
        str(p.get("symbol") or "").upper()
        for p in (positions if isinstance(positions, list) else [])
        if isinstance(p, dict) and p.get("symbol")
    }
    sig_syms = {str(s.get("symbol") or "").upper() for s in _signal_rows(signals) if s.get("symbol")}
    allowed_symbols = held_syms | watch_syms | paper_syms | sig_syms

    reviews: list[dict] = []
    if norm_mode in ("real", "both"):
        real_decs = _real_decisions(holdings, daily_action, advisor_orders, max_decisions)
        for dec in real_decs:
            ev = _evidence_for_real(dec, weather=weather, catalysts=catalysts)
            reviews.append(
                _review_one(dec, ev, None, allowed_symbols=allowed_symbols or None, lang=lang)
            )

    if norm_mode in ("paper", "both"):
        paper_decs = _recent_exits(positions, max_decisions)
        for dec in paper_decs:
            ev = _evidence_for(dec)
            obs = _observed_values(e["result"] for e in ev)
            reviews.append(
                _review_one(dec, ev, obs, allowed_symbols=allowed_symbols or None, lang=lang)
            )

    findings = _portfolio_checks(
        holdings,
        signals,
        mode="real" if norm_mode in ("real", "both") else "paper",
        daily_action=daily_action,
        weather=weather,
    )
    counts = {v: sum(1 for r in reviews if r["verdict"] == v) for v in ("agree", "flag", "skipped")}
    run_id = uuid.uuid4().hex[:12]

    hints_recorded = 0
    if record_hints and norm_mode in ("real", "both"):
        hints_recorded = _record_hint_flags(reviews, run_id)

    report = {
        "run_id": run_id,
        "mode": norm_mode,
        "lang": lang,
        "asOf": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regime": _regime_of(signals),
        "decisionsReviewed": len(reviews),
        "verdicts": counts,
        "findingsCount": len(findings),
        "reviews": reviews,
        "portfolioFindings": findings,
        "hintsRecorded": hints_recorded,
        "elapsed_s": round(time.time() - started, 2),
    }

    if save_history:
        save_portfolio_review(report)

    return report
