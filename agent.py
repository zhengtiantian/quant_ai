"""
ReAct tool-calling research agent for quant_ai (F12).

Hand-written agent loop (no framework) against LM Studio's OpenAI-compatible
/chat/completions with `tools`. The LLM autonomously decides which read-only
data tools to call (Thought -> Action -> Observation), then synthesizes a
grounded research note.

Tools are not implemented here. They are discovered at runtime from the quant
MCP server over stdio (see mcp_client.py), so the MCP server is the single
definition of every platform tool: registering one there is enough for this
agent to gain it.

Guardrails (controlled agency):
  - tools are read-only (eval write tools excluded via openai_tools())
  - smart scope subsetting (stock vs portfolio vs full) reduces small-model latency
  - duplicate tool_calls within one step execute once
  - repeated (tool, canonical_args) across steps return the cached observation with
    a nudge to stop looping
  - F09 injection sanitisation & delimiter wrapping + observation length capping
  - hard max_steps cap, then a forced final synthesis without tools
  - thinking-model fallback: if `content` is empty, use `reasoning_content`
"""

from __future__ import annotations

import contextvars
import datetime as dt
import json
import os
import re
import time
import uuid
from typing import Any, Optional

import requests

import injection_guard
import mcp_client

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1").strip()
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen3.5-9b").strip()
LOCAL_MONGO_URI = os.getenv(
    "LOCAL_MONGO_URI", "mongodb://root:root@localhost:37018/?authSource=admin"
).strip()

MAX_OBS_CHARS = int(os.getenv("AGENT_MAX_OBS_CHARS", "3600"))
_HEADERS = {"Authorization": "Bearer lm-studio", "Content-Type": "application/json"}

STOCK_TOOLS = frozenset({
    "get_daily_action",
    "get_stock_features",
    "get_signals",
    "get_catalysts",
    "get_news_sentiment",
    "search_news",
    "get_lot_advice",
    "get_market_weather",
})

PORTFOLIO_TOOLS = frozenset({
    "get_market_weather",
    "get_positions",
    "get_daily_action",
    "get_catalysts",
    "get_trades",
    "get_pipeline_health",
    "search_news",
})

SYSTEM_PROMPT = (
    "You are a quantitative equity research agent. Use the provided read-only data tools "
    "to ground your analysis in real platform data, then write a concise research note: "
    "stance (bullish/bearish/neutral), confidence (high/medium/low), and reasoning tied to "
    "the numbers you retrieved. Prioritize get_daily_action, get_catalysts, get_market_weather, "
    "get_stock_features, get_signals, and get_news_sentiment where relevant. Call each tool at most "
    "once per symbol/args. If a tool returns no data, say so and move on — do NOT retry it. "
    "You never execute trades; you only analyze.\n" + injection_guard.UNTRUSTED_DATA_RULE
)

# Per-run context vars so `_chat(messages, use_tools=True)` keeps its 2-arg signature
# expected by unit tests that mock `agent._chat`.
_ACTIVE_TOOL_SPECS: contextvars.ContextVar[Optional[list[dict]]] = contextvars.ContextVar(
    "_ACTIVE_TOOL_SPECS", default=None
)
_ACTIVE_MODEL_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_ACTIVE_MODEL_ID", default=None
)

_HISTORY_COLL = None


def build_system_prompt(lang: str = "zh") -> str:
    lang_norm = (lang or "zh").strip().lower()
    if lang_norm.startswith("en"):
        lang_rule = "Write the final research note in clear English. End or start with an explicit line: `Stance: Bullish|Bearish|Neutral | Confidence: High|Medium|Low`."
    else:
        lang_rule = "请使用简体中文撰写最终投研纪要，并在开头或结尾明确给出判断行：`立场：看多/看空/中性 | 置信度：高/中/低`。"
    return f"{SYSTEM_PROMPT}\n{lang_rule}"


def build_default_question(symbol: str, lang: str = "zh") -> str:
    sym = (symbol or "").strip().upper()
    if (lang or "zh").strip().lower().startswith("en"):
        return (
            f"Run a deep quantitative research check on {sym}: inspect today's daily action, "
            f"upcoming catalysts, market weather, latest technical/sentiment features, signals, "
            f"and recent news. Summarize key numbers, bull vs bear risks, and give your stance and confidence."
        )
    return (
        f"请对 {sym} 做一次深度投研分析：优先调用工具核查其今日操作建议（get_daily_action）、"
        f"催化事件（get_catalysts）、大盘风控天气（get_market_weather）、最新量化特征与信号"
        f"（get_stock_features / get_signals）以及近期新闻情绪（get_news_sentiment / search_news）。"
        f"请结合真实数据给出多空核心依据、主要风险点，以及明确的看多/看空/中性立场与置信度。"
    )


def infer_scope(question: str, symbol: Optional[str] = None, scope: str = "auto") -> str:
    s = (scope or "auto").strip().lower()
    if s in ("stock", "portfolio", "all"):
        return s
    if symbol and symbol.strip():
        return "stock"
    q = question or ""
    if re.search(r"\b(portfolio|positions?|holdings?|weather|market|risk|pipeline)\b|组合|持仓|大盘|天气|风险|流水线", q, re.I):
        return "portfolio"
    if re.search(r"\b[A-Z]{2,5}\b", q):
        return "stock"
    return "all"


def select_tool_specs(scope: str = "all") -> list[dict]:
    specs = get_tool_specs()
    if scope == "stock":
        allowed = STOCK_TOOLS
    elif scope == "portfolio":
        allowed = PORTFOLIO_TOOLS
    else:
        return specs
    filtered = [t for t in specs if (t.get("function") or {}).get("name") in allowed]
    # Fallback to full specs when custom/mock tools are injected without subset overlap.
    return filtered or specs


# =====================================================
# Tool surface — provided by the MCP server
# =====================================================

def get_tool_specs() -> list[dict]:
    """OpenAI-format tool definitions discovered from the MCP server."""
    return mcp_client.get_client().openai_tools()


def execute_tool(name: str, args: dict) -> str:
    """Run one tool through the MCP session; never raises."""
    try:
        return mcp_client.get_client().call_tool(name, args)
    except Exception as e:  # noqa: BLE001 - observation text is the error channel
        return json.dumps({"error": f"tool {name} failed: {e}"})


def _resolve_model_id() -> str:
    cached = _ACTIVE_MODEL_ID.get()
    if cached:
        return cached
    try:
        resp = requests.get(f"{LM_STUDIO_URL}/models", headers=_HEADERS, timeout=4)
        ids = [m["id"] for m in resp.json().get("data", [])]
        chosen = next(
            (m for m in ids if LOCAL_MODEL_NAME.lower() in m.lower()),
            ids[0] if ids else LOCAL_MODEL_NAME,
        )
        return chosen
    except Exception:
        return LOCAL_MODEL_NAME


def _chat(messages: list[dict], use_tools: bool = True) -> dict:
    payload: dict[str, Any] = {
        "model": _resolve_model_id(),
        "messages": messages,
        "temperature": 0.2,
    }
    if use_tools:
        active_specs = _ACTIVE_TOOL_SPECS.get()
        payload["tools"] = active_specs if active_specs is not None else get_tool_specs()
    resp = requests.post(
        f"{LM_STUDIO_URL}/chat/completions", json=payload, headers=_HEADERS, timeout=180
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def _content(msg: dict) -> str:
    return (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()


def _parse_args(args_raw: Any) -> tuple[dict, str]:
    if isinstance(args_raw, dict):
        parsed = args_raw
    else:
        try:
            loaded = json.loads(args_raw or "{}")
            parsed = loaded if isinstance(loaded, dict) else {"value": loaded}
        except Exception:
            return {}, str(args_raw or "{}").strip()
    try:
        canonical = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    except Exception:
        canonical = str(args_raw)
    return parsed, canonical


def _prepare_observation(name: str, raw_obs: str) -> tuple[str, str, bool, list[str]]:
    """Truncates, sanitises (F09), and wraps observation for model context."""
    obs_str = str(raw_obs or "")
    was_truncated = False
    if len(obs_str) > MAX_OBS_CHARS:
        dropped = len(obs_str) - MAX_OBS_CHARS
        obs_str = obs_str[:MAX_OBS_CHARS] + f"\n[... truncated {dropped} chars]"
        was_truncated = True
    wrapped, screening = injection_guard.wrap_untrusted(obs_str, tag="tool_output", tool=name)
    return wrapped, screening.text, was_truncated, list(screening.flags)


def parse_stance_confidence(answer: str) -> tuple[str, str]:
    text = (answer or "").lower()
    stance = "unknown"
    # Prefer explicit Stance / 立场 lines first.
    m_st = re.search(r"(?:stance|立场)\s*[:：]\s*([^\n|]+)", answer or "", re.I)
    hay = (m_st.group(1) if m_st else text[:500]).lower()
    if any(k in hay for k in ("bullish", "看多", "偏多", "买入", "加仓")):
        stance = "bullish"
    elif any(k in hay for k in ("bearish", "看空", "偏空", "卖出", "减仓")):
        stance = "bearish"
    elif any(k in hay for k in ("neutral", "中性", "观望", "持有")):
        stance = "neutral"

    confidence = "unknown"
    m_cf = re.search(r"(?:confidence|置信度)\s*[:：]\s*([^\n|]+)", answer or "", re.I)
    chay = (m_cf.group(1) if m_cf else text[:500]).lower()
    if any(k in chay for k in ("high", "高", "较高")):
        confidence = "high"
    elif any(k in chay for k in ("medium", "moderate", "中", "中等")):
        confidence = "medium"
    elif any(k in chay for k in ("low", "低", "较低")):
        confidence = "low"

    return stance, confidence


def _history_collection():
    global _HISTORY_COLL
    if _HISTORY_COLL is not None:
        return _HISTORY_COLL
    try:
        from pymongo import MongoClient

        client = MongoClient(LOCAL_MONGO_URI, serverSelectionTimeoutMS=2500)
        _HISTORY_COLL = client["quant_data"]["agent_research_runs"]
        return _HISTORY_COLL
    except Exception:
        return None


def save_research_run(doc: dict[str, Any]) -> Optional[str]:
    coll = _history_collection()
    if coll is None:
        return None
    try:
        coll.insert_one(dict(doc))
        return str(doc.get("run_id") or "")
    except Exception:
        return None


def list_research_runs(symbol: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
    coll = _history_collection()
    if coll is None:
        return []
    try:
        query: dict[str, Any] = {}
        if symbol and symbol.strip():
            query["symbol"] = symbol.strip().upper()
        cur = coll.find(
            query,
            {
                "_id": 0,
                "run_id": 1,
                "symbol": 1,
                "scope": 1,
                "question": 1,
                "stance": 1,
                "confidence": 1,
                "steps": 1,
                "elapsed_s": 1,
                "tools_used": 1,
                "created_at": 1,
                "answer_preview": 1,
                "note": 1,
            },
        ).sort("created_at", -1).limit(max(1, min(limit, 50)))
        return list(cur)
    except Exception:
        return []


def get_research_run(run_id: str) -> Optional[dict[str, Any]]:
    coll = _history_collection()
    if coll is None:
        return None
    try:
        doc = coll.find_one({"run_id": run_id}, {"_id": 0})
        return doc
    except Exception:
        return None


def agent_events(
    question: str,
    max_steps: int = 5,
    symbol: Optional[str] = None,
    lang: str = "zh",
    scope: str = "auto",
    save_history: bool = False,
):
    """Core ReAct loop as a fine-grained SSE event generator."""
    started = time.time()
    norm_symbol = symbol.strip().upper() if symbol and symbol.strip() else None
    resolved_scope = infer_scope(question, symbol=norm_symbol, scope=scope)
    selected_specs = select_tool_specs(resolved_scope)
    tool_names = [(t.get("function") or {}).get("name", "") for t in selected_specs if t.get("function")]
    model_id = _resolve_model_id()

    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(lang)},
        {"role": "user", "content": question},
    ]
    cache: dict[tuple[str, str], str] = {}
    trace: list[dict[str, Any]] = []
    tools_used: list[str] = []
    flagged_tools: list[str] = []

    def _call_chat(msgs: list[dict], use_tools: bool = True) -> dict:
        tok_s = _ACTIVE_TOOL_SPECS.set(selected_specs)
        tok_m = _ACTIVE_MODEL_ID.set(model_id)
        try:
            return _chat(msgs, use_tools=use_tools)
        finally:
            _ACTIVE_MODEL_ID.reset(tok_m)
            _ACTIVE_TOOL_SPECS.reset(tok_s)

    yield {
        "type": "run_start",
        "symbol": norm_symbol,
        "scope": resolved_scope,
        "tools_available": tool_names,
        "max_steps": max_steps,
        "model": model_id,
    }

    final_answer = ""
    final_steps = 0
    final_note: Optional[str] = None

    for step in range(1, max_steps + 1):
        yield {"type": "step_start", "step": step, "max_steps": max_steps}
        msg = _call_chat(messages)
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            final_answer = _content(msg)
            final_steps = step
            break

        messages.append(
            {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            }
        )

        seen_this_step: dict[tuple[str, str], str] = {}
        for tc in tool_calls:
            name = tc["function"]["name"]
            args_raw = tc["function"].get("arguments") or "{}"
            args, canonical = _parse_args(args_raw)
            key = (name, canonical)

            if key in seen_this_step:
                obs = seen_this_step[key]
            elif key in cache:
                obs = (
                    cache[key]
                    + "\n[note: already retrieved earlier — do not call "
                    "this tool again; synthesize your answer now]"
                )
            else:
                yield {"type": "tool_start", "step": step, "action": name, "args": args}
                t0 = time.time()
                raw_obs = execute_tool(name, args)
                dur_ms = round((time.time() - t0) * 1000, 1)
                obs, clean_preview, was_truncated, flags = _prepare_observation(name, raw_obs)
                ok = '"error"' not in raw_obs[:200].lower()
                cache[key] = obs
                if name not in tools_used:
                    tools_used.append(name)
                if flags and name not in flagged_tools:
                    flagged_tools.append(name)
                ev = {
                    "type": "tool_call",
                    "step": step,
                    "action": name,
                    "args": args,
                    "observation": clean_preview[:600],
                    "duration_ms": dur_ms,
                    "ok": ok,
                    "truncated": was_truncated,
                    "injection_flags": flags,
                }
                trace.append({k: v for k, v in ev.items() if k != "type"})
                yield ev
            seen_this_step[key] = obs
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", name),
                    "content": obs,
                }
            )
    else:
        messages.append(
            {
                "role": "user",
                "content": "Stop gathering data. Write your final research note now from what you have.",
            }
        )
        final = _call_chat(messages, use_tools=False)
        final_answer = _content(final)
        final_steps = max_steps
        final_note = "hit max_steps"

    elapsed_s = round(time.time() - started, 2)
    stance, confidence = parse_stance_confidence(final_answer)
    # Layer-4 validation without false-triggering `no_citations` on tool-based prose.
    answer_problems = [
        p for p in injection_guard.check_answer(final_answer, n_sources=0, cited=[]) if p != "no_citations"
    ]
    security = {"flagged_tools": flagged_tools, "answer_problems": answer_problems}
    run_id = uuid.uuid4().hex[:12]

    final_ev: dict[str, Any] = {
        "type": "final",
        "run_id": run_id,
        "symbol": norm_symbol,
        "scope": resolved_scope,
        "answer": final_answer,
        "stance": stance,
        "confidence": confidence,
        "steps": final_steps,
        "elapsed_s": elapsed_s,
        "tools_used": tools_used,
        "security": security,
    }
    if final_note:
        final_ev["note"] = final_note

    if save_history:
        save_research_run(
            {
                "run_id": run_id,
                "symbol": norm_symbol,
                "scope": resolved_scope,
                "lang": lang,
                "question": question,
                "answer": final_answer,
                "answer_preview": final_answer[:220],
                "stance": stance,
                "confidence": confidence,
                "steps": final_steps,
                "elapsed_s": elapsed_s,
                "tools_used": tools_used,
                "trace": trace,
                "security": security,
                "note": final_note,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )

    yield final_ev


def run_research_agent(
    question: str,
    max_steps: int = 5,
    symbol: Optional[str] = None,
    lang: str = "zh",
    scope: str = "auto",
    save_history: bool = False,
) -> dict:
    """Blocking wrapper: drain agent_events() into the classic response shape."""
    trace: list[dict] = []
    for ev in agent_events(
        question,
        max_steps=max_steps,
        symbol=symbol,
        lang=lang,
        scope=scope,
        save_history=save_history,
    ):
        ev_type = ev["type"]
        if ev_type == "tool_call":
            trace.append({k: v for k, v in ev.items() if k != "type"})
        elif ev_type == "final":
            result = {k: v for k, v in ev.items() if k != "type"}
            result["trace"] = trace
            return result
    return {
        "answer": "",
        "steps": 0,
        "trace": trace,
        "elapsed_s": 0.0,
        "note": "no final event",
    }
