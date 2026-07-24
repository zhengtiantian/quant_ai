"""
ReAct tool-calling research agent for quant_ai.

Hand-written agent loop (no framework) against LM Studio's OpenAI-compatible
/chat/completions with `tools`. The LLM autonomously decides which read-only
data tools to call (Thought -> Action -> Observation), then synthesizes a
grounded research note.

Guardrails (controlled agency):
  - tools are read-only (mongo aggregations; no writes, no trading actions)
  - duplicate tool_calls within one step are executed once
  - repeated (tool, args) across steps return the cached observation with a
    nudge to stop looping (small local models love to retry empty tools)
  - hard max_steps cap, then a forced final synthesis without tools
  - thinking-model fallback: if `content` is empty, use `reasoning_content`
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from typing import Any

import requests

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1").strip()
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen3.5-9b").strip()
QUANT_API = os.getenv("QUANT_API", "http://localhost:18081").strip()
MONGO_URI = os.getenv(
    "LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/?authSource=admin"
).strip()
DB_NAME = os.getenv("FEATURE_DB_NAME", "quant_data")

_HEADERS = {"Authorization": "Bearer lm-studio", "Content-Type": "application/json"}

_mongo_client = None


def _db():
    global _mongo_client
    if _mongo_client is None:
        from pymongo import MongoClient
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
    return _mongo_client[DB_NAME]


# =====================================================
# Tools — quant_api (Java data layer) first, mongo direct as fallback
# =====================================================

def _api_get(path: str) -> str | None:
    """GET from quant_api; None on any failure so callers can fall back."""
    try:
        resp = requests.get(f"{QUANT_API}{path}", timeout=8)
        if resp.status_code == 200:
            return json.dumps(resp.json(), ensure_ascii=False, default=str)
    except Exception:
        pass
    return None


def tool_get_news_sentiment(args: dict) -> str:
    """Aggregate LLM-labeled news sentiment for a symbol (API-first)."""
    symbol = str(args.get("symbol", "")).upper().strip()
    days = int(args.get("days", 90))
    if not symbol:
        return json.dumps({"error": "symbol required"})
    via_api = _api_get(f"/api/agent-data/news/{symbol}/sentiment?days={days}")
    if via_api is not None:
        return via_api
    try:
        col = _db()["news_articles_company_matched_v2"]
        match: dict[str, Any] = {"symbol": symbol, "llm_sentiment_final": {"$exists": True}}
        if days > 0:
            # article `date` is a "YYYYMMDDHHMMSS" string — lexicographic compare works
            cutoff = (dt.datetime.utcnow() - dt.timedelta(days=days)).strftime("%Y%m%d%H%M%S")
            match["date"] = {"$gte": cutoff}
        agg = list(col.aggregate([
            {"$match": match},
            {"$group": {
                "_id": None,
                "articles": {"$sum": 1},
                "avg_sentiment": {"$avg": "$llm_sentiment_final"},
                "avg_disagreement": {"$avg": "$llm_disagreement"},
            }},
        ]))
        if not agg:
            return json.dumps({"symbol": symbol, "articles": 0,
                               "note": "no labeled articles for this symbol"})
        g = agg[0]
        heads = list(col.find({"symbol": symbol}, {"_id": 0, "title": 1, "date": 1,
                                                   "llm_sentiment_final": 1})
                     .sort("date", -1).limit(3))
        return json.dumps({
            "symbol": symbol,
            "articles": g["articles"],
            "avg_sentiment": round(g["avg_sentiment"], 4),
            "avg_model_disagreement": round(g.get("avg_disagreement") or 0.0, 4),
            "recent_headlines": heads,
            "scale": "sentiment -1 (bearish) .. +1 (bullish); low disagreement = high model consensus",
        }, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": f"mongo query failed: {e}"})


def tool_get_features(args: dict) -> str:
    """Latest engineered daily features for a symbol (API-first)."""
    symbol = str(args.get("symbol", "")).upper().strip()
    if not symbol:
        return json.dumps({"error": "symbol required"})
    via_api = _api_get(f"/api/agent-data/features/{symbol}/latest")
    if via_api is not None:
        return via_api
    try:
        col = _db()["daily_symbol_features"]
        doc = col.find_one({"symbol": symbol}, {"_id": 0}, sort=[("date", -1)])
        if not doc:
            return json.dumps({"symbol": symbol,
                               "note": "no features computed yet for this symbol"})
        slim = {k: v for k, v in list(doc.items())[:25]}
        return json.dumps({"symbol": symbol, "latest_features": slim},
                          ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": f"mongo query failed: {e}"})


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_news_sentiment",
            "description": ("Aggregated LLM-labeled news sentiment for a stock symbol: "
                            "article count, average sentiment (-1..1), model disagreement, "
                            "recent headlines. Source: 840K+ dual-LLM labeled articles."),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "ticker, e.g. AAPL"},
                    "days": {"type": "integer", "description": "lookback window in days (default 90)"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_features",
            "description": "Latest engineered daily features (momentum, volatility, sentiment aggregates, etc.) for a symbol.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "ticker, e.g. AAPL"}},
                "required": ["symbol"],
            },
        },
    },
]

TOOL_IMPL = {
    "get_news_sentiment": tool_get_news_sentiment,
    "get_features": tool_get_features,
}

SYSTEM_PROMPT = (
    "You are a quantitative equity research agent. Use the provided read-only data tools "
    "to ground your analysis in real platform data, then write a concise research note: "
    "stance (bullish/bearish/neutral), confidence, and reasoning tied to the numbers you "
    "retrieved. Call each tool at most once per symbol. If a tool returns no data, say so "
    "and move on — do NOT retry it. You never execute trades; you only analyze."
)


def _resolve_model_id() -> str:
    try:
        resp = requests.get(f"{LM_STUDIO_URL}/models", headers=_HEADERS, timeout=4)
        ids = [m["id"] for m in resp.json().get("data", [])]
        return next((m for m in ids if LOCAL_MODEL_NAME.lower() in m.lower()),
                    ids[0] if ids else LOCAL_MODEL_NAME)
    except Exception:
        return LOCAL_MODEL_NAME


def _chat(messages: list[dict], use_tools: bool = True) -> dict:
    payload: dict[str, Any] = {
        "model": _resolve_model_id(),
        "messages": messages,
        "temperature": 0.2,
    }
    if use_tools:
        payload["tools"] = TOOLS
    resp = requests.post(f"{LM_STUDIO_URL}/chat/completions", json=payload,
                         headers=_HEADERS, timeout=180)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def _content(msg: dict) -> str:
    return (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()


def agent_events(question: str, max_steps: int = 5):
    """Core ReAct loop as an event generator.

    Yields dicts: {"type": "tool_call", ...} per executed tool, then exactly one
    {"type": "final", "answer": ..., "steps": ..., "elapsed_s": ..., [
    "note"]}. Both the blocking and the SSE endpoint wrap this generator.
    """
    started = time.time()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    cache: dict[tuple, str] = {}  # (name, args_json) -> observation, across steps

    for step in range(1, max_steps + 1):
        msg = _chat(messages)
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            yield {"type": "final", "answer": _content(msg), "steps": step,
                   "elapsed_s": round(time.time() - started, 2)}
            return

        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": tool_calls})

        seen_this_step: dict[tuple, str] = {}
        for tc in tool_calls:
            name = tc["function"]["name"]
            args_raw = tc["function"].get("arguments") or "{}"
            key = (name, args_raw)
            if key in seen_this_step:
                obs = seen_this_step[key]
            elif key in cache:
                obs = (cache[key] + "\n[note: already retrieved earlier — do not call "
                       "this tool again; synthesize your answer now]")
            else:
                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {}
                impl = TOOL_IMPL.get(name)
                obs = impl(args) if impl else json.dumps({"error": f"unknown tool {name}"})
                cache[key] = obs
                yield {"type": "tool_call", "step": step, "action": name,
                       "args": args, "observation": obs[:400]}
            seen_this_step[key] = obs
            messages.append({"role": "tool", "tool_call_id": tc.get("id", name),
                             "content": obs})

    messages.append({"role": "user", "content":
                     "Stop gathering data. Write your final research note now from what you have."})
    final = _chat(messages, use_tools=False)
    yield {"type": "final", "answer": _content(final), "steps": max_steps,
           "elapsed_s": round(time.time() - started, 2), "note": "hit max_steps"}


def run_research_agent(question: str, max_steps: int = 5) -> dict:
    """Blocking wrapper: drain agent_events() into the classic response shape."""
    trace: list[dict] = []
    for ev in agent_events(question, max_steps):
        if ev["type"] == "tool_call":
            trace.append({k: ev[k] for k in ("step", "action", "args", "observation")})
        else:
            result = {k: v for k, v in ev.items() if k != "type"}
            result["trace"] = trace
            return result
    return {"answer": "", "steps": 0, "trace": trace, "elapsed_s": 0.0,
            "note": "no final event"}
