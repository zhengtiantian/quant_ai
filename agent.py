"""
ReAct tool-calling research agent for quant_ai.

Hand-written agent loop (no framework) against LM Studio's OpenAI-compatible
/chat/completions with `tools`. The LLM autonomously decides which read-only
data tools to call (Thought -> Action -> Observation), then synthesizes a
grounded research note.

Tools are not implemented here. They are discovered at runtime from the quant
MCP server over stdio (see mcp_client.py), so the MCP server is the single
definition of every platform tool: registering one there is enough for this
agent to gain it.

Guardrails (controlled agency):
  - tools are read-only (the MCP server exposes no mutating operation)
  - duplicate tool_calls within one step are executed once
  - repeated (tool, args) across steps return the cached observation with a
    nudge to stop looping (small local models love to retry empty tools)
  - hard max_steps cap, then a forced final synthesis without tools
  - thinking-model fallback: if `content` is empty, use `reasoning_content`
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

import mcp_client

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1").strip()
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen3.5-9b").strip()

_HEADERS = {"Authorization": "Bearer lm-studio", "Content-Type": "application/json"}

SYSTEM_PROMPT = (
    "You are a quantitative equity research agent. Use the provided read-only data tools "
    "to ground your analysis in real platform data, then write a concise research note: "
    "stance (bullish/bearish/neutral), confidence, and reasoning tied to the numbers you "
    "retrieved. Call each tool at most once per symbol. If a tool returns no data, say so "
    "and move on — do NOT retry it. You never execute trades; you only analyze."
)


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
        payload["tools"] = get_tool_specs()
    resp = requests.post(f"{LM_STUDIO_URL}/chat/completions", json=payload,
                         headers=_HEADERS, timeout=180)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def _content(msg: dict) -> str:
    return (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()


def agent_events(question: str, max_steps: int = 5):
    """Core ReAct loop as an event generator.

    Yields dicts: {"type": "tool_call", ...} per executed tool, then exactly one
    {"type": "final", "answer": ..., "steps": ..., "elapsed_s": ..., ["note"]}.
    Both the blocking and the SSE endpoint wrap this generator.
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
                obs = execute_tool(name, args)
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
