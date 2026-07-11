"""
Quant LangChain Agent v2
- RAG: numpy cosine similarity + LM Studio embeddings (Harrier-OSS-v1)
- Tools: 4 real callable tools wired to quant_api
- Agent: LangGraph ReAct loop with tool-calling (falls back to manual loop)
- LLM: LM Studio (OpenAI-compatible) → ChatAnthropic → ChatOpenAI
"""

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from fastapi import FastAPI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

app = FastAPI(title="Quant LangChain Agent v2")

# =====================================================
# Config
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen3.5-9b").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "harrier-oss-v1-0.6b").strip()
QUANT_API = os.getenv("QUANT_API", "http://quant_api:8081").strip()
KNOWLEDGE_PATHS = os.getenv("KNOWLEDGE_PATHS", "/app/knowledge").strip()
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://host.docker.internal:1234/v1").strip()

_LM_STUDIO_HEADERS = {"Authorization": "Bearer lm-studio", "Content-Type": "application/json"}

print(f"[Config] LM Studio={LM_STUDIO_URL}  model={LOCAL_MODEL_NAME}  embed={EMBED_MODEL}")


# =====================================================
# LLM: LM Studio (OpenAI-compatible) → ChatAnthropic → ChatOpenAI
# =====================================================
def get_chat_llm(temperature: float = 0.2):
    try:
        resp = requests.get(f"{LM_STUDIO_URL}/models", headers=_LM_STUDIO_HEADERS, timeout=3)
        available = [m["id"] for m in resp.json().get("data", [])]
        match = next((m for m in available if LOCAL_MODEL_NAME.lower() in m.lower()), None)
        model_id = match or (available[0] if available else LOCAL_MODEL_NAME)
        from langchain_openai import ChatOpenAI
        print(f"[LLM] Using LM Studio: {model_id}")
        return ChatOpenAI(
            model=model_id,
            openai_api_key="lm-studio",
            openai_api_base=LM_STUDIO_URL,
            temperature=temperature,
        )
    except Exception as e:
        print(f"[LLM] LM Studio unavailable: {e}")

    if ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        print("[LLM] Falling back to ChatAnthropic: claude-haiku-4-5-20251001")
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    if OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        print("[LLM] Falling back to ChatOpenAI: gpt-4o-mini")
        return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=temperature)

    raise ValueError(
        "No LLM available. Set LM_STUDIO_URL, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
    )


# =====================================================
# Vector Store: numpy cosine similarity + Ollama embeddings
# Falls back to keyword search if embeddings unavailable
# =====================================================
class SimpleVectorStore:
    def __init__(self):
        self.docs: list[dict] = []
        self._embeddings: Optional[np.ndarray] = None
        self.ready = False

    def _embed_single(self, text: str) -> Optional[np.ndarray]:
        try:
            resp = requests.post(
                f"{LM_STUDIO_URL}/embeddings",
                json={"model": EMBED_MODEL, "input": text[:2000]},
                headers=_LM_STUDIO_HEADERS,
                timeout=15,
            )
            data = resp.json()
            embedding = data["data"][0]["embedding"]
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"[Embed] Error: {e}")
        return None

    def build(self, docs: list[dict]) -> bool:
        self.docs = docs
        embs = []
        dim = None
        for doc in docs:
            e = self._embed_single(doc["text"])
            if e is not None:
                dim = len(e)
            embs.append(e)

        if dim is None:
            print("[RAG] Embedding model unavailable — using keyword fallback")
            return False

        stacked = np.stack([e if e is not None else np.zeros(dim, np.float32) for e in embs])
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings = stacked / norms
        self.ready = True
        print(f"[RAG] Vector store ready: {len(docs)} docs, dim={dim}, model={EMBED_MODEL}")
        return True

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        q = self._embed_single(query)
        if q is None or self._embeddings is None:
            return []
        q_norm = q / (np.linalg.norm(q) or 1.0)
        scores = self._embeddings @ q_norm
        idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"text": self.docs[i]["text"], "source": self.docs[i]["source"], "score": float(scores[i])}
            for i in idx
        ]


_vs = SimpleVectorStore()
_vs_loaded = False


def _load_knowledge_docs() -> list[dict]:
    docs = []
    allowed = {".md", ".txt", ".json", ".yaml", ".yml"}
    for base in [p.strip() for p in KNOWLEDGE_PATHS.split(",") if p.strip()]:
        p = Path(base)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in allowed:
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore").strip()
                    if text:
                        docs.append({"text": text[:6000], "source": f.name})
                except Exception:
                    pass
    print(f"[RAG] Loaded {len(docs)} knowledge docs")
    return docs


def _ensure_vs():
    global _vs_loaded
    if _vs_loaded:
        return
    _vs_loaded = True
    docs = _load_knowledge_docs()
    if docs:
        _vs.build(docs)


def retrieve_context(query: str, top_k: int = 4) -> str:
    _ensure_vs()
    if _vs.ready:
        results = _vs.search(query, top_k=top_k)
    else:
        # Keyword fallback
        docs = _load_knowledge_docs()
        q_toks = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        scored = []
        for d in docs:
            overlap = len(q_toks & set(re.findall(r"[a-zA-Z0-9_]+", d["text"].lower())))
            if overlap > 0:
                scored.append((overlap, d))
        scored.sort(reverse=True)
        results = [d for _, d in scored[:top_k]]

    if not results:
        return "No relevant knowledge found."
    return "\n\n---\n\n".join(f"[{r['source']}]\n{r['text'][:1200]}" for r in results)


# =====================================================
# MCP Tools — real callable functions
# =====================================================
@tool
def query_knowledge(query: str) -> str:
    """Search the quant platform knowledge base. Use to look up factor weights, IC values,
    strategy patterns, MongoDB schema, signal logic, and system architecture."""
    return retrieve_context(query)


@tool
def get_latest_signals(limit: int = 10) -> str:
    """Fetch the latest trading signals from the quant signal system.
    Returns top-ranked signals with composite_score, signal_type, factor values.
    Use to understand current market conditions before designing a strategy."""
    try:
        resp = requests.get(
            f"{QUANT_API}/api/signals/latest",
            params={"limit": limit},
            timeout=5,
        )
        data = resp.json()
        signals = data.get("signals", data) if isinstance(data, dict) else data
        return json.dumps(signals[:limit], ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error fetching signals: {e}"


@tool
def get_positions() -> str:
    """Get current open paper trading positions.
    Use to check existing exposure and avoid conflicting trades."""
    try:
        resp = requests.get(f"{QUANT_API}/api/positions", timeout=5)
        return json.dumps(resp.json(), ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error fetching positions: {e}"


@tool
def get_performance_stats() -> str:
    """Get live portfolio performance metrics: Sharpe ratio, max drawdown, win rate,
    annualized return from the paper trading backtest. Use to calibrate risk parameters."""
    try:
        resp = requests.get(f"{QUANT_API}/api/performance", timeout=5)
        return json.dumps(resp.json(), ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error fetching performance: {e}"


AGENT_TOOLS = [query_knowledge, get_latest_signals, get_positions, get_performance_stats]
TOOL_MAP = {t.name: t for t in AGENT_TOOLS}


# =====================================================
# Agent Loop
# Tries LangGraph first (prebuilt ReAct); falls back to manual tool-calling loop.
# =====================================================
def _run_with_langgraph(system: str, user_msg: str) -> tuple[str, list[str]]:
    from langgraph.prebuilt import create_react_agent

    llm = get_chat_llm(temperature=0.2)
    agent = create_react_agent(llm, AGENT_TOOLS)
    result = agent.invoke({"messages": [SystemMessage(content=system), HumanMessage(content=user_msg)]})

    tools_used = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc.get("name", ""))

    final = result["messages"][-1].content
    return final, tools_used


def _run_manual_loop(system: str, user_msg: str, max_iters: int = 6) -> tuple[str, list[str]]:
    llm = get_chat_llm(temperature=0.2).bind_tools(AGENT_TOOLS)
    messages = [SystemMessage(content=system), HumanMessage(content=user_msg)]
    tools_used: list[str] = []

    for _ in range(max_iters):
        response = llm.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            return response.content, tools_used

        for tc in response.tool_calls:
            name, args, call_id = tc["name"], tc["args"], tc["id"]
            tools_used.append(name)
            print(f"[Agent] → {name}({args})")
            try:
                result = TOOL_MAP[name].invoke(args) if name in TOOL_MAP else f"Unknown tool: {name}"
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

    # Force final answer
    llm_plain = get_chat_llm(temperature=0.2)
    messages.append(HumanMessage(content="Generate the final JSON output now based on everything gathered."))
    final = llm_plain.invoke(messages)
    return final.content, tools_used


def run_agent(system: str, user_msg: str) -> tuple[str, list[str]]:
    """Run ReAct agent. Tries LangGraph, falls back to manual loop."""
    try:
        import langgraph  # noqa: F401
        print("[Agent] Using LangGraph ReAct agent")
        return _run_with_langgraph(system, user_msg)
    except ImportError:
        print("[Agent] LangGraph not installed — using manual tool-calling loop")
        return _run_manual_loop(system, user_msg)


# =====================================================
# Request models
# =====================================================
class QueryRequest(BaseModel):
    question: str


class WorkflowSpecRequest(BaseModel):
    prompt: str
    strategyId: str
    userId: str = "local-user"


class WorkflowTasksRequest(BaseModel):
    strategySpec: dict


# =====================================================
# FastAPI Routes
# =====================================================
@app.on_event("startup")
async def on_startup():
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _ensure_vs)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": LOCAL_MODEL_NAME,
        "embed_model": EMBED_MODEL,
        "vector_store_ready": _vs.ready,
        "knowledge_docs": len(_vs.docs),
    }


@app.post("/api/ask")
def ask_agent(request: QueryRequest):
    """RAG-augmented Q&A about the quant system."""
    try:
        start = time.time()
        context = retrieve_context(request.question, top_k=3)
        prompt = (
            "You are a quant research assistant for an AI-driven equity signal platform.\n\n"
            f"Knowledge context:\n{context}\n\n"
            f"Question: {request.question}"
        )
        llm = get_chat_llm(temperature=0.7)
        answer = llm.invoke([HumanMessage(content=prompt)]).content
        return {"answer": answer, "elapsed_s": round(time.time() - start, 2)}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/api/workflow/generate-spec")
def generate_workflow_spec(request: WorkflowSpecRequest):
    """
    Agent-driven strategy spec generation.
    The agent uses tools to gather context (knowledge, live signals, positions)
    then outputs a structured JSON strategy specification.
    """
    try:
        start = time.time()

        system = f"""You are a quant strategy planner for an AI-driven equity signal platform.

You have tools to search the knowledge base, retrieve live signals, and check portfolio state.
Use them to gather context BEFORE generating the strategy spec.

Recommended tool usage:
1. Call query_knowledge with the user's strategy description to find matching patterns
2. Call get_latest_signals to understand current market conditions
3. Call get_positions to check existing exposure (avoid conflicts)
4. Optionally call get_performance_stats to calibrate risk parameters

Then output ONLY a valid JSON object — no markdown, no explanation — with these fields:
{{
  "strategyId": "{request.strategyId}",
  "workflowId": "{request.strategyId}",
  "owner": "{request.userId}",
  "name": "<short name>",
  "description": "<1-2 sentences>",
  "market": "US_EQUITY",
  "tasks": [
    {{
      "taskId": "task_data",
      "type": "data_collection",
      "module": "quant_data.stock_collector.price_collector.collector",
      "dependencies": [],
      "parameters": {{...}}
    }},
    {{
      "taskId": "task_features",
      "type": "feature_engineering",
      "module": "quant_data.feature_builders.daily_symbol_features",
      "dependencies": ["task_data"],
      "parameters": {{"indicators": [<list of factor names>]}}
    }},
    {{
      "taskId": "task_signals",
      "type": "signal_generation",
      "module": "quant_data.research.score_daily_signals",
      "dependencies": ["task_features"],
      "parameters": {{"rule": "<pandas boolean expression>", "min_score": 0.6}}
    }},
    {{
      "taskId": "task_risk",
      "type": "risk_management",
      "module": "quant_data.research.track_positions",
      "dependencies": ["task_signals"],
      "parameters": {{"max_position_size": 0.1, "stop_loss": 0.02, "max_hold": 60}}
    }},
    {{
      "taskId": "task_backtest",
      "type": "backtesting",
      "module": "quant_data.research.backtest_portfolio",
      "dependencies": ["task_risk"],
      "parameters": {{"initial_cash": 100000, "fee_bps": 5, "window": "2y", "rebalance": "daily"}}
    }}
  ],
  "risk": {{"max_position_size": 0.1, "stop_loss": 0.02, "max_drawdown": 0.2}},
  "backtest": {{"initial_cash": 100000, "fee_bps": 5, "window": "2y", "rebalance": "daily"}},
  "createdAt": "<ISO 8601 timestamp>"
}}"""

        answer, tools_used = run_agent(system, request.prompt)

        # Extract JSON from model output
        start_j = answer.find("{")
        end_j = answer.rfind("}")
        if start_j < 0 or end_j <= start_j:
            raise ValueError(f"Agent returned no JSON. Output: {answer[:400]}")

        spec = json.loads(answer[start_j : end_j + 1])
        spec["strategyId"] = request.strategyId
        spec["workflowId"] = request.strategyId
        spec["owner"] = request.userId

        # Validate dependency graph
        task_ids = {str(t.get("taskId", "")) for t in spec.get("tasks", [])}
        dep_errors = [
            f"{t['taskId']}: dependency '{d}' not found"
            for t in spec.get("tasks", [])
            for d in t.get("dependencies", [])
            if d not in task_ids
        ]

        return {
            "strategySpec": spec,
            "source": "agent+rag+tools",
            "toolsUsed": tools_used,
            "dependencyCheck": {"valid": not dep_errors, "errors": dep_errors},
            "elapsed_s": round(time.time() - start, 2),
            "model": LOCAL_MODEL_NAME,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "source": "agent_failed"}


@app.post("/api/workflow/generate-tasks")
def generate_workflow_tasks(request: WorkflowTasksRequest):
    """
    Generate executable Python code for each task in the strategy spec.
    Uses RAG to retrieve relevant code patterns before code generation.
    """
    try:
        start = time.time()
        spec = request.strategySpec

        # RAG: find relevant code patterns from knowledge base
        query = f"{spec.get('name', '')} {spec.get('description', '')} {json.dumps(spec.get('tasks', []))}"
        context = retrieve_context(query[:500], top_k=3)

        prompt = f"""You are a quant Python code generator for the Quant Trade platform.

Strategy spec:
{json.dumps(spec, indent=2, ensure_ascii=False)[:3000]}

Knowledge context (use these patterns):
{context}

Generate a JSON array where each element is one task implementation:
[
  {{
    "taskId": "<same as spec taskId>",
    "taskType": "<type>",
    "fileName": "<snake_case_name>.py",
    "code": "<complete runnable Python code>"
  }}
]

Code requirements:
- Use pymongo: MongoClient("mongodb://mongo:27017/"), db = client["quant_data"]
- Import only: pandas, numpy, pymongo, scipy, datetime
- Each script is self-contained and runnable
- Match the task parameters exactly from the spec
- Include a main block: if __name__ == "__main__": main()

Output ONLY the JSON array, no markdown fences, no explanation."""

        llm = get_chat_llm(temperature=0.25)
        answer = llm.invoke([HumanMessage(content=prompt)]).content

        start_a = answer.find("[")
        end_a = answer.rfind("]")
        tasks: list = []
        if start_a >= 0 and end_a > start_a:
            tasks = json.loads(answer[start_a : end_a + 1])

        return {
            "tasks": tasks if isinstance(tasks, list) else [],
            "source": "rag+llm",
            "elapsed_s": round(time.time() - start, 2),
            "model": LOCAL_MODEL_NAME,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "source": "llm_failed", "tasks": []}


@app.post("/api/generate-script")
def generate_script(request: QueryRequest):
    """Generate a standalone Python quant script from a description."""
    try:
        start = time.time()
        context = retrieve_context(request.question, top_k=2)
        prompt = (
            "You are an expert quant Python developer for the Quant Trade platform.\n\n"
            f"Knowledge context:\n{context}\n\n"
            "Generate a complete, runnable Python script for:\n"
            f"{request.question}"
        )
        llm = get_chat_llm(temperature=0.3)
        script = llm.invoke([HumanMessage(content=prompt)]).content
        return {"script": script, "elapsed_s": round(time.time() - start, 2)}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8083))
    print(f"Starting Quant LangChain Agent v2 on :{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
