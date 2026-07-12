"""
quant_ai — RAG + Local LLM service for the Quant Trade platform.

Architecture:
  - RAG: numpy cosine similarity over LM Studio embeddings
  - LLM: LM Studio (OpenAI-compatible) → Anthropic → OpenAI fallback
  - API: FastAPI endpoints consumed by quant_ui (browser-direct) and quant_api
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
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

app = FastAPI(title="quant_ai")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Config
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen3.5-9b").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5").strip()
QUANT_API = os.getenv("QUANT_API", "http://quant_api:8081").strip()
KNOWLEDGE_PATHS = os.getenv("KNOWLEDGE_PATHS", "/app/knowledge").strip()
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://host.docker.internal:1234/v1").strip()

_LM_STUDIO_HEADERS = {"Authorization": "Bearer lm-studio", "Content-Type": "application/json"}

print(f"[Config] LM Studio={LM_STUDIO_URL}  model={LOCAL_MODEL_NAME}  embed={EMBED_MODEL}")


def _resolve_model_id(name_hint: str) -> str:
    """Find best-matching model ID from LM Studio's /v1/models list."""
    try:
        resp = requests.get(f"{LM_STUDIO_URL}/models", headers=_LM_STUDIO_HEADERS, timeout=3)
        available = [m["id"] for m in resp.json().get("data", [])]
        match = next((m for m in available if name_hint.lower() in m.lower()), None)
        return match or name_hint
    except Exception:
        return name_hint


# =====================================================
# LLM: LM Studio (OpenAI-compatible) → Anthropic → OpenAI
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
        print("[LLM] Falling back to Anthropic: claude-haiku-4-5-20251001")
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    if OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        print("[LLM] Falling back to OpenAI: gpt-4o-mini")
        return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=temperature)

    raise ValueError(
        "No LLM available. Set LM_STUDIO_URL, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
    )


# =====================================================
# Vector Store: numpy cosine similarity + LM Studio embeddings
# Falls back to keyword search if embeddings unavailable
# =====================================================
class SimpleVectorStore:
    def __init__(self):
        self.docs: list[dict] = []
        self._embeddings: Optional[np.ndarray] = None
        self.ready = False

    def _embed_single(self, text: str) -> Optional[np.ndarray]:
        try:
            model_id = _resolve_model_id(EMBED_MODEL)
            resp = requests.post(
                f"{LM_STUDIO_URL}/embeddings",
                json={"model": model_id, "input": text[:2000]},
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
        # keyword fallback when embedding model is unavailable
        docs = _load_knowledge_docs()
        q_toks = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        scored = []
        for d in docs:
            overlap = len(q_toks & set(re.findall(r"[a-zA-Z0-9_]+", d["text"].lower())))
            if overlap > 0:
                scored.append((overlap, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [d for _, d in scored[:top_k]]

    if not results:
        return "No relevant knowledge found."
    return "\n\n---\n\n".join(f"[{r['source']}]\n{r['text'][:1200]}" for r in results)


# =====================================================
# Live data helpers (used to enrich prompts with real-time context)
# =====================================================
def fetch_latest_signals(limit: int = 10) -> str:
    try:
        resp = requests.get(f"{QUANT_API}/api/signals/latest", params={"limit": limit}, timeout=5)
        data = resp.json()
        signals = data.get("signals", data) if isinstance(data, dict) else data
        return json.dumps(signals[:limit], ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error fetching signals: {e}"


def fetch_positions() -> str:
    try:
        resp = requests.get(f"{QUANT_API}/api/positions", timeout=5)
        return json.dumps(resp.json(), ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error fetching positions: {e}"


def fetch_performance() -> str:
    try:
        resp = requests.get(f"{QUANT_API}/api/performance", timeout=5)
        return json.dumps(resp.json(), ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error fetching performance: {e}"


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


class SimpleSpecRequest(BaseModel):
    prompt: str
    userId: str = "local-user"


class SimpleTasksRequest(BaseModel):
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
def ask(request: QueryRequest):
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
    """RAG + single LLM call: generate a strategy spec JSON from a natural-language prompt."""
    try:
        start = time.time()
        import datetime

        context = retrieve_context(request.prompt, top_k=3)

        prompt = f"""You are a quant strategy planner. Output ONLY a valid JSON object, no markdown, no explanation.

User requirement: {request.prompt}

Knowledge context (use these patterns):
{context}

Output this exact JSON structure:
{{
  "strategyId": "{request.strategyId}",
  "workflowId": "{request.strategyId}",
  "owner": "{request.userId}",
  "name": "<short strategy name>",
  "description": "<1-2 sentences describing the strategy>",
  "market": "US_EQUITY",
  "tasks": [
    {{
      "taskId": "task_data",
      "type": "data_collection",
      "module": "quant_data.stock_collector.price_collector.collector",
      "dependencies": [],
      "parameters": {{"symbols": ["<ticker>"], "timeframe": "1d", "lookback_days": 365}}
    }},
    {{
      "taskId": "task_features",
      "type": "feature_engineering",
      "module": "quant_data.feature_builders.daily_symbol_features",
      "dependencies": ["task_data"],
      "parameters": {{"indicators": ["rsi_14", "macd", "bb_20"]}}
    }},
    {{
      "taskId": "task_signals",
      "type": "signal_generation",
      "module": "quant_data.research.score_daily_signals",
      "dependencies": ["task_features"],
      "parameters": {{"rule": "<pandas boolean expression on features>", "min_score": 0.6}}
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
  "createdAt": "{datetime.datetime.utcnow().isoformat()}Z"
}}"""

        llm = get_chat_llm(temperature=0.2)
        answer = llm.invoke([HumanMessage(content=prompt)]).content

        start_j = answer.find("{")
        end_j = answer.rfind("}")
        if start_j < 0 or end_j <= start_j:
            raise ValueError(f"LLM returned no JSON. Output: {answer[:400]}")

        spec = json.loads(answer[start_j: end_j + 1])
        spec["strategyId"] = request.strategyId
        spec["workflowId"] = request.strategyId
        spec["owner"] = request.userId

        task_ids = {str(t.get("taskId", "")) for t in spec.get("tasks", [])}
        dep_errors = [
            f"{t['taskId']}: dependency '{d}' not found"
            for t in spec.get("tasks", [])
            for d in t.get("dependencies", [])
            if d not in task_ids
        ]

        return {
            "strategySpec": spec,
            "source": "rag+llm",
            "dependencyCheck": {"valid": not dep_errors, "errors": dep_errors},
            "elapsed_s": round(time.time() - start, 2),
            "model": LOCAL_MODEL_NAME,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "source": "llm_failed"}


@app.post("/api/workflow/generate-tasks")
def generate_workflow_tasks(request: WorkflowTasksRequest):
    """RAG + single LLM call: generate executable Python code for each task in the spec."""
    try:
        start = time.time()
        spec = request.strategySpec

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
            tasks = json.loads(answer[start_a: end_a + 1])

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
    """Generate a standalone Python quant script from a plain-text description."""
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


@app.post("/api/v1/strategies/generate-spec")
def api_generate_spec(request: SimpleSpecRequest):
    """Frontend adapter: accepts {prompt, userId}, auto-generates strategyId, returns flat spec."""
    import uuid
    strategy_id = f"strat-{uuid.uuid4().hex[:8]}"
    inner = generate_workflow_spec(WorkflowSpecRequest(
        prompt=request.prompt,
        strategyId=strategy_id,
        userId=request.userId,
    ))
    if "error" in inner:
        return inner
    spec = inner.get("strategySpec", {})
    spec.setdefault("_source", "rag+llm")
    return spec


@app.post("/api/v1/strategies/generate-tasks")
def api_generate_tasks(request: SimpleTasksRequest):
    """Frontend adapter: accepts {strategySpec}, returns task list."""
    inner = generate_workflow_tasks(WorkflowTasksRequest(strategySpec=request.strategySpec))
    tasks = inner.get("tasks", [])
    return tasks if isinstance(tasks, list) else inner


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8083))
    print(f"Starting quant_ai on :{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
