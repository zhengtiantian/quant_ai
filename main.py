"""
quant_ai — RAG + Local LLM service for the Quant Trade platform.

Architecture:
  - RAG: numpy cosine similarity over LM Studio embeddings
  - LLM: LM Studio (OpenAI-compatible) → Anthropic → OpenAI fallback
  - API: FastAPI endpoints consumed by quant_ui (browser-direct) and quant_api
"""

import asyncio
import json
import os
import re
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from fastapi import Body, FastAPI, HTTPException, Query
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

LOCAL_MONGO_URI = os.getenv(
    "LOCAL_MONGO_URI", "mongodb://root:root@localhost:37018/?authSource=admin"
).strip()

_LM_STUDIO_HEADERS = {"Authorization": "Bearer lm-studio", "Content-Type": "application/json"}

print(f"[Config] LM Studio={LM_STUDIO_URL}  model={LOCAL_MODEL_NAME}  embed={EMBED_MODEL}")


def _service_headers() -> dict:
    """Service credentials for quant_api calls (R.5 phase 1b).

    Tolerant of the module being absent so the app still starts; while the API permits
    all requests, no header simply means the call is anonymous, as it was before.
    """
    try:
        from service_auth import auth_headers
    except ImportError:
        return {}
    return auth_headers()


_news_coll = None


def _news_collection():
    """Mongo handle for R.10's news RAG, opened once and reused.

    Returns None rather than raising when mongo is unreachable, so the endpoint can
    report the corpus as unavailable instead of 500-ing — the same reasoning as the
    quote-service fallback: an honest "unavailable" beats an opaque failure.
    """
    global _news_coll
    if _news_coll is not None:
        return _news_coll
    try:
        from pymongo import MongoClient

        client = MongoClient(LOCAL_MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        _news_coll = client["quant_data"]["news_articles_company_matched_v2"]
        return _news_coll
    except Exception as e:  # noqa: BLE001 — any connection failure is the same answer here
        print(f"[news_rag] mongo unavailable: {e}")
        return None


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
        resp = requests.get(f"{QUANT_API}/api/signals/latest", params={"limit": limit},
                            timeout=5, headers=_service_headers())
        data = resp.json()
        signals = data.get("signals", data) if isinstance(data, dict) else data
        return json.dumps(signals[:limit], ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error fetching signals: {e}"


def fetch_positions() -> str:
    try:
        resp = requests.get(f"{QUANT_API}/api/positions", timeout=5,
                            headers=_service_headers())
        return json.dumps(resp.json(), ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error fetching positions: {e}"


def fetch_performance() -> str:
    try:
        resp = requests.get(f"{QUANT_API}/api/performance", timeout=5,
                            headers=_service_headers())
        return json.dumps(resp.json(), ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error fetching performance: {e}"


# =====================================================
# Request models
# =====================================================
class QueryRequest(BaseModel):
    question: str


class AgentRequest(BaseModel):
    question: str = ""
    max_steps: int = 5
    symbol: Optional[str] = None
    lang: str = "zh"
    scope: str = "auto"
    save_history: bool = True


class PortfolioReviewRequest(BaseModel):
    mode: str = "real"
    max_decisions: int = 5
    lang: str = "zh"
    save_history: bool = True
    record_hints: bool = True


class NewsQueryRequest(BaseModel):
    question: str
    symbol: Optional[str] = None
    since: Optional[int] = None   # YYYYMMDD
    until: Optional[int] = None   # YYYYMMDD
    k: int = 8
    rerank: Optional[bool] = None  # None = server default (NEWS_RERANK); see news_rag.py
    theme: Optional[str] = None
    impacted_symbol: Optional[str] = None


# =====================================================
# FastAPI Routes
# =====================================================
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Build the RAG vector store off the event loop so startup does not block.
    asyncio.get_running_loop().run_in_executor(None, _ensure_vs)
    # R.3: preload the reranker when NEWS_RERANK=1, so the first /api/ask/news is not slow.
    from news_rag import warm_up_reranker
    asyncio.get_running_loop().run_in_executor(None, warm_up_reranker)
    if MARKETS_ENABLED:
        from markets import background
        background.start()
    yield


app.router.lifespan_context = lifespan

# Portfolio, global markets and daily briefs (markets/). Host only: it keeps personal data in
# SQLite under ~/Library and calls the claude CLI, neither of which exists in the Docker image.
MARKETS_ENABLED = os.getenv("MARKETS_ENABLED", "").strip() == "1"
if MARKETS_ENABLED:
    from markets.api import router as markets_router
    app.include_router(markets_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": LOCAL_MODEL_NAME,
        "embed_model": EMBED_MODEL,
        "vector_store_ready": _vs.ready,
        "knowledge_docs": len(_vs.docs),
    }


def _ask_docs_impl(request: QueryRequest):
    """RAG-augmented Q&A over the 4 platform architecture markdown files in knowledge/."""
    from langchain_core.messages import AIMessage

    try:
        start = time.time()
        context = retrieve_context(request.question, top_k=3)
        prompt = (
            "You are a quant research assistant for an AI-driven equity signal platform.\n\n"
            f"Knowledge context:\n{context}\n\n"
            f"Question: {request.question}"
        )
        llm = get_chat_llm(temperature=0.7)
        msgs = [HumanMessage(content=prompt)]
        if "qwen" in str(getattr(llm, "model_name", getattr(llm, "model", ""))).lower():
            msgs.append(AIMessage(content="<think>\n\n</think>\n\n"))
        answer = llm.invoke(msgs).content.strip()
        return {"answer": answer, "elapsed_s": round(time.time() - start, 2)}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/api/ask/docs")
def ask_docs(request: QueryRequest):
    """Q&A over the static markdown platform docs in knowledge/ (distinct from /api/ask/news)."""
    return _ask_docs_impl(request)


@app.post("/api/ask")
def ask(request: QueryRequest):
    """Deprecated alias for /api/ask/docs (platform docs only; use /api/ask/news for news RAG)."""
    return _ask_docs_impl(request)


@app.post("/api/ask/news")
def ask_news(request: NewsQueryRequest):
    """R.10 — answer from the 1.2M-article news corpus, with checkable citations.

    Separate from /api/ask/docs rather than replacing it: the knowledge folder answers
    "how does the regime scorer weight factors" and the news corpus answers "what
    happened to NVDA on export controls". Different corpora, different questions.
    """
    try:
        from news_rag import answer_from_news

        coll = _news_collection()
        if coll is None:
            return {"error": "news corpus unavailable (mongo unreachable)"}
        return answer_from_news(
            request.question,
            llm=get_chat_llm(temperature=0.2),
            coll=coll,
            symbol=request.symbol,
            since=request.since,
            until=request.until,
            k=max(1, min(request.k, 20)),
            rerank=request.rerank,
            theme=request.theme,
            impacted_symbol=request.impacted_symbol,
        )
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/api/retrieve/news")
def retrieve_news(
    q: str,
    symbol: str | None = None,
    from_date: str | None = Query(None, alias="from"),
    to_date: str | None = Query(None, alias="to"),
    limit: int = 20,
    theme: str | None = None,
    impacted_symbol: str | None = None,
):
    """R.12 — hybrid retrieval only: articles, no generated answer.

    The one retriever every news consumer uses. quant_ai's MCP server calls
    `news_rag.search_articles` in-process; quant_api's MCP server (R.5) calls this
    endpoint. The fusion therefore lives in one place, and the Python and Java tool
    surfaces cannot drift into two different rankings of the same corpus.
    """
    from news_rag import search_articles
    try:
        return search_articles(
            q, symbol=symbol, from_date=from_date, to_date=to_date, limit=limit,
            theme=theme, impacted_symbol=impacted_symbol,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/agent/research")
def agent_research(request: AgentRequest):
    """ReAct tool-calling research agent: LLM autonomously queries platform data tools."""
    from agent import build_default_question, run_research_agent

    q = (request.question or "").strip()
    if not q:
        if request.symbol and request.symbol.strip():
            q = build_default_question(request.symbol, request.lang)
        else:
            raise HTTPException(status_code=400, detail="question or symbol is required")
    try:
        return run_research_agent(
            q,
            max_steps=max(1, min(request.max_steps, 8)),
            symbol=request.symbol,
            lang=request.lang,
            scope=request.scope,
            save_history=request.save_history,
        )
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/api/agent/portfolio")
def agent_portfolio_review(
    max_decisions: int = 5,
    mode: Optional[str] = None,
    lang: Optional[str] = None,
    request: Optional[PortfolioReviewRequest] = Body(default=None),
):
    """Review rule & real-portfolio decisions through Schema + Grounding + Business gates (F.17 / F13)."""
    try:
        from portfolio_agent import run_portfolio_review

        eff_max = request.max_decisions if request else max_decisions
        eff_mode = (request.mode if request else mode) or "real"
        eff_lang = (request.lang if request else lang) or "zh"
        save_hist = request.save_history if request else True
        rec_hints = request.record_hints if request else True
        return run_portfolio_review(
            max_decisions=max(1, min(eff_max, 10)),
            mode=eff_mode,
            lang=eff_lang,
            save_history=save_hist,
            record_hints=rec_hints,
        )
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/api/agent/portfolio/history")
def agent_portfolio_history(mode: Optional[str] = None, limit: int = 20):
    """List saved portfolio review reports (most recent first)."""
    from portfolio_agent import list_portfolio_reviews

    items = list_portfolio_reviews(mode=mode, limit=max(1, min(limit, 50)))
    return {"items": items}


@app.get("/api/agent/portfolio/history/{run_id}")
def agent_portfolio_history_detail(run_id: str):
    """Fetch full report for one saved portfolio review run."""
    from portfolio_agent import get_portfolio_review

    doc = get_portfolio_review(run_id)
    if not doc:
        raise HTTPException(status_code=404, detail="portfolio review not found")
    return doc


@app.post("/api/agent/research/stream")
def agent_research_stream(request: AgentRequest):
    """SSE stream of the agent run: run_start, step_start, tool_start, tool_call, then final."""
    from fastapi.responses import StreamingResponse
    from agent import agent_events, build_default_question

    q = (request.question or "").strip()
    if not q:
        if request.symbol and request.symbol.strip():
            q = build_default_question(request.symbol, request.lang)
        else:
            raise HTTPException(status_code=400, detail="question or symbol is required")

    def gen():
        try:
            for ev in agent_events(
                q,
                max_steps=max(1, min(request.max_steps, 8)),
                symbol=request.symbol,
                lang=request.lang,
                scope=request.scope,
                save_history=request.save_history,
            ):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/agent/research/history")
def agent_research_history(symbol: Optional[str] = None, limit: int = 20):
    """List saved ReAct research runs (most recent first)."""
    from agent import list_research_runs

    items = list_research_runs(symbol=symbol, limit=max(1, min(limit, 50)))
    return {"items": items}


@app.get("/api/agent/research/history/{run_id}")
def agent_research_history_detail(run_id: str):
    """Fetch full trace and answer for one saved ReAct research run."""
    from agent import get_research_run

    doc = get_research_run(run_id)
    if not doc:
        raise HTTPException(status_code=404, detail="research run not found")
    return doc


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8083))
    print(f"Starting quant_ai on :{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
