"""R.10 — answer questions from the news corpus, with citations that can be checked.

This is the half of RAG that R.1-R.4 does not cover. Those build retrieval; nothing
consumed it. `/api/ask` still searches the four markdown files in `knowledge/`, so the
platform had 716,074 indexed articles and an answer endpoint that could not see one.

Three properties matter more than the prompt:

**It cites by id.** Every claim carries the article ids it came from. Prose that sounds
sourced but names nothing cannot be checked, and R.11's faithfulness gate needs
something concrete to check against.

**It refuses.** When retrieval returns nothing, the endpoint says so rather than letting
the model answer from what it already knows. A RAG system that silently falls back to
parametric memory is worse than no RAG: the answer still reads as sourced, so the
failure is invisible to whoever asked.

**It filters by date.** Answering "what did the market think in early 2022" with 2026
articles is the M.7 look-ahead error with a natural-language interface on top. The
window is passed to both retrieval legs, not applied afterwards.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tools"))

from hybrid_search import RRF_K, search  # noqa: E402

MAX_SNIPPET = 700
DEFAULT_K = 8

REFUSAL = (
    "I could not find any articles in the corpus matching that question, so I have "
    "nothing to answer from. Rather than guess, here is what to try: widen or remove "
    "the date window, drop the symbol filter, or rephrase using words a headline "
    "would use."
)

SYSTEM = """You answer questions about US equities strictly from the news excerpts provided.

Rules, in order of importance:
1. Use ONLY the excerpts. If they do not contain the answer, say so plainly. Do not fill
   gaps from your own knowledge, even when you are confident and even when the gap is small.
2. Cite every factual claim with the article ids it came from, inline, as [1] or [2][5].
   A sentence with no citation must be a statement about the excerpts themselves
   ("the excerpts do not mention...").
3. If the excerpts disagree, say so and cite both sides rather than picking one.
4. Note the date of what you are reporting. These are historical articles; "recently"
   means relative to the article, not to today.
5. Be brief. Three or four sentences unless the question genuinely needs more."""


def _fetch_bodies(hits, coll) -> dict:
    """Pull article bodies for the retrieved hits, in one query keyed on `_id`.

    Both retrieval legs carry the mongo `_id` for this reason. The obvious version --
    `find_one({"symbol":…, "title":…})` per hit -- has no index to use (the collection
    is indexed on `(symbol, date)` and full-text, not title), so eight hits cost eight
    partial scans and measured 5.9s against 15ms for the retrieval itself. Fetching by
    primary key in a single `$in` is ~4ms.

    Titles are not a key here anyway: syndicated copies share one.
    """
    from bson import ObjectId

    ids = []
    for h in hits:
        if h.mongo_id:
            try:
                ids.append(ObjectId(h.mongo_id))
            except Exception:  # noqa: BLE001 — a malformed id just means no body
                pass
    if not ids:
        return {}
    cursor = coll.find(
        {"_id": {"$in": ids}},
        {"content": 1, "url": 1, "date": 1, "llm_sentiment_final": 1, "llm_event_type_a": 1},
    )
    return {str(d["_id"]): d for d in cursor}


def build_context(hits, coll) -> tuple[str, list[dict]]:
    bodies = _fetch_bodies(hits, coll)

    blocks, sources = [], []
    for i, h in enumerate(hits, 1):
        doc = bodies.get(h.mongo_id) or {}
        body = (doc.get("content") or "").strip()
        excerpt = body[:MAX_SNIPPET] if body else "(headline only — no body text)"
        blocks.append(
            f"[{i}] {h.symbol} | {h.date} | {h.title}\n{excerpt}"
        )
        sources.append({
            "id": i,
            "symbol": h.symbol,
            "date": h.date,
            "title": h.title,
            "url": doc.get("url") or h.url,
            "event_type": doc.get("llm_event_type_a"),
            "sentiment": doc.get("llm_sentiment_final"),
            "dense_rank": h.dense_rank,
            "sparse_rank": h.sparse_rank,
            "rrf": round(h.rrf, 5),
            "has_body": bool(body),
        })
    return "\n\n".join(blocks), sources


def cited_ids(answer: str, n_sources: int) -> list[int]:
    """Ids the answer actually cites, so R.11 can compare them against the sources."""
    found = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    return sorted(i for i in found if 1 <= i <= n_sources)


def answer_from_news(
    question: str,
    llm,
    coll,
    symbol: str | None = None,
    since: int | None = None,
    until: int | None = None,
    k: int = DEFAULT_K,
    rrf_k: int = RRF_K,
) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    t0 = time.perf_counter()
    dense, sparse, fused = search(
        question, k=k, symbol=symbol, since=since, until=until, rrf_k=rrf_k
    )
    t_retrieve = time.perf_counter() - t0

    if not fused:
        return {
            "answer": REFUSAL,
            "refused": True,
            "sources": [],
            "cited": [],
            "retrieval_ms": round(t_retrieve * 1000, 1),
            "generation_ms": 0.0,
        }

    context, sources = build_context(fused, coll)

    t1 = time.perf_counter()
    answer = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"News excerpts:\n\n{context}\n\nQuestion: {question}"),
    ]).content
    t_gen = time.perf_counter() - t1

    return {
        "answer": answer,
        "refused": False,
        "sources": sources,
        "cited": cited_ids(answer, len(sources)),
        # Split, not totalled: the two halves have different fixes, and "the answer took
        # 9 seconds" is not actionable while "8.6s of it was generation" is.
        "retrieval_ms": round(t_retrieve * 1000, 1),
        "generation_ms": round(t_gen * 1000, 1),
        "legs": {"dense": len(dense), "sparse": len(sparse), "fused": len(fused)},
    }
