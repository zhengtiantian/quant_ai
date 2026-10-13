"""R.3 — cross-encoder reranking on top of hybrid retrieval.

Why a second stage at all. Both retrieval legs score a document without ever looking at the
query and the document together: dense compares two vectors computed separately, sparse
counts term overlap. A cross-encoder reads the (query, document) pair in one forward pass,
so it can see that "memory glut pressures chip makers" is answered by "Micron warns DRAM
oversupply" and *not* by "Samsung memory chip unveiled" -- which share the same words and
sit close in embedding space. It is far better at relevance and far too slow to run over
729,101 documents, so it runs over the fused top-N only. Retrieve-then-rerank is how that
trade is resolved; N and the latency it costs are the design parameters.

Model: BAAI/bge-reranker-v2-m3 (568M, multilingual, 512-token window). The document text is
title + body excerpt, truncated by the model to 512 tokens; R.1 measured that only 30.2% of
articles fit one window, so the reranker judges the lede, which is where news puts its
subject anyway. Title-only articles (17.9% of the corpus) are reranked on their title.

Measured (R.3, 2026-10-12, Apple M-series, MPS): 100 candidates in ~3.3 s warm. That is the
cost the evaluation table has to justify.

    python tools/rerank.py "memory glut pressures chip makers"
    python tools/rerank.py "export controls" --symbol NVDA --candidates 50
"""

from __future__ import annotations

import argparse
import os
import time
from functools import lru_cache

from bson import ObjectId

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "mps")
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "32"))
# Tokens per (query, document) pair. Cost grows with it; news puts the subject in the lede, so a
# shorter window may cost little relevance. Measured by eval/rerank_sweep.sh.
RERANK_MAX_LEN = int(os.getenv("RERANK_MAX_LEN", "512"))
BODY_CHARS = 1500   # ~350 tokens of lede after the title; the model truncates at 512 anyway


@lru_cache(maxsize=2)
def _model(max_len: int = RERANK_MAX_LEN):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE, max_length=max_len)


def doc_text(title: str | None, body: str | None) -> str:
    title = (title or "").strip()
    body = (body or "").strip()
    return f"{title}\n{body[:BODY_CHARS]}" if body else title


def fetch_bodies(coll, hits) -> dict[str, str]:
    """One batched Mongo read for the candidates' bodies, keyed by mongo_id."""
    ids = []
    for h in hits:
        if h.mongo_id:
            try:
                ids.append(ObjectId(h.mongo_id))
            except Exception:  # noqa: BLE001
                pass
    return {str(d["_id"]): d.get("content") or "" for d in coll.find({"_id": {"$in": ids}}, {"content": 1})}


def rerank(query: str, hits: list, bodies: dict[str, str], k: int | None = None,
           max_len: int | None = None) -> tuple[list, float]:
    """Return (hits sorted by cross-encoder score, seconds spent). Sets h.rerank_score."""
    if not hits:
        return [], 0.0
    pairs = [(query, doc_text(h.title, bodies.get(h.mongo_id or "", ""))) for h in hits]
    t = time.perf_counter()
    scores = _model(max_len or RERANK_MAX_LEN).predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)
    spent = time.perf_counter() - t
    for h, s in zip(hits, scores):
        h.rerank_score = float(s)
    ranked = sorted(hits, key=lambda h: h.rerank_score, reverse=True)
    return (ranked[:k] if k else ranked), spent


def search_reranked(query, k=10, symbol=None, since=None, until=None, candidates=100, rrf_k=None):
    """Hybrid retrieval (RRF) for `candidates` documents, then cross-encoder rerank to `k`."""
    from hybrid_search import (DB_NAME, MONGO_URI, QDRANT_URL, RRF_K, SRC_COLL,
                               dense_leg, rrf_fuse, sparse_leg)
    from pymongo import MongoClient
    from qdrant_client import QdrantClient

    qc = QdrantClient(url=QDRANT_URL, timeout=60)
    coll = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)[DB_NAME][SRC_COLL]
    dense = dense_leg(qc, query, symbol, since, until, candidates)
    sparse = sparse_leg(coll, query, symbol, since, until, candidates)
    fused = rrf_fuse(dense, sparse, candidates, rrf_k or RRF_K, 1.0, 1.0)
    ranked, spent = rerank(query, fused, fetch_bodies(coll, fused), k)
    return fused, ranked, spent


def main() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--symbol")
    ap.add_argument("--since", type=int)
    ap.add_argument("--until", type=int)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--candidates", type=int, default=100)
    a = ap.parse_args()

    fused, ranked, spent = search_reranked(a.query, a.k, a.symbol, a.since, a.until, a.candidates)
    pos = {h.key: i for i, h in enumerate(fused, 1)}
    print(f'query: "{a.query}"  rerank {len(fused)} candidates in {spent:.2f}s')
    for i, h in enumerate(ranked, 1):
        print(f" {i:>2}. [rrf #{pos[h.key]:>3}  ce {h.rerank_score:.3f}] {str(h.symbol):<6} {h.date}  {str(h.title)[:70]}")


if __name__ == "__main__":
    main()
