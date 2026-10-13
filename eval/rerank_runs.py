"""R.3 + R.4 — add the reranked system to the evaluation, and judge what it newly surfaces.

The reranker takes hybrid_k60's top-`CANDIDATES` and reorders it. Its top-10 can therefore
contain documents that no first-stage system ranked inside the pool depth (15), so they were
never judged. Scoring it against the existing pool alone would count every one of those as
non-relevant and understate the reranker -- the same pool bias build_pool.py describes, but
aimed at one system. So this script also **extends the pool** with the reranker's top
POOL_DEPTH, and those new pairs go through the same blind judge before scoring.

The latency of each query's rerank is stored with the run: the table is only honest if it
says what the gain costs.

    python eval/rerank_runs.py                    # runs + pool extension
    python eval/judge_pool.py --llm --rubric v3 --model qwen/qwen3.8-27b --source <set>
    python eval/score_runs.py --source <set>
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import yaml
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from hybrid_search import (  # noqa: E402
    DB_NAME, MONGO_URI, QDRANT_URL, SRC_COLL, dense_leg, rrf_fuse, sparse_leg,
)
from qdrant_client import QdrantClient  # noqa: E402
from rerank import RERANK_MODEL, fetch_bodies, rerank  # noqa: E402

QUERIES = Path(__file__).with_name("retrieval_queries.yaml")
POOL_COLL = "eval_retrieval_pool"
RUNS_COLL = "eval_retrieval_runs"
POOL_DEPTH = int(os.getenv("POOL_DEPTH", "15"))
CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "100"))
SYSTEM = os.getenv("RERANK_SYSTEM", "hybrid_k60_rerank")
EXCERPT = 600


def main() -> None:
    spec = yaml.safe_load(QUERIES.read_text())["queries"]
    db = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)[DB_NAME]
    src, pool, runs = db[SRC_COLL], db[POOL_COLL], db[RUNS_COLL]
    qc = QdrantClient(url=QDRANT_URL, timeout=120)

    added, latencies, moved_in = 0, [], 0
    for i, sq in enumerate(spec, 1):
        qid, q = sq["id"], sq["query"]
        symbol, since, until = sq.get("symbol"), sq.get("since"), sq.get("until")
        dense = dense_leg(qc, q, symbol, since, until, CANDIDATES)
        sparse = sparse_leg(src, q, symbol, since, until, CANDIDATES)
        fused = rrf_fuse(dense, sparse, CANDIDATES, 60, 1.0, 1.0)
        bodies = fetch_bodies(src, fused)
        ranked, spent = rerank(q, fused, bodies)
        latencies.append(spent)
        top = ranked[:POOL_DEPTH]

        first_stage = {h.key for h in fused[:10]}
        moved_in += sum(1 for h in top[:10] if h.key not in first_stage)
        runs.update_one(
            {"query_id": qid, "system": SYSTEM},
            {"$set": {"ranking": [h.key for h in top], "rerank_model": RERANK_MODEL,
                      "candidates": len(fused), "rerank_seconds": round(spent, 3)}},
            upsert=True,
        )

        # extend the pool with what only the reranker surfaced
        known = {d["doc_key"] for d in pool.find({"query_id": qid}, {"doc_key": 1})}
        new_docs = []
        for h in top:
            if h.key in known:
                continue
            body = (bodies.get(h.mongo_id or "", "") or "").strip()
            new_docs.append({
                "query_id": qid, "query": q, "challenge": sq["challenge"], "doc_key": h.key,
                "mongo_id": h.mongo_id, "symbol": h.symbol, "date": h.date, "title": h.title,
                "excerpt": body[:EXCERPT] if body else "", "has_body": bool(body), "url": h.url,
                "added_by": SYSTEM,
            })
        if new_docs:
            pool.insert_many(new_docs)
        added += len(new_docs)
        print(f"[{i:>2}/{len(spec)}] {qid:<4} {q[:44]:<46} rerank {spent:5.2f}s  new pool docs {len(new_docs):>2}", flush=True)

    print(f"\n{len(spec)} queries reranked with {RERANK_MODEL} over top-{CANDIDATES}")
    print(f"latency per query: median {statistics.median(latencies):.2f}s  max {max(latencies):.2f}s")
    print(f"{moved_in} documents entered a top-10 that hybrid_k60 did not have there "
          f"({moved_in / len(spec):.1f} per query)")
    print(f"pool extended by {added} pairs -> judge them before scoring")


if __name__ == "__main__":
    main()
