"""R.4 step 1 — build the pooled candidate set for relevance judging.

Pooled judging is the standard IR method (TREC has used it for decades): run several
systems, take the top-N of each, union them, judge the union, then score every system
against those judgments. It exists because judging 716,074 documents per query is
impossible and judging only one system's output would bias the comparison toward it.

Two things this script is careful about.

**Ranks and system identity are stored separately from what a judge sees.** A judge told
"this was dense's #1" is no longer judging relevance. The runs go to
`eval_retrieval_runs` for scoring; the pool served for judgment carries only the query
and the document.

**Pool bias is real and is not hidden.** A document none of the four systems retrieved is
never judged and counts as non-relevant, so every recall figure computed this way is an
*upper bound*. Adding diverse systems to the pool widens coverage but cannot eliminate
this. It is a limitation to state in the writeup, not a reason to avoid the method --
the alternative is no measurement at all.

    python eval/build_pool.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from hybrid_search import (  # noqa: E402
    MONGO_URI, QDRANT_URL, DB_NAME, SRC_COLL,
    dense_leg, rrf_fuse, sparse_leg,
)
from qdrant_client import QdrantClient  # noqa: E402

QUERIES = Path(__file__).with_name("retrieval_queries.yaml")
POOL_COLL = "eval_retrieval_pool"
RUNS_COLL = "eval_retrieval_runs"
POOL_DEPTH = int(os.getenv("POOL_DEPTH", "30"))
EXCERPT = 600

# The four systems under comparison. K=60/depth 100 is the published RRF default;
# K=5/depth 30 is the setting R.2 found looked better by eye, deliberately left
# unadopted so this measurement decides rather than taste.
SYSTEMS = {
    "sparse":       dict(mode="sparse", candidates=100),
    "dense":        dict(mode="dense",  candidates=100),
    "hybrid_k60":   dict(mode="hybrid", candidates=100, rrf_k=60),
    "hybrid_k5":    dict(mode="hybrid", candidates=30,  rrf_k=5),
}


def run_system(cfg, qc, coll, q, symbol, since, until, depth):
    n = cfg["candidates"]
    if cfg["mode"] == "sparse":
        return sparse_leg(coll, q, symbol, since, until, n)[:depth]
    if cfg["mode"] == "dense":
        return dense_leg(qc, q, symbol, since, until, n)[:depth]
    d = dense_leg(qc, q, symbol, since, until, n)
    s = sparse_leg(coll, q, symbol, since, until, n)
    return rrf_fuse(d, s, depth, cfg["rrf_k"], 1.0, 1.0)


def main() -> None:
    spec = yaml.safe_load(QUERIES.read_text())["queries"]
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    db = client[DB_NAME]
    src, pool, runs = db[SRC_COLL], db[POOL_COLL], db[RUNS_COLL]
    qc = QdrantClient(url=QDRANT_URL, timeout=120)

    pool.delete_many({})
    runs.delete_many({})
    pool.create_index([("query_id", 1), ("doc_key", 1)], unique=True)
    runs.create_index([("query_id", 1), ("system", 1)], unique=True)

    from bson import ObjectId

    total_pool = 0
    for i, spec_q in enumerate(spec, 1):
        qid = spec_q["id"]
        q = spec_q["query"]
        symbol = spec_q.get("symbol")
        since, until = spec_q.get("since"), spec_q.get("until")

        seen: dict[str, dict] = {}
        for name, cfg in SYSTEMS.items():
            hits = run_system(cfg, qc, src, q, symbol, since, until, POOL_DEPTH)
            runs.insert_one({
                "query_id": qid, "system": name,
                "ranking": [h.key for h in hits],   # ranks live here, not in the pool
            })
            for h in hits:
                if h.key not in seen:
                    seen[h.key] = h

        # One batched body fetch for the whole pool of this query.
        ids = []
        for h in seen.values():
            if h.mongo_id:
                try:
                    ids.append(ObjectId(h.mongo_id))
                except Exception:  # noqa: BLE001
                    pass
        bodies = {
            str(d["_id"]): d
            for d in src.find({"_id": {"$in": ids}},
                              {"content": 1, "url": 1, "llm_event_type_a": 1})
        }

        docs = []
        for h in seen.values():
            body = (bodies.get(h.mongo_id, {}).get("content") or "").strip()
            docs.append({
                "query_id": qid,
                "query": q,
                "challenge": spec_q["challenge"],
                "doc_key": h.key,
                "mongo_id": h.mongo_id,
                "symbol": h.symbol,
                "date": h.date,
                "title": h.title,
                "excerpt": body[:EXCERPT] if body else "",
                "has_body": bool(body),
                "url": bodies.get(h.mongo_id, {}).get("url") or h.url,
            })
        if docs:
            pool.insert_many(docs)
        total_pool += len(docs)
        print(f"[{i:>2}/{len(spec)}] {qid:<4} {q[:46]:<48} pool={len(docs):>3}", flush=True)

    print(f"\n{len(spec)} queries, {total_pool} pooled documents "
          f"({total_pool / max(1, len(spec)):.1f} per query)")
    print(f"pool -> {DB_NAME}.{POOL_COLL}   rankings -> {DB_NAME}.{RUNS_COLL}")


if __name__ == "__main__":
    main()
