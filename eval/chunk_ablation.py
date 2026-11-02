"""Q.14 item 6 — does indexing the whole article (512-token chunks) find better documents than the
title + lede the index holds today? (2026-11-02)

The index embeds `title + first 1,000 chars` per article (tools/index_news.py) because chunking the
corpus is 2.96M vectors and ~20 hours of embedding against 0.85M / 3.2 h, and only 30.2% of
articles fit one 512-token window. The plan left the choice to R.4's measurements. Re-embedding
everything to ask the question would spend the 20 hours to avoid asking it, so this asks it on the
judged pool instead:

  for each of the 60 evaluation queries, every pooled document (2,487 judged pairs, labels from
  the qwen3.8-27b rubric-v3 judge, kappa 0.67 against Claude) is scored three ways by dense
  similarity to the query, and the ranking is scored by nDCG@10 and strict precision@10:

    lede        cosine(query, title + first 1,000 chars): what the index holds now
    chunk_max   max cosine over ~400-token chunks of title + the whole body
    lede_chunk  the mean of the two

Every document ranked is judged, so there is no pool bias between the three (a re-ranking of the
same judged set). What it cannot show is first-stage recall: an article whose relevant passage is
deep in the body and that no system retrieved is in no pool. A gain here is a lower bound on what
chunking could add; no gain here says the body adds little signal beyond the lede.

    python eval/chunk_ablation.py            # embeds what is not cached, then scores
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from bson import ObjectId
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from index_news import DOC_PREFIX, LEDE_CHARS, embed  # noqa: E402
from hybrid_search import DB_NAME, MONGO_URI, QUERY_PREFIX, SRC_COLL  # noqa: E402

SOURCE = os.getenv("JUDGE_SOURCE", "llm_v3_qwen27")
CHUNK_CHARS, CHUNK_STEP, MAX_CHUNKS = 1600, 1400, 12    # ~400 tokens, 200-char overlap, bodies up to ~17K chars
CACHE = Path(os.getenv("CHUNK_CACHE", "/Volumes/Data4T/rag_eval/chunk_ablation.npz"))
K = 10
BATCH = 32


def chunks(title: str, body: str) -> list[str]:
    if not body:
        return [DOC_PREFIX + title]
    out = [f"{DOC_PREFIX}{title}\n\n{body[i:i + CHUNK_CHARS]}" for i in range(0, len(body), CHUNK_STEP)]
    return out[:MAX_CHUNKS]


def load():
    db = MongoClient(MONGO_URI)[DB_NAME]
    pool = list(db["eval_retrieval_pool"].find({}, {"_id": 0}))
    grades = {(j["query_id"], j["doc_key"]): int(j["grade"]) for j in db["eval_retrieval_judgments"].find(
        {"source": SOURCE, "grade": {"$ne": None}}, {"query_id": 1, "doc_key": 1, "grade": 1})}
    ids = list({p["mongo_id"] for p in pool})
    docs = {str(d["_id"]): d for d in db[SRC_COLL].find({"_id": {"$in": [ObjectId(i) for i in ids]}}, {"title": 1, "content": 1, "body": 1, "text": 1})}
    return pool, grades, docs


def body_of(d) -> str:
    for f in ("content", "body", "text"):
        if d.get(f):
            return str(d[f]).strip()
    return ""


def embed_all(texts: list[str]) -> np.ndarray:
    cache = dict(np.load(CACHE, allow_pickle=True)) if CACHE.exists() else {}
    have = dict(zip(cache.get("keys", []), cache.get("vecs", [])))
    todo = [t for t in dict.fromkeys(texts) if t not in have]
    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        for t, v in zip(part, embed(part)):
            have[t] = np.asarray(v, np.float32)
        if (i // BATCH) % 20 == 0:
            print(f"  embedded {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)
            _save(have)
    _save(have)
    m = np.stack([have[t] for t in texts])
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _save(have):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    keys = list(have)
    np.savez(CACHE, keys=np.array(keys, dtype=object), vecs=np.stack([have[k] for k in keys]))


def ndcg(gs: list[int], ideal: list[int], k: int = K) -> float:
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gs[:k]))
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(sorted(ideal, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def boot_ci(x: list[float], n: int = 5000, seed: int = 7):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(x) for _ in x) / len(x) for _ in range(n))
    return round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)


def main() -> None:
    pool, grades, docs = load()
    by_q = defaultdict(list)
    for p in pool:
        if (p["query_id"], p["doc_key"]) in grades and p["mongo_id"] in docs:
            by_q[p["query_id"]].append(p)
    queries = {qid: ps[0]["query"] for qid, ps in by_q.items()}
    challenge = {qid: ps[0].get("challenge", "") for qid, ps in by_q.items()}

    doc_text, doc_chunks, lens = {}, {}, []
    for mid, d in docs.items():
        title, body = (d.get("title") or "").strip(), body_of(d)
        lens.append(len(body))
        doc_text[mid] = f"{DOC_PREFIX}{title}\n\n{body[:LEDE_CHARS]}" if body else DOC_PREFIX + title
        doc_chunks[mid] = chunks(title, body)
    all_chunks = [c for cs in doc_chunks.values() for c in cs]
    print(f"queries {len(by_q)}, judged pairs {sum(len(v) for v in by_q.values())}, documents {len(docs)}, "
          f"chunks {len(all_chunks)} (bodies > {CHUNK_CHARS} chars: {sum(l > CHUNK_CHARS for l in lens)}, "
          f"truncated at {MAX_CHUNKS} chunks: {sum(l > CHUNK_STEP * MAX_CHUNKS for l in lens)})")

    qv = dict(zip(queries, embed_all([QUERY_PREFIX + q for q in queries.values()])))
    lv = dict(zip(doc_text, embed_all(list(doc_text.values()))))
    cv_all = embed_all(all_chunks)
    cv, pos = {}, 0
    for mid, cs in doc_chunks.items():
        cv[mid] = cv_all[pos:pos + len(cs)]
        pos += len(cs)

    systems = ("lede", "chunk_max", "lede_chunk")
    res = {s: {} for s in systems}
    for qid, ps in by_q.items():
        q = qv[qid]
        ideal = [grades[(qid, p["doc_key"])] for p in ps]
        sc = {"lede": [float(lv[p["mongo_id"]] @ q) for p in ps],
              "chunk_max": [float((cv[p["mongo_id"]] @ q).max()) for p in ps]}
        sc["lede_chunk"] = [(a + b) / 2 for a, b in zip(sc["lede"], sc["chunk_max"])]
        for s in systems:
            order = sorted(range(len(ps)), key=lambda i: -sc[s][i])
            gs = [ideal[i] for i in order]
            res[s][qid] = {"ndcg": ndcg(gs, ideal), "p_strict": sum(g == 2 for g in gs[:K]) / K}

    out = {"judge": SOURCE, "queries": len(by_q), "systems": {}}
    for s in systems:
        nd = [res[s][q]["ndcg"] for q in by_q]
        ps_ = [res[s][q]["p_strict"] for q in by_q]
        out["systems"][s] = {"ndcg@10": round(sum(nd) / len(nd), 4), "p_strict@10": round(sum(ps_) / len(ps_), 4),
                             "by_challenge": {c: round(sum(res[s][q]["ndcg"] for q in by_q if challenge[q] == c) /
                                                       max(1, sum(challenge[q] == c for q in by_q)), 4)
                                              for c in sorted(set(challenge.values()))}}
    for s in ("chunk_max", "lede_chunk"):
        d = [res[s][q]["ndcg"] - res["lede"][q]["ndcg"] for q in by_q]
        sd = (sum((x - sum(d) / len(d)) ** 2 for x in d) / (len(d) - 1)) ** 0.5
        out[f"{s}_minus_lede"] = {"mean": round(sum(d) / len(d), 4), "ci95": boot_ci(d),
                                  "t": round(sum(d) / len(d) / (sd / len(d) ** 0.5), 2) if sd > 0 else None,
                                  "queries_better": sum(x > 1e-9 for x in d), "queries_worse": sum(x < -1e-9 for x in d)}
    print(json.dumps(out, indent=1))
    Path(CACHE.parent / "chunk_ablation_report.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
