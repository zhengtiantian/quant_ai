"""R.4 step 2 — relevance judgments over the pooled candidates.

The pool built by `build_pool.py` is useless until every (query, document) pair carries a
graded label. This script produces those labels two ways, and the second one is the point
of the exercise:

**LLM judging** (`--llm`) labels all 1,810 pairs. It is the only way to cover the pool at
this size, and on its own it is an unverified instrument -- a judge whose error rate is
unknown cannot certify a retrieval result.

**Human judging** (`--human`) labels a stratified sample in the terminal. Comparing the
two on the same pairs gives Cohen's kappa, which is what turns the LLM judge from an
assumption into a measured instrument. If agreement is poor the retrieval table is not
reported as fact; that is the honest outcome, not a failure of the method.

Judging is blind on purpose. The pool carries no system identity and no rank, so nothing
here can prefer the system that retrieved a document. `eval_retrieval_runs` holds the
rankings and is only read at scoring time.

Grades (graded, not binary, because nDCG needs the middle grade to mean anything):
  2 — answers the query: this document is what someone asking it wanted
  1 — related: same company or theme, but does not answer what was asked
  0 — not relevant

Rubric v2 (2026-10-09). The first run (source "llm", rubric v1 in commit 4633a59) was read
as grade inflation because 93% of pairs came back relevant. Checked against 150 reference
labels (stratified by challenge and by the v1 grade) that reading was wrong: about 91% of
the pool *is* at least related -- it is the top 15 of four retrievers -- and v1's real fault
was the 1/2 boundary: it graded 29 of 80 answering documents as merely related (kappa 0.41).
v2 spells out that boundary (opposite outcomes are 1; word overlap, market-research
reports and stock promotion are 0; a bare name query is answered by a document mainly about
it): kappa 0.74 on the same pairs, and its estimated pool distribution (10/26/64% for
0/1/2) matches the reference (9/25/66%). Extracting facts and grading in code reached 0.76
but put 76% of the pool at grade 2; qwen3.5-4b scored 0.64 either way.

    python eval/judge_pool.py --llm                  # label everything with rubric v2, resumable
    python eval/judge_pool.py --llm --limit 50       # smoke test
    python eval/judge_pool.py --human --sample 100   # stratified sample for kappa
    python eval/judge_pool.py --human --sample 30 --among reference_claude   # spot-check the reference
    python eval/judge_pool.py --kappa human llm_v2   # agreement between any two sources
    python eval/judge_pool.py --import-labels FILE --as reference_claude
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from pymongo import MongoClient, UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from hybrid_search import MONGO_URI, DB_NAME  # noqa: E402

POOL_COLL = "eval_retrieval_pool"
JUDGE_COLL = "eval_retrieval_judgments"

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1").rstrip("/")
# A non-thinking instruct model on purpose: the qwen3.5 MLX builds emit their
# chain of thought into `reasoning_content` and leave `content` empty, which costs
# tokens and returns nothing to parse. Judge quality is not assumed either way --
# `--kappa` measures it against human labels.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemma-4-e4b-it-mlx")
JUDGE_SOURCE = os.getenv("JUDGE_SOURCE", "llm_v2")  # "llm" holds the rubric-v1 run, kept for comparison
WORKERS = int(os.getenv("JUDGE_WORKERS", "4"))
EXCERPT = 600

PROMPT = """You are grading search results for a financial news retrieval system.

Query: {query}

Document
  Title: {title}
  Company: {symbol}
  Date: {date}
  Body: {body}

Grades:
  2 = the document reports the specific thing the query asks for. For a query that is only a
      name, product or ticker: the document is mainly about that entity.
  1 = the document is about the same company/entity or the same specific theme, but does not
      report what was asked -- including the opposite outcome (strong guidance when the query
      asks for weak guidance, a beat when it asks for a miss).
  0 = everything else: documents that only share words with the query, generic market-research
      reports, stock-promotion pages, and documents about a different subject.

Reply with JSON only: {{"grade": 0|1|2, "why": "<one short sentence>"}}"""


def _db():
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)[DB_NAME]


def _pending(db, source: str, limit: int | None):
    """Pool pairs this source has not judged yet, so a killed run resumes where it stopped."""
    done = {(j["query_id"], j["doc_key"]) for j in
            db[JUDGE_COLL].find({"source": source}, {"query_id": 1, "doc_key": 1})}
    rows = [d for d in db[POOL_COLL].find() if (d["query_id"], d["doc_key"]) not in done]
    return rows[:limit] if limit else rows


def _ask(doc) -> dict | None:
    body = (doc.get("excerpt") or "").strip()
    payload = {
        "model": JUDGE_MODEL,
        "temperature": 0,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": PROMPT.format(
            query=doc["query"],
            title=doc.get("title") or "(no title)",
            symbol=doc.get("symbol") or "(unknown)",
            date=doc.get("date") or "(unknown)",
            body=body[:EXCERPT] if body else "(title only -- this article has no body text)",
        )}],
    }
    try:
        r = requests.post(f"{LM_STUDIO_URL}/chat/completions", json=payload,
                          headers={"Content-Type": "application/json"}, timeout=180)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or ""
    except Exception as e:  # noqa: BLE001 - one dead pair must not kill a 1,810-pair run
        return {"error": str(e)[:200]}

    # Models wrap JSON in prose or fences often enough that locating the object beats parsing.
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {"error": f"unparseable: {text[:120]}"}
    try:
        out = json.loads(text[start:end + 1])
        grade = int(out.get("grade"))
    except Exception:  # noqa: BLE001
        return {"error": f"unparseable: {text[:120]}"}
    if grade not in (0, 1, 2):
        return {"error": f"grade out of range: {grade}"}
    return {"grade": grade, "why": str(out.get("why", ""))[:300]}


def _record(doc, result, source, model):
    # judge_id = source: the collection's unique index is (judge_id, query_id, doc_key), shared
    # with the MCP submit_judgment tool, so every judging source needs its own judge_id or the
    # second source to label a pair collides with the first.
    return UpdateOne(
        {"judge_id": source, "query_id": doc["query_id"], "doc_key": doc["doc_key"], "source": source},
        {"$set": {
            "query": doc["query"],
            "challenge": doc["challenge"],
            "grade": result.get("grade"),
            "why": result.get("why", ""),
            "error": result.get("error"),
            "model": model,
            "judged_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )


def judge_llm(limit: int | None) -> None:
    db = _db()
    rows = _pending(db, JUDGE_SOURCE, limit)
    if not rows:
        print("nothing left to judge")
        return
    print(f"judging {len(rows)} pairs with {JUDGE_MODEL} as {JUDGE_SOURCE} ({WORKERS} workers)", flush=True)

    ops, errors, dist = [], 0, {0: 0, 1: 0, 2: 0}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, (doc, res) in enumerate(zip(rows, pool.map(_ask, rows)), 1):
            if res.get("error"):
                errors += 1
            else:
                dist[res["grade"]] += 1
            ops.append(_record(doc, res, JUDGE_SOURCE, JUDGE_MODEL))
            if len(ops) >= 50:
                db[JUDGE_COLL].bulk_write(ops)
                ops = []
            if i % 50 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}  grades {dist}  errors {errors}", flush=True)
    if ops:
        db[JUDGE_COLL].bulk_write(ops)
    print(f"done: {sum(dist.values())} judged, {errors} errors -> {DB_NAME}.{JUDGE_COLL}")


def judge_human(sample: int, seed: int, among: str | None = None) -> None:
    """Stratified over the five challenge types, so kappa is not dominated by one kind.
    among: only pairs another source judged, so the two can be compared (e.g. reference_claude)."""
    db = _db()
    rows = _pending(db, "human", None)
    if among:
        keep = {(j["query_id"], j["doc_key"]) for j in db[JUDGE_COLL].find({"source": among})}
        rows = [r for r in rows if (r["query_id"], r["doc_key"]) in keep]
    by_challenge: dict[str, list] = {}
    for r in rows:
        by_challenge.setdefault(r["challenge"], []).append(r)
    rnd = random.Random(seed)
    picks = []
    per = max(1, sample // max(1, len(by_challenge)))
    for challenge, items in sorted(by_challenge.items()):
        rnd.shuffle(items)
        picks.extend(items[:per])
    rnd.shuffle(picks)
    picks = picks[:sample]

    print(f"{len(picks)} pairs to judge. 0/1/2 to grade, s to skip, q to stop and save.\n")
    ops = []
    for i, doc in enumerate(picks, 1):
        body = (doc.get("excerpt") or "").strip()
        print("=" * 78)
        print(f"[{i}/{len(picks)}]  QUERY: {doc['query']}   ({doc['challenge']})")
        print(f"  {doc.get('symbol') or '?'}  {doc.get('date') or '?'}  {doc.get('title') or '(no title)'}")
        print(f"  {body[:400] if body else '(title only)'}")
        while True:
            ans = input("  grade [0/1/2/s/q]: ").strip().lower()
            if ans in ("0", "1", "2", "s", "q"):
                break
        if ans == "q":
            break
        if ans == "s":
            continue
        ops.append(_record(doc, {"grade": int(ans)}, "human", "human"))
    if ops:
        db[JUDGE_COLL].bulk_write(ops)
    print(f"\nsaved {len(ops)} human judgments")


def import_labels(path: str, source: str) -> None:
    """Loads graded pairs judged outside this script, e.g. the 150 reference labels of 2026-10-09.
    File: JSON list of {"query_id", "doc_key", "grade"}; the source name says who judged them."""
    db = _db()
    pool = {(d["query_id"], d["doc_key"]): d for d in db[POOL_COLL].find()}
    ops, missing = [], 0
    for row in json.load(open(path)):
        doc = pool.get((row["query_id"], row["doc_key"]))
        if doc is None or row.get("grade") not in (0, 1, 2):
            missing += 1
            continue
        ops.append(_record(doc, {"grade": row["grade"], "why": row.get("why", "")}, source, source))
    if ops:
        db[JUDGE_COLL].bulk_write(ops)
    print(f"imported {len(ops)} labels as source={source} ({missing} skipped)")


def kappa(a_source: str = "human", b_source: str = JUDGE_SOURCE) -> None:
    """Cohen's kappa on pairs both judges labelled; also the plain agreement rate."""
    db = _db()
    human = {(j["query_id"], j["doc_key"]): j["grade"]
             for j in db[JUDGE_COLL].find({"source": a_source, "grade": {"$ne": None}})}
    llm = {(j["query_id"], j["doc_key"]): j["grade"]
           for j in db[JUDGE_COLL].find({"source": b_source, "grade": {"$ne": None}})}
    both = sorted(set(human) & set(llm))
    if not both:
        print(f"no pairs judged by both {a_source} and {b_source}")
        return
    print(f"{a_source} (rows) vs {b_source} (cols)")

    n = len(both)
    agree = sum(1 for k in both if human[k] == llm[k])
    ph = {g: sum(1 for k in both if human[k] == g) / n for g in (0, 1, 2)}
    pl = {g: sum(1 for k in both if llm[k] == g) / n for g in (0, 1, 2)}
    pe = sum(ph[g] * pl[g] for g in (0, 1, 2))
    po = agree / n
    k = (po - pe) / (1 - pe) if pe < 1 else 0.0

    print(f"overlapping pairs : {n}")
    print(f"exact agreement   : {po:.3f}  ({agree}/{n})")
    print(f"Cohen's kappa     : {k:.3f}")
    print(f"\nconfusion (rows {a_source}, cols {b_source})")
    print("        0     1     2")
    for g in (0, 1, 2):
        row = [sum(1 for x in both if human[x] == g and llm[x] == c) for c in (0, 1, 2)]
        print(f"  {g}  " + "  ".join(f"{v:>4}" for v in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="label the pool with the local model")
    ap.add_argument("--human", action="store_true", help="label a stratified sample by hand")
    ap.add_argument("--kappa", nargs="*", metavar="SOURCE",
                    help="agreement between two sources (default: human vs the current judge)")
    ap.add_argument("--import-labels", metavar="FILE", help="load externally judged pairs")
    ap.add_argument("--as", dest="as_source", default="reference", help="source name for --import-labels")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--among", metavar="SOURCE", help="with --human: sample only pairs this source judged")
    a = ap.parse_args()
    if a.llm:
        judge_llm(a.limit)
    elif a.human:
        judge_human(a.sample, a.seed, a.among)
    elif a.kappa is not None:
        kappa(*(a.kappa[:2] if len(a.kappa) >= 2 else ["human", a.kappa[0] if a.kappa else JUDGE_SOURCE]))
    elif a.import_labels:
        import_labels(a.import_labels, a.as_source)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
