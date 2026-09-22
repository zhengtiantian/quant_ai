"""R.4 step 3 — score the four retrieval systems against the pooled judgments.

`build_pool.py` produced the rankings, `judge_pool.py` produced the labels; this is where
they meet and the comparison stops being a matter of taste.

What the numbers mean, and what they do not:

**Unjudged is treated as non-relevant.** Only pooled documents were judged, so a document
no system retrieved scores zero by construction. Every recall figure here is therefore an
*upper bound*, shared equally by the four systems — fair for comparing them, wrong for
claiming absolute recall of the corpus.

**Two relevance bars, both reported.** recall/MRR at grade >= 1 ("related or better") and
at grade == 2 ("answers the query"). A system that surfaces the right company but the
wrong article looks fine on the loose bar and collapses on the strict one, and that gap is
worth seeing rather than choosing a bar that hides it.

**nDCG is the graded one** — gain 2^g - 1, ideal taken over the judged pool for that query,
which is why the middle grade had to exist.

**Per challenge, not only overall.** The aggregate is the least interesting row: the whole
argument for hybrid retrieval is that sparse wins on literal queries and dense wins on
paraphrase, and only the split shows whether that held.

    python eval/score_runs.py
    python eval/score_runs.py --k 5
    python eval/score_runs.py --md      # markdown tables, for the writeup
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from hybrid_search import MONGO_URI, DB_NAME  # noqa: E402

POOL_COLL = "eval_retrieval_pool"
RUNS_COLL = "eval_retrieval_runs"
JUDGE_COLL = "eval_retrieval_judgments"

SYSTEM_ORDER = ["sparse", "dense", "hybrid_k60", "hybrid_k5"]


def load(source: str):
    db = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)[DB_NAME]
    grades: dict[tuple[str, str], int] = {}
    for j in db[JUDGE_COLL].find({"source": source, "grade": {"$ne": None}}):
        grades[(j["query_id"], j["doc_key"])] = j["grade"]
    runs: dict[str, dict[str, list]] = defaultdict(dict)
    for r in db[RUNS_COLL].find():
        runs[r["system"]][r["query_id"]] = r["ranking"]
    challenge = {d["query_id"]: d["challenge"] for d in
                 db[POOL_COLL].find({}, {"query_id": 1, "challenge": 1})}
    return grades, runs, challenge


def metrics(ranking, qid, grades, k, bar):
    """recall@k and reciprocal rank at the given relevance bar, plus graded nDCG@k."""
    top = ranking[:k]
    rel_in_pool = [key for (q, key), g in grades.items() if q == qid and g >= bar]
    hits = [key for key in top if grades.get((qid, key), 0) >= bar]

    recall = len(hits) / len(rel_in_pool) if rel_in_pool else None
    rr = 0.0
    for i, key in enumerate(top, 1):
        if grades.get((qid, key), 0) >= bar:
            rr = 1.0 / i
            break

    gains = [(2 ** grades.get((qid, key), 0) - 1) for key in top]
    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(gains, 1))
    ideal = sorted((2 ** g - 1 for (q, _), g in grades.items() if q == qid), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    ndcg = dcg / idcg if idcg else None
    return recall, rr, ndcg


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def table(rows, headers, md: bool) -> str:
    if md:
        out = ["| " + " | ".join(headers) + " |",
               "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    out = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)),
           "  ".join("-" * w for w in widths)]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--source", default="llm", help="which judge's labels to score against")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()

    grades, runs, challenge = load(a.source)
    if not grades:
        print(f"no judgments from source={a.source}; run judge_pool.py first")
        return

    qids = sorted({q for sys_runs in runs.values() for q in sys_runs})
    judged_q = {q for q, _ in grades}
    missing = [q for q in qids if q not in judged_q]
    print(f"judge={a.source}  queries={len(qids)}  judged pairs={len(grades)}  k={a.k}")
    if missing:
        print(f"WARNING: {len(missing)} queries have no judgments yet: {', '.join(missing[:8])}")
    print()

    systems = [s for s in SYSTEM_ORDER if s in runs] + \
              [s for s in sorted(runs) if s not in SYSTEM_ORDER]

    # ---- overall ----
    rows = []
    per_system_by_challenge: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for s in systems:
        loose = [metrics(runs[s].get(q, []), q, grades, a.k, 1) for q in qids]
        strict = [metrics(runs[s].get(q, []), q, grades, a.k, 2) for q in qids]
        rows.append([
            s,
            f"{mean(r for r, _, _ in loose):.3f}",
            f"{mean(rr for _, rr, _ in loose):.3f}",
            f"{mean(r for r, _, _ in strict):.3f}",
            f"{mean(rr for _, rr, _ in strict):.3f}",
            f"{mean(n for _, _, n in loose):.3f}",
        ])
        for q, (r, _, n) in zip(qids, loose):
            per_system_by_challenge[s][challenge.get(q, "?")].append((r, n))

    print(table(rows, ["system", f"recall@{a.k}", f"MRR@{a.k}",
                       f"recall@{a.k} (g=2)", f"MRR@{a.k} (g=2)", f"nDCG@{a.k}"], a.md))
    print()

    # ---- per challenge, recall at the loose bar ----
    challenges = sorted({c for c in challenge.values()})
    rows = [[s] + [f"{mean(r for r, _ in per_system_by_challenge[s][c]):.3f}" for c in challenges]
            for s in systems]
    print(f"recall@{a.k} by challenge type")
    print(table(rows, ["system"] + challenges, a.md))
    print()

    rows = [[s] + [f"{mean(n for _, n in per_system_by_challenge[s][c]):.3f}" for c in challenges]
            for s in systems]
    print(f"nDCG@{a.k} by challenge type")
    print(table(rows, ["system"] + challenges, a.md))


if __name__ == "__main__":
    main()
