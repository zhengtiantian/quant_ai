"""R.8 — measure retrieval latency, split into embed vs search.

"Where does the time go" should be answered with a breakdown, not a guess. The split
matters because the two halves have different fixes: embed latency is the local model,
search latency is the index configuration (in particular whether vectors are held in
RAM or memory-mapped from disk).

Usage:
    python tools/bench_search.py                 # unfiltered
    python tools/bench_search.py --filtered      # with symbol + date range filters
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

LM = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:26333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "news_v1")

QUERIES = [
    "memory chip oversupply hurting DRAM prices",
    "China export controls on advanced semiconductors",
    "CEO resigns amid accounting scandal",
    "quarterly earnings beat analyst expectations",
    "data center capex guidance raised for AI demand",
    "antitrust regulator opens investigation",
    "supply chain disruption at a key contract manufacturer",
    "activist investor takes a stake and pushes for a breakup",
    "product recall over a safety defect",
    "dividend increase and new share buyback authorisation",
]


def embed(text: str) -> list[float]:
    r = requests.post(
        f"{LM}/embeddings",
        json={"model": EMBED_MODEL, "input": ["search_query: " + text]},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    qc = QdrantClient(url=QDRANT_URL, timeout=60)
    info = qc.get_collection(COLLECTION)
    print(f"collection={COLLECTION} points={info.points_count} k={args.k} "
          f"filtered={args.filtered}")

    qfilter = None
    if args.filtered:
        qfilter = qm.Filter(must=[
            qm.FieldCondition(key="date_int", range=qm.Range(gte=20230101, lte=20261231)),
        ])

    vecs = [embed(q) for q in QUERIES]  # warm the model out of the measurement

    e_lat, s_lat = [], []
    for _ in range(args.rounds):
        for q, v in zip(QUERIES, vecs):
            t = time.perf_counter()
            embed(q)
            e_lat.append((time.perf_counter() - t) * 1000)

            t = time.perf_counter()
            qc.query_points(COLLECTION, query=v, limit=args.k,
                            query_filter=qfilter, with_payload=True)
            s_lat.append((time.perf_counter() - t) * 1000)

    for name, lat in (("embed ", e_lat), ("search", s_lat)):
        print(f"  {name}  n={len(lat):<4} p50={statistics.median(lat):7.2f}ms  "
              f"p95={pct(lat, 95):7.2f}ms  p99={pct(lat, 99):7.2f}ms  "
              f"max={max(lat):7.2f}ms")
    tot = [a + b for a, b in zip(e_lat, s_lat)]
    print(f"  total   p50={statistics.median(tot):7.2f}ms  p95={pct(tot, 95):7.2f}ms")


if __name__ == "__main__":
    main()
