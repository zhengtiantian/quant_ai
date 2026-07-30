"""Query the R.1 news index — a sanity check before, and a building block for, R.2.

Uses the `search_query: ` prefix, which is the counterpart to the `search_document: `
prefix applied at index time. Mixing them up is the failure mode that reports no error
and simply retrieves worse.

Usage:
    python tools/search_news.py "memory chip oversupply"
    python tools/search_news.py "export controls" --symbol NVDA --since 20240101
"""

from __future__ import annotations

import argparse
import os

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

LM = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:26333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "news_v1")

QUERY_PREFIX = "search_query: "


def embed_query(text: str) -> list[float]:
    r = requests.post(
        f"{LM}/embeddings",
        json={"model": EMBED_MODEL, "input": [QUERY_PREFIX + text]},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--symbol", help="restrict to one ticker")
    ap.add_argument("--since", type=int, help="YYYYMMDD lower bound")
    ap.add_argument("--until", type=int, help="YYYYMMDD upper bound")
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    must = []
    if args.symbol:
        must.append(qm.FieldCondition(key="symbol", match=qm.MatchValue(value=args.symbol)))
    if args.since or args.until:
        must.append(
            qm.FieldCondition(
                key="date_int", range=qm.Range(gte=args.since, lte=args.until)
            )
        )

    qc = QdrantClient(url=QDRANT_URL, timeout=60)
    hits = qc.query_points(
        collection_name=COLLECTION,
        query=embed_query(args.query),
        limit=args.k,
        query_filter=qm.Filter(must=must) if must else None,
        with_payload=True,
    ).points

    print(f'query: "{args.query}"' + (f"  symbol={args.symbol}" if args.symbol else ""))
    for h in hits:
        p = h.payload
        print(
            f"  {h.score:.4f}  {str(p.get('symbol')):<6} {p.get('date')}  "
            f"{str(p.get('event_type') or '-'):<14} {str(p.get('title'))[:78]}"
        )


if __name__ == "__main__":
    main()
