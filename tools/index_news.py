"""R.1d — embed the news corpus into Qdrant.

Design follows the R.1b/R.1c measurements in quant_docker/PROJECT_PLAN.md rather than
the assumptions the plan originally carried:

  * one vector per article from `title + first 1000 chars`, not 512-token chunks.
    Only 30.2% of articles fit a single 512-token window, so full chunking is 2.96M
    vectors / 9.1 GB / 20.0 h against 0.85M / 2.6 GB / 3.19 h here. Whether the extra
    19 hours buy recall is R.4's question, and R.4 cannot ask it until something is
    indexed.
  * title-only articles are indexed, not skipped: 17.90% of the corpus has no body at
    all, and dropping them would silently lose one document in six.
  * deduplicated on (symbol, title, date): 14.3% of the corpus is duplicate copies, and
    a retriever that returns the same article three times has spent three of its ten
    slots saying one thing. It would also inflate every recall number R.4 produces.

Idempotent and resumable. Point ids are UUID5 of the mongo `_id`, so a re-run upserts
in place instead of duplicating, and a checkpoint file records the last `_id` processed
so an interrupted 3-hour pass resumes rather than restarts.

Usage:
    python tools/index_news.py --limit 2000      # smoke test
    python tools/index_news.py                   # full corpus
    python tools/index_news.py --reset           # drop the collection first
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests
from pymongo import ASCENDING, MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

LM = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@localhost:37018/?authSource=admin")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:26333")

DB_NAME = "quant_data"
SRC_COLL = "news_articles_company_matched_v2"
COLLECTION = os.getenv("QDRANT_COLLECTION", "news_v1")

VECTOR_DIM = 768
LEDE_CHARS = 1000
EMBED_BATCH = 32          # throughput is character-bound; 32 only balances latency
MONGO_BATCH = 500
CHECKPOINT = Path(__file__).with_name(".index_news_checkpoint.json")

# nomic-embed-text is asymmetric. Omitting these prefixes fails silently: vectors still
# return, dimensions still match, nothing errors, retrieval is just quietly worse.
DOC_PREFIX = "search_document: "
NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def dedupe_key(doc) -> str:
    raw = f"{doc.get('symbol','')}\x00{doc.get('title','')}\x00{doc.get('date','')}"
    return hashlib.blake2b(raw.encode("utf-8", "replace"), digest_size=16).hexdigest()


def build_text(doc) -> str | None:
    title = (doc.get("title") or "").strip()
    body = (doc.get("content") or "").strip()
    if not title and not body:
        return None
    if not body:
        return DOC_PREFIX + title
    # The lede slice also caps the scrape-pollution outliers -- the longest document in
    # the corpus is 4.9M characters of whole-page dump.
    return f"{DOC_PREFIX}{title}\n\n{body[:LEDE_CHARS]}"


def to_int_date(v) -> int | None:
    """Normalise `date` to a YYYYMMDD int so range filters work.

    The field carries two formats: 'YYYYMMDD' and 'YYYYMMDDHHMMSS'. Measured on a
    40K sample, **33.2% are the 14-character form** -- an 8-only parse would leave a
    third of the corpus with a null date and therefore invisible to every date filter,
    silently. Retrieval over a dated corpus without a working date filter is the M.7
    look-ahead mistake wearing a different costume, so this is the guard, not a detail.
    """
    s = str(v or "").strip()
    return int(s[:8]) if len(s) >= 8 and s[:8].isdigit() else None


def embed(texts: list[str]) -> list[list[float]]:
    r = requests.post(
        f"{LM}/embeddings", json={"model": EMBED_MODEL, "input": texts}, timeout=600
    )
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


def ensure_collection(qc: QdrantClient, reset: bool) -> None:
    exists = qc.collection_exists(COLLECTION)
    if exists and reset:
        print(f"dropping existing collection {COLLECTION}")
        qc.delete_collection(COLLECTION)
        exists = False
    if not exists:
        qc.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(size=VECTOR_DIM, distance=qm.Distance.COSINE),
        )
        # Payload indexes, not just payload: an unindexed filter on 851K points is a
        # full scan, and every useful query here filters by symbol or date.
        qc.create_payload_index(COLLECTION, "symbol", qm.PayloadSchemaType.KEYWORD)
        qc.create_payload_index(COLLECTION, "date_int", qm.PayloadSchemaType.INTEGER)
        qc.create_payload_index(COLLECTION, "event_type", qm.PayloadSchemaType.KEYWORD)
        print(f"created collection {COLLECTION} (dim={VECTOR_DIM}, cosine)")


def load_checkpoint(reset: bool):
    if reset or not CHECKPOINT.exists():
        return None, set()
    try:
        data = json.loads(CHECKPOINT.read_text())
        return data.get("last_id"), set(data.get("seen", []))
    except (json.JSONDecodeError, OSError) as e:
        print(f"checkpoint unreadable ({e}); starting from the beginning")
        return None, set()


def save_checkpoint(last_id, seen: set) -> None:
    tmp = CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"last_id": str(last_id), "seen": sorted(seen)}))
    tmp.replace(CHECKPOINT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N source documents")
    ap.add_argument("--reset", action="store_true", help="drop the collection and checkpoint")
    args = ap.parse_args()

    try:
        requests.get(f"{LM}/models", timeout=8).raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"embedding endpoint {LM} unreachable: {e}")

    qc = QdrantClient(url=QDRANT_URL, timeout=120)
    ensure_collection(qc, args.reset)

    coll = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)[DB_NAME][SRC_COLL]
    total = coll.estimated_document_count()

    last_id, seen = load_checkpoint(args.reset)
    query = {"_id": {"$gt": __import__("bson").ObjectId(last_id)}} if last_id else {}
    if last_id:
        print(f"resuming after _id={last_id} ({len(seen)} dedupe keys carried over)")

    cursor = coll.find(
        query,
        {"title": 1, "content": 1, "symbol": 1, "date": 1,
         "llm_event_type": 1, "llm_sentiment_final": 1, "url": 1},
    ).sort("_id", ASCENDING).batch_size(MONGO_BATCH)

    n_read = n_dupe = n_empty = n_indexed = 0
    pending: list[tuple] = []
    t0 = time.perf_counter()

    def flush():
        nonlocal pending, n_indexed
        if not pending:
            return
        vectors = embed([p[1] for p in pending])
        qc.upsert(
            collection_name=COLLECTION,
            points=[
                qm.PointStruct(
                    id=str(uuid.uuid5(NAMESPACE, str(doc["_id"]))),
                    vector=vec,
                    payload={
                        "mongo_id": str(doc["_id"]),
                        "symbol": doc.get("symbol"),
                        "date": doc.get("date"),
                        "date_int": to_int_date(doc.get("date")),
                        "title": doc.get("title"),
                        "url": doc.get("url"),
                        "event_type": doc.get("llm_event_type"),
                        "sentiment": doc.get("llm_sentiment_final"),
                    },
                )
                for (doc, _text), vec in zip(pending, vectors)
            ],
            wait=False,
        )
        n_indexed += len(pending)
        pending = []

    try:
        for doc in cursor:
            n_read += 1
            key = dedupe_key(doc)
            if key in seen:
                n_dupe += 1
            else:
                text = build_text(doc)
                if text is None:
                    n_empty += 1
                else:
                    seen.add(key)
                    pending.append((doc, text))
                    if len(pending) >= EMBED_BATCH:
                        flush()

            if n_read % 5000 == 0:
                el = time.perf_counter() - t0
                rate = n_indexed / el if el else 0
                remain = (total - n_read) / rate / 3600 if rate else 0
                print(
                    f"read {n_read:>7}/{total}  indexed {n_indexed:>7}  "
                    f"dupe {n_dupe:>6}  empty {n_empty:>5}  "
                    f"{rate:6.1f} vec/s  eta {remain:5.2f}h",
                    flush=True,
                )
                save_checkpoint(doc["_id"], seen)

            if args.limit and n_read >= args.limit:
                break

        flush()
        save_checkpoint(doc["_id"], seen)
    except KeyboardInterrupt:
        flush()
        print("\ninterrupted; checkpoint saved, re-run to resume")

    el = time.perf_counter() - t0
    print(
        f"\ndone: read {n_read}, indexed {n_indexed}, "
        f"skipped {n_dupe} duplicates and {n_empty} empty, in {el / 60:.1f} min "
        f"({n_indexed / el:.1f} vec/s)"
    )
    print("collection:", qc.get_collection(COLLECTION).points_count, "points")


if __name__ == "__main__":
    main()
