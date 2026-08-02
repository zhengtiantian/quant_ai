"""Repair the `event_type` payload in the news index — without re-embedding.

The indexer read `llm_event_type`, which is populated on **50 documents**. The field
carrying the labels is `llm_event_type_a`, populated on 845,597. So every point went in
with a null `event_type`, and the payload index built over it indexed nothing.

The repair is a payload update, not a re-index: vectors are unaffected by a metadata
mistake, so re-embedding 716,074 documents to fix a string field would be 2.65 hours
spent recomputing something that was always correct. Same reasoning as
`repair_index_dupes.py`.

Batched by value rather than by point: there are 7 distinct event types, so one
`set_payload` per (value, chunk) beats 716,074 individual updates.

    python tools/backfill_event_type.py --dry-run
    python tools/backfill_event_type.py
"""

from __future__ import annotations

import argparse
import os
import uuid
from collections import defaultdict

from pymongo import ASCENDING, MongoClient
from qdrant_client import QdrantClient

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@localhost:37018/?authSource=admin")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:26333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "news_v1")
DB_NAME = "quant_data"
SRC_COLL = "news_articles_company_matched_v2"

NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")
CHUNK = 5000


def to_date8(v) -> str:
    s = str(v or "").strip()
    return s[:8] if len(s) >= 8 and s[:8].isdigit() else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    coll = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)[DB_NAME][SRC_COLL]
    qc = QdrantClient(url=QDRANT_URL, timeout=180)
    print(f"index holds {qc.get_collection(COLLECTION).points_count:,} points")

    # Replay the indexer's dedupe so only points that actually exist are updated.
    seen: set[tuple] = set()
    by_value: dict[str, list[str]] = defaultdict(list)
    scanned = missing = 0

    cursor = coll.find(
        {}, {"symbol": 1, "title": 1, "date": 1, "llm_event_type_a": 1}
    ).sort("_id", ASCENDING)
    for doc in cursor:
        scanned += 1
        key = (doc.get("symbol", ""), doc.get("title", ""), to_date8(doc.get("date")))
        if key in seen:
            continue
        seen.add(key)
        val = doc.get("llm_event_type_a")
        if not val:
            missing += 1
            continue
        by_value[val].append(str(uuid.uuid5(NAMESPACE, str(doc["_id"]))))

    total = sum(len(v) for v in by_value.values())
    print(f"scanned {scanned:,} rows -> {len(seen):,} indexed points, "
          f"{total:,} with an event_type, {missing:,} without")
    for val, ids in sorted(by_value.items(), key=lambda kv: -len(kv[1])):
        print(f"  {val:<14} {len(ids):>8,}")

    if args.dry_run:
        print("dry run — nothing written")
        return

    done = 0
    for val, ids in by_value.items():
        for i in range(0, len(ids), CHUNK):
            qc.set_payload(
                collection_name=COLLECTION,
                payload={"event_type": val},
                points=ids[i:i + CHUNK],
                wait=False,
            )
            done += len(ids[i:i + CHUNK])
        print(f"  {val:<14} done ({done:,}/{total:,})", flush=True)

    # Verify against the index rather than trusting the write path.
    from qdrant_client.http import models as qm
    for val in by_value:
        n = qc.count(COLLECTION, count_filter=qm.Filter(must=[
            qm.FieldCondition(key="event_type", match=qm.MatchValue(value=val))
        ]), exact=True).count
        print(f"  verify {val:<14} indexed={n:>8,}  expected={len(by_value[val]):>8,}"
              f"  {'OK' if n == len(by_value[val]) else 'MISMATCH'}")


if __name__ == "__main__":
    main()
