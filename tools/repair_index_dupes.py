"""R.8 — repair the index in place instead of re-embedding it.

`index_news.py` originally keyed dedupe on the raw `date`, which carries both
YYYYMMDD and YYYYMMDDHHMMSS. The same story filed under both forms produced two
keys and two points: 13,027 of them, 1.79% of the index.

The point worth making is the repair strategy, not the bug. Point ids are UUID5 of
the mongo `_id`, so re-running the indexer would not remove the extras -- nothing
deletes a point that simply stops being generated -- and `--reset` means 2.65 hours
of re-embedding to fix 1.79% of the collection. Deleting the redundant ids directly
takes seconds and re-embeds nothing, because the vectors that remain are still
correct; only the *set* of points was wrong.

Keeps the same survivor the indexer would: lowest `_id` within each normalised key,
matching its `keep="first"` over an `_id`-ascending scan.

    python tools/repair_index_dupes.py --dry-run
    python tools/repair_index_dupes.py
"""

from __future__ import annotations

import argparse
import os
import uuid

from pymongo import ASCENDING, MongoClient
from qdrant_client import QdrantClient

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@localhost:37018/?authSource=admin")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:26333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "news_v1")
DB_NAME = "quant_data"
SRC_COLL = "news_articles_company_matched_v2"

NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")  # must match index_news.py
BATCH = 1000


def to_date8(v) -> str:
    s = str(v or "").strip()
    return s[:8] if len(s) >= 8 and s[:8].isdigit() else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    coll = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)[DB_NAME][SRC_COLL]
    qc = QdrantClient(url=QDRANT_URL, timeout=120)
    before = qc.get_collection(COLLECTION).points_count
    print(f"index holds {before:,} points")

    # Replay the indexer's own scan order so "which copy survived" is reproduced
    # exactly, then find the ids that were indexed under the raw-date key but are
    # redundant under the normalised one.
    raw_seen: set[tuple] = set()
    norm_seen: set[tuple] = set()
    redundant: list[str] = []
    scanned = 0

    cursor = coll.find({}, {"symbol": 1, "title": 1, "date": 1}).sort("_id", ASCENDING)
    for doc in cursor:
        scanned += 1
        sym, title = doc.get("symbol", ""), doc.get("title", "")
        raw_key = (sym, title, str(doc.get("date") or ""))
        if raw_key in raw_seen:
            continue          # never indexed in the first place
        raw_seen.add(raw_key)

        norm_key = (sym, title, to_date8(doc.get("date")))
        if norm_key in norm_seen:
            # Indexed under the old key, redundant under the new one.
            redundant.append(str(uuid.uuid5(NAMESPACE, str(doc["_id"]))))
        else:
            norm_seen.add(norm_key)

    print(f"scanned {scanned:,} rows -> {len(raw_seen):,} indexed, "
          f"{len(norm_seen):,} distinct stories, {len(redundant):,} redundant points "
          f"({100 * len(redundant) / max(1, len(raw_seen)):.2f}%)")

    if args.dry_run:
        print("dry run — nothing deleted")
        return
    if not redundant:
        print("nothing to do")
        return

    for i in range(0, len(redundant), BATCH):
        qc.delete(collection_name=COLLECTION,
                  points_selector=redundant[i:i + BATCH], wait=True)
        print(f"  deleted {min(i + BATCH, len(redundant)):,}/{len(redundant):,}", flush=True)

    after = qc.get_collection(COLLECTION).points_count
    print(f"\n{before:,} -> {after:,} points (removed {before - after:,}); "
          f"expected {len(norm_seen):,}")


if __name__ == "__main__":
    main()
