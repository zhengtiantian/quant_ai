#!/usr/bin/env python3
"""Export extended universe news for Colab GPU embedding and import vectors back to Qdrant."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from bson import ObjectId
from pymongo import ASCENDING, MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QUANT_AI_ROOT = Path(__file__).resolve().parents[1]
if str(QUANT_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_AI_ROOT))

from tools import index_news as idx  # noqa: E402

TMP_DIR = Path(os.getenv("COLAB_TMP_DIR", "/Volumes/Data4T/tmp"))
EXPORT_JSONL_GZ = TMP_DIR / "universe_news_to_embed.jsonl.gz"
PARITY_REF_JSON = TMP_DIR / "parity_ref.json"
META_JSON = TMP_DIR / "universe_export_meta.json"
VECTORS_NPY = TMP_DIR / "universe_vectors_fp16.npy"


def cmd_export() -> None:
    """Export unindexed extended-universe matched articles and 10 LM Studio parity reference vectors."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    coll = MongoClient(idx.MONGO_URI, serverSelectionTimeoutMS=10000)[idx.DB_NAME][idx.SRC_COLL]
    last_id, seen = idx.load_checkpoint(False, idx.UNIVERSE_CHECKPOINT)

    base_query: dict = {
        "origin": "universe_backfill",
        "company_match_v2.matched": {"$ne": False},
    }
    query = dict(base_query)
    if last_id:
        query["_id"] = {"$gt": ObjectId(last_id)}
        print(f"export resuming after _id={last_id} ({len(seen):,} dedupe keys carried over)")

    total = coll.count_documents(query)
    cursor = coll.find(
        query,
        {
            "title": 1,
            "content": 1,
            "symbol": 1,
            "date": 1,
            "llm_event_type_a": 1,
            "llm_sentiment_final": 1,
            "url": 1,
        },
    ).sort("_id", ASCENDING).batch_size(1000)

    n_read = n_exported = n_dupe = n_empty = 0
    max_id = last_id
    sample_texts: list[str] = []
    t0 = time.perf_counter()

    with gzip.open(EXPORT_JSONL_GZ, "wt", encoding="utf-8") as gz:
        for doc in cursor:
            n_read += 1
            max_id = str(doc["_id"])
            key = idx.dedupe_key(doc)
            if key in seen:
                n_dupe += 1
                continue
            text = idx.build_text(doc)
            if text is None:
                n_empty += 1
                continue
            seen.add(key)
            rec = {
                "point_id": str(uuid.uuid5(idx.NAMESPACE, max_id)),
                "mongo_id": max_id,
                "dedupe_key": key,
                "symbol": doc.get("symbol"),
                "date": doc.get("date"),
                "date_int": idx.to_int_date(doc.get("date")),
                "title": doc.get("title"),
                "url": doc.get("url"),
                "event_type": doc.get("llm_event_type_a"),
                "sentiment": doc.get("llm_sentiment_final"),
                "text": text,
            }
            gz.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_exported += 1
            if len(sample_texts) < 10:
                sample_texts.append(text)
            if n_read % 50000 == 0:
                print(f"export scanned {n_read:,}/{total:,} -> kept {n_exported:,}, dupe {n_dupe:,}")

    if not sample_texts:
        sys.exit("No unindexed extended-universe documents to export.")

    print(f"Generating {len(sample_texts)} LM Studio parity vectors via {idx.LM} ({idx.EMBED_MODEL})...")
    ref_vecs = idx.embed(sample_texts)
    parity_rows = [{"text": t, "embedding": v} for t, v in zip(sample_texts, ref_vecs)]
    PARITY_REF_JSON.write_text(json.dumps(parity_rows), encoding="utf-8")
    META_JSON.write_text(
        json.dumps({"count": n_exported, "max_id": max_id, "scanned": n_read, "dupes": n_dupe, "empty": n_empty}),
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - t0
    size_mb = EXPORT_JSONL_GZ.stat().st_size / (1024 * 1024)
    print(
        f"Export done in {elapsed:.1f}s: exported={n_exported:,}, max_id={max_id}, "
        f"file={EXPORT_JSONL_GZ} ({size_mb:.1f} MB)"
    )


def cmd_import(batch_size: int = 500) -> None:
    """Import Colab FP16 vectors + metadata into Qdrant `news_v1` and advance universe checkpoint."""
    if not EXPORT_JSONL_GZ.exists() or not VECTORS_NPY.exists():
        sys.exit(f"Missing export/vector file(s): {EXPORT_JSONL_GZ}, {VECTORS_NPY}")

    vecs = np.load(VECTORS_NPY, mmap_mode="r")
    if vecs.ndim != 2 or vecs.shape[1] != idx.VECTOR_DIM:
        sys.exit(f"Unexpected vector matrix shape: {vecs.shape}, expected (*, {idx.VECTOR_DIM})")

    with gzip.open(EXPORT_JSONL_GZ, "rt", encoding="utf-8") as gz:
        n_lines = sum(1 for _ in gz)
    if n_lines != vecs.shape[0]:
        sys.exit(f"Row mismatch: jsonl={n_lines:,} vs npy={vecs.shape[0]:,}")

    qc = QdrantClient(url=idx.QDRANT_URL, timeout=120)
    idx.ensure_collection(qc, reset=False, enable_quantization=True)
    _, seen = idx.load_checkpoint(False, idx.UNIVERSE_CHECKPOINT)

    batch_meta: list[dict] = []
    batch_vecs_idx: list[int] = []
    n_done = 0
    max_id = None
    t0 = time.perf_counter()

    def flush(wait_sync: bool = False) -> None:
        nonlocal batch_meta, batch_vecs_idx, n_done
        if not batch_meta:
            return
        sub = np.asarray(vecs[batch_vecs_idx[0]: batch_vecs_idx[-1] + 1], dtype=np.float32)
        if not np.isfinite(sub).all():
            sys.exit(f"Non-finite vector detected around row {batch_vecs_idx[0]}")
        points = [
            qm.PointStruct(
                id=m["point_id"],
                vector=vec.tolist(),
                payload={
                    "mongo_id": m["mongo_id"],
                    "symbol": m["symbol"],
                    "date": m["date"],
                    "date_int": m["date_int"],
                    "title": m["title"],
                    "url": m["url"],
                    "event_type": m["event_type"],
                    "sentiment": m["sentiment"],
                },
            )
            for m, vec in zip(batch_meta, sub)
        ]
        qc.upsert(collection_name=idx.COLLECTION, points=points, wait=wait_sync)
        n_done += len(points)
        batch_meta = []
        batch_vecs_idx = []

    with gzip.open(EXPORT_JSONL_GZ, "rt", encoding="utf-8") as gz:
        for i, line in enumerate(gz):
            m = json.loads(line)
            seen.add(m["dedupe_key"])
            max_id = m["mongo_id"]
            batch_meta.append(m)
            batch_vecs_idx.append(i)
            if len(batch_meta) >= batch_size:
                flush(wait_sync=(i + 1 >= n_lines))
                if n_done % 25000 < batch_size:
                    rate = n_done / max(time.perf_counter() - t0, 1e-3)
                    print(f"imported {n_done:,}/{n_lines:,} ({rate:,.0f} pts/s)", flush=True)

    flush(wait_sync=True)
    if max_id:
        idx.save_checkpoint(max_id, seen, idx.UNIVERSE_CHECKPOINT)

    info = qc.get_collection(idx.COLLECTION)
    print(
        f"Import complete: {n_done:,} vectors upserted in {time.perf_counter() - t0:.1f}s; "
        f"collection={idx.COLLECTION} points={info.points_count:,} status={info.status}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Colab GPU embedding export/import helper for Qdrant.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    imp = sub.add_parser("import")
    imp.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()
    if args.cmd == "export":
        cmd_export()
    else:
        cmd_import(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
