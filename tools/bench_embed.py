"""R.1c — measure embedding throughput before committing to a full corpus pass.

The point of running this on 1000 real documents rather than estimating: the answer
decides whether the initial index covers all 851K articles or a date-bounded subset,
and whether documents are indexed whole or chunked. Both are cheap to get wrong at
plan time and expensive to get wrong after a 10-hour job.

Two document strategies are timed, because R.1b showed only 30.2% of articles fit in
a single 512-token window:

  title_lede  — one vector per article from title + first 1000 chars
  chunked     — 512-token windows over the full body, ~3.62 vectors/article

Read-only against mongo. Writes nothing to qdrant; this is measurement only.
"""

import os
import statistics
import sys
import time

import requests
from pymongo import MongoClient

LM = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1").rstrip("/")
MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
MONGO = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@localhost:37018/?authSource=admin")
DB = "quant_data"
COLL = "news_articles_company_matched_v2"

N_DOCS = int(os.getenv("N_DOCS", "1000"))
LEDE_CHARS = 1000
CHUNK_CHARS = 512 * 4  # ~4 chars/token
CHUNK_OVERLAP = 200

# nomic-embed-text is an asymmetric model: documents and queries take different
# prefixes, and omitting them measurably degrades retrieval. Getting this wrong is
# invisible -- the vectors still come back, they are just in the wrong space.
DOC_PREFIX = "search_document: "


def fetch_docs(n):
    cli = MongoClient(MONGO, serverSelectionTimeoutMS=8000)
    coll = cli[DB][COLL]
    cur = coll.aggregate(
        [
            {"$sample": {"size": n}},
            {"$project": {"title": 1, "content": 1, "symbol": 1, "date": 1}},
        ],
        allowDiskUse=True,
    )
    return list(cur)


def build_title_lede(docs):
    out = []
    for d in docs:
        title = (d.get("title") or "").strip()
        body = (d.get("content") or "").strip()
        # 17.9% of the corpus has no body at all, so title-only must be a first-class
        # case rather than a skip -- dropping them would silently lose 1 doc in 6.
        text = title if not body else f"{title}\n\n{body[:LEDE_CHARS]}"
        if text:
            out.append(DOC_PREFIX + text)
    return out


def build_chunked(docs):
    out = []
    for d in docs:
        title = (d.get("title") or "").strip()
        body = (d.get("content") or "").strip()
        if not body:
            if title:
                out.append(DOC_PREFIX + title)
            continue
        step = CHUNK_CHARS - CHUNK_OVERLAP
        for i in range(0, len(body), step):
            chunk = body[i : i + CHUNK_CHARS]
            if len(chunk) < 100 and i > 0:
                break
            # The title rides on every chunk: a chunk from paragraph 9 otherwise has
            # no way to say which company it is about.
            out.append(f"{DOC_PREFIX}{title}\n\n{chunk}")
    return out


def embed_batch(texts):
    r = requests.post(
        f"{LM}/embeddings",
        json={"model": MODEL, "input": texts},
        timeout=600,
    )
    r.raise_for_status()
    data = r.json()["data"]
    return len(data), len(data[0]["embedding"])


def run(label, texts, batch_size):
    lat = []
    n_vec = 0
    dim = None
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        b0 = time.perf_counter()
        got, dim = embed_batch(batch)
        lat.append(time.perf_counter() - b0)
        n_vec += got
    total = time.perf_counter() - t0
    chars = sum(len(t) for t in texts)
    print(
        f"  batch={batch_size:<4} vectors={n_vec:<6} dim={dim} "
        f"wall={total:7.2f}s  {n_vec / total:8.1f} vec/s  "
        f"{chars / total / 1000:7.1f} Kchar/s  "
        f"batch p50={statistics.median(lat):.3f}s max={max(lat):.3f}s"
    )
    return n_vec / total


def main():
    try:
        requests.get(f"{LM}/models", timeout=8).raise_for_status()
    except Exception as e:
        sys.exit(f"embedding endpoint {LM} unreachable: {e}")

    print(f"model={MODEL}  endpoint={LM}")
    print(f"fetching {N_DOCS} random articles ...")
    docs = fetch_docs(N_DOCS)
    print(f"fetched {len(docs)}")

    strategies = {
        "title_lede": build_title_lede(docs),
        "chunked": build_chunked(docs),
    }

    best = {}
    for label, texts in strategies.items():
        print(f"\n{label}: {len(texts)} vectors from {len(docs)} articles "
              f"({len(texts) / len(docs):.2f} vec/article), "
              f"{sum(len(t) for t in texts) / len(texts):.0f} chars/vector avg")
        rates = []
        for bs in (8, 32, 64):
            try:
                rates.append((run(label, texts, bs), bs))
            except Exception as e:
                print(f"  batch={bs:<4} FAILED: {type(e).__name__}: {str(e)[:120]}")
        if rates:
            best[label] = max(rates)

    print("\n--- extrapolation to the full corpus (851,071 articles) ---")
    for label, (rate, bs) in best.items():
        per_article = len(strategies[label]) / len(docs)
        total_vec = 851071 * per_article
        hours = total_vec / rate / 3600
        gb = total_vec * 768 * 4 / 1e9
        print(
            f"  {label:<11} best batch={bs:<3} {rate:7.1f} vec/s -> "
            f"{total_vec / 1e6:5.2f}M vectors, {hours:6.2f}h to embed, "
            f"{gb:5.1f} GB raw float32"
        )


if __name__ == "__main__":
    main()
