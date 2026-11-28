"""R.2 — hybrid retrieval over the news corpus: sparse + dense, fused with RRF.

Why hybrid rather than either leg alone. The two retrievers fail in complementary
ways, and this corpus can demonstrate both halves rather than assert them:

  sparse (MongoDB `news_text_idx`, weights title:10 content:1) matches terms. It is
    strong on tickers, product names and people -- tokens with no useful embedding
    neighbourhood -- and blind to paraphrase: "memory glut pressures chip makers"
    shares no term with "Micron DRAM oversupply".
  dense (Qdrant, nomic-embed-text 768d over 729,101 deduplicated articles) matches
    meaning, and does the reverse: it finds the paraphrase and dilutes rare literals,
    because a ticker like LRCX sits nowhere useful in embedding space.

Reciprocal Rank Fusion combines them on rank rather than score, which matters here:
the two scores are not comparable -- Mongo's textScore is an unbounded tf-idf-ish
sum, cosine similarity is bounded [-1,1] -- so any weighted-sum fusion would need a
normalisation that is itself a tuned parameter. RRF needs none.

    score(d) = sum over lists of 1 / (RRF_K + rank(d))

One correctness detail that is easy to miss: **the sparse leg must be deduplicated to
match the dense one.** Qdrant holds 729,101 points because M.12's syndication dedupe
was applied at index time; Mongo still holds all 851,071 rows. Fusing a deduplicated
list against one where a single story occupies 402 candidate slots would not compare
two retrievers, it would compare one retriever against a broken one -- and the sparse
leg would look far worse than it is.

Usage:
    python tools/hybrid_search.py "memory chip oversupply"
    python tools/hybrid_search.py "export controls" --symbol NVDA --since 20220101
    python tools/hybrid_search.py "AI capex guidance" --explain
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import requests
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

LM = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@localhost:37018/?authSource=admin")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:26333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "news_v1")
DB_NAME = "quant_data"
SRC_COLL = "news_articles_company_matched_v2"

QUERY_PREFIX = "search_query: "   # pairs with search_document: at index time

RRF_K = int(os.getenv("RRF_K", "60"))
CANDIDATES = int(os.getenv("RRF_CANDIDATES", "100"))   # per leg, before fusion
W_DENSE = float(os.getenv("RRF_W_DENSE", "1.0"))
W_SPARSE = float(os.getenv("RRF_W_SPARSE", "1.0"))

_QDRANT_CLIENT = None
_MONGO_COLL = None
_STUB_RE = re.compile(
    r"^\s*[\w\s.,&'/-]{2,50}\(\s*(?:NASDAQ|NYSE|NYSEARCA|AMEX)\s*:\s*[A-Z.\-]{1,8}\s*\)\s*$",
    re.I,
)


def _get_qdrant_client() -> QdrantClient:
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        _QDRANT_CLIENT = QdrantClient(url=QDRANT_URL, timeout=60)
    return _QDRANT_CLIENT


def _get_mongo_coll():
    global _MONGO_COLL
    if _MONGO_COLL is None:
        _MONGO_COLL = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)[DB_NAME][SRC_COLL]
    return _MONGO_COLL


def auto_weights(query: str, w_dense: float | None = None, w_sparse: float | None = None) -> tuple[float, float]:
    """Adjust dense vs sparse leg weights when callers do not pass explicit weights."""
    if w_dense is not None and w_sparse is not None:
        return float(w_dense), float(w_sparse)
    q = (query or "").strip()
    tokens = q.split()
    has_quote = '"' in q
    has_ticker = any(re.fullmatch(r"[A-Z]{2,5}(?:\.[A-Z])?", t.strip(".,?!()")) for t in tokens)
    has_digits = any(any(ch.isdigit() for ch in t) for t in tokens)
    if has_quote or (len(tokens) <= 2 and (has_ticker or has_digits)):
        wd, ws = 0.85, 1.35
    elif len(tokens) >= 4:
        wd, ws = 1.25, 0.85
    else:
        wd, ws = W_DENSE, W_SPARSE
    return (
        float(w_dense) if w_dense is not None else wd,
        float(w_sparse) if w_sparse is not None else ws,
    )


def _is_stub_title(title: str | None) -> bool:
    t = (title or "").strip()
    if not t:
        return False
    return len(t) < 16 or bool(_STUB_RE.match(t))


@dataclass
class Hit:
    key: str
    symbol: str | None
    date: str | None
    title: str | None
    url: str | None
    mongo_id: str | None = None   # primary key, so bodies are fetched by _id not by title
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf: float = 0.0
    legs: list[str] = field(default_factory=list)


def _norm_date(v) -> str:
    """`date` carries both YYYYMMDD and YYYYMMDDHHMMSS (33.2% is the long form)."""
    s = str(v or "").strip()
    return s[:8] if len(s) >= 8 and s[:8].isdigit() else ""


def _key(symbol, title, date) -> str:
    """Same identity M.12 uses, so the two legs agree on what one story is."""
    return f"{symbol}\x00{(title or '').strip()}\x00{_norm_date(date)}"


def embed_query(text: str) -> list[float]:
    r = requests.post(
        f"{LM}/embeddings",
        json={"model": EMBED_MODEL, "input": [QUERY_PREFIX + text]},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def before_ok(raw_date, before: str | None) -> bool:
    """True when an article dated `raw_date` was published before `before` (YYYYMMDDHHMMSS, UTC).

    N.6 (2026-11-14): replays step through a day hour by hour, so a day-level `until`
    would hand the judge articles from later the same day. A date-only `date` (YYYYMMDD,
    ~30% of recent rows) on the cutoff day has no time, so it is treated as possibly
    after the cutoff and left out: missing one article is safer than seeing the future.
    """
    if not before:
        return True
    s = str(raw_date or "").strip()
    if len(s) >= 14 and s[:14].isdigit():
        return s[:14] < before
    return bool(s[:8]) and s[:8] < before[:8]


def _until_for(until, before):
    """The day-level bound both legs filter on: the earlier of `until` and the cutoff's day."""
    day = int(before[:8]) if before else None
    return day if until is None else (until if day is None else min(until, day))


def dense_leg(qc, query, symbol, since, until, limit, before: str | None = None,
              theme: str | None = None, impacted_symbol: str | None = None) -> list[Hit]:
    until = _until_for(until, before)
    must = []
    if symbol:
        must.append(qm.FieldCondition(key="symbol", match=qm.MatchValue(value=symbol)))
    if since or until:
        must.append(qm.FieldCondition(key="date_int", range=qm.Range(gte=since, lte=until)))
    if theme:
        must.append(qm.FieldCondition(key="themes", match=qm.MatchValue(value=theme)))
    if impacted_symbol:
        must.append(qm.FieldCondition(key="impacted_symbols", match=qm.MatchValue(value=impacted_symbol.upper())))
    pts = qc.query_points(
        collection_name=COLLECTION,
        query=embed_query(query),
        # over-fetch when a cutoff can drop part of its own day, then keep the best `limit` before it
        limit=limit * 2 if before else limit,
        query_filter=qm.Filter(must=must) if must else None,
        with_payload=True,
    ).points
    if before:
        pts = [p for p in pts if before_ok((p.payload or {}).get("date"), before)][:limit]
    out = []
    for rank, p in enumerate(pts, 1):
        pl = p.payload or {}
        out.append(Hit(
            key=_key(pl.get("symbol"), pl.get("title"), pl.get("date")),
            symbol=pl.get("symbol"), date=_norm_date(pl.get("date")),
            title=pl.get("title"), url=pl.get("url"),
            mongo_id=pl.get("mongo_id"),
            dense_rank=rank, dense_score=float(p.score),
        ))
    return out


def sparse_leg(coll, query, symbol, since, until, limit, before: str | None = None,
               theme: str | None = None, impacted_symbol: str | None = None) -> list[Hit]:
    until = _until_for(until, before)
    q: dict = {"$text": {"$search": query}}
    if symbol:
        q["symbol"] = symbol
    and_clauses = []
    if theme:
        and_clauses.append({"$or": [{"llm_themes": theme}, {"student_gd1.themes": theme}]})
    if impacted_symbol:
        sym_u = impacted_symbol.upper()
        and_clauses.append({"$or": [
            {f"student_gd1.impacts.{sym_u}": {"$in": [-2, -1, 1, 2]}},
            {f"llm_impacts.{sym_u}": {"$in": [-2, -1, 1, 2]}},
        ]})
    if and_clauses:
        q["$and"] = and_clauses

    # The date window goes into the query, not into a Python loop afterwards.
    #
    # Filtering after the fact made the two legs asymmetric: qdrant filters inside the
    # search, so it returns the best `limit` results *within* the window, while a
    # post-hoc filter returns whichever of the best results across all history happen
    # to land in it. On one NVDA query that was 37 hits instead of 100, and the 37 were
    # the wrong 37. That corrupts R.4's numbers invisibly -- the sparse leg looks worse
    # than it is, and hybrid inherits the damage.
    #
    # `date` is a string in two formats, YYYYMMDD and YYYYMMDDHHMMSS. Lexicographic
    # order matches chronological order for both, and "20221001120000" sorts after
    # "20221001" and before "20230701", so a plain string range is correct for the
    # mixed field and can still use the (symbol, date) index.
    if since is not None or until is not None:
        rng: dict = {}
        if since is not None:
            rng["$gte"] = str(since)
        if until is not None:
            rng["$lt"] = str(until + 1)   # +1 day-number keeps the whole `until` day
        if before:
            # the exact cutoff: "20261112093000" < "20261112100000" in string order too; a date-only
            # "20261112" also sorts below it, so the cutoff day's date-only rows are excluded by name
            rng["$lt"] = min(rng["$lt"], before)
            rng["$ne"] = before[:8]
        q["date"] = rng

    # Over-fetch, then deduplicate to the identity the dense index already has: without
    # it a syndicated story consumes its full copy count of candidate slots and the
    # sparse leg is crippled by a data problem rather than a retrieval one. 3x covers
    # the measured 16% duplicate rate with room to spare.
    cursor = (
        coll.find(q, {"symbol": 1, "title": 1, "date": 1, "url": 1, "_id": 1,
                      "score": {"$meta": "textScore"}})
        .sort([("score", {"$meta": "textScore"})])
        .limit(limit * 3)
    )

    seen: set[str] = set()
    out: list[Hit] = []
    for doc in cursor:
        d = _norm_date(doc.get("date"))
        k = _key(doc.get("symbol"), doc.get("title"), doc.get("date"))
        if k in seen:
            continue
        seen.add(k)
        out.append(Hit(
            key=k, symbol=doc.get("symbol"), date=d,
            title=doc.get("title"), url=doc.get("url"),
            mongo_id=str(doc["_id"]),
            sparse_rank=len(out) + 1, sparse_score=float(doc.get("score", 0.0)),
        ))
        if len(out) >= limit:
            break
    return out


def rrf_fuse(dense: list[Hit], sparse: list[Hit], k: int,
             rrf_k: int = RRF_K, w_dense: float = W_DENSE,
             w_sparse: float = W_SPARSE) -> list[Hit]:
    """Fuse on rank. Each leg contributes at most once per document.

    That "at most once" is not defensive padding. A leg can return the same story
    twice under a key this function considers identical — `index_news.py` dedupes on
    the raw `date`, which carries both YYYYMMDD and YYYYMMDDHHMMSS forms, while the
    key here normalises to YYYYMMDD. Points that were distinct at index time collapse
    here. Summing every occurrence let one story bank 1/(K+r) several times and
    outrank genuinely better results; keeping only its best rank per leg is both the
    correct reading of RRF and immune to whatever the upstream key happens to be.
    """
    merged: dict[str, Hit] = {}
    for leg_name, hits, weight in (("dense", dense, w_dense),
                                   ("sparse", sparse, w_sparse)):
        best_rank: dict[str, int] = {}
        for h in hits:
            rank = h.dense_rank if leg_name == "dense" else h.sparse_rank
            if rank is None:
                continue
            if h.key in best_rank and best_rank[h.key] <= rank:
                continue
            best_rank[h.key] = rank

            cur = merged.get(h.key)
            if cur is None:
                cur = merged[h.key] = Hit(key=h.key, symbol=h.symbol, date=h.date,
                                          title=h.title, url=h.url,
                                          mongo_id=h.mongo_id)
            elif cur.mongo_id is None:
                cur.mongo_id = h.mongo_id
            if leg_name == "dense":
                cur.dense_rank, cur.dense_score = h.dense_rank, h.dense_score
            else:
                cur.sparse_rank, cur.sparse_score = h.sparse_rank, h.sparse_score
            if leg_name not in cur.legs:
                cur.legs.append(leg_name)

        for key, rank in best_rank.items():
            merged[key].rrf += weight / (rrf_k + rank)

    for hit in merged.values():
        if _is_stub_title(hit.title):
            hit.rrf *= 0.5

    ranked = sorted(merged.values(), key=lambda h: h.rrf, reverse=True)
    return ranked[:k]


def search_with_meta(query, k=10, symbol=None, since=None, until=None, candidates=CANDIDATES,
                     rrf_k=RRF_K, w_dense: float | None = None, w_sparse: float | None = None,
                     before: str | None = None, theme: str | None = None,
                     impacted_symbol: str | None = None, qc=None, coll=None) -> tuple[list[Hit], list[Hit], list[Hit], dict]:
    """Run dense and sparse legs in parallel with auto-weights and single-leg degradation."""
    qc = qc if qc is not None else _get_qdrant_client()
    coll = coll if coll is not None else _get_mongo_coll()
    wd, ws = auto_weights(query, w_dense=w_dense, w_sparse=w_sparse)
    degraded: dict[str, str] = {}
    dense: list[Hit] = []
    sparse: list[Hit] = []
    dense_err: Exception | None = None
    sparse_err: Exception | None = None

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_dense = pool.submit(dense_leg, qc, query, symbol, since, until, candidates, before,
                              theme=theme, impacted_symbol=impacted_symbol)
        f_sparse = pool.submit(sparse_leg, coll, query, symbol, since, until, candidates, before,
                               theme=theme, impacted_symbol=impacted_symbol)
        try:
            dense = f_dense.result()
        except Exception as e:  # noqa: BLE001
            dense_err = e
            degraded["dense_error"] = str(e)[:200]
        try:
            sparse = f_sparse.result()
        except Exception as e:  # noqa: BLE001
            sparse_err = e
            degraded["sparse_error"] = str(e)[:200]

    if dense_err is not None and sparse_err is not None:
        raise dense_err
    fused = rrf_fuse(dense, sparse, k, rrf_k, wd, ws)
    return dense, sparse, fused, {"weights": {"dense": wd, "sparse": ws}, "degraded": degraded}


def search(query, k=10, symbol=None, since=None, until=None, candidates=CANDIDATES,
           rrf_k=RRF_K, w_dense=None, w_sparse=None, before: str | None = None,
           theme: str | None = None, impacted_symbol: str | None = None):
    """`before` (YYYYMMDDHHMMSS, UTC) keeps only articles published before that moment, in both legs."""
    dense, sparse, fused, _ = search_with_meta(
        query, k=k, symbol=symbol, since=since, until=until, candidates=candidates,
        rrf_k=rrf_k, w_dense=w_dense, w_sparse=w_sparse, before=before,
        theme=theme, impacted_symbol=impacted_symbol,
    )
    return dense, sparse, fused


def _show(label: str, hits: list[Hit], explain: bool) -> None:
    print(f"\n--- {label} ---")
    for i, h in enumerate(hits, 1):
        tag = ""
        if explain:
            d = f"d{h.dense_rank}" if h.dense_rank else "d–"
            s = f"s{h.sparse_rank}" if h.sparse_rank else "s–"
            tag = f"[{d:>5} {s:>5}] "
        print(f" {i:>2}. {tag}{str(h.symbol):<6} {h.date}  {str(h.title)[:72]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--symbol")
    ap.add_argument("--since", type=int)
    ap.add_argument("--until", type=int)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--explain", action="store_true",
                    help="show each result's rank in each leg")
    ap.add_argument("--legs", action="store_true", help="also print each leg alone")
    ap.add_argument("--rrf-k", type=int, default=RRF_K)
    ap.add_argument("--candidates", type=int, default=CANDIDATES)
    ap.add_argument("--w-dense", type=float, default=W_DENSE)
    ap.add_argument("--w-sparse", type=float, default=W_SPARSE)
    ap.add_argument("--theme")
    ap.add_argument("--impacted-symbol")
    args = ap.parse_args()

    dense, sparse, fused = search(args.query, args.k, args.symbol, args.since,
                                  args.until, args.candidates, args.rrf_k,
                                  args.w_dense, args.w_sparse,
                                  theme=args.theme, impacted_symbol=args.impacted_symbol)
    print(f'query: "{args.query}"'
          + (f"  symbol={args.symbol}" if args.symbol else "")
          + (f"  since={args.since}" if args.since else ""))

    if args.legs:
        _show("dense only", dense[:args.k], False)
        _show("sparse only", sparse[:args.k], False)
    _show(f"hybrid (RRF k={RRF_K})", fused, args.explain)

    both = sum(1 for h in fused if len(h.legs) == 2)
    print(f"\n{both}/{len(fused)} of the fused top-{args.k} were found by both legs; "
          f"{len(fused) - both} by only one — the one-leg hits are what fusion adds.")


if __name__ == "__main__":
    main()
