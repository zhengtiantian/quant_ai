"""S.1 — measure the injection scanner's false-positive rate on real articles.

A detector that fires on ordinary financial journalism is worse than no detector: the
flag gets ignored, and then it is ignored on the day it matters. So the rate has to be
measured on the corpus the scanner will actually run against, before it is wired in.

Prints, per pattern, how often it fires and a sample of what it fired on, so a rule with
a bad hit rate can be tightened or dropped rather than left to erode trust in the flag.

    python tools/measure_injection_fp.py --sample 40000
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from injection_guard import _PATTERNS, sanitise  # noqa: E402

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@localhost:37018/?authSource=admin")
DB_NAME = "quant_data"
COLL = "news_articles_company_matched_v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40000)
    ap.add_argument("--examples", type=int, default=2)
    args = ap.parse_args()

    coll = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)[DB_NAME][COLL]
    cursor = coll.aggregate(
        [{"$sample": {"size": args.sample}},
         {"$project": {"_id": 0, "title": 1, "content": 1, "symbol": 1, "date": 1}}],
        allowDiskUse=True,
    )

    hits = Counter()
    samples: dict[str, list] = {}
    n = 0
    total_removed = 0
    docs_with_removal = 0

    for doc in cursor:
        text = ((doc.get("title") or "") + "\n\n" + (doc.get("content") or "")).strip()
        if not text:
            continue
        n += 1
        cleaned, removed = sanitise(text)
        if removed:
            total_removed += removed
            docs_with_removal += 1
        for name, pat in _PATTERNS:
            m = pat.search(cleaned)
            if m:
                hits[name] += 1
                if len(samples.setdefault(name, [])) < args.examples:
                    s = max(0, m.start() - 60)
                    samples[name].append(
                        f"{doc.get('symbol')} {doc.get('date')}: "
                        f"...{cleaned[s:m.end() + 60].strip()}...".replace("\n", " ")
                    )

    print(f"scanned {n:,} real articles\n")
    print(f"{'pattern':<24}{'hits':>8}{'rate':>10}")
    flagged_any = 0
    for name, _ in _PATTERNS:
        c = hits.get(name, 0)
        print(f"  {name:<22}{c:>8,}{100 * c / max(1, n):>9.3f}%")
        flagged_any += c
    print(f"\nsanitisation: {docs_with_removal:,} of {n:,} articles "
          f"({100 * docs_with_removal / max(1, n):.2f}%) had characters stripped, "
          f"{total_removed:,} in total")

    print("\nwhat fired (these are FALSE POSITIVES unless the corpus is already poisoned):")
    for name in samples:
        for ex in samples[name]:
            print(f"  [{name}] {ex[:190]}")
    if not samples:
        print("  nothing fired")


if __name__ == "__main__":
    main()
