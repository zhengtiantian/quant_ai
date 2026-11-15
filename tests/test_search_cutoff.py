"""N.6: search_articles(before=...) returns only articles published before an exact moment, in both
retrieval legs. No Mongo, Qdrant or LM Studio needed: the legs run against fakes that record the filter."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import hybrid_search as hs  # noqa: E402
import news_rag  # noqa: E402

CUT = "20261112100000"


def test_before_ok_compares_to_the_second_and_drops_date_only_rows_of_the_cutoff_day():
    assert hs.before_ok("20261112095959", CUT)
    assert not hs.before_ok("20261112100000", CUT)
    assert not hs.before_ok("20261112180000", CUT)
    assert not hs.before_ok("20261112", CUT)      # no time: may be after the cutoff
    assert hs.before_ok("20261111", CUT)
    assert not hs.before_ok("20261113", CUT)
    assert not hs.before_ok("", CUT)
    assert hs.before_ok("20991231", None)         # no cutoff: everything passes


def test_until_is_the_earlier_of_the_window_and_the_cutoff_day():
    assert hs._until_for(None, CUT) == 20261112
    assert hs._until_for(20261101, CUT) == 20261101
    assert hs._until_for(20261130, CUT) == 20261112
    assert hs._until_for(20261130, None) == 20261130


class _Cursor(list):
    def sort(self, *a, **k): return self
    def limit(self, *a, **k): return self


class _Coll:
    def __init__(self, docs): self.docs, self.q = docs, None

    def find(self, q, proj):
        self.q = q
        rng = q.get("date", {})
        keep = [d for d in self.docs if rng.get("$gte", "") <= d["date"] < rng.get("$lt", "~") and d["date"] != rng.get("$ne")]
        return _Cursor(keep)


def test_sparse_leg_puts_the_cutoff_into_the_query():
    docs = [{"_id": i, "symbol": "NVDA", "title": f"t{i}", "url": "", "date": d, "score": 1.0}
            for i, d in enumerate(["20261111", "20261112", "20261112093000", "20261112100000", "20261112120000", "20261113"])]
    coll = _Coll(docs)
    hits = hs.sparse_leg(coll, "huang", "NVDA", None, None, 10, before=CUT)
    assert coll.q["date"]["$lt"] == CUT and coll.q["date"]["$ne"] == "20261112"
    assert sorted(h.title for h in hits) == ["t0", "t2"]


def test_dense_leg_filters_inside_the_leg_and_ranks_what_is_left():
    pts = [SimpleNamespace(score=1 - i / 10, payload={"symbol": "NVDA", "title": f"t{i}", "date": d, "url": "", "mongo_id": str(i)})
           for i, d in enumerate(["20261112120000", "20261112", "20261112090000", "20261110"])]
    qc = mock.Mock()
    qc.query_points.return_value = SimpleNamespace(points=pts)
    with mock.patch.object(hs, "embed_query", return_value=[0.0]):
        hits = hs.dense_leg(qc, "huang", "NVDA", None, None, 5, before=CUT)
    flt = qc.query_points.call_args.kwargs["query_filter"]
    rng = [c.range for c in flt.must if getattr(c, "key", "") == "date_int"][0]
    assert rng.lte == 20261112
    assert qc.query_points.call_args.kwargs["limit"] == 10          # over-fetched
    assert [h.title for h in hits] == ["t2", "t3"] and [h.dense_rank for h in hits] == [1, 2]


def test_search_articles_accepts_a_datetime_and_passes_utc():
    seen = {}

    def fake_search(q, k, **kw):
        seen.update(kw)
        return [], [], []

    dublin = timezone(timedelta(hours=1))
    with mock.patch.object(news_rag, "search", fake_search):
        news_rag.search_articles("huang", symbol="nvda", coll=mock.Mock(find=lambda *a, **k: []),
                                 before=datetime(2026, 11, 12, 11, 0, tzinfo=dublin))
    assert seen["before"] == "20261112100000"
    assert news_rag._cutoff("2026-11-12T10:00:00Z") == "20261112100000"
    assert news_rag._cutoff(CUT) == CUT
    assert news_rag._cutoff(None) is None
