"""R.3 in the request path: news_rag reranks the fused top-30 when asked, and never fails
an answer because the reranker could not run. No Mongo, Qdrant, LM Studio or model needed."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import news_rag  # noqa: E402
from hybrid_search import Hit  # noqa: E402


class FakeLLM:
    def invoke(self, msgs):
        class R:
            content = "Stub answer [1]."
        return R()


def _hits(n):
    return [Hit(key=f"k{i}", symbol="NVDA", date="20240101", title=f"title {i}", url=None,
                mongo_id=None) for i in range(n)]


def _run(rerank, search_k_seen, reranked=None, rerank_error=None, n_hits=40):
    def fake_search(q, k, **kw):
        search_k_seen.append(k)
        return [], [], _hits(min(n_hits, k))

    def fake_rerank(q, hits, bodies, k, max_len):
        if rerank_error:
            raise rerank_error
        return list(reversed(hits))[:k], 0.5

    with mock.patch.object(news_rag, "search", fake_search), \
         mock.patch.object(news_rag, "build_context", lambda hits, coll: ("ctx", [{"id": i + 1, "title": h.title, "injection_flags": [], "chars_sanitised": 0} for i, h in enumerate(hits)])), \
         mock.patch("rerank.rerank", fake_rerank), mock.patch("rerank.fetch_bodies", lambda coll, hits: {}):
        return news_rag.answer_from_news("q", FakeLLM(), coll=None, k=8, rerank=rerank)


def test_off_retrieves_k_and_keeps_fused_order():
    seen = []
    res = _run(False, seen)
    assert seen == [8]
    assert res["rerank"] == {"enabled": False}
    assert [s["title"] for s in res["sources"]][:2] == ["title 0", "title 1"]


def test_on_retrieves_the_candidate_depth_and_returns_k_reranked():
    seen = []
    res = _run(True, seen)
    assert seen == [news_rag.RERANK_CANDIDATES]
    assert len(res["sources"]) == 8
    assert res["sources"][0]["title"] == f"title {news_rag.RERANK_CANDIDATES - 1}"   # reversed by the fake
    assert res["rerank"]["candidates"] == news_rag.RERANK_CANDIDATES and res["rerank"]["ms"] == 500.0


def test_reranker_failure_falls_back_to_fused_order_instead_of_failing():
    seen = []
    res = _run(True, seen, rerank_error=RuntimeError("no torch"))
    assert not res["refused"]
    assert len(res["sources"]) == 8 and res["sources"][0]["title"] == "title 0"
    assert "no torch" in res["rerank"]["error"]


def test_empty_retrieval_still_refuses_with_rerank_on():
    seen = []
    res = _run(True, seen, n_hits=0)
    assert res["refused"]


def test_default_follows_the_env_switch():
    seen = []
    with mock.patch.object(news_rag, "RERANK_ENABLED", True):
        assert _run(None, seen)["rerank"]["enabled"] is True
    with mock.patch.object(news_rag, "RERANK_ENABLED", False):
        assert _run(None, seen)["rerank"]["enabled"] is False
