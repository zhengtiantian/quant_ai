"""Unit tests for F08 News RAG & hybrid search upgrades."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import hybrid_search as hs  # noqa: E402
import news_rag  # noqa: E402


def test_auto_weights_and_stub_demotion():
    wd_tick, ws_tick = hs.auto_weights("NVDA")
    assert ws_tick > wd_tick
    wd_sem, ws_sem = hs.auto_weights("why did memory stocks sell off sharply")
    assert wd_sem > ws_sem

    stub = hs.Hit(key="k_stub", symbol="LRCX", date="20261120", title="Lam Research (Nasdaq: LRCX)", url=None, dense_rank=1, sparse_rank=1)
    real = hs.Hit(key="k_real", symbol="LRCX", date="20261120", title="Lam Research cuts Q4 shipments on export curbs", url=None, dense_rank=2, sparse_rank=2)
    fused = hs.rrf_fuse([stub, real], [stub, real], k=2)
    assert fused[0].key == "k_real"


def test_search_with_meta_degrades_gracefully_when_dense_fails():
    h = hs.Hit(key="k1", symbol="NVDA", date="20261125", title="Nvidia drops on export rules", url=None, sparse_rank=1)
    with mock.patch.object(hs, "dense_leg", side_effect=RuntimeError("qdrant down")), \
         mock.patch.object(hs, "sparse_leg", return_value=[h]):
        dense, sparse, fused, meta = hs.search_with_meta("NVDA export controls", k=5, qc="dummy", coll="dummy")
    assert dense == [] and len(sparse) == 1 and len(fused) == 1
    assert "qdrant down" in meta["degraded"]["dense_error"]


def test_prepare_query_relative_date_and_chinese_rewrite():
    now = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
    # 1. Glossary fallback (llm=None)
    rq, s, u, info = news_rag.prepare_query("最近30天英伟达出口管制和存储芯片抛售原因", llm=None, now=now)
    assert s == 20261027 and u is None
    assert info["rewritten"] is True and info["rewrite_mode"] == "glossary"
    assert "NVDA" in rq and "export controls" in rq and "memory" in rq

    # 2. LLM rewrite preserves English keywords and strips CJK
    class FakeLLM:
        model_name = "qwen3.5-9b-mlx"

        def invoke(self, msgs):
            return SimpleNamespace(content="<think>x</think>\nNVDA export controls memory chip selloff")

    rq2, s2, _, info2 = news_rag.prepare_query("最近7天美光和英伟达为什么跌", llm=FakeLLM(), now=now)
    assert s2 == 20261119
    assert rq2 == "NVDA export controls memory chip selloff"
    assert info2["rewrite_mode"] == "llm"

    # 3. Year-to-date phrase
    _, s3, _, _ = news_rag.prepare_query("今年以来半导体涨幅", llm=None, now=now)
    assert s3 == 20260101


def test_answer_from_news_single_fetch_bodies_and_gd1_fields():
    hit = hs.Hit(key="k1", symbol="TSM", date="20261120", title="NVDA restrictions ripple to TSM", url="https://u", mongo_id="507f1f77bcf86cd799439011", dense_rank=1)
    fetch_calls = []

    class FakeColl:
        def find(self, query, proj):
            fetch_calls.append(proj)
            return [{
                "_id": "507f1f77bcf86cd799439011",
                "content": "Taiwan Semiconductor faces order shifts from US curbs.",
                "url": "https://u",
                "date": "20261120",
                "student_gd1": {"themes": ["export_controls"], "impacts": {"TSM": -1}, "event_type": "regulation", "sentiment": -0.6},
            }]

    class FakeLLM:
        def invoke(self, msgs):
            return SimpleNamespace(content="TSM faces indirect headwinds [1].")

    with mock.patch.object(news_rag, "search_with_meta", return_value=([hit], [], [hit], {"weights": {"dense": 1.0, "sparse": 1.0}, "degraded": {}})), \
         mock.patch("rerank.rerank", return_value=([hit], 0.12)):
        res = news_rag.answer_from_news("export controls impact TSM", FakeLLM(), coll=FakeColl(), rerank=True, theme="export_controls", impacted_symbol="TSM")
    assert len(fetch_calls) == 1
    assert "student_gd1" in fetch_calls[0]
    assert res["sources"][0]["themes"] == ["export_controls"]
    assert res["sources"][0]["impacts"] == {"TSM": -1}
