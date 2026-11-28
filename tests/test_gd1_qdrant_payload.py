"""Unit tests for GD1 v3 Qdrant payload extraction and hybrid search filters."""
import sys
from pathlib import Path
from types import SimpleNamespace

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import hybrid_search as hs  # noqa: E402
import index_news as inews  # noqa: E402


def test_extract_themes_prefers_llm_themes_and_falls_back_to_gd1():
    assert inews.extract_themes({"llm_themes": ["ai_capex", "export_controls"], "student_gd1": {"themes": ["crypto"]}}) == [
        "ai_capex", "export_controls"
    ]
    assert inews.extract_themes({"student_gd1": {"themes": ["memory_cycle", ""]}}) == ["memory_cycle"]
    assert inews.extract_themes({}) == []


def test_extract_impacted_symbols_excludes_self_and_zero():
    doc = {
        "symbol": "NVDA",
        "llm_impacts": {"NVDA": 2, "AMD": -1, "tsm": 2, "INTC": 0, "BAD": None, "FLAG": True},
    }
    assert inews.extract_impacted_symbols(doc) == ["AMD", "TSM"]
    assert inews.extract_impacted_symbols({"symbol": "AAPL", "student_gd1": {"impacts": {"QCOM": -2}}}) == ["QCOM"]
    assert inews.extract_impacted_symbols({"symbol": "AAPL", "llm_impacts": None}) == []


def test_dense_and_sparse_legs_apply_theme_and_impacted_symbol_filters(monkeypatch):
    monkeypatch.setattr(hs, "embed_query", lambda q: [0.1, 0.2])

    captured_dense = {}

    class FakeQC:
        def query_points(self, **kwargs):
            captured_dense["filter"] = kwargs["query_filter"]
            return SimpleNamespace(points=[])

    hs.dense_leg(FakeQC(), "chip export", "NVDA", 20250101, 20250131, 10,
                 theme="export_controls", impacted_symbol="amd")
    keys = {c.key: c.match.value for c in captured_dense["filter"].must if getattr(c, "match", None)}
    assert keys["symbol"] == "NVDA"
    assert keys["themes"] == "export_controls"
    assert keys["impacted_symbols"] == "AMD"

    captured_sparse = {}

    class FakeColl:
        def find(self, q, proj):
            captured_sparse["query"] = q
            class FakeCursor:
                def sort(self, *a, **kw):
                    return self
                def limit(self, *a, **kw):
                    return []
            return FakeCursor()

    hs.sparse_leg(FakeColl(), "chip export", "NVDA", 20250101, 20250131, 10,
                  theme="export_controls", impacted_symbol="amd")
    and_clauses = captured_sparse["query"]["$and"]
    assert {"llm_themes": "export_controls"} in and_clauses[0]["$or"]
    assert {"student_gd1.themes": "export_controls"} in and_clauses[0]["$or"]
    assert {"student_gd1.impacts.AMD": {"$in": [-2, -1, 1, 2]}} in and_clauses[1]["$or"]
    assert {"llm_impacts.AMD": {"$in": [-2, -1, 1, 2]}} in and_clauses[1]["$or"]
