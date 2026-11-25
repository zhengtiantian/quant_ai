"""The Keywords tab's files: screen / audit fields carried through, the auto switch, badge counts."""
import json
from pathlib import Path

import pytest

from markets import keyword_review as kr


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    kw, cov = tmp_path / "kw", tmp_path / "cov"
    (kw / "runs").mkdir(parents=True)
    cov.mkdir()
    monkeypatch.setattr(kr, "DIR", kw)
    monkeypatch.setattr(kr, "COVERAGE_DIR", cov)
    cands = [
        {"id": "core:AAA:widget", "set": "core", "symbol": "AAA", "name": "Aaa Inc", "term": "Widget", "kind": "product",
         "tier": "add", "co_mention": 0.9, "examples": [], "company_headlines": 10, "current_keywords": ["Aaa"],
         "screen": {"risks": [], "claude": {"risky": False, "other": ""}}},
        {"id": "core:AAA:air thing", "set": "core", "symbol": "AAA", "name": "Aaa Inc", "term": "Air Thing", "kind": "product",
         "tier": "add", "co_mention": 0.8, "examples": [], "company_headlines": 10, "current_keywords": ["Aaa"],
         "screen": {"risks": [{"code": "claude", "text": "Claude: an aircraft"}]}},
        {"id": "core:AAA:other", "set": "core", "symbol": "AAA", "name": "Aaa Inc", "term": "Other", "kind": "none",
         "tier": "model_none", "co_mention": 0.4, "examples": [], "company_headlines": 10, "current_keywords": ["Aaa"]},
    ]
    (kw / "runs" / "2026-11-14.json").write_text(json.dumps({"created": "x", "candidates": cands}))
    (kw / "auto_preview.json").write_text(json.dumps({"at": "t", "ids": ["core:AAA:widget"]}))
    (kw / "decisions.json").write_text(json.dumps({"core:AAA:widget": {"decision": "add", "at": "t", "by": "auto"}}))
    (kw / "audit.json").write_text(json.dumps({"at": "t", "terms": {"core:AAA:widget": {"brought_in": 12, "kept": 9, "verdict": "watching"}}}))
    (cov / "latest.json").write_text(json.dumps({"at": "c", "core_flagged": ["AAA", "BBB"], "core_alerts": ["AAA"], "universe_flagged": ["ZZZ"]}))
    return kw, cov


def test_review_carries_screen_audit_and_who_decided(dirs):
    rows = {r["id"]: r for r in kr.review()["proposals"]}
    w = rows["core:AAA:widget"]
    assert w["decided_by"] == "auto" and w["auto_eligible"] and w["audit"]["kept"] == 9
    assert rows["core:AAA:air thing"]["screen"]["risks"][0]["code"] == "claude"
    assert rows["core:AAA:air thing"]["decided_by"] is None


def test_summary_counts_pending_and_alerts(dirs):
    s = kr.summary()
    assert s["pending"] == 1                       # "air thing": add tier, undecided; model_none is not counted
    assert s["coverage_alerts"] == 2 and s["coverage_notices"] == 1
    assert s["auto_apply"] is False


def test_auto_switch_and_user_decision_replaces_auto(dirs):
    kw, _ = dirs
    assert kr.set_auto(True)["auto_apply"] is True
    assert json.loads((kw / "settings.json").read_text())["auto_apply"] is True
    assert kr.review()["auto"]["enabled"] is True
    kr.decide(["core:AAA:widget"], "reject")
    d = json.loads((kw / "decisions.json").read_text())["core:AAA:widget"]
    assert d["decision"] == "reject" and "by" not in d         # the user's decision: auto never touches it


def test_expired_proposals_and_screen_health_degradation(dirs):
    kw, _ = dirs
    # Add two newer runs with 0 Claude responses, making the 2026-10-01 run expired (>21d and not in last 2)
    (kw / "runs" / "2026-10-01.json").write_text(json.dumps({
        "created": "2026-10-01T08:00:00",
        "candidates": [
            {"id": "core:AAA:stale_term", "set": "core", "symbol": "AAA", "name": "Aaa Inc", "term": "Stale Term",
             "kind": "product", "tier": "add", "co_mention": 0.7, "examples": [], "company_headlines": 5, "current_keywords": ["Aaa"]},
        ],
    }))
    (kw / "runs" / "2026-11-21.json").write_text(json.dumps({
        "created": "2026-11-21T08:00:00",
        "candidates": [
            {"id": "core:AAA:no_claude", "set": "core", "symbol": "AAA", "name": "Aaa Inc", "term": "No Claude",
             "kind": "product", "tier": "add", "co_mention": 0.9, "examples": [], "company_headlines": 10,
             "current_keywords": ["Aaa"], "screen": {"risks": [], "claude": None}},
        ],
    }))
    rev = kr.review()
    rows = {r["id"]: r for r in rev["proposals"]}
    assert rows["core:AAA:stale_term"]["expired"] is True
    assert rows["core:AAA:no_claude"]["expired"] is False
    assert rev["screen_health"]["degraded"] is True
    assert rev["screen_health"]["claude_answered"] == 0
    assert rev["screen_health"]["claude_rate"] == 0.0

    s = kr.summary()
    assert s["expired"] == 1
    assert s["pending"] == 2  # air thing (in last 2 runs) + no_claude; stale_term excluded
    assert s["screen_health"]["degraded"] is True
