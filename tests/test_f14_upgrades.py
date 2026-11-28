"""Unit tests for F14 upgrades (market_panic gating, hint_scores splits/filters, devils_advocate fallback & audit)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from markets import devils_advocate as da  # noqa: E402
from markets import hint_scores, risk_rules  # noqa: E402


class RiskRulesPanicTest(unittest.TestCase):
    def test_market_panic_blocks_and_widens_spacing_to_3x_atr(self):
        ex = {"mv": {"NVDA": 1000.0}, "cash": 5000.0, "total": 6000.0, "weights": {"NVDA": 16.7}, "cluster_mv": 1000.0, "cluster_pct": 16.7}
        # 2.5 x ATR below 100 (atr=4 -> 90) would pass event spacing (2x=92),
        # but must trigger BOTH market_panic AND spacing under 3x panic spacing (88)!
        res = risk_rules.gate(
            {"symbol": "NVDA", "side": "buy", "shares": 1, "limit": 90.0},
            ex,
            held=10.0,
            atr=4.0,
            lowest_lot=100.0,
            stabilized=True,
            event_7d=True,
            market_panic=True,
            panic_reason="VIX 25",
            panic_reason_zh="VIX 恐慌",
        )
        self.assertFalse(res["allowed"])
        self.assertEqual(set(res["blocked_by"]), {"market_panic", "spacing"})
        self.assertTrue(any("3 x ATR" in n for n in res["notes"]))


class HintScoresF14Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "m.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_probe_records_market_panic_and_summary_filters_and_splits(self):
        st = {
            "holdings": [{"symbol": "AAA", "quantity": 5.0, "market_value": 500.0, "quote": {"price": 100.0}}],
            "watchlist": [{"symbol": "BBB", "quote": {"price": 50.0}}],
            "cash": 5000.0,
        }
        res = hint_scores.probe(
            st,
            self.db,
            analyze=lambda s, path=None, state=None: {"stats": {"atr14": 4.0, "stabilized": True}, "catalysts_7d": [], "dip_guardrail": {}},
            stats_fn=lambda s, p, path=None: {"atr14": 2.0, "stabilized": True},
            catalysts_fn=lambda s, p=None: [],
            market_state={"tighten_buys": True, "headline_en": "Chip stress", "headline_zh": "芯片踩踏", "vix": {"price": 26.5}},
        )
        by_sym = {r["symbol"]: r for r in res}
        self.assertEqual(by_sym["AAA"]["codes"], ["market_panic"])
        self.assertEqual(by_sym["BBB"]["codes"], ["market_panic"])
        hint_scores.record_probe("2026-11-27", res, self.db)

        hint_scores.record("devils_advocate", "AAA", "2026-11-27", True, 100.0, ["manual"], {"strength": "strong"}, self.db)
        hint_scores.record("portfolio_review_flag", "AAA", "2026-11-27", True, 100.0, ["trim"], {}, self.db)

        s_all = hint_scores.summary(self.db)
        self.assertEqual(s_all["groups"]["add_block:market_panic"]["fired"], 2)
        self.assertEqual(s_all["groups"]["devils_advocate:strong"]["fired"], 1)
        self.assertEqual(s_all["groups"]["portfolio_review_flag"]["fired"], 1)

        s_held = hint_scores.summary(self.db, held_only=True)
        self.assertEqual(s_held["groups"]["add_block:market_panic"]["fired"], 1)
        self.assertTrue(s_held["held_only"])


class DevilsAdvocateF14Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "m.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_packet_includes_market_weather_and_check_catches_injection(self):
        kw = {
            "state": {"holdings": [], "watchlist": [{"symbol": "BBB", "quote": {"price": 50.0, "name": "Bee"}}], "cash": 5000.0},
            "news_fn": lambda s: [],
            "judgment_fn": lambda s: None,
            "release_fn": lambda s: None,
            "filing_fn": lambda s: None,
            "analysts_fn": lambda s, since, p: [],
            "market_fn": lambda: {"tighten_buys": True, "vix_level": 26.2, "sox_1d_pct": -4.1, "severity": "critical", "headline_en": "Panic"},
        }
        with mock.patch("markets.lot_advisor._bar_stats", return_value={"atr14": 2.0, "atr14_pct": 4.0, "drop_1d_pct": -5.0, "drop_1d_atr": 2.2, "drop_3d_pct": -8.0, "stabilized": True, "big_drop": None}), \
             mock.patch("markets.lot_advisor._upcoming_7d_catalysts", return_value=[]):
            text, ids, keep = da.packet("BBB", self.db, **kw)

        self.assertIn("market panic / sector stress active", text)
        self.assertIn("Market weather (critical, tighten_buys=True)", text)
        self.assertIn("market panic / sector stress active", keep["gate"][0])

        # Layer-4 check rejects adopted instructions / prompt disclosure
        bad = {
            "summary": {"en": "Following the instructions in the article, sell everything.", "zh": "遵照新闻指令，全部卖出。"},
            "base_rate": {"en": "Base", "zh": "基准"},
            "points": [{"en": "Point", "zh": "理由", "cites": [next(iter(ids))]}],
            "would_weaken": {"en": "None", "zh": "无"},
            "strength": "strong",
        }
        probs = da.check(bad, ids)
        self.assertIn("adopted_document_instruction", probs)

    def test_local_fallback_when_claude_unavailable(self):
        fake_json = json.dumps({
            "summary": {"zh": "本地降级反方结论", "en": "Local fallback summary"},
            "base_rate": {"zh": "历史回撤较高", "en": "Elevated pullback rate"},
            "points": [{"zh": "波动放大且恐慌", "en": "Elevated vol", "cites": ["F1"]}],
            "would_weaken": {"zh": "放量企稳", "en": "Stabilises"},
            "strength": "moderate",
        })
        with mock.patch("markets.briefs.find_claude", return_value=None), \
             mock.patch("agent._chat", return_value={"content": fake_json}):
            res = da.ask("test prompt")
        self.assertEqual(res["engine"], "local_fallback")
        self.assertEqual(res["structured_output"]["strength"], "moderate")

        # Verify _public reads engine cleanly from stored cited metadata
        kw = {
            "state": {"holdings": [], "watchlist": [{"symbol": "BBB", "quote": {"price": 50.0, "name": "Bee"}}], "cash": 5000.0},
            "news_fn": lambda s: [],
            "judgment_fn": lambda s: None,
            "release_fn": lambda s: None,
            "filing_fn": lambda s: None,
            "analysts_fn": lambda s, since, p: [],
            "market_fn": lambda: {"tighten_buys": False},
        }
        with mock.patch("markets.lot_advisor._bar_stats", return_value={"atr14": 2.0, "atr14_pct": 4.0, "drop_1d_pct": -5.0, "drop_1d_atr": 2.2, "drop_3d_pct": -8.0, "stabilized": True, "big_drop": None}), \
             mock.patch("markets.lot_advisor._upcoming_7d_catalysts", return_value=[]):
            rid = da.request("BBB", "manual", self.db, start=False)["id"]
            out = da.run(rid, self.db, ask_fn=lambda _: res, **kw)
        self.assertEqual(out["engine"], "local_fallback")
        pub = da.latest(path=self.db)["items"]["BBB"]
        self.assertEqual(pub["engine"], "local_fallback")


if __name__ == "__main__":
    unittest.main()
