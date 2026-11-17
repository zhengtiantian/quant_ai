"""Blocked buys logged and scored against having bought (Q.17.0). No network: charts and analysis are given."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from markets import hint_scores  # noqa: E402


def state():
    return {"holdings": [{"symbol": "AAA", "quantity": 10.0, "market_value": 1000.0, "quote": {"price": 100.0}}],
            "watchlist": [{"symbol": "BBB", "quote": {"price": 50.0}}], "cash": 5000.0}


def analyze(sym, path=None, state=None):
    return {"stats": {"atr14": 4.0, "stabilized": False, "drop_1d_atr": 2.5}, "catalysts_7d": [],
            "dip_guardrail": {"anchor_cost": 102.0}}


def stats_fn(sym, price, path=None):
    return {"atr14": 2.0, "stabilized": True, "drop_1d_atr": 0.1}


def chart(closes):
    """A fake get_chart: daily candles 2026-01-01.. with the given closes per symbol."""
    def fn(symbol, rng, path=None):
        cs = closes.get(symbol, [])
        return {"candles": [{"time": f"2026-01-{i + 1:02d}", "close": c, "high": c, "low": c, "open": c} for i, c in enumerate(cs)]}
    return fn


class HintScoresTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "m.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_row_per_kind_stock_and_session(self):
        self.assertTrue(hint_scores.record("add_block", "aaa", "2026-01-01", True, 100.0, ["falling"], path=self.db))
        self.assertFalse(hint_scores.record("add_block", "AAA", "2026-01-01", True, 99.0, ["size"], path=self.db))
        self.assertTrue(hint_scores.record("devils_advocate", "AAA", "2026-01-01", True, 99.0, ["manual"], path=self.db))

    def test_probe_asks_the_gate_for_held_and_watched_stocks(self):
        res = {r["symbol"]: r for r in hint_scores.probe(state(), self.db, analyze=analyze, stats_fn=stats_fn, catalysts_fn=lambda s, p=None: [])}
        self.assertFalse(res["AAA"]["allowed"])
        self.assertEqual(set(res["AAA"]["codes"]), {"falling", "spacing"})     # 100 > 102 - 1.5 x 4
        self.assertTrue(res["BBB"]["allowed"])                                  # watched, stabilised, cash to spare
        self.assertFalse(res["BBB"]["held"])

    def test_newly_blocked_needs_a_previous_probe(self):
        blocked = [{"symbol": "AAA", "held": True, "price": 100.0, "allowed": False, "codes": ["falling"]}]
        self.assertEqual(hint_scores.record_probe("2026-01-02", blocked, self.db), [])          # nothing to compare with
        hint_scores.record("add_block", "ZZZ", "2026-01-05", False, 1.0, ["size"], path=self.db)  # a probe that did not block AAA
        self.assertEqual(hint_scores.record_probe("2026-01-06", blocked, self.db), ["AAA"])
        hint_scores.record("add_block", "AAA", "2026-01-07", True, 100.0, ["falling"], path=self.db)
        self.assertEqual(hint_scores.record_probe("2026-01-08", blocked, self.db), [])          # still blocked: not new

    def test_scored_from_the_close_against_spy(self):
        hint_scores.record("add_block", "AAA", "2026-01-01", True, 100.0, ["falling"], path=self.db)
        closes = {"AAA": [100.0, 99, 98, 97, 96, 90.0], "SPY": [400.0, 400, 400, 400, 400, 404.0]}
        self.assertEqual(hint_scores.score_due(self.db, chart(closes)), 2)                       # r5 and x5; 20 / 60 not due
        g = hint_scores.summary(self.db)["groups"]
        five = g["add_block:falling"]["horizons"]["5"]
        self.assertEqual(five["own"]["fell"], 1)
        self.assertAlmostEqual(five["own"]["loss_avoided"], 10.0)                               # -10% after the block
        self.assertAlmostEqual(five["vs_spy"]["loss_avoided"], 11.0)                            # SPY +1%
        self.assertIsNone(five["own"]["gain_missed"])
        self.assertEqual(g["add_block"]["horizons"]["20"]["own"]["n"], 0)

    def test_a_rise_is_a_gain_missed(self):
        hint_scores.record("add_block", "AAA", "2026-01-01", True, 100.0, ["size"], path=self.db)
        hint_scores.score_due(self.db, chart({"AAA": [100.0, 1, 1, 1, 1, 110.0], "SPY": [1.0] * 6}))
        own = hint_scores.summary(self.db)["groups"]["add_block:size"]["horizons"]["5"]["own"]
        self.assertEqual((own["rose"], own["gain_missed"], own["net"]), (1, 10.0, -10.0))


if __name__ == "__main__":
    unittest.main()
