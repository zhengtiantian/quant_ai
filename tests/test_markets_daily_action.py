"""Risk gate (Q.12) and today's action (Q.13). No network: the state and the Q.10 analysis are given."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from markets import daily_action, risk_rules  # noqa: E402


def state(intc_qty=142.18, intc_px=64.89, others=(("SKHY", 6755.82), ("TSM", 2760.33), ("STX", 2154.21), ("GOOGL", 281.45)), cash=0.0):
    hs = [{"symbol": "INTC", "quantity": intc_qty, "avg_cost": 87.69, "market_value": intc_qty * intc_px, "quote": {"price": intc_px}}]
    hs += [{"symbol": s, "quantity": 1.0, "avg_cost": v, "market_value": v, "quote": {"price": v}} for s, v in others]
    return {"holdings": hs, "cash": cash}


def advice(stabilized=False, lowest=64.44, events=True, report=False):
    cats = [{"title": "Apple earnings", "impact": "high", "bucket": "spillover", "category": "earnings", "days_away": 2}] if events else []
    if report:
        cats.append({"title": "INTC earnings", "impact": "high", "bucket": "direct", "category": "earnings", "days_away": 0})
    return {"stats": {"atr14": 6.86, "stabilized": stabilized}, "catalysts_7d": cats, "dip_guardrail": {"anchor_cost": lowest}}


class GateTest(unittest.TestCase):
    def test_exposure_counts_cash_in_the_total(self):
        ex = risk_rules.exposure(state(cash=1000.0))
        self.assertAlmostEqual(ex["total"], 142.18 * 64.89 + 6755.82 + 2760.33 + 2154.21 + 281.45 + 1000.0)
        self.assertGreater(ex["cluster_pct"], 90)                  # GOOGL is the only name outside the cluster

    def test_trim_back_to_the_cap(self):
        ex = risk_rules.exposure(state())
        n = risk_rules.trim_shares(142.18, ex["mv"]["INTC"], ex["total"], 30)
        self.assertEqual(n, 45)                                    # 43.6% -> 30%: 1 - 30/43.6 of 142.18, rounded up
        self.assertEqual(risk_rules.trim_shares(10, 100, 1000, 30), 0)

    def test_a_sale_is_capped_at_the_whole_shares_held(self):
        g = risk_rules.gate({"symbol": "STX", "side": "sell", "shares": 5, "limit": 700}, risk_rules.exposure(state()), held=3.32)
        self.assertEqual((g["shares"], g["allowed"]), (3, True))

    def test_a_buy_meets_cash_caps_spacing_and_the_falling_rule(self):
        ex = risk_rules.exposure(state(cash=5000.0))
        g = risk_rules.gate({"symbol": "INTC", "side": "buy", "shares": 10, "limit": 64.0}, ex, held=142.18, atr=6.86,
                            lowest_lot=64.44, stabilized=False, event_7d=True)
        self.assertFalse(g["allowed"])
        self.assertEqual(set(g["blocked_by"]), {"falling", "spacing", "size"})   # over the symbol cap: no room at all
        self.assertTrue(any("$50.72" in n for n in g["notes"]))                  # 64.44 - 2 x 6.86

    def test_a_buy_that_fits(self):
        ex = risk_rules.exposure({"holdings": [{"symbol": "GOOGL", "market_value": 1000.0}], "cash": 9000.0})
        g = risk_rules.gate({"symbol": "GOOGL", "side": "buy", "shares": 5, "limit": 300.0}, ex, held=3, atr=10, lowest_lot=340,
                            stabilized=True)
        self.assertEqual((g["allowed"], g["shares"]), (True, 5))
        g = risk_rules.gate({"symbol": "GOOGL", "side": "buy", "shares": 20, "limit": 300.0}, ex, held=3)
        self.assertEqual(g["shares"], 6)                            # 30% of 10,000 = 3,000; 2,000 of room left


class DailyActionTest(unittest.TestCase):
    def test_overweight_intc_is_trimmed_into_a_rebound_and_buys_are_blocked(self):
        st = state()
        a = daily_action.decide("INTC", st["holdings"][0], risk_rules.exposure(st), advice())
        self.assertEqual(a["action"], "trim")
        self.assertEqual([(o["shares"], o["limit"]) for o in a["orders"]], [(23, 75.18), (22, 78.61)])   # +1.5 / +2 x ATR
        self.assertFalse(a["buy_allowed"])
        self.assertTrue(any("Irish CGT" in r for r in a["reasons"]))           # the limits are below the average cost

    def test_report_night_comes_first(self):
        st = state()
        a = daily_action.decide("INTC", st["holdings"][0], risk_rules.exposure(st), advice(report=True))
        self.assertEqual((a["action"], a["orders"]), ("earnings", []))

    def test_hold_when_within_the_caps(self):
        st = {"holdings": [{"symbol": "INTC", "quantity": 10, "avg_cost": 80, "market_value": 650.0, "quote": {"price": 65.0}},
                           {"symbol": "GOOGL", "quantity": 10, "avg_cost": 280, "market_value": 2800.0, "quote": {"price": 280.0}}],
              "cash": 0.0}
        st["holdings"].append({"symbol": "AAPL", "quantity": 10, "avg_cost": 200, "market_value": 2000.0, "quote": {"price": 200.0}})
        st["holdings"].append({"symbol": "MSFT", "quantity": 10, "avg_cost": 300, "market_value": 3000.0, "quote": {"price": 300.0}})
        a = daily_action.decide("INTC", st["holdings"][0], risk_rules.exposure(st), advice(stabilized=True, events=False))
        self.assertEqual(a["action"], "hold")
        self.assertIn("Cash $0.00 buys 0 shares at $65.00", a["blocks"])

    def test_compute_keeps_the_answer_for_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "m.db"
            out = daily_action.compute(path=db, state=state(), analyze=lambda s, path=None, state=None: advice())
            self.assertEqual(out["actions"][0]["action"], "trim")
            import sqlite3
            rows = sqlite3.connect(db).execute("SELECT symbol, action FROM daily_actions").fetchall()
            self.assertEqual(rows, [("INTC", "trim")])


if __name__ == "__main__":
    unittest.main()
