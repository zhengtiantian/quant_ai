"""Unit tests for short-term swing lot exit & dip-entry spacing advisor (Q.10)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import lot_advisor  # noqa: E402


def _fake_chart(_sym, _rng, _path=None):
    # 25 sessions of calm bars around 90 with ATR = 3.0, then a sharp plunge to 65.75
    candles = []
    for i in range(24):
        candles.append({"date": f"2026-10-{i+1:02d}", "open": 90.0, "high": 91.5, "low": 88.5, "close": 90.0})
    candles.append({"date": "2026-10-30", "open": 72.0, "high": 73.0, "low": 65.0, "close": 65.75})
    return {"candles": candles}


class LotAdvisorTest(unittest.TestCase):
    def _mock_state(self, live_price=65.75):
        return {
            "cash": 1500.0,
            "totals": {"market_value": 20000.0},
            "holdings": [
                {
                    "symbol": "INTC",
                    "quantity": 98.0,
                    "avg_cost": 92.24,
                    "market_value": 6443.5,
                    "quote": {"price": live_price, "session": "regular"},
                    "buy_orders": [
                        {"id": 1, "trade_date": "2026-09-10", "unit_cost": 100.0, "open_qty": 58.0,
                         "open_pnl": (live_price - 100.0) * 58.0, "open_pnl_pct": (live_price / 100.0 - 1) * 100, "status": "open"},
                        {"id": 10, "trade_date": "2026-10-30", "unit_cost": 73.14, "open_qty": 20.0,
                         "open_pnl": (live_price - 73.14) * 20.0, "open_pnl_pct": (live_price / 73.14 - 1) * 100, "status": "open"},
                        {"id": 11, "trade_date": "2026-10-30", "unit_cost": 71.00, "open_qty": 20.0,
                         "open_pnl": (live_price - 71.00) * 20.0, "open_pnl_pct": (live_price / 71.00 - 1) * 100, "status": "open"},
                    ],
                },
                {
                    "symbol": "TSM",
                    "quantity": 30.0,
                    "avg_cost": 400.0,
                    "market_value": 12000.0,
                    "quote": {"price": 400.0, "session": "regular"},
                    "buy_orders": [],
                },
            ],
            "closed": [],
        }

    def test_detects_tight_recent_pair_and_escalates_spacing_before_binary_events(self):
        cats = [{"date": "2026-11-03", "days_away": 4, "title": "美国中期选举", "category": "macro", "impact": "high", "bucket": "macro"}]
        res = lot_advisor.analyze_symbol(
            "INTC",
            proposed_buy_price=69.50,
            state=self._mock_state(65.75),
            chart_fn=_fake_chart,
            catalysts_fn=lambda _s: cats,
        )
        dg = res["dip_guardrail"]
        self.assertEqual(dg["verdict"], "red")
        self.assertEqual(dg["min_spacing_atr"], 2.0)
        self.assertIsNotNone(dg["recent_pair_tight"])
        self.assertAlmostEqual(dg["recent_pair_tight"]["gap_usd"], 2.14, places=2)
        self.assertGreater(res["semi_cluster_weight_pct"], 90.0)
        self.assertTrue(res["has_binary_7d"])
        # When proposed_buy_price is deep enough (>= 2.0*ATR below 71.00), spacing passes cleanly
        res_deep = lot_advisor.analyze_symbol(
            "INTC",
            proposed_buy_price=58.00,
            state=self._mock_state(65.75),
            chart_fn=_fake_chart,
            catalysts_fn=lambda _s: cats,
        )
        self.assertIn(res_deep["dip_guardrail"]["verdict"], ("amber", "green"))
        self.assertGreaterEqual(res_deep["dip_guardrail"]["spacing_vs_lowest_atr"], 2.0)

    def test_lot_exit_plans_recommend_harvest_on_profitable_tranche(self):
        # Rebound to 80.00 -> lot #11 ($71.00) is +$9.00 (+$180), well above +1.5xATR
        cats = [{"date": "2026-11-06", "days_away": 6, "title": "美国非农就业", "category": "macro", "impact": "high", "bucket": "macro"}]
        res = lot_advisor.analyze_symbol(
            "INTC",
            state=self._mock_state(80.00),
            chart_fn=_fake_chart,
            catalysts_fn=lambda _s: cats,
        )
        by_id = {p["buy_id"]: p for p in res["lot_exit_plans"]}
        self.assertEqual(by_id[11]["status_code"], "harvest_now")
        self.assertIsNotNone(by_id[11]["trail_stop_price"])
        self.assertGreater(by_id[11]["tp1_gain_usd"], 0)
        # Core lot #1 ($100) is still underwater, while dip lots #10 and #11 are ready to harvest
        self.assertEqual(by_id[1]["status_code"], "underwater_swing")


if __name__ == "__main__":
    unittest.main()
