"""Risk plan for the portfolio caps (markets/optimizer.py). No network: covariance and bars are given."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from markets import optimizer  # noqa: E402


def ex(mv, cash=0.0):
    return {"mv": mv, "cash": cash, "total": sum(mv.values()) + cash}


class PlanTest(unittest.TestCase):
    def test_own_excess_then_cluster_from_the_riskiest(self):
        # INTC 45% / SKHY 35% / TSM 15% (all chips) / GOOGL 5%; INTC the most volatile
        e = ex({"INTC": 4500.0, "SKHY": 3500.0, "TSM": 1500.0, "GOOGL": 500.0})
        vol = np.array([0.80, 0.50, 0.35, 0.30])
        corr = np.full((4, 4), 0.6) + np.eye(4) * 0.4
        p = optimizer.plan(e, {"INTC": 60.0, "SKHY": 150.0, "TSM": 400.0, "GOOGL": 280.0}, np.outer(vol, vol) * corr)
        self.assertAlmostEqual(p["total_usd"], 2500.0, delta=1)                  # chips 95% -> 70% of 10,000
        self.assertGreaterEqual(p["sell"]["INTC"]["usd"], 1500.0)                # its own excess over 30% at least
        self.assertGreaterEqual(p["sell"]["SKHY"]["usd"], 500.0)
        self.assertNotIn("GOOGL", p["sell"])                                     # outside the cluster, cash already met
        self.assertEqual(p["cluster_after_pct"], 70.0)
        self.assertLess(p["vol_after_pct"], p["vol_before_pct"])

    def test_cash_floor_without_a_risk_model(self):
        e = ex({"GOOGL": 2800.0, "AAPL": 2400.0, "MSFT": 2400.0, "AMZN": 2400.0})   # each within 30%, no cash
        p = optimizer.plan(e, {"GOOGL": 300.0, "AAPL": 200.0, "MSFT": 400.0, "AMZN": 200.0}, None)
        self.assertAlmostEqual(p["total_usd"], 1000.0, delta=1)                  # 10% cash
        self.assertEqual(max(p["sell"], key=lambda s: p["sell"][s]["usd"]), "GOOGL")   # the largest first without covariance
        self.assertFalse(p["risk_model"])

    def test_within_every_cap_sells_nothing(self):
        e = ex({"GOOGL": 2500.0, "AAPL": 2500.0, "MSFT": 2500.0, "INTC": 1000.0}, cash=1500.0)
        self.assertEqual(optimizer.plan(e, {}, None)["sell"], {})

    def test_covariance_from_common_dates(self):
        rng = np.random.default_rng(0)
        bars = {s: [{"time": f"2026-{i:04d}", "close": float(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 80))[i]))} for i in range(80)]
                for s in ("A", "B")}
        cov = optimizer.covariance(["A", "B"], chart_fn=lambda s, r, p=None: {"candles": bars[s]})
        self.assertEqual(cov.shape, (2, 2))
        self.assertIsNone(optimizer.covariance(["A"], chart_fn=lambda s, r, p=None: {"candles": bars[s][:30]}))


if __name__ == "__main__":
    unittest.main()
