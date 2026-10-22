"""get_my_holdings / get_my_transactions read quant_ai's markets portfolio and keep the shape
portfolio_agent relies on; without the markets routes they fall back to quant_api."""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mcp_server as ms  # noqa: E402
import portfolio_agent as pa  # noqa: E402

STATE = {
    "status": "success", "session": "post",
    "holdings": [
        {"symbol": "INTC", "quantity": 100.0, "avg_cost": 106.8, "fifo_avg_cost": 106.8, "cost": 10680.0,
         "market_value": 8874.0, "unrealized": -1806.0, "unrealized_pct": -16.9, "fifo_unrealized": -1806.0,
         "realized": 0.0, "day_change_pct": -4.54, "day_pnl": -422.0, "weight": 85.0, "trades": 3,
         "quote": {"price": 88.74, "session": "pre", "regular_price": 92.96, "name": "Intel"}},
        {"symbol": "STX", "quantity": 4.0, "avg_cost": 400.0, "fifo_avg_cost": 390.0, "cost": 1600.0,
         "market_value": 1566.0, "unrealized": -34.0, "unrealized_pct": -2.1, "fifo_unrealized": -6.0,
         "realized": 12.5, "day_change_pct": -1.0, "day_pnl": -15.0, "weight": 15.0, "trades": 2,
         "quote": {"price": 391.5, "session": "pre", "stale": True}},
    ],
    "totals": {"market_value": 10440.0, "cost": 12280.0, "unrealized": -1840.0, "unrealized_pct": -14.98,
               "fifo_cost": 12240.0, "realized": 12.5, "dividends": 0.0, "day_pnl": -437.0, "day_pnl_pct": -4.0},
    "watchlist": [{"symbol": "NVDA", "quote": {"price": 180.0}, "day_change_pct": -2.0}],
    "transactions": [
        {"id": 3, "symbol": "STX", "side": "sell", "quantity": 1, "price": 410, "trade_date": "2026-10-20", "fee": 0, "note": ""},
        {"id": 1, "symbol": "INTC", "side": "buy", "quantity": 100, "price": 106.8, "trade_date": "2026-09-01", "fee": 1, "note": "x"},
    ],
}


class MarketsHoldingsTest(unittest.TestCase):
    def test_shape_matches_what_portfolio_agent_reads(self):
        out = ms.holdings_from_markets(STATE)
        self.assertEqual([h["weightPct"] for h in out["holdings"]], [85.0, 15.0])
        self.assertEqual(out["totals"]["holdingsValue"], out["totals"]["totalValue"])
        self.assertEqual(out["holdings"][1]["quoteSource"], "nasdaq (stale)")
        self.assertEqual(out["holdings"][1]["fifoAvgCost"], 390.0)
        # the agent's own concentration check runs on it unchanged
        flags = pa._portfolio_checks(out, [])
        self.assertTrue(any("INTC" in json.dumps(f) for f in flags))

    def test_tool_prefers_markets_and_falls_back_to_quant_api(self):
        with mock.patch.object(ms, "_markets_get", return_value=STATE), mock.patch.object(ms, "_get") as api:
            body = json.loads(ms.get_my_holdings())
            tx = json.loads(ms.get_my_transactions("intc"))
        api.assert_not_called()
        self.assertEqual(body["source"], "quant_ai markets")
        self.assertEqual([(t["id"], t["side"]) for t in tx], [(1, "BUY")])
        with mock.patch.object(ms, "_markets_get", return_value=None), \
             mock.patch.object(ms, "_get", return_value='{"holdings": []}') as api:
            self.assertEqual(ms.get_my_holdings(), '{"holdings": []}')
        api.assert_called_once_with("/api/portfolio/holdings")


if __name__ == "__main__":
    unittest.main()
