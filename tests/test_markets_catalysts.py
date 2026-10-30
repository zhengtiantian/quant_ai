"""3-Layer Catalyst Collector, Per-Stock Breakdown, and Query Engine tests (no live network/quota)."""

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import catalysts, events  # noqa: E402


FAKE_STATE = {
    "holdings": [
        {"symbol": "TSM", "quantity": 10.0, "avg_cost": 380.0, "market_value": 4200.0, "weight": 60.0,
         "unrealized_pct": 10.5, "day_change_pct": 1.2, "quote": {"name": "Taiwan Semiconductor", "price": 420.0}},
        {"symbol": "STX", "quantity": 20.0, "avg_cost": 100.0, "market_value": 2800.0, "weight": 40.0,
         "unrealized_pct": 40.0, "day_change_pct": -0.5, "quote": {"name": "Seagate Technology", "price": 140.0}},
    ],
    "watchlist": [
        {"symbol": "NVDA", "day_change_pct": 2.1, "quote": {"name": "NVIDIA Corporation", "price": 195.0}},
        {"symbol": "MSFT", "day_change_pct": 0.4, "quote": {"name": "Microsoft Corporation", "price": 465.0}},
    ],
}


class EventsCurrencyTest(unittest.TestCase):
    def test_foreign_listing_currency_mapping(self):
        self.assertEqual(events._listing_currency("000660.KS"), "KRW")
        self.assertEqual(events._listing_currency("2330.TW"), "TWD")
        self.assertEqual(events._listing_currency("ASML.AS"), "EUR")
        self.assertEqual(events._listing_currency("8035.T"), "JPY")
        self.assertEqual(events._listing_currency("NVDA"), "USD")


class ThreeLayerCatalystsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "cat.db"

    def tearDown(self):
        self.tmp.cleanup()

    def _patch_l1(self):
        return [
            mock.patch.object(catalysts.pf, "get_state", return_value=FAKE_STATE),
            mock.patch.object(catalysts.events, "earnings_on", side_effect=lambda d, path=None: (
                [{"symbol": "NVDA", "name": "NVIDIA", "when": "after the close", "eps_forecast": 2.49,
                  "estimates": 13, "fiscal_quarter": "Oct 2026"}]
                if d == date(2026, 11, 17) else []
            )),
            mock.patch.object(catalysts.events, "finnhub_earnings", side_effect=lambda sym, s, e, path=None: (
                [{"symbol": "NVDA", "listed_as": "NVDA", "date": "2026-11-17", "when": "after the close",
                  "eps_forecast": 2.52, "revenue_forecast": 111_356_000_000, "forecast_currency": "USD", "fiscal_quarter": "Q3 2027"}]
                if sym == "NVDA" else
                [{"symbol": "TSM", "listed_as": "2330.TW", "date": "2027-01-13", "when": "before the open",
                  "eps_forecast": 32.22, "revenue_forecast": 1_631_700_000_000, "forecast_currency": "TWD", "fiscal_quarter": "Q4 2026"}]
                if sym == "TSM" and e >= date(2027, 1, 13) else []
            )),
            mock.patch.object(catalysts, "_cached_implied_move", return_value={"implied_move_pct": 7.8}),
            mock.patch.object(catalysts.events, "dividends_for", side_effect=lambda sym, path=None: (
                [{"symbol": "STX", "ex_date": "2026-11-24", "pay_date": "2026-12-10", "amount": 0.76, "currency": "USD"}]
                if sym == "STX" else []
            )),
            mock.patch.object(catalysts.events, "splits_on", return_value=[]),
            mock.patch.object(catalysts.events, "econ_on", return_value=[]),
        ]

    def test_l1_merges_earnings_dividends_tsm_monthly_and_per_stock_spillover(self):
        patches = self._patch_l1()
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        feed = catalysts.upcoming_feed(scope="all", days=30, path=self.db, today=date(2026, 10, 30))
        # NVDA earnings merged (no duplicate between Nasdaq & Finnhub; carries Finnhub revenue + Nasdaq EPS + implied move)
        nvda_earn = [e for e in feed["timeline"] if e["kind"] == "earnings" and e["symbol"] == "NVDA"]
        self.assertEqual(len(nvda_earn), 1)
        self.assertEqual(nvda_earn[0]["eps_forecast"], 2.49)
        self.assertEqual(nvda_earn[0]["revenue_forecast"], 111_356_000_000)
        self.assertEqual(nvda_earn[0]["implied_move_pct"], 7.8)
        self.assertIn("TSM", nvda_earn[0]["related_holdings"])

        # STX held dividend computes exact expected USD payout (20 shares * $0.76 = $15.20)
        stx_div = [e for e in feed["timeline"] if e["kind"] == "dividend" and e["symbol"] == "STX"]
        self.assertEqual(len(stx_div), 1)
        self.assertEqual(stx_div[0]["estimated_payout_usd"], 15.2)

        # TSM 10th-of-month revenue event generated on 2026-11-10
        tsm_rev = [e for e in feed["timeline"] if e["id"].startswith("corp:TSM:rev:")]
        self.assertEqual(len(tsm_rev), 1)
        self.assertEqual(tsm_rev[0]["date"], "2026-11-10")

        # Every holding & watchlist symbol has its own per_stock entry with direct/spillover/macro visibility
        self.assertEqual(set(feed["per_stock"].keys()), {"TSM", "STX", "NVDA", "MSFT"})
        self.assertGreaterEqual(feed["per_stock"]["TSM"]["direct_count"], 1)
        # TSM also sees NVDA earnings as a supply-chain spillover event!
        self.assertTrue(any(e["symbol"] == "NVDA" for e in feed["per_stock"]["TSM"]["spillover"]))

    def test_l2_extraction_and_l3_web_sweep_appear_in_feed_and_per_stock(self):
        patches = self._patch_l1()
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        fake_llm_reply = mock.Mock()
        fake_llm_reply.content = json.dumps([
            {
                "source_id": 1,
                "symbol": "MSFT",
                "affected_symbols": ["NVDA", "TSM"],
                "kind": "product",
                "event_date": "2026-11-18",
                "date_precision": "exact",
                "date_label": "",
                "impact_level": "high",
                "title_zh": "微软 Ignite 2026 开发者大会发布新款 AI 芯片与云计划",
                "title_en": "Microsoft Ignite 2026 AI silicon & cloud keynote",
                "detail_zh": "关注自研芯片及对 NVDA / TSM AI 算力采购指引",
                "detail_en": "Watch custom AI silicon & Azure AI capex guidance for NVDA/TSM",
            }
        ])
        fake_llm = mock.Mock()
        fake_llm.model_name = "qwen3.5-9b-mlx"
        fake_llm.invoke.return_value = fake_llm_reply

        with mock.patch.object(catalysts, "_sec_cik_map", return_value={}), \
             mock.patch.object(catalysts, "fetch_google_news_rss", return_value=[
                 {"title": "Microsoft Ignite 2026 keynote set for November 18", "url": "https://news.example.com/ignite",
                  "date": "2026-10-29", "source": "Reuters"}
             ]), \
             mock.patch.object(catalysts.advisor, "news", return_value={}), \
             mock.patch.object(catalysts, "_get_llm", return_value=fake_llm):
            saved_l2 = catalysts.collect_l2_extracted(days=45, path=self.db, today=date(2026, 10, 30))
        self.assertEqual(saved_l2, 1)

        claude_reply = {
            "result": (
                "=====EVENTS=====\n"
                '{"symbol": "NVDA", "affected_symbols": ["TSM", "STX"], "kind": "product", "event_date": "2026-11-16", '
                '"date_precision": "exact", "date_label": "Nov 16-20", "impact_level": "high", '
                '"title_zh": "SC26 全球超算大会开幕", "title_en": "SC26 Supercomputing Conference", '
                '"detail_zh": "AI 集群与存储方案密集发布", "detail_en": "Major HPC & AI cluster announcements", '
                '"source_title": "SC26 Official", "source_url": "https://sc26.supercomputing.org"}\n'
                "=====END====="
            )
        }
        with mock.patch("markets.briefs.find_claude", return_value="/usr/local/bin/claude"), \
             mock.patch("markets.briefs.run_claude", return_value=claude_reply):
            saved_l3 = catalysts.collect_l3_web(days=45, path=self.db, today=date(2026, 10, 30))
        self.assertEqual(saved_l3, 1)

        feed = catalysts.upcoming_feed(scope="all", days=30, path=self.db, today=date(2026, 10, 30))
        self.assertEqual(feed["summary"]["l2"], 1)
        self.assertEqual(feed["summary"]["l3"], 1)
        self.assertTrue(any(e["layer"] == "L2_extracted" and e["symbol"] == "MSFT" for e in feed["per_stock"]["MSFT"]["direct"]))
        self.assertTrue(any(e["layer"] == "L3_web" and e["symbol"] == "NVDA" for e in feed["per_stock"]["NVDA"]["direct"]))

    def test_query_catalysts_persists_and_deletes_history(self):
        patches = self._patch_l1()
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        fake_reply = mock.Mock()
        fake_reply.content = "## 核心结论\n未来一个月 NVDA 财报（11/17，隐含波动 ±7.8%）与 TSM 月度营收（11/10）最关键 [1]。"
        fake_llm = mock.Mock()
        fake_llm.model_name = "qwen3.5-9b-mlx"
        fake_llm.invoke.return_value = fake_reply

        with mock.patch.object(catalysts, "fetch_google_news_rss", return_value=[
            {"title": "NVDA Q3 preview", "url": "https://example.com/nvda", "date": "2026-10-29", "source": "CNBC"}
        ]), mock.patch.object(catalysts, "_get_llm", return_value=fake_llm):
            res = catalysts.query_catalysts("未来一个月什么事件最影响 TSM 和 NVDA？", days=30, path=self.db)

        self.assertIn("核心结论", res["answer"])
        self.assertEqual(len(res["sources"]), 1)
        hist = catalysts.list_queries(path=self.db)
        self.assertEqual(len(hist), 1)
        catalysts.delete_query(qid=res["id"], path=self.db)
        self.assertEqual(catalysts.list_queries(path=self.db), [])




class CatalystsApiRouteTest(unittest.TestCase):
    def _client(self, host="127.0.0.1"):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from markets import api

        app = FastAPI()
        app.include_router(api.router)
        return TestClient(app, client=(host, 50000))

    def test_routes_work_locally_and_reject_remote(self):
        with mock.patch.object(catalysts, "upcoming_feed", return_value={"timeline": [], "per_stock": {}, "summary": {"total": 0}}), \
             mock.patch.object(catalysts, "list_queries", return_value=[]), \
             mock.patch.object(catalysts, "trigger_collection", return_value={"started": ["l2"], "layers": {}}), \
             mock.patch.object(catalysts, "query_catalysts", return_value={"id": 1, "answer": "ok", "sources": []}), \
             mock.patch.object(catalysts, "delete_query"):
            c = self._client("127.0.0.1")
            r_feed = c.get("/api/markets/catalysts?days=30&scope=all")
            self.assertEqual(r_feed.status_code, 200)
            self.assertEqual(r_feed.json()["status"], "success")

            r_col = c.post("/api/markets/catalysts/collect", json={"layer": "l2", "days": 45})
            self.assertEqual(r_col.status_code, 200)
            self.assertEqual(r_col.json()["started"], ["l2"])

            r_q = c.post("/api/markets/catalysts/query", json={"question": "TSM 财报什么时候", "days": 30, "mode": "fast"})
            self.assertEqual(r_q.status_code, 200)
            self.assertEqual(r_q.json()["result"]["answer"], "ok")

            r_del = c.post("/api/markets/catalysts/history/delete", json={"id": 1})
            self.assertEqual(r_del.status_code, 200)

            remote = self._client("198.51.100.7")
            self.assertEqual(remote.get("/api/markets/catalysts").status_code, 403)


if __name__ == "__main__":
    unittest.main()
