import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import global_markets as pf  # noqa: E402
from markets import portfolio  # noqa: E402

ITEMS = [(".N225", "Nikkei 225", "TSE", "index"), (".GDAXI", "DAX", "XETRA", "index")]


def at(zone, y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo(zone))


class MarketStatusTest(unittest.TestCase):
    def test_hours_in_local_time(self):
        self.assertTrue(pf.market_status("TSE", at("Asia/Tokyo", 2026, 10, 22, 10, 0))["open"])
        self.assertFalse(pf.market_status("TSE", at("Asia/Tokyo", 2026, 10, 22, 15, 31))["open"])
        self.assertFalse(pf.market_status("XETRA", at("Europe/Berlin", 2026, 10, 22, 8, 59))["open"])
        self.assertTrue(pf.market_status("XETRA", at("Europe/Berlin", 2026, 10, 22, 17, 0))["open"])

    def test_weekend_closed(self):
        st = pf.market_status("HKEX", at("Asia/Hong_Kong", 2026, 10, 24, 11, 0))  # Saturday
        self.assertFalse(st["open"])
        self.assertIsNone(st["close_ts"])


class GlobalDueTest(unittest.TestCase):
    def status(self, open_, close_ts):
        return lambda m: {"open": open_, "close_ts": close_ts, "local_time": "00:00"}

    def test_never_fetched_is_due(self):
        with mock.patch.object(pf, "market_status", self.status(False, None)):
            self.assertEqual(pf._global_due(ITEMS, {}, 1000.0), [".N225", ".GDAXI"])

    def test_open_market_waits_for_ttl(self):
        stored = {s[0]: {"fetched_at": 1000.0} for s in ITEMS}
        with mock.patch.object(pf, "market_status", self.status(True, 9e9)):
            self.assertEqual(pf._global_due(ITEMS, stored, 1000.0 + pf.GLOBAL_TTL - 1), [])
            self.assertEqual(len(pf._global_due(ITEMS, stored, 1000.0 + pf.GLOBAL_TTL)), 2)

    def test_closed_market_fetched_once_after_close(self):
        with mock.patch.object(pf, "market_status", self.status(False, 2000.0)):
            self.assertEqual(len(pf._global_due(ITEMS, {s[0]: {"fetched_at": 1990.0} for s in ITEMS}, 5000.0)), 2)
            self.assertEqual(pf._global_due(ITEMS, {s[0]: {"fetched_at": 2010.0} for s in ITEMS}, 9e6), [])
        with mock.patch.object(pf, "market_status", self.status(False, None)):  # weekend
            self.assertEqual(pf._global_due(ITEMS, {s[0]: {"fetched_at": 1.0} for s in ITEMS}, 9e6), [])


class RefreshGlobalTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "p.db"

    def test_stores_parsed_quotes_and_skips_unrequested(self):
        reply = {"FormattedQuoteResult": {"FormattedQuote": [
            {"symbol": ".N225", "last": "62,907.84", "change_pct": "-2.65%", "previous_day_closing": "64,620.00",
             "currencyCode": "JPY", "name": "Nikkei 225"},
            {"symbol": "OTHER", "last": "1"}, {"symbol": ".GDAXI", "last": "UNCH"}]}}
        with mock.patch.object(pf, "GLOBAL_ITEMS", ITEMS), mock.patch.object(portfolio, "_get_json", return_value=reply) as get:
            self.assertEqual(pf.refresh_global(self.path), 1)
            g = pf.global_state(self.path, background=False)
        self.assertIn(".N225%7C.GDAXI", get.call_args[0][0])
        asia, europe = g["regions"]
        nikkei = asia["countries"][0]["exchanges"][0]["indices"][0]
        self.assertEqual(nikkei["price"], 62907.84)
        self.assertEqual(nikkei["change_pct"], -2.65)
        self.assertEqual(nikkei["currency"], "JPY")
        dax = europe["countries"][0]["exchanges"][0]["indices"][0]
        self.assertNotIn("price", dax)

    def test_tree_keeps_only_exchanges_with_items(self):
        items = ITEMS + [(".KS11", "KOSPI", "KOSPI", "index"), (".KQ11", "KOSDAQ", "KOSDAQ", "index")]
        with mock.patch.object(pf, "GLOBAL_ITEMS", items):
            g = pf.global_state(self.path, background=False)
        self.assertEqual([r["name"] for r in g["regions"]], ["Asia", "Europe"])
        self.assertEqual([c["name"] for c in g["regions"][0]["countries"]], ["Japan", "South Korea"])
        self.assertEqual([e["code"] for e in g["regions"][0]["countries"][1]["exchanges"]], ["KOSPI", "KOSDAQ"])
        self.assertEqual([c["name"] for c in g["regions"][1]["countries"]], ["Germany"])

    def test_korean_stocks_go_to_naver(self):
        items = [("KRX:005930", "Samsung", "KOSPI", "stock"), (".KS11", "KOSPI", "KOSPI", "index")]
        replies = {"naver": {"datas": [{"itemCode": "005930", "closePrice": "214,000", "compareToPreviousClosePrice": "-9,500",
                                        "fluctuationsRatio": "-4.25", "stockName": "x"}]},
                   "cnbc": {"FormattedQuoteResult": {"FormattedQuote": [{"symbol": ".KS11", "last": "5,533.77"}]}}}
        fake = lambda url: replies["naver" if "naver" in url else "cnbc"]
        with mock.patch.object(pf, "GLOBAL_ITEMS", items), mock.patch.object(portfolio, "_get_json", side_effect=fake):
            self.assertEqual(pf.refresh_global(self.path), 2)
            ex = pf.global_state(self.path, background=False)["regions"][0]["countries"][0]["exchanges"][0]
        samsung = ex["stocks"][0]
        self.assertEqual((samsung["price"], samsung["prev_close"], samsung["currency"]), (214000.0, 223500.0, "KRW"))
        self.assertEqual(ex["indices"][0]["price"], 5533.77)

    def test_one_source_failing_keeps_the_other(self):
        items = [("KRX:005930", "Samsung", "KOSPI", "stock"), (".KS11", "KOSPI", "KOSPI", "index")]
        def fake(url):
            if "naver" in url:
                raise OSError("down")
            return {"FormattedQuoteResult": {"FormattedQuote": [{"symbol": ".KS11", "last": "1"}]}}
        with mock.patch.object(pf, "GLOBAL_ITEMS", items), mock.patch.object(portfolio, "_get_json", side_effect=fake):
            self.assertEqual(pf.refresh_global(self.path), 1)

    def test_macro_board_groups_and_caches(self):
        reply = {"FormattedQuoteResult": {"FormattedQuote": [{"symbol": "@LCO.1", "last": "111.94", "change_pct": "+2.1%"},
                                                             {"symbol": "US10Y", "last": "5.071%"}]}}
        with mock.patch.object(portfolio, "_get_json", return_value=reply) as get:
            m = pf.macro_state(self.path)
            pf.macro_state(self.path)                                   # Brent and US10Y are fresh now
        self.assertEqual(get.call_count, 2)                              # the rest were missing both times
        items = {i["symbol"]: i for g in m["groups"] for i in g["items"]}
        self.assertEqual(items["@LCO.1"]["price"], 111.94)
        self.assertEqual(items["US10Y"]["price"], 5.071)
        self.assertNotIn("%40LCO.1", get.call_args_list[1][0][0])

    def test_nothing_due_means_no_request(self):
        with mock.patch.object(pf, "GLOBAL_ITEMS", ITEMS), mock.patch.object(pf, "_global_due", return_value=[]), \
             mock.patch.object(portfolio, "_get_json") as get:
            self.assertEqual(pf.refresh_global(self.path), 0)
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
