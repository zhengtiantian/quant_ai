"""Predictions: parsing and validation, scoring against CNBC daily bars, and the scorecard.
Network stubbed."""

import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import events, predictions as pr  # noqa: E402

ALLOWED = pr.universe(["INTC", "LRCX"])
ISSUED = date(2026, 10, 22)


def line(**kw):
    import json
    base = {"symbol": "8035.T", "date": "2026-10-23", "op": "<=", "threshold": -3, "confidence": 0.7, "zh": "z", "en": "e"}
    base.update(kw)
    return json.dumps(base)


class ParseTest(unittest.TestCase):
    def test_valid_and_dropped(self):
        text = "\n".join([line(), line(symbol="KRX:005930"), line(op="=="), line(date="2026-10-24"),
                          line(date="2026-12-01"), line(op="abs>=", threshold=-2), "not json", line(symbol="INTC", op="abs>=", threshold=5),
                          line(symbol=".KS11", confidence=3)])
        good, dropped = pr.parse_block(text, ALLOWED, ISSUED)
        self.assertEqual([(g["symbol"], g["exchange"]) for g in good], [("8035.T", "TSE"), ("INTC", "US"), (".KS11", "KOSPI")])
        self.assertEqual(good[2]["confidence"], 1.0)                     # clamped
        self.assertEqual(len(dropped), 5)                                # korean stock, op, weekend, too far, negative abs

    def test_universe_excludes_korean_stocks_but_keeps_indices(self):
        self.assertNotIn("KRX:005930", ALLOWED)
        self.assertIn(".KQ11", ALLOWED)
        self.assertEqual(ALLOWED[".SOX"][1], "US")


BARS = {"barData": {"priceBars": [
    {"tradeTime": "20261021000000", "close": "10160.0"},
    {"tradeTime": "20261022000000", "close": "9450.0"},
    {"tradeTime": "20261023000000", "close": "9072.0"},
]}}


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "m.db"

    def test_day_change_from_bars(self):
        with mock.patch.object(pr.pf, "_get_json", return_value=BARS):
            self.assertAlmostEqual(pr.day_change("8035.T", date(2026, 10, 23)), (9072 / 9450 - 1) * 100)
            self.assertIsNone(pr.day_change("8035.T", date(2026, 10, 26)))

    def test_ops(self):
        self.assertTrue(pr.holds("<=", -3, -4))
        self.assertFalse(pr.holds(">=", 1, 0.5))
        self.assertTrue(pr.holds("abs>=", 5, -6))
        self.assertTrue(pr.holds("abs<=", 2, -1.5))

    def test_scored_only_after_the_close_then_hit_or_miss(self):
        good, _ = pr.parse_block("\n".join([line(), line(op=">=", threshold=1, confidence=0.6)]), ALLOWED, ISSUED)
        pr.save(1, "flash", good, self.path)
        tokyo_close = datetime(2026, 10, 23, 15, 30, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp()
        with mock.patch.object(pr, "day_change", return_value=-4.0):
            self.assertEqual(pr.score_due(self.path, now=tokyo_close + 60), 0)          # bars not settled yet
            self.assertEqual(pr.score_due(self.path, now=tokyo_close + pr.SCORE_DELAY + 1), 2)
        rows = pr.for_brief(1, self.path)
        self.assertEqual([(r["status"], r["actual"]) for r in rows], [("hit", -4.0), ("miss", -4.0)])
        card = pr.scorecard(self.path)
        self.assertEqual((card["overall"]["scored"], card["overall"]["hits"]), (2, 1))
        self.assertAlmostEqual(card["overall"]["brier"], ((0.7 - 1) ** 2 + (0.6 - 0) ** 2) / 2)
        self.assertEqual(card["by_kind"]["flash"]["hit_rate"], 0.5)

    def test_no_trading_that_day_is_voided_later(self):
        good, _ = pr.parse_block(line(), ALLOWED, ISSUED)
        pr.save(2, "us", good, self.path)
        close = datetime(2026, 10, 23, 15, 30, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp()
        with mock.patch.object(pr, "day_change", return_value=None):
            self.assertEqual(pr.score_due(self.path, now=close + 86400), 0)             # maybe the bar is late
            self.assertEqual(pr.score_due(self.path, now=close + 5 * 86400), 1)
        self.assertEqual(pr.for_brief(2, self.path)[0]["status"], "void")

    def test_rewritten_brief_replaces_its_predictions(self):
        good, _ = pr.parse_block(line(), ALLOWED, ISSUED)
        pr.save(3, "us", good + good, self.path)
        pr.save(3, "us", good, self.path)
        self.assertEqual(len(pr.for_brief(3, self.path)), 1)


class EventsTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "m.db"

    def test_implied_move_from_the_first_expiry_atm_straddle(self):
        chain = {"data": {"table": {"rows": [
            {"expirygroup": "October 23, 2026"},
            {"expirygroup": "", "strike": "95.00", "c_Bid": "6.0", "c_Ask": "6.4", "p_Bid": "1.0", "p_Ask": "1.2"},
            {"expirygroup": "", "strike": "100.00", "c_Bid": "3.0", "c_Ask": "3.4", "p_Bid": "2.9", "p_Ask": "3.1"},
            {"expirygroup": "October 30, 2026"},
            {"expirygroup": "", "strike": "100.00", "c_Bid": "5.0", "c_Ask": "5.4", "p_Bid": "4.9", "p_Ask": "5.1"},
        ]}}}
        with mock.patch.object(events.pf, "_get_json", return_value=chain):
            m = events.implied_move("LRCX", date(2026, 10, 21), 100.5)
        self.assertEqual((m["expiry"], m["strike"], m["implied_move_pct"]), ("October 23, 2026", 100.0, 6.2))

    def test_upcoming_keeps_only_watched_names_and_main_releases(self):
        earn = [{"symbol": "LRCX", "name": "Lam"}, {"symbol": "XYZ", "name": "Other"}]
        econ = [{"country": "United States", "event": "Core PCE Price Index"}, {"country": "Brazil", "event": "CPI"},
                {"country": "United States", "event": "Redbook"}]
        with mock.patch.object(events, "earnings_on", return_value=earn), mock.patch.object(events, "econ_on", return_value=econ), \
             mock.patch.object(events, "implied_move", return_value={"implied_move_pct": 7.1}) as im:
            u = events.upcoming(["LRCX"], days=2, path=self.path, prices={"LRCX": 100})
        self.assertEqual([e["symbol"] for e in u["earnings"]], ["LRCX", "LRCX"])
        self.assertEqual(u["earnings"][0]["bellwether"], "chip equipment")
        self.assertEqual(u["earnings"][0]["implied"], {"implied_move_pct": 7.1})
        self.assertEqual({e["event"] for e in u["economic"]}, {"Core PCE Price Index"})
        self.assertEqual(im.call_count, 2)

    def test_big_moves_only_outside_regular_hours(self):
        q = {"LRCX": {"session": "post", "session_change_pct": -7.4, "price": 1, "regular_price": 1.1},
             "NVDA": {"session": "post", "session_change_pct": 2.0},
             "AMD": {"session": "regular", "session_change_pct": -9.0},
             "MU": {"session": "pre", "session_change_pct": 6.0, "stale": True}}
        with mock.patch.object(events.pf, "get_quotes", return_value=q):
            self.assertEqual([m["symbol"] for m in events.big_moves(self.path)], ["LRCX"])

    def test_calendar_is_cached(self):
        with mock.patch.object(events.pf, "_get_json", return_value={"data": {"rows": [{"symbol": "LRCX", "time": "time-after-hours"}]}}) as g:
            events.earnings_on(date(2026, 10, 21), self.path)
            rows = events.earnings_on(date(2026, 10, 21), self.path)
        self.assertEqual(g.call_count, 1)
        self.assertEqual(rows[0]["when"], "after the close")


if __name__ == "__main__":
    unittest.main()
