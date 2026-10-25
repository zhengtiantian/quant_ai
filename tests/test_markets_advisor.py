"""Holdings advisor (Q.5): levels, order validation, fills and scoring, triggers. No network."""
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import advisor  # noqa: E402
from markets import portfolio as pf  # noqa: E402

NY = ZoneInfo("America/New_York")


def holding(symbol, price, day_pct, session="regular", value=5000.0, session_pct=None):
    return {"symbol": symbol, "quantity": 10, "avg_cost": price * 1.2, "market_value": value, "day_change_pct": day_pct,
            "quote": {"price": price, "session": session, "session_change_pct": session_pct}}


class LevelsTest(unittest.TestCase):
    def test_earnings_band_uses_the_implied_move_else_two_atr(self):
        stats = {"atr14_pct": 3.0, "high_20d": 95, "low_20d": 80, "high_60d": 100, "low_60d": 75}
        lv = advisor.levels(87.0, 106.74, stats, 9.9)
        self.assertEqual(lv["sell_limits"], [91.31, 95.61])
        self.assertEqual(lv["buy_limits"], [82.69, 78.39])
        self.assertEqual(lv["to_break_even_pct"], 22.7)
        self.assertEqual(advisor.levels(100.0, None, stats, None)["band_pct"], 6.0)

    def test_impact_rules(self):
        self.assertEqual(advisor.impact("Intel cuts full-year guidance as PC demand slows"), "major")
        self.assertEqual(advisor.impact("Analyst upgrades Seagate, raises price target"), "high")
        self.assertIsNone(advisor.impact("A Ph.D. led me to my dream job at Intel"))


class OrdersTest(unittest.TestCase):
    PRICES = {"INTC": 87.0, "AMZN": 248.0}

    def parse(self, lines):
        return advisor.parse_orders("\n".join(lines), self.PRICES, date(2026, 10, 26))

    def test_valid_orders_kept_bad_ones_dropped_with_reasons(self):
        orders, dropped = self.parse([
            '{"symbol": "INTC", "side": "sell", "tag": "trim", "pct_of_position": 20, "limit": 92.5, "valid_until": "2026-10-29", "zh": "z", "en": "e"}',
            '{"symbol": "AMZN", "side": "buy", "tag": "new", "amount_usd": 500, "limit": 240, "valid_until": "2026-11-06"}',
            '{"symbol": "NVDA", "side": "buy", "limit": 180, "valid_until": "2026-10-30"}',          # not offered
            '{"symbol": "INTC", "side": "sell", "limit": 140, "valid_until": "2026-10-30"}',         # 61% away
            '{"symbol": "INTC", "side": "buy", "limit": 80, "valid_until": "2027-01-30"}',           # too long
            'not json'])
        self.assertEqual([(o["symbol"], o["side"], o["pct"], o["amount_usd"]) for o in orders],
                         [("INTC", "sell", 20.0, None), ("AMZN", "buy", None, 500.0)])
        self.assertEqual(len(dropped), 3)

    def test_fills_and_edge(self):
        sell = {"side": "sell", "tag": "trim", "limit_price": 92.5}
        buy = {"side": "buy", "tag": "rebuy", "limit_price": 80.0}
        stop = {"side": "sell", "tag": "stop", "limit_price": 78.0}
        bar = {"high": 93.0, "low": 79.5, "close": 85.0}
        self.assertTrue(advisor.filled(sell, bar) and advisor.filled(buy, bar))
        self.assertFalse(advisor.filled(stop, bar))
        self.assertAlmostEqual(advisor.edge_pct(sell, 85.0), 8.82, places=2)     # sold 92.5, later 85: +8.8%
        self.assertAlmostEqual(advisor.edge_pct(buy, 85.0), 6.25, places=2)      # bought 80, later 85: +6.25%


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "a.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_order_filled_then_scored_at_expiry_unfilled_expires(self):
        issued = datetime(2026, 10, 26, 9, 0, tzinfo=NY).timestamp()                # Monday pre-market
        advisor.save_orders(1, [
            {"symbol": "INTC", "side": "sell", "tag": "trim", "pct": 20, "shares": None, "amount_usd": None,
             "limit_price": 92.5, "issued_price": 87.0, "valid_until": "2026-10-28", "zh": "", "en": ""},
            {"symbol": "INTC", "side": "buy", "tag": "rebuy", "pct": 20, "shares": None, "amount_usd": None,
             "limit_price": 70.0, "issued_price": 87.0, "valid_until": "2026-10-28", "zh": "", "en": ""}], issued, self.db)
        bars = [{"date": date(2026, 10, 26), "high": 90, "low": 86, "close": 89},
                {"date": date(2026, 10, 27), "high": 93, "low": 88, "close": 91},
                {"date": date(2026, 10, 28), "high": 92, "low": 84, "close": 85}]
        midweek = datetime(2026, 10, 27, 17, 0, tzinfo=NY).timestamp()
        advisor.score_due(self.db, now=midweek, bars_fn=lambda s, a, b: bars)
        status = {o["limit_price"]: o["status"] for o in advisor.active_orders(self.db)}
        self.assertEqual(status, {92.5: "filled", 70.0: "open"})
        after = datetime(2026, 10, 28, 17, 0, tzinfo=NY).timestamp()
        advisor.score_due(self.db, now=after, bars_fn=lambda s, a, b: bars)
        card = advisor.scorecard(self.db)
        done = {o["limit_price"]: (o["status"], o["edge_pct"]) for o in card["recent"]}
        self.assertEqual(done[92.5], ("scored", 8.82))       # sold at 92.5, 85 at expiry
        self.assertEqual(done[70.0][0], "expired")
        self.assertEqual((card["fill_rate"], card["win_rate"]), (0.5, 1.0))


class TriggerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "a.db"
        self.now = datetime(2026, 10, 28, 11, 0, tzinfo=NY).timestamp()             # Wednesday, regular session
        self._thr = mock.patch.object(advisor, "threshold", side_effect=lambda s, p=None: {"INTC": 6.1, "TSM": 3.0}.get(s))
        self._thr.start()

    def tearDown(self):
        self._thr.stop()
        self.tmp.cleanup()

    def run_check(self, holdings, news=None, earnings=None, now=None):
        with mock.patch.object(pf, "get_state", return_value={"holdings": holdings}):
            return advisor.check_triggers(self.db, now=now or self.now, news_fn=lambda s: news or {},
                                          earnings_fn=lambda s: list(earnings or []))

    def test_move_beyond_own_threshold_once_per_session_and_urgent_when_large(self):
        hs = [holding("INTC", 80.0, -6.5), holding("TSM", 420.0, 2.5), holding("WDC", 340.0, -9.0, value=0.01)]
        found = self.run_check(hs)
        self.assertEqual([f["symbol"] for f in found], ["INTC"])         # TSM under 3%, WDC is dust
        self.assertFalse(found[0]["urgent"])
        self.assertEqual(self.run_check(hs), [])                         # same session: not again
        big = self.run_check([holding("TSM", 400.0, -4.8)])
        self.assertTrue(big[0]["urgent"])                                # 4.8 >= 1.5 x 3.0

    def test_earnings_tomorrow_and_reported_today(self):
        hs = [holding("INTC", 87.0, 0.5, session="regular")]
        prep = self.run_check(hs, earnings=[{"symbol": "INTC", "date": "2026-10-29", "when": "after the close"}])
        self.assertEqual(prep[0]["reasons"][0]["kind"], "earnings_prep")
        out_now = datetime(2026, 10, 29, 16, 30, tzinfo=NY).timestamp()
        post = [holding("INTC", 80.0, 0.2, session="post", session_pct=-1.0)]
        out = self.run_check(post, earnings=[{"symbol": "INTC", "date": "2026-10-29", "when": "after the close"}], now=out_now)
        self.assertEqual([(r["kind"], r["urgent"]) for r in out[0]["reasons"]], [("earnings_out", True)])

    def test_news_fires_once_and_major_is_urgent(self):
        hs = [holding("INTC", 87.0, 0.1)]
        items = {"INTC": [{"title": "Intel cuts full-year guidance", "source": "Reuters", "impact": "major"},
                          {"title": "Intel stock: what to watch", "source": "yahoo", "impact": None}]}
        found = self.run_check(hs, news=items)
        self.assertTrue(found[0]["urgent"] and found[0]["reasons"][0]["kind"] == "news")
        self.assertEqual(self.run_check(hs, news=items), [])

    def test_advised_limit_reached(self):
        advisor.save_orders(7, [{"symbol": "INTC", "side": "sell", "tag": "trim", "pct": 20, "shares": None, "amount_usd": None,
                                 "limit_price": 92.5, "issued_price": 87.0, "valid_until": "2026-10-30", "zh": "", "en": ""}],
                            self.now - 3600, self.db)
        found = self.run_check([holding("INTC", 92.8, 4.0)])
        self.assertEqual([(r["kind"], r["urgent"]) for r in found[0]["reasons"]], [("limit", True)])

    def test_weekend_is_quiet(self):
        saturday = datetime(2026, 10, 31, 11, 0, tzinfo=NY).timestamp()
        self.assertEqual(self.run_check([holding("INTC", 70.0, -12.0)], now=saturday), [])


class CashTest(unittest.TestCase):
    def test_cash_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "p.db"
            pf.init_db(db)
            self.assertEqual(pf.get_cash(db), 0.0)
            self.assertEqual(pf.set_cash("2500.456", db), 2500.46)
            with self.assertRaises(ValueError):
                pf.set_cash("-5", db)


if __name__ == "__main__":
    unittest.main()
