"""Volume of the held and watched stocks (markets/volume.py) and gold / bitcoin / oil in the Holdings
page's index list (global_markets.INDEX_ADDED), 2026-11-21."""
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import global_markets  # noqa: E402
from markets import portfolio as pf  # noqa: E402
from markets import volume  # noqa: E402

ET = ZoneInfo("America/New_York")


def bars(n: int, last: date, vol=lambda i: 1000 + i):
    out, d, i = [], last, n - 1
    while len(out) < n:
        if d.weekday() < 5:
            out.append({"time": d.isoformat(), "volume": float(vol(i))})
            i -= 1
        d -= timedelta(days=1)
    return list(reversed(out))


class SessionTests(unittest.TestCase):
    def test_bar_final_after_the_close(self):
        self.assertEqual(volume.last_complete_session(datetime(2026, 11, 20, 16, 5, tzinfo=ET)), date(2026, 11, 20))
        self.assertEqual(volume.last_complete_session(datetime(2026, 11, 20, 15, 59, tzinfo=ET)), date(2026, 11, 19))

    def test_weekend_and_monday_morning_use_friday(self):
        self.assertEqual(volume.last_complete_session(datetime(2026, 11, 21, 12, 0, tzinfo=ET)), date(2026, 11, 20))
        self.assertEqual(volume.last_complete_session(datetime(2026, 11, 23, 9, 0, tzinfo=ET)), date(2026, 11, 20))


class SummarizeTests(unittest.TestCase):
    def test_month_and_year_windows(self):
        b = bars(300, date(2026, 11, 20), vol=lambda i: 100)
        s = volume.summarize(b, date(2026, 11, 20))
        self.assertEqual((s["last_date"], s["last_volume"]), ("2026-11-20", 100.0))
        self.assertEqual((s["month"]["sessions"], s["month"]["total"], s["month"]["avg"]), (21, 2100.0, 100))
        self.assertEqual((s["year"]["sessions"], s["year"]["total"], s["year"]["avg"]), (252, 25200.0, 100))

    def test_todays_unfinished_bar_is_left_out(self):
        b = bars(30, date(2026, 11, 20)) + [{"time": "2026-11-23", "volume": 5.0}]
        s = volume.summarize(b, date(2026, 11, 20))
        self.assertEqual(s["last_date"], "2026-11-20")
        self.assertEqual(s["month"]["sessions"], 21)

    def test_a_young_listing_uses_what_there_is(self):
        s = volume.summarize(bars(10, date(2026, 11, 20), vol=lambda i: 50), date(2026, 11, 20))
        self.assertEqual((s["month"]["sessions"], s["year"]["sessions"], s["year"]["avg"]), (10, 10, 50))

    def test_no_bars(self):
        self.assertIsNone(volume.summarize([], date(2026, 11, 20)))


class RefreshTests(unittest.TestCase):
    def setUp(self):
        tmp = Path(tempfile.mkdtemp())
        self.db, self.out = tmp / "p.db", tmp / "volume.json"
        pf.init_db(self.db)
        pf.save_transaction({"trade_date": "2026-09-01", "symbol": "NVDA", "side": "buy", "quantity": 10, "price": 100}, self.db)
        pf.add_watch("AMD", path=self.db)
        pf.add_watch("NVDA", path=self.db)              # held: listed with the holdings only
        self.now = datetime(2026, 11, 21, 12, 0, tzinfo=ET)

    def test_holdings_and_watchlist_with_their_figures(self):
        seen = []

        def fetch(sym):
            seen.append(sym)
            if sym == "AMD":
                raise RuntimeError("CNBC timed out")
            return bars(40, date(2026, 11, 20))

        data = volume.refresh(self.db, bars_fn=fetch, now=self.now, out=self.out)
        self.assertEqual(data["groups"], {"holdings": ["NVDA"], "watchlist": ["AMD"]})
        self.assertEqual(sorted(seen), ["AMD", "NVDA"])
        self.assertEqual(data["symbols"]["NVDA"]["month"]["sessions"], 21)
        self.assertIn("AMD", data["errors"])
        self.assertEqual(volume.load(self.out)["session"], "2026-11-20")

    def test_last_good_figures_kept_when_a_fetch_fails(self):
        volume.refresh(self.db, bars_fn=lambda s: bars(40, date(2026, 11, 19)), now=self.now, out=self.out)

        def fail(sym):
            raise RuntimeError("down")

        data = volume.refresh(self.db, bars_fn=fail, now=self.now, out=self.out)
        self.assertTrue(data["symbols"]["NVDA"]["stale"])
        self.assertEqual(data["symbols"]["NVDA"]["last_date"], "2026-11-19")

    def test_due_on_a_new_session_or_a_new_symbol(self):
        data = volume.refresh(self.db, bars_fn=lambda s: bars(40, date(2026, 11, 20)), now=self.now, out=self.out)
        groups = volume.tracked_symbols(self.db)
        self.assertFalse(volume.due(data, groups, now=self.now))
        self.assertTrue(volume.due(data, groups, now=datetime(2026, 11, 23, 16, 30, tzinfo=ET)))
        pf.add_watch("TSM", path=self.db)
        self.assertTrue(volume.due(data, volume.tracked_symbols(self.db), now=self.now))
        self.assertTrue(volume.due({}, groups, now=self.now))


class CommodityIndexTests(unittest.TestCase):
    def setUp(self):
        self.db = Path(tempfile.mkdtemp()) / "p.db"

    def test_gold_bitcoin_and_brent_are_added_once(self):
        shown = global_markets._index_symbols(self.db)
        self.assertEqual(shown[-3:], ["@GC.1", "BTC.CB=", "@LCO.1"])
        global_markets.set_index("BTC.CB=", False, self.db)        # removed by the user: it stays removed
        self.assertNotIn("BTC.CB=", global_markets._index_symbols(self.db))
        self.assertIn("@CL.1", {s for s, _, _ in global_markets.INDEX_CATALOG})   # WTI: in the catalog, not shown

    def test_a_saved_list_keeps_its_order(self):
        global_markets._index_symbols(self.db)
        conn = pf._conn(self.db)
        conn.execute("DELETE FROM index_watch_added")
        conn.execute("DELETE FROM index_watch WHERE symbol IN ('@GC.1', 'BTC.CB=', '@LCO.1')")
        conn.commit()
        conn.close()
        shown = global_markets._index_symbols(self.db)
        self.assertEqual(shown[:5], global_markets.INDEX_DEFAULT)
        self.assertEqual(shown[5:], ["@GC.1", "BTC.CB=", "@LCO.1"])


if __name__ == "__main__":
    unittest.main()
