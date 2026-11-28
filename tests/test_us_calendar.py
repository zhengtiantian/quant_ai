"""The US market calendar (markets/us_calendar.py) and its use in the holdings' session and trading
day, 2026-11-26: on Thanksgiving the holdings show the previous session's after-hours price."""
import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import portfolio as pf  # noqa: E402
from markets import us_calendar as cal  # noqa: E402
from markets import volume  # noqa: E402

ET = pf.US_EAST


def et(*args) -> datetime:
    return datetime(*args, tzinfo=ET)


class CalendarTests(unittest.TestCase):
    def test_published_holidays(self):
        # NYSE's published schedules
        self.assertEqual(sorted(cal.holidays(2026)), [date(2026, m, d) for m, d in (
            (1, 1), (1, 19), (2, 16), (4, 3), (5, 25), (6, 19), (7, 3), (9, 7), (11, 26), (12, 25))])
        self.assertEqual(sorted(cal.holidays(2027)), [date(2027, m, d) for m, d in (
            (1, 1), (1, 18), (2, 15), (3, 26), (5, 31), (6, 18), (7, 5), (9, 6), (11, 25), (12, 24))])
        self.assertIn(date(2025, 4, 18), cal.holidays(2025))           # Good Friday

    def test_new_year_on_a_saturday_is_not_moved(self):
        self.assertNotIn(date(2021, 12, 31), cal.holidays(2021))
        self.assertTrue(cal.is_trading_day(date(2021, 12, 31)))
        self.assertIn(date(2023, 1, 2), cal.holidays(2023))            # Sunday -> Monday

    def test_early_closes(self):
        self.assertEqual(sorted(cal.early_closes(2026)), [date(2026, 11, 27), date(2026, 12, 24)])
        self.assertEqual(sorted(cal.early_closes(2025)), [date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24)])
        self.assertEqual((cal.close_hour(date(2026, 11, 27)), cal.post_end_hour(date(2026, 11, 27))), (13, 17))
        self.assertEqual(cal.close_hour(date(2026, 11, 25)), 16)

    def test_neighbouring_trading_days(self):
        self.assertEqual(cal.previous_trading_day(date(2026, 11, 27)), date(2026, 11, 25))
        self.assertEqual(cal.next_trading_day(date(2026, 11, 25)), date(2026, 11, 27))
        self.assertEqual(cal.next_trading_day(date(2026, 12, 24)), date(2026, 12, 28))


class SessionTests(unittest.TestCase):
    def test_a_holiday_is_closed_all_day(self):
        for h in (5, 7, 10, 15, 18):
            self.assertEqual(pf.session_now(et(2026, 11, 26, h, 0)), "closed")
        self.assertEqual(pf.session_now(et(2026, 11, 25, 6, 46)), "pre")

    def test_an_early_close_day(self):
        self.assertEqual(pf.session_now(et(2026, 11, 27, 12, 59)), "regular")
        self.assertEqual(pf.session_now(et(2026, 11, 27, 13, 0)), "post")
        self.assertEqual(pf.session_now(et(2026, 11, 27, 17, 0)), "closed")
        self.assertEqual(pf.session_now(et(2026, 11, 25, 16, 30)), "post")

    def test_trading_day_on_and_after_a_holiday(self):
        self.assertEqual(pf.trading_day(et(2026, 11, 26, 10, 0)), date(2026, 11, 25))
        self.assertEqual(pf.trading_day(et(2026, 11, 27, 3, 59)), date(2026, 11, 25))
        self.assertEqual(pf.trading_day(et(2026, 11, 27, 4, 0)), date(2026, 11, 27))
        self.assertEqual(pf.trading_day(et(2026, 11, 28, 10, 0)), date(2026, 11, 27))

    def test_next_session_skips_the_holiday(self):
        self.assertEqual(pf.seconds_until_next_session(et(2026, 11, 25, 21, 0)), 31 * 3600)   # to 11/27 04:00

    def test_after_hours_end_after_an_early_close(self):
        self.assertEqual(pf._post_close_cutoff(et(2026, 11, 27, 18, 0)), et(2026, 11, 27, 17, 0).timestamp())
        self.assertEqual(pf._post_close_cutoff(et(2026, 11, 26, 10, 0)), et(2026, 11, 25, 20, 0).timestamp())

    def test_volume_session(self):
        self.assertEqual(volume.last_complete_session(et(2026, 11, 26, 18, 0)), date(2026, 11, 25))
        self.assertEqual(volume.last_complete_session(et(2026, 11, 27, 13, 5)), date(2026, 11, 27))


class HolidayQuoteTests(unittest.TestCase):
    def test_thanksgiving_shows_the_previous_after_hours_price(self):
        payload = {"FormattedQuoteResult": {"FormattedQuote": [{
            "symbol": "INTC", "code": "0", "name": "Intel Corp",
            "last": "54.47", "change": "+0.50", "change_pct": "+0.93",
            "last_time": "2026-11-25T16:00:00.000-0500", "curmktstatus": "POST_MKT", "volume": "90000000",
            "ExtendedMktQuote": {"type": "POST_MKT", "last": "54.85", "change": "+0.38", "change_pct": "+0.70",
                                 "last_time": "2026-11-25T19:59:50.000-0500"},
        }]}}
        thanksgiving = et(2026, 11, 26, 6, 46)
        with mock.patch.object(pf, "_get_json", return_value=payload), \
             mock.patch.object(pf, "datetime") as mock_dt:
            mock_dt.now.return_value = thanksgiving
            q = pf.fetch_quotes_cnbc([("INTC", "stocks")])["INTC"]
        self.assertEqual((q["session"], q["price"], q["regular_price"], q["price_basis"]),
                         ("closed", 54.85, 54.47, "post_close"))
        self.assertEqual(q["session_change"], 0.38)


if __name__ == "__main__":
    unittest.main()
