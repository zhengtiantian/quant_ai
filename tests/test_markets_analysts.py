"""Analysts' ratings and price targets per stock (markets/analysts.py); no network, no news store."""

import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import analysts as a  # noqa: E402


def _at(days_ago: float) -> str:
    return (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S")


def _action(firm, days_ago, action="main", frm=None, to="Buy", pt=None, prior=None, pt_action=None):
    return {"firm": firm, "graded_at": _at(days_ago), "action": action, "from_grade": frm, "to_grade": to,
            "pt_action": pt_action, "pt_current": pt, "pt_prior": prior}


SUMMARY = {"target_mean": 320.0, "target_median": 315.0, "target_high": 500.0, "target_low": 180.0, "price": 175.0,
           "buy": 50, "hold": 5, "sell": 1, "analysts": 56}


class WordsTest(unittest.TestCase):
    def test_bucket(self):
        for g, b in [("Overweight", "Buy"), ("Outperform", "Buy"), ("Strong Buy", "Buy"), ("Equal-Weight", "Hold"),
                     ("Market Perform", "Hold"), ("Neutral", "Hold"), ("Underweight", "Sell"), ("Reduce", "Sell"),
                     ("", None), (None, None), ("Speculative", None)]:
            self.assertEqual(a.bucket(g), b, g)

    def test_describe_and_direction(self):
        r = _action("Baird", 0, "down", "Outperform", "Neutral", 200, 500, "Lowers")
        self.assertEqual(a.describe(r), "Downgraded to Neutral from Outperform; cut target to $200 from $500")
        self.assertEqual(a.direction(r), -1)
        r = _action("Mizuho", 0, "main", "Outperform", "Outperform", 315, 300, "Raises")
        self.assertEqual(a.describe(r), "Kept Outperform; raised target to $315 from $300")
        self.assertEqual(a.direction(r), 1)
        r = _action("Piper Sandler", 0, "init", None, "Overweight", 300, None, "Announces")
        self.assertEqual(a.describe(r), "Started coverage at Overweight; target $300")
        self.assertEqual(a.direction(r), 0)
        # an upgrade with a lower target counts as an upgrade
        self.assertEqual(a.direction(_action("X", 0, "up", "Hold", "Buy", 90, 100)), 1)

    def test_small_targets_keep_cents(self):
        self.assertEqual(a.describe(_action("X", 0, "main", "Buy", "Buy", 12.5, 11.0)), "Kept Buy; raised target to $12.50 from $11.00")


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "m.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_refetch_never_duplicates_and_newest_per_firm_wins(self):
        data = {"actions": [_action("JP Morgan", 30, pt=300, prior=290), _action("Baird", 3, "main", pt=500, prior=480)],
                "summary": SUMMARY}
        self.assertEqual(a.store("NVDA", data, self.db), 2)
        self.assertEqual(a.store("NVDA", data, self.db), 0)
        # the same firm spelt another way, later: it replaces what is shown for that firm
        data["actions"].append(_action("JPMorgan", 1, "down", "Overweight", "Neutral", 250, 300))
        data["actions"].append(_action("Baird", 0.5, "down", "Outperform", "Neutral", 200, 500))
        self.assertEqual(a.store("NVDA", data, self.db), 2)
        firms = {f["firm"]: f for f in a.detail("NVDA", self.db)["firms"]}
        self.assertEqual(set(firms), {"JPMorgan", "Baird"})
        self.assertEqual(firms["Baird"]["target"], 200)
        self.assertEqual(firms["Baird"]["rating_bucket"], "Hold")
        self.assertEqual(firms["JPMorgan"]["direction"], -1)

    def test_stale_firms_go_last(self):
        a.store("AAPL", {"actions": [_action("Old Firm", 900, pt=100), _action("New Firm", 10, pt=200)], "summary": SUMMARY}, self.db)
        firms = a.detail("AAPL", self.db)["firms"]
        self.assertEqual([f["firm"] for f in firms], ["New Firm", "Old Firm"])
        self.assertTrue(firms[1]["stale"])

    def test_summary_flags_a_recent_change(self):
        a.store("NVDA", {"actions": [_action("Baird", 1, "down", "Outperform", "Neutral", 200, 500),
                                     _action("Mizuho", 20, "main", "Buy", "Buy", 315, 300)], "summary": SUMMARY}, self.db)
        a.store("AAPL", {"actions": [_action("Barclays", 20, "main", "Buy", "Buy", 300, 290)], "summary": SUMMARY}, self.db)
        s = a.summaries(["NVDA", "AAPL", "MSFT"], self.db)
        self.assertNotIn("MSFT", s)                                   # never fetched: no entry
        self.assertEqual(s["NVDA"]["latest"]["firm"], "Baird")
        self.assertEqual(s["NVDA"]["latest"]["direction"], -1)
        self.assertEqual((s["NVDA"]["up_30d"], s["NVDA"]["down_30d"]), (1, 1))
        self.assertIsNone(s["AAPL"]["latest"])                       # 20 days old: not flagged
        self.assertEqual(s["NVDA"]["target_mean"], 320.0)

    def test_rate_limit_pauses_the_pass(self):
        with mock.patch.object(a, "fetch", side_effect=a.RateLimited("429")), \
             mock.patch.object(a, "followed_symbols", return_value=["NVDA", "AAPL"]), mock.patch.object(a.time, "sleep"):
            a._paused_until["t"] = 0
            done = a.run_due(self.db)
        self.assertEqual([d["status"] for d in done], ["rate_limited"])
        self.assertGreater(a._paused_until["t"], 0)
        a._paused_until["t"] = 0


class NewsMatchTest(unittest.TestCase):
    def test_firm_patterns(self):
        cases = [("JP Morgan", "JPMorgan cuts Nvidia price target to $250", True),
                 ("JP Morgan", "J.P. Morgan downgrades Apple to Neutral", True),
                 ("Bank of America", "BofA raises Microsoft price target", True),
                 ("Argus Research", "Argus upgrades Apple to Buy", True),
                 ("RBC Capital", "RBC Capital Reiterates Outperform on NVIDIA", True),
                 ("UBS", "Microsoft subsidiary wins contract", False)]
        for firm, title, hit in cases:
            pat = a._firm_pattern(firm)
            found = bool(pat.search(a._normal_title(title)) and a._RATING_WORDS.search(title))
            self.assertEqual(found, hit, (firm, title))


class ScheduleTest(unittest.TestCase):
    def test_interval(self):
        d = ZoneInfo("Europe/Dublin")
        self.assertEqual(a.interval(datetime(2026, 11, 11, 12, 0, tzinfo=d)), 1800)    # Wednesday, before the US open
        self.assertEqual(a.interval(datetime(2026, 11, 11, 18, 0, tzinfo=d)), 3600)
        self.assertEqual(a.interval(datetime(2026, 11, 11, 2, 0, tzinfo=d)), 6 * 3600)
        self.assertEqual(a.interval(datetime(2026, 11, 14, 12, 0, tzinfo=d)), 6 * 3600)  # Saturday


if __name__ == "__main__":
    unittest.main()
