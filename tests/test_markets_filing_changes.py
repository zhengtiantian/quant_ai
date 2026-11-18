"""Q.17.5's flag: read from quant_data's latest.json, shown while recent, logged once as a blocking hint."""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from markets import filing_changes as fc
from markets import hint_scores

NY = ZoneInfo("America/New_York")


def chart(days, closes):
    return lambda sym, rng, path=None: {"candles": [{"time": d, "close": c} for d, c in zip(days, closes)]}


class FilingChangesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file = Path(self.tmp.name) / "latest.json"
        self.file.write_text(json.dumps({"generated_at": "2026-11-18T23:00:00+00:00", "items": {
            "AAA": {"form": "10-Q", "accession": "0000000001-26-000001", "accepted": "2026-11-05T21:30:00+00:00", "filing_date": "2026-11-05",
                    "sim_1a": 0.71, "cut": 0.8813, "len_ratio": 1.8, "direction": "grown", "flag": True, "url": "https://www.sec.gov/x"},
            "BBB": {"form": "10-K", "accepted": "2026-11-01T12:00:00+00:00", "sim_1a": 0.999, "cut": 0.9967, "flag": False},
            "CCC": {"form": "10-Q", "accepted": "2025-11-01T12:00:00+00:00", "sim_1a": 0.5, "cut": 0.8813, "flag": True}}}))
        self.old = fc.FILE
        fc.FILE, fc._cache["mtime"] = self.file, None
        self.db = Path(self.tmp.name) / "markets.db"

    def tearDown(self):
        fc.FILE, fc._cache["mtime"] = self.old, None
        self.tmp.cleanup()

    def test_only_recent_flags_are_shown(self):
        now = datetime(2026, 11, 18, 12, tzinfo=NY)
        self.assertIsNotNone(fc.get("AAA", now))
        self.assertIsNone(fc.get("BBB", now))                          # not flagged
        self.assertIsNone(fc.get("CCC", now))                          # over 180 days old
        out = fc.for_symbols(["AAA", "BBB"], now)
        self.assertEqual(list(out["items"]), ["AAA"])
        self.assertIn("1.9% (2018-2023) and 3.2% (2024-2026)", out["items"]["AAA"]["text"]["en"])

    def test_logged_once_on_the_first_tradable_session(self):
        # accepted 16:30 New York on 11-05: first tradable session is 11-06
        state = {"holdings": [{"symbol": "AAA"}], "watchlist": []}
        ch = chart(["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"], [10.0, 10.5, 9.8, 9.9])
        self.assertEqual(fc.record_due(self.db, state, ch), ["AAA"])
        self.assertEqual(fc.record_due(self.db, state, ch), [])         # once
        conn = hint_scores._conn(self.db)
        row = conn.execute("SELECT * FROM hint_log WHERE kind = 'filing_change'").fetchone()
        conn.close()
        self.assertEqual((row["symbol"], row["day"], row["ref_price"], json.loads(row["codes"])), ("AAA", "2026-11-06", 9.8, ["grown"]))


if __name__ == "__main__":
    unittest.main()
