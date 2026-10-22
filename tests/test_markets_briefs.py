"""Briefs: when each one is due, one writer per brief, parsing Claude's reply, and the stored
outcome of a run. Claude and the market data are stubbed: no network, no quota."""

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import briefs  # noqa: E402

DUB, NY = ZoneInfo("Europe/Dublin"), ZoneInfo("America/New_York")

REPLY = """=====HEADLINE_ZH=====
芯片股领跌，亚洲普跌
=====HEADLINE_EN=====
Chips lead a broad Asian sell-off
=====BODY_ZH=====
## 一句话结论
芯片股领跌。
=====BODY_EN=====
## Bottom line
Chips led.
=====SOURCES=====
- [Reuters: Asia stocks fall](https://www.reuters.com/x)
- [CNBC](https://www.cnbc.com/y)
"""


class DueTest(unittest.TestCase):
    def test_asia_europe_from_14_dublin_to_end_of_day_weekdays(self):
        self.assertIsNone(briefs.due_session("asia_europe", datetime(2026, 10, 22, 13, 59, tzinfo=DUB)))
        self.assertEqual(briefs.due_session("asia_europe", datetime(2026, 10, 22, 14, 0, tzinfo=DUB)), "2026-10-22")
        self.assertEqual(briefs.due_session("asia_europe", datetime(2026, 10, 22, 23, 50, tzinfo=DUB)), "2026-10-22")
        self.assertIsNone(briefs.due_session("asia_europe", datetime(2026, 10, 24, 15, 0, tzinfo=DUB)))   # Saturday

    def test_us_after_2005_until_next_premarket(self):
        self.assertIsNone(briefs.due_session("us", datetime(2026, 10, 22, 20, 4, tzinfo=NY)))
        self.assertEqual(briefs.due_session("us", datetime(2026, 10, 22, 20, 5, tzinfo=NY)), "2026-10-22")
        self.assertEqual(briefs.due_session("us", datetime(2026, 10, 23, 3, 59, tzinfo=NY)), "2026-10-22")
        self.assertIsNone(briefs.due_session("us", datetime(2026, 10, 23, 4, 0, tzinfo=NY)))
        # Friday's brief can still be caught up over the weekend, until Monday's pre-market
        self.assertEqual(briefs.due_session("us", datetime(2026, 10, 25, 12, 0, tzinfo=NY)), "2026-10-23")
        self.assertIsNone(briefs.due_session("us", datetime(2026, 10, 26, 4, 1, tzinfo=NY)))

    def test_us_brief_time_in_dublin_follows_both_dst_switches(self):
        # 20:05 New York is 01:05 Dublin most of the year, 00:05 in the weeks the US switches later
        late_oct = datetime(2026, 10, 27, 0, 30, tzinfo=DUB)            # Europe on GMT, US still on EDT
        self.assertEqual(briefs.due_session("us", late_oct), "2026-10-26")


class ParseTest(unittest.TestCase):
    def test_blocks_and_sources(self):
        p = briefs.parse_reply("preamble\n" + REPLY)
        self.assertEqual(p["headline_en"], "Chips lead a broad Asian sell-off")
        self.assertIn("## 一句话结论", p["body_zh"])
        self.assertEqual([s["url"] for s in p["sources"]], ["https://www.reuters.com/x", "https://www.cnbc.com/y"])

    def test_missing_block_is_an_error(self):
        with self.assertRaises(ValueError):
            briefs.parse_reply(REPLY.split("=====BODY_EN=====")[0])


class GenerateTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "m.db"
        for name, value in (("snapshot", {"macro": {}}), ("_notify", None)):
            p = mock.patch.object(briefs, name, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def claude(self, reply=REPLY, error=None):
        if error:
            return mock.patch.object(briefs, "run_claude", side_effect=RuntimeError(error))
        return mock.patch.object(briefs, "run_claude", return_value={"result": reply, "total_cost_usd": 0.4,
                                                                      "modelUsage": {"claude-opus-5-5": {}}})

    def test_success_is_stored_once(self):
        after = briefs.scheduled_at("asia_europe", "2026-10-22") + 60
        with self.claude() as c, mock.patch.object(briefs.time, "time", return_value=after):
            rid = briefs.generate("asia_europe", "2026-10-22", path=self.path)
            self.assertIsNone(briefs.generate("asia_europe", "2026-10-22", path=self.path))   # already done
        self.assertEqual(c.call_count, 1)
        b = briefs.get_brief(rid, self.path)
        self.assertEqual((b["status"], b["headline_zh"], len(b["sources"]), b["model"]),
                         ("done", "芯片股领跌，亚洲普跌", 2, "claude-opus-5-5"))

    def test_failure_retries_after_a_pause_then_gives_up(self):
        with self.claude(error="usage limit reached"):
            rid = briefs.generate("us", "2026-10-22", path=self.path)
            self.assertEqual(briefs.get_brief(rid, self.path)["status"], "failed")
            self.assertIsNone(briefs.generate("us", "2026-10-22", path=self.path))           # too soon
            with mock.patch.object(briefs, "RETRY_AFTER", 0):
                briefs.generate("us", "2026-10-22", path=self.path)
                briefs.generate("us", "2026-10-22", path=self.path)
                self.assertIsNone(briefs.generate("us", "2026-10-22", path=self.path))       # 3 attempts used
        b = briefs.get_brief(rid, self.path)
        self.assertEqual((b["attempts"], b["status"]), (3, "failed"))
        self.assertIn("usage limit", b["error"])
        with self.claude():                                                                   # by hand: allowed again
            briefs.generate("us", "2026-10-22", trigger="manual", force=True, path=self.path)
        self.assertEqual(briefs.get_brief(rid, self.path)["status"], "done")

    def test_a_manual_brief_before_the_scheduled_time_is_rewritten_on_schedule(self):
        sched = briefs.scheduled_at("asia_europe", "2026-10-22")
        with self.claude(), mock.patch.object(briefs.time, "time", return_value=sched - 7200):
            briefs.generate("asia_europe", "2026-10-22", trigger="manual", force=True, path=self.path)
        with self.claude() as c, mock.patch.object(briefs.time, "time", return_value=sched + 60):
            rid = briefs.generate("asia_europe", "2026-10-22", path=self.path)
            self.assertIsNone(briefs.generate("asia_europe", "2026-10-22", path=self.path))   # now it counts
        self.assertEqual(c.call_count, 1)
        self.assertEqual(briefs.get_brief(rid, self.path)["trigger"], "schedule")

    def test_a_running_brief_is_not_started_twice(self):
        self.assertTrue(briefs._claim("us", "2026-10-22", "schedule", False, self.path))
        self.assertFalse(briefs._claim("us", "2026-10-22", "manual", True, self.path))
        with mock.patch.object(briefs.time, "time", return_value=briefs.time.time() + briefs.STALE_RUNNING + 1):
            self.assertTrue(briefs._claim("us", "2026-10-22", "schedule", False, self.path))   # crashed run

    def test_unparseable_reply_keeps_the_raw_text_for_debugging(self):
        with self.claude(reply="sorry, no"):
            rid = briefs.generate("asia_europe", "2026-10-22", path=self.path)
        conn = briefs._conn(self.path)
        row = conn.execute("SELECT status, raw FROM briefs WHERE id = ?", (rid,)).fetchone()
        conn.close()
        self.assertEqual(tuple(row), ("failed", "sorry, no"))

    def test_list_is_newest_first(self):
        with self.claude():
            briefs.generate("asia_europe", "2026-10-21", path=self.path)
            briefs.generate("us", "2026-10-22", path=self.path)
        self.assertEqual([b["session_date"] for b in briefs.list_briefs(path=self.path)], ["2026-10-22", "2026-10-21"])
        self.assertEqual(len(briefs.list_briefs("us", path=self.path)), 1)


class FindClaudeTest(unittest.TestCase):
    def test_newest_vscode_extension_wins(self):
        fake = ["/x/anthropic.claude-code-2.1.99-darwin-arm64/resources/native-binary/claude",
                "/x/anthropic.claude-code-2.1.306-darwin-arm64/resources/native-binary/claude"]
        with mock.patch.dict(briefs.os.environ, {"CLAUDE_BIN": ""}), \
             mock.patch.object(briefs.shutil, "which", return_value=None), \
             mock.patch.object(briefs.glob, "glob", return_value=fake), \
             mock.patch.object(briefs.os, "access", side_effect=lambda p, m: p in fake):
            self.assertEqual(briefs.find_claude(), fake[1])


if __name__ == "__main__":
    unittest.main()
