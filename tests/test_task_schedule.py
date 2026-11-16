"""task_schedule: cron expansion in the DAG's time zone and the frequent-task grouping. No Airflow needed."""
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from unittest import mock
from markets import task_schedule as ts

UTC, DUB = ZoneInfo("UTC"), ZoneInfo("Europe/Dublin")


class CronTimes(unittest.TestCase):
    def day(self, y, m, d, tz=DUB):
        s = datetime(y, m, d, tzinfo=tz)
        return s, s + timedelta(days=1)

    def test_every_30_minutes(self):
        self.assertEqual(len(ts.cron_times("*/30 * * * *", UTC, *self.day(2026, 11, 4))), 48)

    def test_weekday_list_and_range(self):
        t = ts.cron_times("30 13,17,21 * * 1-5", UTC, *self.day(2026, 11, 4))      # Wednesday
        self.assertEqual([x.astimezone(DUB).strftime("%H:%M") for x in t], ["13:30", "17:30", "21:30"])
        self.assertEqual(ts.cron_times("30 13,17,21 * * 1-5", UTC, *self.day(2026, 11, 7)), [])   # Saturday

    def test_sunday_is_0_and_7(self):
        self.assertEqual(len(ts.cron_times("0 4 * * 0", UTC, *self.day(2026, 11, 8))), 1)
        self.assertEqual(len(ts.cron_times("0 4 * * 7", UTC, *self.day(2026, 11, 8))), 1)

    def test_dublin_zone_across_summer_time(self):
        t = ts.cron_times("30 11 * * *", DUB, *self.day(2026, 7, 1))
        self.assertEqual(t[0].astimezone(UTC).strftime("%H:%M"), "10:30")


class Grouping(unittest.TestCase):
    def test_frequent_runs_become_one_row(self):
        now = datetime(2026, 11, 4, 12, tzinfo=UTC)
        es = [ts._entry("airflow", "q", "Q", datetime(2026, 11, 4, h, tzinfo=UTC), "done" if h < 12 else "pending", now)
              for h in range(24)]
        rows = ts._group(es)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["counts"], {"done": 12, "failed": 0, "running": 0, "pending": 12})
        self.assertEqual(rows[0]["state"], "partial")



    def test_job_agent_names_and_dedup(self):
        self.assertIn("job_agent_morning", ts.NAMES)
        self.assertIn("job_agent_afternoon", ts.NAMES)
        self.assertIn("gdelt_hourly", ts.NAMES)
        now = datetime(2026, 11, 14, 12, tzinfo=UTC)
        fake_af = [
            ts._entry("airflow", "job_agent_morning", ts.NAMES["job_agent_morning"], datetime(2026, 11, 14, 11, tzinfo=DUB), "done", now, dag_id="job_agent_morning"),
            ts._entry("airflow", "job_agent_afternoon", ts.NAMES["job_agent_afternoon"], datetime(2026, 11, 14, 16, tzinfo=DUB), "pending", now, dag_id="job_agent_afternoon"),
        ]
        with mock.patch.object(ts, "_airflow", return_value=fake_af), \
             mock.patch.object(ts, "_briefs", return_value=[]), \
             mock.patch.object(ts, "_legacy_job_agent_launchd_loaded", return_value=True):
            res = ts.schedule(now)
        today_items = res["days"][0]["items"]
        self.assertEqual([i["source"] for i in today_items], ["airflow", "airflow"])


if __name__ == "__main__":
    unittest.main()
