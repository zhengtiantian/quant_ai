"""Pipeline freshness: lag rules per stage and the once-a-day alert. Mongo is faked."""

import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import pipeline_health as ph  # noqa: E402

NOW = datetime(2026, 10, 22, 17, 0, tzinfo=timezone.utc)          # a Thursday


class JudgeTest(unittest.TestCase):
    def test_weekday_lag_ignores_weekends(self):
        self.assertEqual(ph.weekdays_between(date(2026, 10, 16), date(2026, 10, 19)), 1)   # Fri -> Mon
        self.assertIsNone(ph.judge(datetime(2026, 10, 20, tzinfo=timezone.utc), {"weekdays": 2}, NOW))
        self.assertEqual(ph.judge(datetime(2026, 10, 9, tzinfo=timezone.utc), {"weekdays": 2}, NOW), "9 weekdays behind")

    def test_hours_and_days(self):
        self.assertIsNone(ph.judge(NOW - timedelta(hours=5), {"hours": 6}, NOW))
        self.assertIn("h old", ph.judge(NOW - timedelta(days=8), {"hours": 6}, NOW))
        self.assertIn("days old", ph.judge(NOW - timedelta(days=3), {"days": 2}, NOW))
        self.assertEqual(ph.judge(None, {"days": 2}, NOW), "no data")


class FakeColl:
    def __init__(self, doc):
        self.doc = doc

    def find_one(self, *a, **kw):
        return self.doc


class FakeOid:
    def __init__(self, t):
        self.generation_time = t


class CheckTest(unittest.TestCase):
    def client(self, signals_date):
        docs = {"news_articles": {"_id": FakeOid(NOW - timedelta(hours=1))},
                "news_articles_company_matched_v2": {"_id": FakeOid(NOW - timedelta(hours=20))},
                "stock_prices_history": {"timestamp": NOW - timedelta(days=1)},
                "macro_indicators": {"date": "2026-10-22"},
                "daily_symbol_features": {"trade_date": "2026-10-21"},
                "daily_signals": {"trade_date": signals_date}}
        db = {k: FakeColl(v) for k, v in docs.items()}
        return {"quant_data": db}

    def test_healthy_and_stale(self):
        ok = ph.check(NOW, client=self.client("2026-10-21"))
        self.assertTrue(ok["ok"], ok["problems"])
        bad = ph.check(NOW, client=self.client("2026-07-17"))
        self.assertFalse(bad["ok"])
        self.assertTrue(bad["problems"][0].startswith("signals:"))
        self.assertEqual(ph.latest()["problems"], bad["problems"])

    def test_unreachable_mongo_is_a_problem(self):
        class Boom(dict):
            def __getitem__(self, k):
                raise ConnectionError("refused")
        r = ph.check(NOW, client=Boom())
        self.assertFalse(r["ok"])
        self.assertIn("MongoDB unreachable", r["problems"][0])

    def test_alert_once_a_day(self):
        path = Path(tempfile.mkdtemp()) / "m.db"
        bad = {"ok": False, "problems": ["signals: 9 weekdays behind"]}
        with mock.patch.object(ph, "check", return_value=bad), mock.patch.object(ph, "_notify") as n:
            ph.check_and_alert(path)
            ph.check_and_alert(path)
        self.assertEqual(n.call_count, 1)



    def test_job_agent_freshness_rules(self):
        import json, sqlite3
        tmp = Path(tempfile.mkdtemp())
        (tmp / "scrape_state.json").write_text(json.dumps({"last_success": "2026-10-22T11:00:00"}))
        conn = sqlite3.connect(tmp / "job_history.db")
        conn.execute("CREATE TABLE global_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE china_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO global_meta VALUES ('eu_refreshed_at', '2026-10-22 16:00:00')")
        conn.execute("INSERT INTO china_meta VALUES ('foreign_refreshed_at', '2026-10-20 08:00:00')")
        conn.commit()
        conn.close()

        res = ph.check(NOW, client=self.client("2026-10-21"), job_agent_dir=tmp)
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["problems"]), 1)
        self.assertTrue(res["problems"][0].startswith("job_agent (china):"))


if __name__ == "__main__":
    unittest.main()
