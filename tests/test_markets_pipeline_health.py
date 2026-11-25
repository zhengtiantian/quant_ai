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

    def test_minutes_hours_and_days(self):
        self.assertIsNone(ph.judge(NOW - timedelta(minutes=8), {"minutes": 10}, NOW))
        self.assertIn("min old", ph.judge(NOW - timedelta(minutes=15), {"minutes": 10}, NOW))
        self.assertIsNone(ph.judge(NOW - timedelta(hours=5), {"hours": 6}, NOW))
        self.assertIn("h old", ph.judge(NOW - timedelta(days=8), {"hours": 6}, NOW))
        self.assertIn("days old", ph.judge(NOW - timedelta(days=3), {"days": 2}, NOW))
        self.assertEqual(ph.judge(None, {"days": 2}, NOW), "no data")


class FakeColl:
    def __init__(self, doc, distinct_syms=None):
        self.doc = doc
        self._distinct_syms = distinct_syms

    def find_one(self, *a, **kw):
        return self.doc

    def distinct(self, field, query=None):
        if self._distinct_syms is None:
            return [f"SYM{i}" for i in range(60)]
        return self._distinct_syms


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

    def test_alert_per_stage_dedupe(self):
        path = Path(tempfile.mkdtemp()) / "m.db"
        bad1 = {"ok": False, "problems": ["signals: 9 weekdays behind"]}
        bad2 = {"ok": False, "problems": ["signals: 9 weekdays behind", "qdrant news_v1: newest is 3.0 days old"]}
        with mock.patch.object(ph, "check", side_effect=[bad1, bad1, bad2]), mock.patch.object(ph, "_notify") as n:
            ph.check_and_alert(path)
            ph.check_and_alert(path)  # same failing stage today -> muted
            ph.check_and_alert(path)  # new failing stage today -> fires
        self.assertEqual(n.call_count, 2)
        self.assertIn("qdrant news_v1", n.call_args[0][1])

    def test_prices_symbol_coverage(self):
        c = self.client("2026-10-21")
        c["quant_data"]["stock_prices_history"] = FakeColl({"timestamp": NOW - timedelta(days=1)}, distinct_syms=["AAPL", "MSFT"])
        res = ph.check(NOW, client=c)
        self.assertFalse(res["ok"])
        self.assertTrue(any("only 2 symbols" in p for p in res["problems"]))

    def test_qdrant_and_data4t_and_airflow_checks(self):
        import json, os, io
        tmp = Path(tempfile.mkdtemp())
        cp = tmp / ".index_news_checkpoint.json"
        oid_hex = format(int((NOW - timedelta(hours=4)).timestamp()), "08x") + "0" * 16
        cp.write_text(json.dumps({"last_id": oid_hex}), encoding="utf-8")

        data_dir = tmp / "Data4T"
        (data_dir / "daily_advice").mkdir(parents=True)
        (data_dir / "filing_changes").mkdir(parents=True)
        (data_dir / "market_snapshots" / "options").mkdir(parents=True)
        ts = NOW.timestamp()
        for f in [
            data_dir / "daily_advice" / "latest.json",
            data_dir / "filing_changes" / "latest.json",
            data_dir / "market_snapshots" / "options" / "2026-10-21.parquet",
        ]:
            f.write_text("ok", encoding="utf-8")
            os.utime(f, (ts, ts))

        class FakeResp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        q_body = json.dumps({"result": {"status": "green", "points_count": 1000}}).encode("utf-8")

        class FakeCursor:
            def __init__(self):
                self._q = ""
            def execute(self, q, params=None):
                self._q = q
            def fetchone(self):
                return (NOW - timedelta(minutes=2),)
            def fetchall(self):
                if "task_instance" in self._q:
                    return [("filing_changes", "risk_factor_changes", 1)]
                if "dag_run" in self._q:
                    return [("weekly_inst13f_holdings", NOW - timedelta(days=2)), ("weekly_model_training", NOW - timedelta(days=2))]
                return [("filing_changes",)]

        class FakePG:
            def cursor(self): return FakeCursor()
            def close(self): pass

        with mock.patch("urllib.request.urlopen", return_value=FakeResp(q_body)), \
             mock.patch.object(ph, "_open_airflow_conn", return_value=FakePG()):
            res = ph.check(
                NOW,
                client=self.client("2026-10-21"),
                qdrant_url="http://localhost:26333",
                qdrant_checkpoint=cp,
                data_dir=data_dir,
                airflow_conn="postgresql://airflow:airflow@127.0.0.1:25432/airflow",
            )
        self.assertFalse(res["ok"])
        self.assertTrue(any(p.startswith("airflow failures (24h):") for p in res["problems"]))
        self.assertTrue(any(p.startswith("airflow paused:") for p in res["problems"]))
        self.assertFalse(any(p.startswith("qdrant news_v1:") for p in res["problems"]))
        self.assertFalse(any(p.startswith("daily advice:") for p in res["problems"]))



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
