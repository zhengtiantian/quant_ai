"""Freshness watchdog for the quant_data pipeline (PROJECT_PLAN Q.2 / J.4).

The daily pipeline runs under the host Airflow scheduler, whose metadata DB and data live in
Docker. In October 2026 Docker was off for a week: the scheduler crash-looped, nothing was
collected, and nobody noticed. Airflow's failure callback (quant_data/_host_common.py) only
fires when a task runs and fails; this catches the other case, when nothing runs at all.

Every CHECK_EVERY seconds quant_ai (always on) reads the newest date of each stage's output in
MongoDB and compares it with what a healthy pipeline would have by now. A problem is notified
once a day; the latest result is served at /api/markets/pipeline/health.
"""

import json
import os
import plistlib
import sqlite3
import subprocess
import threading
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from . import budget_link

CHECK_EVERY = 30 * 60
QUANT_DATA_ENV = Path(os.getenv("QUANT_DATA_ENV", "/Users/xiz/Quant_trade/quant_data/.env"))
JOB_AGENT_DIR = Path(os.getenv("JOB_AGENT_APP_DIR", os.path.expanduser("~/Library/Application Support/job_agent")))
QDRANT_URL = os.getenv("PIPELINE_QDRANT_URL", os.getenv("QDRANT_URL", "http://127.0.0.1:26333"))
QDRANT_CHECKPOINT = Path(
    os.getenv("PIPELINE_QDRANT_CHECKPOINT", "/Users/xiz/Quant_trade/quant_ai/tools/.index_news_checkpoint.json")
)
DATA_DIR = Path(os.getenv("PIPELINE_DATA_DIR", "/Volumes/Data4T"))
AIRFLOW_PLIST = Path(
    os.getenv("PIPELINE_AIRFLOW_PLIST", os.path.expanduser("~/Library/LaunchAgents/com.quant.airflow-scheduler.plist"))
)
MIN_PRICE_SYMBOLS = int(os.getenv("PIPELINE_MIN_PRICE_SYMBOLS", "50"))
DUBLIN_TZ = ZoneInfo("Europe/Dublin")

CRITICAL_AIRFLOW_DAGS = (
    "backfill_1_collect_and_match",
    "backfill_2_enrich_and_features",
    "daily_signal_pipeline",
    "daily_advice",
    "weekly_model_training",
    "weekly_inst13f_holdings",
    "filing_changes",
)
EXPECTED_UNPAUSED_DAGS = (
    *CRITICAL_AIRFLOW_DAGS,
    "market_snapshots",
    "quant_intraday_news",
    "gdelt_hourly",
    "news_linkage",
)

# Host-side multi-region Job Agent freshness rules (migrated to Airflow job_agent_morning & job_agent_afternoon)
JOB_AGENT_STAGES = [
    ("job_agent (ireland)", "scrape_state", "last_success", {"hours": 30}, "job_agent_morning / job_agent_afternoon (11:00 & 16:00)"),
    ("job_agent (eu/me/remote)", "global_meta", "eu_refreshed_at", {"hours": 36}, "job_agent_afternoon eu_me_remote_daily (16:00)"),
    ("job_agent (china)", "china_meta", "foreign_refreshed_at", {"hours": 36}, "job_agent_morning china_fulltime_daily (11:00)"),
]

# stage -> (collection, how to read its newest date, allowed lag, what produces it)
#   lag in weekdays for daily outputs, hours for the intraday news
STAGES = [
    ("intraday news", "news_articles", "objectid", {"hours": 6}, "quant_intraday_news (every 30 min)"),
    ("company match", "news_articles_company_matched_v2", "objectid", {"days": 2}, "backfill_1 / gdelt_hourly company_match"),
    ("prices", "stock_prices_history", "price", {"weekdays": 2}, "price_history_backfill / daily_price"),
    ("macro", "macro_indicators", "date", {"weekdays": 2}, "daily_signal_pipeline macro"),
    ("features", "daily_symbol_features", "trade_date", {"weekdays": 2}, "daily_signal_pipeline daily_features"),
    ("signals", "daily_signals", "trade_date", {"weekdays": 2}, "daily_signal_pipeline score_signals"),
]

_state: Dict[str, Any] = {"checked_at": None, "ok": None, "stages": [], "problems": []}
_lock = threading.Lock()


def _mongo_uri() -> Optional[str]:
    if os.getenv("PIPELINE_MONGO_URI"):
        return os.environ["PIPELINE_MONGO_URI"]
    try:
        for line in QUANT_DATA_ENV.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() == "HOST_LOCAL_MONGO_URI":
                return v.strip()
    except OSError:
        return None
    return None


def weekdays_between(a: date, b: date) -> int:
    """Weekdays after a, up to and including b (0 when b <= a)."""
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def _newest(db, coll: str, how: str) -> Optional[datetime]:
    c = db[coll]
    if how == "objectid":
        doc = c.find_one(sort=[("_id", -1)], projection={"_id": 1})
        return doc["_id"].generation_time if doc else None
    if how == "price":            # indexed on (symbol, timestamp): ask one liquid symbol
        doc = c.find_one({"symbol": "AAPL"}, sort=[("timestamp", -1)], projection={"timestamp": 1})
        v = doc and doc.get("timestamp")
    else:
        doc = c.find_one(sort=[(how, -1)], projection={how: 1})
        v = doc and doc.get(how)
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v / 1000 if v > 1e11 else v, timezone.utc)
    return datetime.fromisoformat(str(v)[:10]).replace(tzinfo=timezone.utc)


def judge(newest: Optional[datetime], lag: Dict[str, int], now: datetime) -> Optional[str]:
    """None when fresh enough, else why not."""
    if newest is None:
        return "no data"
    if "minutes" in lag:
        age = (now - newest).total_seconds() / 60
        return None if age <= lag["minutes"] else f"newest is {age:.0f} min old"
    if "hours" in lag:
        age = (now - newest).total_seconds() / 3600
        return None if age <= lag["hours"] else f"newest is {age:.0f} h old"
    if "days" in lag:
        age = (now - newest).total_seconds() / 86400
        return None if age <= lag["days"] else f"newest is {age:.1f} days old"
    behind = weekdays_between(newest.date(), now.date())
    return None if behind <= lag["weekdays"] else f"{behind} weekdays behind"


def _parse_local_ts(val: Optional[str]) -> Optional[datetime]:
    if not val:
        return None
    try:
        dt = datetime.fromisoformat(str(val).strip().replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=DUBLIN_TZ)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _check_job_agent(now: datetime, app_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    stages: List[Dict[str, Any]] = []
    problems: List[str] = []
    state_file = app_dir / "scrape_state.json"
    db_file = app_dir / "job_history.db"
    if not state_file.exists() and not db_file.exists():
        return stages, problems

    state_data: Dict[str, Any] = {}
    if state_file.exists():
        try:
            state_data = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    meta_vals: Dict[Tuple[str, str], str] = {}
    if db_file.exists():
        try:
            conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, timeout=1.0)
            cur = conn.cursor()
            for tbl in ("global_meta", "china_meta"):
                try:
                    for k, v in cur.execute(f"SELECT key, value FROM {tbl}").fetchall():
                        meta_vals[(tbl, str(k))] = str(v)
                except Exception:
                    pass
            conn.close()
        except Exception:
            pass

    for name, src_kind, key, lag, producer in JOB_AGENT_STAGES:
        raw_ts = state_data.get(key) if src_kind == "scrape_state" else meta_vals.get((src_kind, key))
        newest = _parse_local_ts(raw_ts)
        why = judge(newest, lag, now)
        stages.append({
            "stage": name,
            "newest": newest.isoformat() if newest else None,
            "ok": why is None,
            "problem": why,
            "producer": producer,
        })
        if why:
            problems.append(f"{name}: {why} ({producer})")
    return stages, problems


def _prices_coverage_problem(db, newest: datetime) -> Optional[str]:
    """Verify prices exist broadly on the latest AAPL bar date rather than relying on one ticker."""
    coll = db["stock_prices_history"]
    if not hasattr(coll, "distinct"):
        return None
    doc = coll.find_one({"symbol": "AAPL"}, sort=[("timestamp", -1)], projection={"timestamp": 1})
    raw_ts = doc.get("timestamp") if doc else None
    if raw_ts is None:
        return "no AAPL timestamp"
    syms = coll.distinct("symbol", {"timestamp": raw_ts})
    if len(syms) < MIN_PRICE_SYMBOLS:
        return f"only {len(syms)} symbols on {newest.date().isoformat()} (expected >= {MIN_PRICE_SYMBOLS})"
    return None


def _oid_timestamp(oid_hex: Optional[str]) -> Optional[datetime]:
    if not oid_hex or len(str(oid_hex)) < 8:
        return None
    try:
        return datetime.fromtimestamp(int(str(oid_hex)[:8], 16), tz=timezone.utc)
    except ValueError:
        return None


def _check_qdrant(
    now: datetime,
    qdrant_url: str = QDRANT_URL,
    checkpoint_path: Path = QDRANT_CHECKPOINT,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    producer = "backfill_2 index_news_qdrant (daily)"
    newest: Optional[datetime] = None
    points: Optional[int] = None
    why: Optional[str] = None
    try:
        with urllib.request.urlopen(f"{qdrant_url.rstrip('/')}/collections/news_v1", timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        res = payload.get("result") or {}
        status = res.get("status")
        points = res.get("points_count")
        if status != "green" or not points:
            why = f"collection status={status}, points={points}"
        elif checkpoint_path.exists():
            cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            newest = _oid_timestamp(cp.get("last_id"))
            why = judge(newest, {"days": 2}, now) if newest else "checkpoint missing valid last_id"
        else:
            why = f"checkpoint missing ({checkpoint_path.name})"
    except Exception as e:  # noqa: BLE001
        why = f"Qdrant unreachable ({type(e).__name__})"

    stage: Dict[str, Any] = {
        "stage": "qdrant news_v1",
        "newest": newest.isoformat() if newest else None,
        "ok": why is None,
        "problem": why,
        "producer": producer,
    }
    if points is not None:
        stage["points"] = points
    return [stage], ([f"qdrant news_v1: {why} ({producer})"] if why else [])


def _mtime_utc(p: Path) -> Optional[datetime]:
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)


def _newest_snapshot_date(snap_dir: Path) -> Optional[datetime]:
    if not snap_dir.exists():
        return None
    files = sorted(snap_dir.glob("*.parquet"))
    if not files:
        return None
    latest_file = files[-1]
    try:
        return datetime.fromisoformat(latest_file.stem).replace(tzinfo=timezone.utc)
    except ValueError:
        return _mtime_utc(latest_file)


def _check_data4t_files(now: datetime, data_dir: Path = DATA_DIR) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not data_dir.exists():
        prob = "Data4T volume not mounted"
        return [{"stage": "data4t volume", "newest": None, "ok": False, "problem": prob, "producer": "local storage"}], [
            f"data4t volume: {prob}"
        ]
    checks = [
        ("daily advice", _mtime_utc(data_dir / "daily_advice" / "latest.json"), {"weekdays": 2}, "daily_advice (08:30/12:30/16:30 NY)"),
        ("market snapshots", _newest_snapshot_date(data_dir / "market_snapshots" / "options"), {"weekdays": 2}, "market_snapshots (16:15 NY)"),
        ("filing changes", _mtime_utc(data_dir / "filing_changes" / "latest.json"), {"weekdays": 3}, "filing_changes (17:45 NY)"),
    ]
    stages: List[Dict[str, Any]] = []
    problems: List[str] = []
    for name, newest, lag, producer in checks:
        why = judge(newest, lag, now)
        stages.append({
            "stage": name,
            "newest": newest.isoformat() if newest else None,
            "ok": why is None,
            "problem": why,
            "producer": producer,
        })
        if why:
            problems.append(f"{name}: {why} ({producer})")
    return stages, problems


def _airflow_dsn(airflow_conn: Optional[str] = None) -> Optional[str]:
    dsn = airflow_conn or os.getenv("AIRFLOW_DB_URI")
    if not dsn and AIRFLOW_PLIST.exists():
        try:
            plist = plistlib.loads(AIRFLOW_PLIST.read_bytes())
            dsn = (plist.get("EnvironmentVariables") or {}).get("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN")
        except Exception:  # noqa: BLE001
            dsn = None
    if not dsn:
        return None
    return str(dsn).replace("postgresql+psycopg2://", "postgresql://", 1)


def _open_airflow_conn(dsn: str):
    import psycopg2
    return psycopg2.connect(dsn, connect_timeout=2)


def _as_utc(val: Optional[datetime]) -> Optional[datetime]:
    if val is None:
        return None
    return val if val.tzinfo else val.replace(tzinfo=timezone.utc)


def _check_airflow(now: datetime, airflow_conn: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    stages: List[Dict[str, Any]] = []
    problems: List[str] = []
    dsn = _airflow_dsn(airflow_conn)
    if not dsn:
        return stages, problems
    conn = None
    try:
        conn = _open_airflow_conn(dsn)
        cur = conn.cursor()

        # 1. Scheduler heartbeat
        cur.execute("SELECT latest_heartbeat FROM job WHERE job_type = 'SchedulerJob' ORDER BY latest_heartbeat DESC LIMIT 1")
        row = cur.fetchone()
        hb = _as_utc(row[0] if row else None)
        why_hb = judge(hb, {"minutes": 10}, now)
        stages.append({"stage": "airflow scheduler", "newest": hb.isoformat() if hb else None,
                       "ok": why_hb is None, "problem": why_hb, "producer": "launchd com.quant.airflow-scheduler"})
        if why_hb:
            problems.append(f"airflow scheduler: {why_hb}")

        # 2. Failed tasks in last 24h (critical DAGs fail on >=1; noisy intraday/linkage >=3)
        cur.execute(
            "SELECT dag_id, task_id, COUNT(*) FROM task_instance "
            "WHERE state = 'failed' AND start_date >= %s GROUP BY dag_id, task_id",
            (now - timedelta(hours=24),),
        )
        bad_fails = [
            f"{d}.{t} ({cnt}x)"
            for d, t, cnt in cur.fetchall()
            if d in CRITICAL_AIRFLOW_DAGS or int(cnt) >= 3
        ]
        why_fail = f"recent failures: {', '.join(sorted(bad_fails))}" if bad_fails else None
        stages.append({"stage": "airflow failures (24h)", "newest": now.isoformat(),
                       "ok": why_fail is None, "problem": why_fail, "producer": "Airflow task_instance"})
        if why_fail:
            problems.append(f"airflow failures (24h): {why_fail}")

        # 3. Weekly jobs freshness (13F + model training)
        cur.execute(
            "SELECT dag_id, MAX(end_date) FROM dag_run "
            "WHERE dag_id IN ('weekly_model_training', 'weekly_inst13f_holdings') AND state = 'success' "
            "GROUP BY dag_id"
        )
        weekly_map = {d: _as_utc(ts) for d, ts in cur.fetchall()}
        weekly_probs = []
        weekly_dates = []
        for wd in ("weekly_inst13f_holdings", "weekly_model_training"):
            ts = weekly_map.get(wd)
            if ts:
                weekly_dates.append(ts)
            w = judge(ts, {"days": 9}, now)
            if w:
                weekly_probs.append(f"{wd} ({w})")
        why_weekly = "; ".join(weekly_probs) if weekly_probs else None
        oldest_weekly = min(weekly_dates) if len(weekly_dates) == 2 else None
        stages.append({"stage": "airflow weekly", "newest": oldest_weekly.isoformat() if oldest_weekly else None,
                       "ok": why_weekly is None, "problem": why_weekly, "producer": "weekly_inst13f_holdings & weekly_model_training"})
        if why_weekly:
            problems.append(f"airflow weekly: {why_weekly}")

        # 4. Expected active DAGs unexpectedly paused
        cur.execute(
            "SELECT dag_id FROM dag WHERE is_active = true AND is_paused = true AND dag_id IN %s",
            (tuple(EXPECTED_UNPAUSED_DAGS),),
        )
        paused = sorted(r[0] for r in cur.fetchall())
        why_paused = f"paused DAGs: {', '.join(paused)}" if paused else None
        stages.append({"stage": "airflow paused", "newest": now.isoformat(),
                       "ok": why_paused is None, "problem": why_paused, "producer": "Airflow dag.is_paused"})
        if why_paused:
            problems.append(f"airflow paused: {why_paused}")
    except Exception as e:  # noqa: BLE001
        prob = f"PostgreSQL unreachable ({type(e).__name__})"
        stages.append({"stage": "airflow metadata", "newest": None, "ok": False, "problem": prob, "producer": "airflow-postgres"})
        problems.append(f"airflow metadata: {prob}")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    return stages, problems


def check(
    now: Optional[datetime] = None,
    client=None,
    job_agent_dir: Optional[Path] = None,
    qdrant_url: Optional[str] = None,
    qdrant_checkpoint: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    airflow_conn: Optional[str] = None,
) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    stages: List[Dict[str, Any]] = []
    problems: List[str] = []
    check_all = client is None
    check_host_jobs = job_agent_dir is not None or check_all
    try:
        if client is None:
            import pymongo
            uri = _mongo_uri()
            if not uri:
                raise RuntimeError(f"HOST_LOCAL_MONGO_URI not found in {QUANT_DATA_ENV}")
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=4000)
        db = client["quant_data"]
        for name, coll, how, lag, producer in STAGES:
            newest = _newest(db, coll, how)
            why = judge(newest, lag, now)
            if how == "price" and why is None and newest is not None:
                why = _prices_coverage_problem(db, newest)
            stages.append({"stage": name, "newest": newest.isoformat() if newest else None, "ok": why is None,
                           "problem": why, "producer": producer})
            if why:
                problems.append(f"{name}: {why} ({producer})")
    except Exception as e:  # noqa: BLE001 - unreachable Mongo is itself the finding
        problems = [f"MongoDB unreachable ({type(e).__name__}): is Docker running with core.sh, and Data4T mounted?"]

    if check_host_jobs:
        ja_stages, ja_problems = _check_job_agent(now, job_agent_dir or JOB_AGENT_DIR)
        stages.extend(ja_stages)
        problems.extend(ja_problems)

    if check_all or qdrant_url is not None or qdrant_checkpoint is not None:
        q_stages, q_problems = _check_qdrant(
            now,
            qdrant_url=qdrant_url or QDRANT_URL,
            checkpoint_path=qdrant_checkpoint or QDRANT_CHECKPOINT,
        )
        stages.extend(q_stages)
        problems.extend(q_problems)

    if check_all or data_dir is not None:
        d_stages, d_problems = _check_data4t_files(now, data_dir=data_dir or DATA_DIR)
        stages.extend(d_stages)
        problems.extend(d_problems)

    if check_all or airflow_conn is not None:
        a_stages, a_problems = _check_airflow(now, airflow_conn=airflow_conn)
        stages.extend(a_stages)
        problems.extend(a_problems)

    result = {"checked_at": now.isoformat(), "ok": not problems, "stages": stages, "problems": problems}
    with _lock:
        _state.update(result)
    return result


def latest() -> Dict[str, Any]:
    with _lock:
        return dict(_state)


def _notify(title: str, text: str) -> None:
    script = f"display notification {json.dumps(text[:200])} with title {json.dumps(title)}"
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001
        pass


def check_and_alert(path=None) -> Dict[str, Any]:
    """Check, and notify at most once per stage per day while anything is wrong."""
    result = check()
    if not result["ok"]:
        today = date.today().isoformat()
        raw_alerted = budget_link._meta(path, "pipeline_alerted_stages")
        try:
            alerted = json.loads(raw_alerted) if raw_alerted else {}
            if not isinstance(alerted, dict):
                alerted = {}
        except Exception:  # noqa: BLE001
            alerted = {}
        fresh_problems: List[str] = []
        for prob in result["problems"]:
            stage = prob.split(":", 1)[0].strip()
            if alerted.get(stage) != today:
                fresh_problems.append(prob)
                alerted[stage] = today
        if fresh_problems:
            # Prune prior days so the metadata JSON never grows unbounded.
            pruned = {k: v for k, v in alerted.items() if v == today}
            budget_link._meta(path, "pipeline_alerted_stages", json.dumps(pruned), write=True)
            _notify("Quant pipeline is stale", "; ".join(fresh_problems)[:200])
    return result


_last = {"at": 0.0}


def run_due(path=None) -> None:
    """Called every minute by markets/background.py."""
    if time.time() - _last["at"] >= CHECK_EVERY:
        _last["at"] = time.time()
        check_and_alert(path)
