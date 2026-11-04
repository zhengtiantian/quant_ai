"""Today's and tomorrow's scheduled tasks with their state, for the Task Monitor page (2026-11-04).

The user asked for one list, in time order, of everything that runs today and tomorrow (Dublin time, so
the window moves at local midnight), each run marked done, failed, running or not run yet. Sources:

  airflow     every unpaused DAG with a cron schedule: its runs in the window are worked out from the
              cron expression in the DAG's own time zone and matched to dag_run by data_interval_end (a
              cron run fires at the end of its interval); DAGs started by another one (backfill_2 after
              backfill_1, daily_signal_pipeline after backfill_2) get one run per day after their upstream;
              manual runs in the window are listed as they are.
  job_agent   launchd, 11:00 and 16:00: read back from its log ("Starting Job Agent execution..." /
              "Job Agent finished." and the ERROR / exit-status lines in between).
  briefs      quant_ai's own scheduled briefs (markets/briefs.py KINDS with a time), from the briefs table.

DAGs that run many times a day (the 30-minute intraday news, the hourly keyword apply) come as one row per
day with a count per state and the runs inside it.
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

DUBLIN = ZoneInfo("Europe/Dublin")
UTC = timezone.utc
AIRFLOW_URI = os.getenv("AIRFLOW_DATABASE_URI", "postgresql://airflow:airflow@127.0.0.1:25432/airflow")
JOB_AGENT_LOG = Path(os.path.expanduser("~/Library/Logs/job_agent/job_agent.log"))
JOB_AGENT_TIMES = (dtime(11, 0), dtime(16, 0))
GROUP_ABOVE = 6                  # more runs than this per day: one summary row
# DAGs started by another DAG, not by a schedule
TRIGGERED_BY = {"backfill_2_enrich_and_features": "backfill_1_collect_and_match",
                "daily_signal_pipeline": "backfill_2_enrich_and_features"}
NAMES = {
    "backfill_1_collect_and_match": "Nightly news collect & company match",
    "backfill_2_enrich_and_features": "News enrichment & features",
    "daily_signal_pipeline": "Daily signal pipeline",
    "daily_advice": "Daily advice (model)",
    "gdelt_batch_verify": "GDELT data check",
    "keyword_refresh_apply": "Keyword review: apply decisions",
    "keyword_refresh_weekly": "Weekly keyword refresh",
    "market_snapshots": "Market snapshots",
    "price_history_backfill": "Daily price history",
    "quant_intraday_news": "Intraday news",
    "quant_news_validation": "News validation",
    "quant_retail_sentiment": "Retail sentiment",
    "universe_news_backfill_0800": "Universe news backfill (morning)",
    "universe_news_backfill_0830_sat": "Universe news backfill (Saturday morning)",
    "universe_news_backfill_1130": "Universe news backfill (midday)",
    "universe_news_backfill_1630": "Universe news backfill (evening)",
    "weekly_inst13f_holdings": "Weekly 13F holdings",
    "weekly_model_training": "Weekly model training",
}
AIRFLOW_STATE = {"success": "done", "failed": "failed", "running": "running", "queued": "running",
                 "upstream_failed": "failed"}


# ---------------------------------------------------------------- cron
def _field(spec: str, lo: int, hi: int) -> set:
    out: set = set()
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/")
            step = int(s)
        if part in ("*", ""):
            a, b = lo, hi
        elif "-" in part:
            a, b = map(int, part.split("-"))
        else:
            a = b = int(part)
        out |= set(range(a, b + 1, step))
    return out


def cron_times(expr: str, tz: ZoneInfo, start: datetime, end: datetime) -> List[datetime]:
    """Fire times of a 5-field cron expression in [start, end), evaluated in `tz` (Airflow's way)."""
    f = expr.split()
    if len(f) != 5:
        return []
    mins, hours, doms, months = _field(f[0], 0, 59), _field(f[1], 0, 23), _field(f[2], 1, 31), _field(f[3], 1, 12)
    dows = {d % 7 for d in _field(f[4], 0, 7)}
    out = []
    d = start.astimezone(tz).date() - timedelta(days=1)
    while d <= end.astimezone(tz).date() + timedelta(days=1):
        dom_ok = d.day in doms or f[2] == "*"
        dow_ok = (d.weekday() + 1) % 7 in dows or f[4] == "*"
        # cron: when both day fields are restricted, either may match
        day_ok = (dom_ok or dow_ok) if (f[2] != "*" and f[4] != "*") else (dom_ok and dow_ok)
        if d.month in months and day_ok:
            for h in sorted(hours):
                for m in sorted(mins):
                    t = datetime.combine(d, dtime(h, m), tzinfo=tz)
                    if start <= t < end:
                        out.append(t)
        d += timedelta(days=1)
    return sorted(out)


# ---------------------------------------------------------------- sources
def _window(now: datetime) -> Tuple[datetime, datetime, List[date]]:
    today = now.astimezone(DUBLIN).date()
    days = [today, today + timedelta(days=1)]
    return (datetime.combine(days[0], dtime(0), tzinfo=DUBLIN),
            datetime.combine(days[1] + timedelta(days=1), dtime(0), tzinfo=DUBLIN), days)


def _entry(source: str, key: str, name: str, at: datetime, state: str, now: datetime, **kw) -> Dict[str, Any]:
    overdue = state == "pending" and at < now - timedelta(minutes=10)
    return {"source": source, "key": key, "name": name, "at": at.astimezone(UTC).isoformat(),
            "day": at.astimezone(DUBLIN).date().isoformat(), "time": at.astimezone(DUBLIN).strftime("%H:%M"),
            "state": state, "overdue": overdue, **kw}


def _run_fields(run: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not run:
        return {}
    s, e = run.get("start_date"), run.get("end_date")
    dur = (e - s).total_seconds() if s and e else ((datetime.now(UTC) - s).total_seconds() if s else None)
    return {"run_id": run["run_id"], "started": s.isoformat() if s else None, "ended": e.isoformat() if e else None,
            "duration_seconds": round(dur) if dur is not None else None, "log_task_id": run.get("log_task_id"),
            "tasks_done": run.get("tasks_done"), "tasks_total": run.get("tasks_total")}


def _airflow(now: datetime, start: datetime, end: datetime) -> List[Dict[str, Any]]:
    import psycopg2
    conn = psycopg2.connect(AIRFLOW_URI, connect_timeout=3)
    try:
        cur = conn.cursor()
        cur.execute("""SELECT d.dag_id, d.is_paused, d.schedule_interval,
                              (s.data::json -> 'dag' ->> 'timezone')
                       FROM dag d LEFT JOIN serialized_dag s ON s.dag_id = d.dag_id
                       WHERE d.is_active""")
        dags = cur.fetchall()
        cur.execute("""SELECT dag_id, run_id, run_type, state, data_interval_end, start_date, end_date
                       FROM dag_run
                       WHERE (data_interval_end >= %s AND data_interval_end < %s) OR (start_date >= %s AND start_date < %s)""",
                    (start - timedelta(hours=1), end, start, end))
        runs = [dict(zip(("dag_id", "run_id", "run_type", "state", "interval_end", "start_date", "end_date"), r))
                for r in cur.fetchall()]
        ids = [r["run_id"] for r in runs]
        tis: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        if ids:
            cur.execute("SELECT dag_id, run_id, task_id, state FROM task_instance WHERE run_id = ANY(%s)", (ids,))
            for d, r, t, st in cur.fetchall():
                tis.setdefault((d, r), []).append((t, st))
    finally:
        conn.close()
    for r in runs:
        ts = tis.get((r["dag_id"], r["run_id"]), [])
        pick = ([t for t, st in ts if st == "failed"] or [t for t, st in ts if st == "running"] or [t for t, _ in ts])
        r["log_task_id"] = pick[0] if pick else None
        r["tasks_total"], r["tasks_done"] = len(ts), sum(1 for _, st in ts if st == "success")

    out: List[Dict[str, Any]] = []
    used: set = set()
    by_dag: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        by_dag.setdefault(r["dag_id"], []).append(r)

    def entry(dag_id: str, at: datetime, run: Optional[Dict[str, Any]], **kw) -> Dict[str, Any]:
        state = AIRFLOW_STATE.get(run["state"], "running") if run else "pending"
        return _entry("airflow", dag_id, NAMES.get(dag_id, dag_id), at, state, now, dag_id=dag_id, **_run_fields(run), **kw)

    paused = {d for d, p, _, _ in dags if p}
    for dag_id, is_paused, sched, tz in dags:
        if is_paused:
            continue
        expr = (sched or "").strip('"')
        if expr and expr.lower() not in ("null", "none"):
            zone = ZoneInfo(tz) if tz and tz not in ("UTC",) else ZoneInfo("UTC")
            for t in cron_times(expr, zone, start, end):
                run = next((r for r in by_dag.get(dag_id, []) if r["run_type"] == "scheduled"
                            and r["interval_end"] and abs((r["interval_end"] - t).total_seconds()) < 60), None)
                if run:
                    used.add(run["run_id"])
                out.append(entry(dag_id, t, run))
    # DAGs another DAG starts: one per day, after the upstream's run of that day
    for dag_id, up in TRIGGERED_BY.items():
        if dag_id in paused:
            continue
        for e in [x for x in out if x.get("dag_id") == up]:
            day = e["day"]
            run = next((r for r in by_dag.get(dag_id, []) if r["start_date"] and r["run_id"] not in used
                        and r["start_date"].astimezone(DUBLIN).date().isoformat() == day), None)
            if run:
                used.add(run["run_id"])
            # not run yet: expected when the upstream ends (or a minute after its start, to keep the order)
            est = (datetime.fromisoformat(e["ended"]) if e.get("ended")
                   else datetime.fromisoformat(e["at"]) + timedelta(minutes=1))
            out.append(entry(dag_id, run["start_date"] if run else est, run, after=NAMES.get(up, up), estimated=not run))
    # anything else that ran in the window (manual runs, catch-ups)
    for r in runs:
        if r["run_id"] in used or not r["start_date"] or not (start <= r["start_date"] < end):
            continue
        out.append(entry(r["dag_id"], r["start_date"], r, manual=r["run_type"] != "scheduled"))
    return out


def _job_agent(now: datetime, days: List[date]) -> List[Dict[str, Any]]:
    """Runs read back from job_agent.log: start marker, then finished / error lines until the next start."""
    runs: List[Dict[str, Any]] = []
    try:
        with JOB_AGENT_LOG.open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 3_000_000))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        text = ""
    cur: Optional[Dict[str, Any]] = None
    for line in text.splitlines():
        m = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.*)", line)
        if not m:
            continue
        ts, msg = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=DUBLIN), m.group(2)
        if msg.startswith("Starting Job Agent execution"):
            cur = {"start": ts, "end": None, "error": None}
            runs.append(cur)
        elif cur is not None:
            if msg.startswith("Job Agent finished"):
                cur["end"] = ts
            elif msg.startswith("ERROR") or "exited with status" in msg or msg.startswith("WARNING"):
                cur["error"] = msg[:200]
    try:
        alive = "run_daily.sh" in subprocess.check_output(["ps", "-axo", "command"], text=True)
    except Exception:  # noqa: BLE001
        alive = False
    out = []
    for d in days:
        for t in JOB_AGENT_TIMES:
            at = datetime.combine(d, t, tzinfo=DUBLIN)
            run = next((r for r in runs if timedelta(minutes=-5) <= r["start"] - at <= timedelta(minutes=90)), None)
            if not run:
                state = "pending"
            elif run["end"]:
                state = "failed" if run["error"] else "done"
            else:
                state = "running" if alive and run is runs[-1] else "failed"
            dur = ((run["end"] or now) - run["start"]).total_seconds() if run else None
            out.append(_entry("job_agent", "job_agent", "Job Agent (job search)", at, state, now,
                              started=run["start"].isoformat() if run else None,
                              ended=run["end"].isoformat() if run and run["end"] else None,
                              duration_seconds=round(dur) if dur is not None else None,
                              error=run["error"] if run else None))
    return out


def _briefs(now: datetime, start: datetime, end: datetime) -> List[Dict[str, Any]]:
    from . import briefs, portfolio as pf
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    try:
        pf.init_db()
        conn = pf._conn()
        try:
            for r in conn.execute("SELECT kind, session_date, status, started_at, finished_at, error FROM briefs "
                                  "WHERE topic = '' AND session_date >= ?", ((start - timedelta(days=2)).date().isoformat(),)):
                rows[(r["kind"], r["session_date"])] = dict(r)
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - no briefs table yet
        pass
    out = []
    for kind, k in briefs.KINDS.items():
        if not k.get("at"):
            continue
        zone = k["zone"]
        d = start.astimezone(zone).date() - timedelta(days=1)
        while d <= end.astimezone(zone).date():
            at = datetime.combine(d, k["at"], tzinfo=zone)
            if d.weekday() < 5 and start <= at < end:
                r = rows.get((kind, d.isoformat()))
                state = {"done": "done", "failed": "failed", "running": "running"}.get(r["status"], "running") if r else "pending"
                s = datetime.fromtimestamp(r["started_at"], UTC) if r and r.get("started_at") else None
                e = datetime.fromtimestamp(r["finished_at"], UTC) if r and r.get("finished_at") else None
                out.append(_entry("briefs", kind, f"Brief: {k['title']}", at, state, now,
                                  started=s.isoformat() if s else None, ended=e.isoformat() if e else None,
                                  duration_seconds=round((e - s).total_seconds()) if s and e else None,
                                  error=(r.get("error") or "")[:200] or None if r else None))
            d += timedelta(days=1)
    return out


# ---------------------------------------------------------------- assembly
def _group(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Frequent DAGs: one row per (source, key, day) holding its runs."""
    counts: Dict[Tuple[str, str, str], int] = {}
    for e in entries:
        k = (e["source"], e["key"], e["day"])
        counts[k] = counts.get(k, 0) + 1
    out, groups = [], {}
    for e in entries:
        k = (e["source"], e["key"], e["day"])
        if counts[k] <= GROUP_ABOVE:
            out.append(e)
            continue
        if k not in groups:
            groups[k] = {**{f: e[f] for f in ("source", "key", "name", "day", "at", "time")},
                         "dag_id": e.get("dag_id"), "group": True, "runs": []}
            out.append(groups[k])
        groups[k]["runs"].append(e)
    for g in groups.values():
        rs = g["runs"]
        c = {s: sum(1 for r in rs if r["state"] == s) for s in ("done", "failed", "running", "pending")}
        g["counts"] = c
        g["overdue"] = any(r["overdue"] for r in rs)
        g["time"] = f"{rs[0]['time']}–{rs[-1]['time']}"
        g["state"] = ("failed" if c["failed"] else "running" if c["running"] else
                      "done" if c["done"] == len(rs) else "pending" if not c["done"] else "partial")
    return out


def schedule(now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or datetime.now(UTC)
    start, end, days = _window(now)
    entries: List[Dict[str, Any]] = []
    errors = []
    for name, fn in (("airflow", lambda: _airflow(now, start, end)), ("job_agent", lambda: _job_agent(now, days)),
                     ("briefs", lambda: _briefs(now, start, end))):
        try:
            entries += fn()
        except Exception as e:  # noqa: BLE001 - one source down must not hide the others
            errors.append(f"{name}: {type(e).__name__}: {e}")
    entries.sort(key=lambda e: (e["at"], e["name"]))
    rows = _group(entries)
    out_days = []
    for d in days:
        items = [r for r in rows if r["day"] == d.isoformat()]
        tally = {s: 0 for s in ("done", "failed", "running", "pending")}
        for e in entries:
            if e["day"] == d.isoformat():
                tally[e["state"]] += 1
        out_days.append({"date": d.isoformat(), "label": "Today" if d == days[0] else "Tomorrow",
                         "weekday": d.strftime("%A"), "counts": tally, "items": items})
    return {"now": now.astimezone(DUBLIN).isoformat(), "timezone": "Europe/Dublin", "days": out_days, "errors": errors}
