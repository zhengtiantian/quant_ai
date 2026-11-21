"""Unified Task & Schedule Monitor for Quant & System Background Services.

Provides real-time visibility into:
1. Airflow DAGs: running tasks, progress %, execution durations, next schedules, and task logs.
2. 找工平台 (Job Agent on http://127.0.0.1:5050/): daemon status, next 11:00/16:00 schedule, scrape state, and live logs.
3. System & AI Background Daemons: com.quant.ai, com.quant.airflow-scheduler, LM Studio, Disk Watch.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import socket
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

DUBLIN_TZ = ZoneInfo("Europe/Dublin")

AIRFLOW_POSTGRES_URI = os.getenv(
    "AIRFLOW_DATABASE_URI",
    "postgresql://airflow:airflow@127.0.0.1:25432/airflow"
)
AIRFLOW_LOGS_DIR = Path(os.getenv("AIRFLOW_LOGS_DIR", "/Volumes/Data4T/docker-volumes/airflow_logs"))
AIRFLOW_SCHEDULER_LOG = Path("/Users/xiz/Quant_trade/quant_data/airflow_scheduler.out.log")

JOB_AGENT_PORT = 5050
JOB_AGENT_LOG_DIR = Path(os.path.expanduser("~/Library/Logs/job_agent"))
JOB_AGENT_APP_DIR = Path(os.path.expanduser("~/Library/Application Support/job_agent"))
JOB_AGENT_STATE_FILE = JOB_AGENT_APP_DIR / "scrape_state.json"
JOB_AGENT_DB_FILE = JOB_AGENT_APP_DIR / "job_history.db"
JOB_AGENT_SNAPSHOT_FILE = JOB_AGENT_APP_DIR / "monitor_snapshot.json"
JOB_AGENT_REPORT_FILE = Path(
    os.path.expanduser(
        "~/Library/CloudStorage/GoogleDrive-biaszh00001@gmail.com/My Drive/17.找工作/Today_Jobs.md"
    )
)

QUANT_AI_LOG = Path(os.path.expanduser("~/Library/Logs/quant_ai/quant_ai.log"))
DISK_WATCH_LOG = Path(os.path.expanduser("~/Library/Logs/quant_ai/disk_watch.log"))


def _is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.4) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def _get_launchd_status() -> Dict[str, Dict[str, Any]]:
    """Parse launchctl list for quant and jobagent services."""
    res = {}
    try:
        out = subprocess.check_output(["launchctl", "list"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pid_str, exit_str, label = parts[0], parts[1], parts[2]
                if any(k in label for k in ["quant", "jobagent"]):
                    pid = int(pid_str) if pid_str != "-" else None
                    last_exit = int(exit_str) if exit_str != "-" else 0
                    res[label] = {
                        "label": label,
                        "pid": pid,
                        "is_running": pid is not None,
                        "last_exit_code": last_exit,
                    }
    except Exception:
        pass
    return res


def _get_process_running(patterns: List[str]) -> bool:
    """Check if any process matching one of the patterns is alive."""
    try:
        out = subprocess.check_output(["ps", "aux"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if any(p in line for p in patterns) and "grep" not in line and "ps aux" not in line:
                return True
    except Exception:
        pass
    return False


def _calc_job_agent_next_run() -> Dict[str, Any]:
    """Calculate the next scheduled run for Job Agent (11:00 & 16:00 Europe/Dublin daily in Airflow)."""
    now = datetime.datetime.now(DUBLIN_TZ)
    t11 = now.replace(hour=11, minute=0, second=0, microsecond=0)
    t16 = now.replace(hour=16, minute=0, second=0, microsecond=0)

    if now < t11:
        next_dt = t11
    elif now < t16:
        next_dt = t16
    else:
        next_dt = (now + datetime.timedelta(days=1)).replace(hour=11, minute=0, second=0, microsecond=0)

    diff_seconds = max(0, int((next_dt - now).total_seconds()))
    hours = diff_seconds // 3600
    minutes = (diff_seconds % 3600) // 60

    return {
        "next_run_iso": next_dt.isoformat(),
        "next_run_display": next_dt.strftime("%Y-%m-%d %H:%M (Dublin)"),
        "countdown": f"{hours}小时 {minutes}分钟后" if hours > 0 else f"{minutes}分钟后",
    }


def _read_job_agent_regions() -> Tuple[Optional[str], List[str]]:
    """Read live open-job counts and latest refresh timestamps across all 5 regions from local APFS DB."""
    timestamps: List[str] = []
    if JOB_AGENT_DB_FILE.exists():
        try:
            conn = sqlite3.connect(f"file:{JOB_AGENT_DB_FILE}?mode=ro", uri=True, timeout=1.0)
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*) FROM jobs j
                WHERE j.is_recommended = 1
                  AND j.job_key NOT IN (SELECT job_key FROM applied_jobs)
                  AND j.job_key NOT IN (SELECT job_key FROM rejected_jobs)
                  AND j.job_key NOT IN (SELECT job_key FROM closed_jobs)
                  AND j.job_key NOT IN (SELECT job_key FROM no_sponsorship_jobs)
            """)
            ie_cnt = cur.fetchone()[0]

            global_cnts = dict(
                cur.execute("""
                    SELECT region, COUNT(*) FROM global_jobs
                    WHERE COALESCE(status, 'open') = 'open'
                    GROUP BY region
                """).fetchall()
            )
            cn_cnt = cur.execute("""
                SELECT COUNT(*) FROM china_jobs
                WHERE COALESCE(mark, '') = ''
                  AND (kind != 'parttime' OR COALESCE(score, 0) >= 60)
            """).fetchone()[0]

            for tbl in ("china_meta", "global_meta"):
                for (_k, val) in cur.execute(f"SELECT key, value FROM {tbl} WHERE key LIKE '%refreshed_at'").fetchall():
                    if val:
                        timestamps.append(str(val).replace(" ", "T")[:19])
            conn.close()

            summary = (
                f"🇮🇪 Ireland {ie_cnt} · 🇪🇺🇬🇧 EU & UK {global_cnts.get('eu', 0)} · "
                f"🇦🇪 中东 {global_cnts.get('me', 0)} · 🌍 Remote {global_cnts.get('remote', 0)} · "
                f"🇨🇳 China {cn_cnt}"
            )
            return summary, timestamps
        except Exception:
            pass

    if JOB_AGENT_SNAPSHOT_FILE.exists():
        try:
            snap = json.loads(JOB_AGENT_SNAPSHOT_FILE.read_text(encoding="utf-8"))
            return snap.get("regions_summary"), timestamps
        except Exception:
            pass
    return None, timestamps


def _get_job_agent_info() -> Dict[str, Any]:
    """Gather Job Agent (:5050) metadata, state, next run and multi-region summary."""
    is_online = _is_port_open(JOB_AGENT_PORT)
    launchd_map = _get_launchd_status()
    launchd = launchd_map.get("com.xiangyu.jobagent.web") or launchd_map.get("com.xiangyu.jobagent") or {}
    is_scraping = _get_process_running([
        "job_agent/airflow_entry.py",
        "job_agent/run_daily.sh",
        "job_agent/main.py",
    ])

    state_data = {}
    if JOB_AGENT_STATE_FILE.exists():
        try:
            state_data = json.loads(JOB_AGENT_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    regions_summary, extra_ts = _read_job_agent_regions()
    all_ts = [t for t in ([state_data.get("last_success")] + extra_ts) if t]
    latest_success = max(all_ts) if all_ts else state_data.get("last_success")

    report_info = {}
    try:
        if JOB_AGENT_REPORT_FILE.exists():
            stat = JOB_AGENT_REPORT_FILE.stat()
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
            lines = JOB_AGENT_REPORT_FILE.read_text(encoding="utf-8").splitlines()
            headline = ""
            for line in lines[:10]:
                if "全网扫描" in line or "合规精选" in line:
                    headline = line.strip().replace(">", "").strip()
                    break
            report_info = {
                "exists": True,
                "path": str(JOB_AGENT_REPORT_FILE),
                "modified_at": mtime.strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": round(stat.st_size / 1024, 1),
                "summary": headline or "今日求职精选报告已生成",
            }
    except Exception:
        pass
    if not report_info and JOB_AGENT_SNAPSHOT_FILE.exists():
        try:
            snap = json.loads(JOB_AGENT_SNAPSHOT_FILE.read_text(encoding="utf-8"))
            if isinstance(snap.get("report"), dict) and snap["report"].get("exists"):
                report_info = snap["report"]
        except Exception:
            pass

    has_log = (JOB_AGENT_LOG_DIR / "job_agent.log").exists()

    return {
        "name": "找工平台 (Job Agent)",
        "url": "http://127.0.0.1:5050/",
        "port": JOB_AGENT_PORT,
        "is_online": is_online,
        "is_scraping_now": is_scraping,
        "schedule": "每日 11:00 & 16:00 都柏林时间 (Airflow 统一调度: job_agent_morning / job_agent_afternoon)",
        "regions_summary": regions_summary,
        "next_run": _calc_job_agent_next_run(),
        "last_success": latest_success,
        "last_full_sweep": state_data.get("last_full_sweep"),
        "launchd_status": "loaded" if launchd else "not_found",
        "report": report_info,
        "has_log": has_log,
    }


def _get_dag_display_meta(dag_id: str) -> Dict[str, str]:
    """Provide user-friendly Chinese names and categories for Airflow DAGs."""
    mapping = {
        "daily_signal_pipeline": {
            "name": "每日量化信号主管道 (EOD Signals)",
            "category": "量化主管道",
            "desc": "收盘宏观/新闻指标聚合，特征工程，多因子打分与仓位风控决策",
        },
        "backfill_1_collect_and_match": {
            "name": "GDELT 新闻抓取与 SLM 匹配",
            "category": "数据采集与匹配",
            "desc": "全球财经与科技新闻每日自动采集，经过SLM模型精准实体匹配入库",
        },
        "backfill_2_enrich_and_features": {
            "name": "新闻特征丰富与向量化嵌入",
            "category": "数据处理与特征",
            "desc": "新闻事件情绪打分、聚类与向量数据库入库，下游触发信号主管道",
        },
        "price_history_backfill": {
            "name": "历史日K行情增量回填",
            "category": "行情与历史数据",
            "desc": "同步每日最新股票收盘价、复权因子与交易量到 TimescaleDB",
        },
        "quant_intraday_news": {
            "name": "实时盘中突发新闻采集",
            "category": "实时盘中监控",
            "desc": "每30分钟采集盘中Breaking News与美股异动快讯",
        },
        "quant_retail_sentiment": {
            "name": "散户与社交情绪监控",
            "category": "实时盘中监控",
            "desc": "美股主流散户社区情绪指数与热度异动采集",
        },
        "weekly_model_training": {
            "name": "AI 预测模型周度再训练",
            "category": "模型每周训练",
            "desc": "每周日自动化重新训练LightGBM多因子回归模型与MLflow版本记录",
        },
        "weekly_inst13f_holdings": {
            "name": "机构 13F 持仓季报更新",
            "category": "机构与资金流",
            "desc": "更新华尔街顶级机构 SEC 13F 季度调仓数据",
        },
        "gdelt_batch_verify": {
            "name": "GDELT 数据健康度验证",
            "category": "系统健康检查",
            "desc": "每周定期校验 GDELT 数据完整性与空洞检测",
        },
        "quant_news_validation": {
            "name": "新闻验证与打分校准",
            "category": "数据处理与特征",
            "desc": "新闻样本交叉校验与标签模型评估",
        },
        "job_agent_morning": {
            "name": "全球求职早班扫描 (IE + 中国全量 + 欧中东官网)",
            "category": "全球求职监控",
            "desc": "每日 11:00 都柏林(19:00 北京)：爱尔兰早班全扫 + 中国全职(201家外企官网/国内/外包) + 中国远程兼职 + 欧英/中东官网快扫",
        },
        "job_agent_afternoon": {
            "name": "全球求职午后扫描 (IE + 欧英/中东/Remote全量 + 中国外企官网)",
            "category": "全球求职监控",
            "desc": "每日 16:00 都柏林：爱尔兰午后全扫 + 欧洲12国/中东3国/全球Remote国家级全扫 + 中国201家外企官网快扫",
        },
        "gdelt_hourly": {
            "name": "GDELT 小时级新闻采集与匹配",
            "category": "数据采集与匹配",
            "desc": "每小时 20 分增量抓取全球 GDELT GKG 新闻并通过 SLM 模型完成公司实体匹配",
        },
        "daily_advice": {
            "name": "持仓盘中/收盘 AI 操作建议 (Daily Advice)",
            "category": "量化主管道",
            "desc": "交易日 13:30 / 17:30 / 21:30 UTC 为当前持仓生成模型动作评估建议",
        },
        "market_snapshots": {
            "name": "期权与分析师评级收盘快照",
            "category": "行情与历史数据",
            "desc": "交易日 21:15 UTC 抓取股票池与持仓的期权隐含波动率及华尔街分析师目标价快照",
        },
        "keyword_refresh_weekly": {
            "name": "公司匹配规则关键词周度挖掘",
            "category": "数据处理与特征",
            "desc": "每周六 08:00 都柏林时间自动生成新增公司匹配关键词候选供 UI 审核",
        },
        "keyword_refresh_apply": {
            "name": "公司关键词审核自动生效",
            "category": "数据处理与特征",
            "desc": "每日 08:55–23:55 每小时将 UI 审核通过的关键词同步写入匹配规则库",
        },
    }
    return mapping.get(dag_id, {
        "name": dag_id,
        "category": "其他任务",
        "desc": dag_id,
    })


def _get_airflow_data() -> Dict[str, Any]:
    """Query Airflow PostgreSQL metadata directly."""
    import psycopg2

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    dags_list = []
    running_tasks = []
    success_today = 0
    failed_today = 0
    running_count = 0

    try:
        conn = psycopg2.connect(AIRFLOW_POSTGRES_URI, connect_timeout=3)
        cur = conn.cursor()

        # 1. Fetch active DAGs
        cur.execute("""
            SELECT dag_id, is_paused, schedule_interval,
                   COALESCE(next_dagrun_create_after, next_dagrun_data_interval_end, next_dagrun),
                   description
            FROM dag
            WHERE is_active = TRUE
            ORDER BY is_paused ASC, dag_id ASC;
        """)
        raw_dags = cur.fetchall()

        # 2. Fetch latest run per DAG
        cur.execute("""
            SELECT DISTINCT ON (dag_id)
                dag_id, run_id, state, start_date, end_date, execution_date
            FROM dag_run
            ORDER BY dag_id, start_date DESC;
        """)
        latest_runs = {
            r[0]: {
                "run_id": r[1],
                "state": r[2],
                "start_date": r[3],
                "end_date": r[4],
                "execution_date": r[5],
            }
            for r in cur.fetchall()
        }

        # 3. Fetch running task instances right now
        cur.execute("""
            SELECT dag_id, task_id, run_id, state, start_date, duration
            FROM task_instance
            WHERE state = 'running'
            ORDER BY start_date DESC;
        """)
        for d_id, t_id, r_id, st, s_date, dur in cur.fetchall():
            dur_sec = dur
            if dur_sec is None and s_date:
                dur_sec = round((now_utc - s_date).total_seconds(), 1)
            running_count += 1
            running_tasks.append({
                "source": "airflow",
                "dag_id": d_id,
                "task_id": t_id,
                "run_id": r_id,
                "name": f"{_get_dag_display_meta(d_id)['name']} ➔ {t_id}",
                "state": st,
                "start_date": s_date.isoformat() if s_date else None,
                "duration_seconds": dur_sec or 0,
                "can_view_log": True,
            })

        # 4. Fetch task instance counts for today's runs to compute today's successes/failures
        cur.execute("""
            SELECT state, count(*)
            FROM dag_run
            WHERE start_date >= %s
            GROUP BY state;
        """, (today_start,))
        for st, cnt in cur.fetchall():
            if st == "success":
                success_today += cnt
            elif st == "failed":
                failed_today += cnt

        # 5. For each latest run, fetch its task instances
        latest_run_ids = [r["run_id"] for r in latest_runs.values() if r.get("run_id")]
        ti_by_run: Dict[str, List[Dict[str, Any]]] = {}
        if latest_run_ids:
            cur.execute("""
                SELECT dag_id, run_id, task_id, state, start_date, end_date, duration
                FROM task_instance
                WHERE run_id = ANY(%s)
                ORDER BY start_date ASC NULLS LAST, task_id ASC;
            """, (latest_run_ids,))
            for d_id, r_id, t_id, st, s_dt, e_dt, dur in cur.fetchall():
                ti_by_run.setdefault(r_id, []).append({
                    "task_id": t_id,
                    "state": st,
                    "start_date": s_dt.isoformat() if s_dt else None,
                    "end_date": e_dt.isoformat() if e_dt else None,
                    "duration": round(dur, 1) if dur is not None else None,
                })

        conn.close()

        # Build DAG representation
        for d_id, is_paused, sched, next_run, desc in raw_dags:
            meta = _get_dag_display_meta(d_id)
            run = latest_runs.get(d_id)
            run_data = None
            progress = {"total": 0, "success": 0, "running": 0, "failed": 0, "percent": 0.0}
            tasks = []

            if run:
                r_id = run["run_id"]
                tasks = ti_by_run.get(r_id, [])
                total_t = len(tasks)
                succ_t = sum(1 for t in tasks if t["state"] == "success")
                run_t = sum(1 for t in tasks if t["state"] == "running")
                fail_t = sum(1 for t in tasks if t["state"] == "failed")
                pct = round((succ_t / total_t * 100), 1) if total_t > 0 else (100.0 if run["state"] == "success" else 0.0)

                dur_sec = None
                if run["start_date"] and run["end_date"]:
                    dur_sec = round((run["end_date"] - run["start_date"]).total_seconds(), 1)
                elif run["start_date"]:
                    dur_sec = round((now_utc - run["start_date"]).total_seconds(), 1)

                progress = {
                    "total": total_t,
                    "success": succ_t,
                    "running": run_t,
                    "failed": fail_t,
                    "percent": pct,
                }
                run_data = {
                    "run_id": run["run_id"],
                    "state": run["state"],
                    "start_date": run["start_date"].isoformat() if run["start_date"] else None,
                    "end_date": run["end_date"].isoformat() if run["end_date"] else None,
                    "duration_seconds": dur_sec,
                    "progress": progress,
                }

            # Human-friendly schedule description
            sched_str = str(sched) if sched else ""
            if sched_str.startswith('"') and sched_str.endswith('"'):
                sched_str = sched_str[1:-1]
            if not sched_str or sched_str.lower() in ("null", "none"):
                if d_id == "daily_signal_pipeline":
                    sched_display = "上游联动触发 (backfill_2 完成后自动执行 · 约 04:30–05:10 UTC)"
                elif d_id == "backfill_2_enrich_and_features":
                    sched_display = "上游联动触发 (backfill_1 完成后自动执行 · 约 04:15–04:30 UTC)"
                elif d_id == "quant_news_validation":
                    sched_display = "按需手动触发 (On-demand 离线质检与抽样)"
                else:
                    sched_display = "按需 / 上游联动触发"
            elif sched_str == "0 4 * * *":
                sched_display = "每日 04:00 UTC (都柏林冬令时 04:00 / 夏令时 05:00)"
            elif sched_str == "30 6 * * *":
                sched_display = "每日 06:30 UTC (14:30 本地)"
            elif sched_str == "*/30 * * * *":
                sched_display = "每 30 分钟"
            elif sched_str == "30 20 * * *":
                sched_display = "每日 20:30 UTC (04:30 本地)"
            elif sched_str == "0 6 * * 0":
                sched_display = "每周日 06:00 UTC"
            elif sched_str == "0 7 * * 0":
                sched_display = "每周日 07:00 UTC"
            elif sched_str == "0 4 * * 0":
                sched_display = "每周日 04:00 UTC"
            elif sched_str == "0 11 * * *":
                sched_display = "每日 11:00 都柏林时间 (19:00 北京 · 自动冬/夏令时)"
            elif sched_str == "0 16 * * *":
                sched_display = "每日 16:00 都柏林时间 (自动冬/夏令时)"
            elif sched_str == "20 0-2,7-23 * * *":
                sched_display = "每小时 20 分 UTC (03:00–06:00 让位夜间主管道)"
            elif sched_str == "30 13,17,21 * * 1-5":
                sched_display = "交易日 13:30 / 17:30 / 21:30 UTC"
            elif sched_str == "15 21 * * 1-5":
                sched_display = "交易日 21:15 UTC (美股收盘后)"
            elif sched_str == "55 8-23 * * *":
                sched_display = "每日 08:55–23:55 都柏林时间 (每小时 55 分)"
            elif sched_str == "0 8 * * 6":
                sched_display = "每周六 08:00 都柏林时间"
            else:
                sched_display = sched_str

            dags_list.append({
                "dag_id": d_id,
                "name": meta["name"],
                "category": meta["category"],
                "description": desc or meta["desc"],
                "is_paused": is_paused,
                "schedule": sched_str,
                "schedule_display": sched_display,
                "next_run": next_run.isoformat() if next_run else None,
                "last_run": run_data,
                "tasks": tasks,
            })

    except Exception as e:
        dags_list = []

    return {
        "dags": dags_list,
        "running_tasks": running_tasks,
        "success_today": success_today,
        "failed_today": failed_today,
        "running_count": running_count,
    }


def _get_system_daemons() -> List[Dict[str, Any]]:
    """Inspect core host services, launchd agents, and docker ports."""
    launchd = _get_launchd_status()

    daemons = [
        {
            "id": "quant_ai",
            "name": "Quant AI 核心服务",
            "type": "macOS Launchd",
            "label": "com.quant.ai",
            "port": 18000,
            "is_port_open": _is_port_open(18000),
            "pid": launchd.get("com.quant.ai", {}).get("pid"),
            "status": "running" if _is_port_open(18000) else "stopped",
            "description": "提供量化分析、投资组合、宏观资讯、AI Agent 和监控 API",
            "log_path": str(QUANT_AI_LOG),
        },
        {
            "id": "airflow_scheduler",
            "name": "Airflow 定时调度器",
            "type": "macOS Launchd",
            "label": "com.quant.airflow-scheduler",
            "port": None,
            "is_port_open": None,
            "pid": launchd.get("com.quant.airflow-scheduler", {}).get("pid"),
            "status": "running" if launchd.get("com.quant.airflow-scheduler", {}).get("is_running") else "stopped",
            "description": "驱动全部 quant_data 量化管道、模型周训与 job_agent 全球求职定时任务",
            "log_path": str(AIRFLOW_SCHEDULER_LOG),
        },
        {
            "id": "job_agent",
            "name": "找工中控 Web 服务 (Job Agent)",
            "type": "macOS Launchd + Airflow",
            "label": "com.xiangyu.jobagent.web",
            "port": 5050,
            "is_port_open": _is_port_open(5050),
            "pid": launchd.get("com.xiangyu.jobagent.web", {}).get("pid"),
            "status": "running" if _is_port_open(5050) else "stopped",
            "description": "5050 端口多国求职看板 (IE/EU/中东/Remote/中国)；定时抓取由 Airflow 每日 11:00 & 16:00 驱动",
            "log_path": str(JOB_AGENT_LOG_DIR / "job_agent.log"),
        },
        {
            "id": "lmstudio",
            "name": "LM Studio 本地大模型",
            "type": "macOS Launchd / Standalone",
            "label": "com.quant.lmstudio-server",
            "port": 1234,
            "is_port_open": _is_port_open(1234),
            "pid": launchd.get("com.quant.lmstudio-server", {}).get("pid"),
            "status": "running" if _is_port_open(1234) else "stopped",
            "description": "本地 LLM 推理接口，服务于 quant_ai 与 job_agent 简历匹配",
            "log_path": str(Path(os.path.expanduser("~/Library/Logs/quant_ai/lmstudio.launchd.log"))),
        },
        {
            "id": "disk_watch",
            "name": "Data4T 磁盘健康看门狗",
            "type": "macOS Launchd",
            "label": "com.quant.disk-watch",
            "port": None,
            "is_port_open": None,
            "pid": launchd.get("com.quant.disk-watch", {}).get("pid"),
            "status": "loaded" if "com.quant.disk-watch" in launchd else "stopped",
            "description": "监控 4TB 外接高速盘空间并防止数据溢出",
            "log_path": str(DISK_WATCH_LOG),
        },
        {
            "id": "airflow_postgres",
            "name": "Airflow 元数据库 (Postgres)",
            "type": "Docker 容器",
            "label": "airflow-postgres",
            "port": 25432,
            "is_port_open": _is_port_open(25432),
            "pid": None,
            "status": "running" if _is_port_open(25432) else "stopped",
            "description": "存储所有 DAG 执行记录、任务状态与调度元数据",
            "log_path": None,
        },
        {
            "id": "quant_mlflow",
            "name": "MLflow 实验与模型追踪",
            "type": "Docker 容器",
            "label": "mlflow",
            "port": 15050,
            "is_port_open": _is_port_open(15050),
            "pid": None,
            "status": "running" if _is_port_open(15050) else "stopped",
            "description": "记录周度模型训练指标与模型注册表",
            "log_path": None,
        },
        {
            "id": "quant_ui",
            "name": "Quant UI 前端交互面板",
            "type": "Docker 容器",
            "label": "quant_ui",
            "port": 18080,
            "is_port_open": _is_port_open(18080),
            "pid": None,
            "status": "running" if _is_port_open(18080) else "stopped",
            "description": "量化投资组合管理、信号监控与定时任务大盘界面",
            "log_path": None,
        },
    ]
    return daemons



# ---------------------------------------------------------------- English for the dashboard
# The dashboard shows English unless the viewer switches to Chinese, so every Chinese display text
# below also goes out in English as "<field>_en" (_add_english); the Chinese fields are unchanged.
EN: Dict[str, str] = {
    "今日求职精选报告已生成": "Today's job report is ready",
    "找工平台 (Job Agent)": "Job Agent",
    "每日 11:00 & 16:00 都柏林时间 (Airflow 统一调度: job_agent_morning / job_agent_afternoon)":
        "Daily 11:00 & 16:00 Dublin time (Airflow: job_agent_morning / job_agent_afternoon)",
    "每日量化信号主管道 (EOD Signals)": "Daily signal pipeline (EOD Signals)",
    "量化主管道": "Quant pipeline",
    "收盘宏观/新闻指标聚合，特征工程，多因子打分与仓位风控决策":
        "End-of-day macro and news features, multi-factor scoring and position risk decisions",
    "GDELT 新闻抓取与 SLM 匹配": "GDELT news collection and SLM matching",
    "数据采集与匹配": "Collection & matching",
    "全球财经与科技新闻每日自动采集，经过SLM模型精准实体匹配入库":
        "Daily collection of global business and tech news, matched to companies by the SLM",
    "新闻特征丰富与向量化嵌入": "News enrichment and embeddings",
    "数据处理与特征": "Processing & features",
    "新闻事件情绪打分、聚类与向量数据库入库，下游触发信号主管道":
        "News sentiment scoring, clustering and vector store; triggers the signal pipeline",
    "历史日K行情增量回填": "Daily price history backfill",
    "行情与历史数据": "Prices & history",
    "同步每日最新股票收盘价、复权因子与交易量到 TimescaleDB":
        "Syncs daily closes, adjustment factors and volume to TimescaleDB",
    "实时盘中突发新闻采集": "Intraday breaking news",
    "实时盘中监控": "Intraday monitoring",
    "每30分钟采集盘中Breaking News与美股异动快讯": "Breaking news and US market movers every 30 minutes",
    "散户与社交情绪监控": "Retail and social sentiment",
    "美股主流散户社区情绪指数与热度异动采集": "Sentiment and buzz from the main US retail investor communities",
    "AI 预测模型周度再训练": "Weekly model retraining",
    "模型每周训练": "Weekly training",
    "每周日自动化重新训练LightGBM多因子回归模型与MLflow版本记录":
        "Retrains the LightGBM multi-factor model every Sunday, versioned in MLflow",
    "机构 13F 持仓季报更新": "Institutional 13F filings",
    "机构与资金流": "Institutions & flows",
    "更新华尔街顶级机构 SEC 13F 季度调仓数据": "Quarterly SEC 13F position changes of the largest institutions",
    "GDELT 数据健康度验证": "GDELT data health check",
    "系统健康检查": "System health",
    "每周定期校验 GDELT 数据完整性与空洞检测": "Weekly GDELT completeness and gap check",
    "新闻验证与打分校准": "News validation and score calibration",
    "新闻样本交叉校验与标签模型评估": "Cross-checks news samples and evaluates the labelling model",
    "全球求职早班扫描 (IE + 中国全量 + 欧中东官网)": "Job scan, morning (IE + China full + EU/ME company sites)",
    "全球求职监控": "Job search",
    "每日 11:00 都柏林(19:00 北京)：爱尔兰早班全扫 + 中国全职(201家外企官网/国内/外包) + 中国远程兼职 + 欧英/中东官网快扫":
        "Daily 11:00 Dublin (19:00 Beijing): Ireland full scan + China full-time (201 company sites / domestic / outsourcing) + China remote part-time + EU/UK & Middle East company sites",
    "全球求职午后扫描 (IE + 欧英/中东/Remote全量 + 中国外企官网)":
        "Job scan, afternoon (IE + EU/UK, Middle East, Remote full + China company sites)",
    "每日 16:00 都柏林：爱尔兰午后全扫 + 欧洲12国/中东3国/全球Remote国家级全扫 + 中国201家外企官网快扫":
        "Daily 16:00 Dublin: Ireland full scan + 12 European / 3 Middle East countries / Remote + 201 company sites in China",
    "GDELT 小时级新闻采集与匹配": "Hourly GDELT news and matching",
    "每小时 20 分增量抓取全球 GDELT GKG 新闻并通过 SLM 模型完成公司实体匹配":
        "At :20 every hour: new GDELT GKG news, matched to companies by the SLM",
    "持仓盘中/收盘 AI 操作建议 (Daily Advice)": "Daily model advice for holdings (Daily Advice)",
    "交易日 13:30 / 17:30 / 21:30 UTC 为当前持仓生成模型动作评估建议":
        "Trading days 13:30 / 17:30 / 21:30 UTC: the models' action for each holding",
    "期权与分析师评级收盘快照": "Options and analyst ratings snapshot",
    "交易日 21:15 UTC 抓取股票池与持仓的期权隐含波动率及华尔街分析师目标价快照":
        "Trading days 21:15 UTC: implied volatility and analyst price targets for the universe and holdings",
    "公司匹配规则关键词周度挖掘": "Weekly keyword mining for company rules",
    "每周六 08:00 都柏林时间自动生成新增公司匹配关键词候选供 UI 审核":
        "Saturdays 08:00 Dublin: new keyword candidates for the company rules, for review in the UI",
    "公司关键词审核自动生效": "Approved keywords applied",
    "每日 08:55–23:55 每小时将 UI 审核通过的关键词同步写入匹配规则库":
        "Hourly 08:55–23:55: keywords approved in the UI are written to the matching rules",
    "其他任务": "Other",
    "上游联动触发 (backfill_2 完成后自动执行 · 约 04:30–05:10 UTC)": "After backfill_2 finishes (about 04:30–05:10 UTC)",
    "上游联动触发 (backfill_1 完成后自动执行 · 约 04:15–04:30 UTC)": "After backfill_1 finishes (about 04:15–04:30 UTC)",
    "按需手动触发 (On-demand 离线质检与抽样)": "On demand (offline quality checks and sampling)",
    "按需 / 上游联动触发": "On demand / triggered upstream",
    "每日 04:00 UTC (都柏林冬令时 04:00 / 夏令时 05:00)": "Daily 04:00 UTC (Dublin 04:00 winter / 05:00 summer)",
    "每日 06:30 UTC (14:30 本地)": "Daily 06:30 UTC (14:30 local)",
    "每 30 分钟": "Every 30 minutes",
    "每日 20:30 UTC (04:30 本地)": "Daily 20:30 UTC (04:30 local)",
    "每周日 06:00 UTC": "Sundays 06:00 UTC",
    "每周日 07:00 UTC": "Sundays 07:00 UTC",
    "每周日 04:00 UTC": "Sundays 04:00 UTC",
    "每日 11:00 都柏林时间 (19:00 北京 · 自动冬/夏令时)": "Daily 11:00 Dublin time (19:00 Beijing)",
    "每日 16:00 都柏林时间 (自动冬/夏令时)": "Daily 16:00 Dublin time",
    "每小时 20 分 UTC (03:00–06:00 让位夜间主管道)": "At :20 every hour UTC (not 03:00–06:00, the nightly pipeline's slot)",
    "交易日 13:30 / 17:30 / 21:30 UTC": "Trading days 13:30 / 17:30 / 21:30 UTC",
    "交易日 21:15 UTC (美股收盘后)": "Trading days 21:15 UTC (after the US close)",
    "每日 08:55–23:55 都柏林时间 (每小时 55 分)": "Daily 08:55–23:55 Dublin time (at :55)",
    "每周六 08:00 都柏林时间": "Saturdays 08:00 Dublin time",
    "Quant AI 核心服务": "Quant AI core service",
    "提供量化分析、投资组合、宏观资讯、AI Agent 和监控 API":
        "Quant analysis, portfolio, macro news, AI agent and monitoring APIs",
    "Airflow 定时调度器": "Airflow scheduler",
    "驱动全部 quant_data 量化管道、模型周训与 job_agent 全球求职定时任务":
        "Runs every scheduled pipeline: quant_data, weekly training and the job scans",
    "找工中控 Web 服务 (Job Agent)": "Job Agent web service",
    "5050 端口多国求职看板 (IE/EU/中东/Remote/中国)；定时抓取由 Airflow 每日 11:00 & 16:00 驱动":
        "Job board on port 5050 (IE / EU / Middle East / Remote / China); scans run from Airflow at 11:00 & 16:00",
    "LM Studio 本地大模型": "LM Studio local models",
    "本地 LLM 推理接口，服务于 quant_ai 与 job_agent 简历匹配": "Local LLM inference for quant_ai and the job matching",
    "Data4T 磁盘健康看门狗": "Data4T disk watchdog",
    "监控 4TB 外接高速盘空间并防止数据溢出": "Watches free space on the 4 TB external disk",
    "Airflow 元数据库 (Postgres)": "Airflow metadata DB (Postgres)",
    "Docker 容器": "Docker container",
    "存储所有 DAG 执行记录、任务状态与调度元数据": "Every DAG run, task state and schedule",
    "MLflow 实验与模型追踪": "MLflow experiments and models",
    "记录周度模型训练指标与模型注册表": "Weekly training metrics and the model registry",
    "Quant UI 前端交互面板": "Quant UI dashboard",
    "量化投资组合管理、信号监控与定时任务大盘界面": "Portfolio, signals and scheduled-task dashboard",
    "找工平台 ➔ 全网职位扫描与匹配中": "Job Agent ➔ scanning and matching jobs",
}
# Fixed order of the category filter on the Task Monitor page
CATEGORIES = ["量化主管道", "全球求职监控", "数据采集与匹配", "数据处理与特征", "行情与历史数据",
              "实时盘中监控", "机构与资金流", "模型每周训练", "系统健康检查"]
_TEXT_KEYS = ("name", "category", "description", "desc", "schedule_display", "type", "countdown",
              "schedule", "summary", "regions_summary")
_CJK = re.compile(r"[一-鿿]")


def _english(text: str) -> str:
    """The English for one display text; built ones (a countdown, the regions line) by pattern."""
    if text in EN:
        return EN[text]
    head, arrow, tail = text.partition(" ➔ ")      # a running step: "<DAG name> ➔ <task id>"
    if arrow and head in EN:
        return f"{EN[head]} ➔ {tail}"
    out = re.sub(r"(\d+)小时 (\d+)分钟后", r"in \1h \2m", text)
    out = re.sub(r"(\d+)分钟后", r"in \1m", out)
    return out.replace("中东", "Middle East")


def _add_english(obj: Any) -> Any:
    """Adds "<field>_en" beside every Chinese display field, in place, at any depth."""
    if isinstance(obj, dict):
        for key in list(obj):
            value = obj[key]
            if key in _TEXT_KEYS and isinstance(value, str) and _CJK.search(value):
                obj[f"{key}_en"] = _english(value)
            else:
                _add_english(value)
    elif isinstance(obj, list):
        for item in obj:
            _add_english(item)
    return obj


def get_task_monitor_status() -> Dict[str, Any]:
    """Unified entrypoint: collects full monitor state across all systems."""
    now = datetime.datetime.now(datetime.timezone.utc)
    airflow_info = _get_airflow_data()
    job_agent_info = _get_job_agent_info()
    daemons = _get_system_daemons()

    # Combine running tasks (avoid duplicating Airflow-driven job_agent_* tasks)
    running_tasks = list(airflow_info["running_tasks"])
    airflow_has_job_agent = any((t.get("dag_id") or "").startswith("job_agent_") for t in running_tasks)
    if job_agent_info["is_scraping_now"] and not airflow_has_job_agent:
        running_tasks.append({
            "source": "job_agent",
            "dag_id": "job_agent",
            "task_id": "scrape_and_match",
            "run_id": "scheduled",
            "name": "找工平台 ➔ 全网职位扫描与匹配中",
            "state": "running",
            "start_date": None,
            "duration_seconds": None,
            "can_view_log": True,
        })

    total_running = len(running_tasks)
    active_dags = sum(1 for d in airflow_info["dags"] if not d["is_paused"])

    return _add_english({
        "status": "success",
        "timestamp": now.isoformat(),
        "summary": {
            "running_count": total_running,
            "success_today": airflow_info["success_today"],
            "failed_today": airflow_info["failed_today"],
            "total_dags": len(airflow_info["dags"]),
            "active_dags": active_dags,
            "job_agent_online": job_agent_info["is_online"],
            "scheduler_running": any(d["id"] == "airflow_scheduler" and d["status"] == "running" for d in daemons),
        },
        "running_tasks": running_tasks,
        "job_agent": job_agent_info,
        "airflow_dags": airflow_info["dags"],
        "system_daemons": daemons,
        "categories": [{"zh": c, "en": EN[c]} for c in CATEGORIES],
    })


def get_task_log(
    source: str,
    target_id: str,
    run_id: Optional[str] = None,
    task_id: Optional[str] = None,
    lines: int = 150
) -> Dict[str, Any]:
    """Tail task logs based on source and identifier."""
    target_file: Optional[Path] = None
    lines = min(max(lines, 20), 1000)

    if source == "job_agent":
        target_file = JOB_AGENT_LOG_DIR / "job_agent.log"
        if target_id == "web":
            target_file = JOB_AGENT_LOG_DIR / "web_server.log"
        elif target_id == "launchd":
            cand = JOB_AGENT_LOG_DIR / "launchd.log"
            web_cand = JOB_AGENT_LOG_DIR / "web_server.log"
            target_file = cand if (cand.exists() and cand.stat().st_size > 0) else web_cand
    elif source == "scheduler":
        target_file = AIRFLOW_SCHEDULER_LOG
    elif source == "quant_ai":
        target_file = QUANT_AI_LOG
    elif source == "disk_watch":
        target_file = DISK_WATCH_LOG
    elif source == "airflow":
        dag_id = target_id
        dag_dir = AIRFLOW_LOGS_DIR / f"dag_id={dag_id}"
        if dag_dir.exists():
            # If specific run and task are provided
            if run_id and task_id:
                task_dir = dag_dir / f"run_id={run_id}" / f"task_id={task_id}"
                if task_dir.exists():
                    attempts = list(task_dir.glob("attempt=*.log"))
                    if attempts:
                        target_file = max(attempts, key=lambda p: int(p.stem.split("=")[-1]))
            # If only run_id is provided, find latest updated task log
            elif run_id:
                run_dir = dag_dir / f"run_id={run_id}"
                if run_dir.exists():
                    logs = list(run_dir.glob("task_id=*/attempt=*.log"))
                    if logs:
                        target_file = max(logs, key=lambda p: p.stat().st_mtime)
            # If nothing provided, find latest run and latest task log
            if not target_file:
                logs = list(dag_dir.glob("run_id=*/task_id=*/attempt=*.log"))
                if logs:
                    target_file = max(logs, key=lambda p: p.stat().st_mtime)

        # Fallback to scheduler log if no specific log found
        if not target_file or not target_file.exists():
            target_file = AIRFLOW_SCHEDULER_LOG

    if not target_file or not target_file.exists():
        return {
            "status": "error",
            "message": f"Log file not found for source={source}, target={target_id}",
            "lines": [],
            "path": str(target_file) if target_file else None,
        }

    try:
        raw_lines = target_file.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = raw_lines[-lines:]
        return {
            "status": "success",
            "source": source,
            "target_id": target_id,
            "run_id": run_id,
            "task_id": task_id,
            "path": str(target_file),
            "total_lines": len(raw_lines),
            "lines_returned": len(tail),
            "log": "\n".join(tail),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "lines": [],
            "path": str(target_file),
        }

