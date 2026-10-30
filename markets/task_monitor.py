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
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

AIRFLOW_POSTGRES_URI = os.getenv(
    "AIRFLOW_DATABASE_URI",
    "postgresql://airflow:airflow@127.0.0.1:25432/airflow"
)
AIRFLOW_LOGS_DIR = Path(os.getenv("AIRFLOW_LOGS_DIR", "/Volumes/Data4T/docker-volumes/airflow_logs"))
AIRFLOW_SCHEDULER_LOG = Path("/Users/xiz/Quant_trade/quant_data/airflow_scheduler.out.log")

JOB_AGENT_PORT = 5050
JOB_AGENT_LOG_DIR = Path(os.path.expanduser("~/Library/Logs/job_agent"))
JOB_AGENT_STATE_FILE = Path(os.path.expanduser("~/Library/Application Support/job_agent/scrape_state.json"))
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
    """Calculate the next scheduled run for Job Agent (11:00 & 16:00 daily)."""
    now = datetime.datetime.now()
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
        "next_run_display": next_dt.strftime("%Y-%m-%d %H:%M"),
        "countdown": f"{hours}小时 {minutes}分钟后" if hours > 0 else f"{minutes}分钟后",
    }


def _get_job_agent_info() -> Dict[str, Any]:
    """Gather Job Agent (:5050) metadata, state, next run and report summary."""
    is_online = _is_port_open(JOB_AGENT_PORT)
    launchd = _get_launchd_status().get("com.xiangyu.jobagent", {})
    is_scraping = _get_process_running(["job_agent/run_daily.sh", "job_agent/main.py"])

    # Read state file
    state_data = {}
    if JOB_AGENT_STATE_FILE.exists():
        try:
            state_data = json.loads(JOB_AGENT_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Read report summary
    report_info = {}
    if JOB_AGENT_REPORT_FILE.exists():
        try:
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

    has_log = (JOB_AGENT_LOG_DIR / "job_agent.log").exists()

    return {
        "name": "找工平台 (Job Agent)",
        "url": "http://127.0.0.1:5050/",
        "port": JOB_AGENT_PORT,
        "is_online": is_online,
        "is_scraping_now": is_scraping,
        "schedule": "每日 11:00 & 16:00 (macOS Launchd)",
        "next_run": _calc_job_agent_next_run(),
        "last_success": state_data.get("last_success"),
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
            SELECT dag_id, is_paused, schedule_interval, next_dagrun, description
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
                    sched_display = "上游联动触发 (backfill_2 完成后自动执行 · 约 03:06 UTC / 11:06 本地)"
                elif d_id == "backfill_2_enrich_and_features":
                    sched_display = "上游联动触发 (backfill_1 完成后自动执行 · 约 02:28 UTC / 10:28 本地)"
                elif d_id == "quant_news_validation":
                    sched_display = "按需手动触发 (On-demand 离线质检与抽样)"
                else:
                    sched_display = "按需 / 上游联动触发"
            elif sched_str == "0 2 * * *":
                sched_display = "每日 02:00 UTC (10:00 本地)"
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
            "description": "驱动全部 quant_data 定时任务、数据管道和周度训练",
            "log_path": str(AIRFLOW_SCHEDULER_LOG),
        },
        {
            "id": "job_agent",
            "name": "找工中控平台 (Job Agent)",
            "type": "macOS Launchd & Web",
            "label": "com.xiangyu.jobagent",
            "port": 5050,
            "is_port_open": _is_port_open(5050),
            "pid": None,
            "status": "running" if _is_port_open(5050) else "stopped",
            "description": "爱尔兰资深工程师全网职位扫描，每日 11:00 & 16:00 定时执行",
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


def get_task_monitor_status() -> Dict[str, Any]:
    """Unified entrypoint: collects full monitor state across all systems."""
    now = datetime.datetime.now(datetime.timezone.utc)
    airflow_info = _get_airflow_data()
    job_agent_info = _get_job_agent_info()
    daemons = _get_system_daemons()

    # Combine running tasks
    running_tasks = list(airflow_info["running_tasks"])
    if job_agent_info["is_scraping_now"]:
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

    return {
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
    }


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
        if target_id == "launchd":
            target_file = JOB_AGENT_LOG_DIR / "launchd.log"
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

