import pytest
from markets import task_monitor


def test_get_task_monitor_status():
    res = task_monitor.get_task_monitor_status()
    assert res["status"] == "success"
    assert "summary" in res
    assert "airflow_dags" in res
    assert "job_agent" in res
    assert "system_daemons" in res
    assert res["job_agent"]["name"] == "找工平台 (Job Agent)"
    assert res["job_agent"]["port"] == 5050
    assert "Airflow" in res["job_agent"]["schedule"]
    assert "next_run" in res["job_agent"]

    # Ensure job_agent Airflow DAGs are mapped under 全球求职监控 and not 其他任务
    m1 = task_monitor._get_dag_display_meta("job_agent_morning")
    m2 = task_monitor._get_dag_display_meta("job_agent_afternoon")
    assert m1["category"] == "全球求职监控"
    assert m2["category"] == "全球求职监控"

    # Ensure daemon label tracks com.xiangyu.jobagent.web
    ja_daemon = next(d for d in res["system_daemons"] if d["id"] == "job_agent")
    assert ja_daemon["label"] == "com.xiangyu.jobagent.web"


def test_get_task_log():
    # Test job agent log
    res = task_monitor.get_task_log(source="job_agent", target_id="job_agent", lines=10)
    assert res["status"] in ("success", "error")
    if res["status"] == "success":
        assert "log" in res
        assert res["source"] == "job_agent"

    # Test web server log via web and launchd fallback
    res_web = task_monitor.get_task_log(source="job_agent", target_id="web", lines=10)
    assert res_web["status"] in ("success", "error")

    # Test airflow scheduler log
    res_sched = task_monitor.get_task_log(source="scheduler", target_id="scheduler", lines=10)
    assert res_sched["status"] == "success"
    assert len(res_sched["log"]) > 0


def test_every_chinese_display_field_also_goes_out_in_english():
    from markets import task_monitor as tm
    out = tm._add_english({
        "job_agent": {"name": "找工平台 (Job Agent)", "next_run": {"countdown": "2小时 5分钟后"}},
        "airflow_dags": [{"name": "每日量化信号主管道 (EOD Signals)", "category": "量化主管道",
                          "schedule_display": "每 30 分钟", "dag_id": "x"}],
        "system_daemons": [{"name": "Airflow 定时调度器", "type": "Docker 容器", "description": "plain English"}],
    })
    assert out["job_agent"]["name_en"] == "Job Agent"
    assert out["job_agent"]["next_run"]["countdown_en"] == "in 2h 5m"
    dag = out["airflow_dags"][0]
    assert (dag["name_en"], dag["category_en"], dag["schedule_display_en"]) == (
        "Daily signal pipeline (EOD Signals)", "Quant pipeline", "Every 30 minutes")
    assert dag["name"] == "每日量化信号主管道 (EOD Signals)"          # the Chinese field is kept
    assert "description_en" not in out["system_daemons"][0]           # English text needs no copy
    assert all(c in tm.EN for c in tm.CATEGORIES)
    assert tm._english("实时盘中突发新闻采集 ➔ finnhub_news") == "Intraday breaking news ➔ finnhub_news"
