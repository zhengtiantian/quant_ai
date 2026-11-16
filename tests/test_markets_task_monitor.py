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
