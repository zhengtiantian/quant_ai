import pytest
from quant_ai.markets import task_monitor


def test_get_task_monitor_status():
    res = task_monitor.get_task_monitor_status()
    assert res["status"] == "success"
    assert "summary" in res
    assert "airflow_dags" in res
    assert "job_agent" in res
    assert "system_daemons" in res
    assert res["job_agent"]["name"] == "找工平台 (Job Agent)"
    assert res["job_agent"]["port"] == 5050
    assert "next_run" in res["job_agent"]


def test_get_task_log():
    # Test job agent log
    res = task_monitor.get_task_log(source="job_agent", target_id="job_agent", lines=10)
    assert res["status"] in ("success", "error")
    if res["status"] == "success":
        assert "log" in res
        assert res["source"] == "job_agent"

    # Test airflow scheduler log
    res_sched = task_monitor.get_task_log(source="scheduler", target_id="scheduler", lines=10)
    assert res_sched["status"] == "success"
    assert len(res_sched["log"]) > 0
