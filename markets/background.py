"""One background thread for the markets jobs that run without anyone looking at a page:
retrying a budget push that job_agent missed, and (briefs.py) writing the daily briefs."""

import threading
import time

from . import budget_link

TICK = 60                      # seconds between checks
BUDGET_RETRY = 300             # a pending budget push is retried this often
_started = threading.Event()


def _loop() -> None:
    last_budget = 0.0
    while True:
        now = time.time()
        if now - last_budget >= BUDGET_RETRY:
            last_budget = now
            try:
                if budget_link.retry_pending():
                    print("[markets] pending budget lines delivered to job_agent")
            except Exception as e:  # noqa: BLE001
                print(f"[markets] budget retry failed: {type(e).__name__}: {e}")
        try:
            from . import briefs
            briefs.run_due()
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001 - the next tick tries again
            print(f"[markets] briefs check failed: {type(e).__name__}: {e}")
        time.sleep(TICK)


def start() -> None:
    if _started.is_set():
        return
    _started.set()
    threading.Thread(target=_loop, name="markets-background", daemon=True).start()
