"""One background thread for the markets jobs that run without anyone looking at a page:
watching the quant_data pipeline's freshness
(pipeline_health.py), (briefs.py) writing the daily briefs, (pulse.py) watching a holding's
after-the-close earnings release, (hint_scores.py) logging and scoring blocked buys and
(devils_advocate.py) the case against adding after a large drop."""

import threading
import time

TICK = 60                      # seconds between checks
_started = threading.Event()


def _loop() -> None:
    while True:
        try:
            from . import pipeline_health
            pipeline_health.run_due()
        except Exception as e:  # noqa: BLE001
            print(f"[markets] pipeline check failed: {type(e).__name__}: {e}")
        try:
            from . import pulse
            pulse.run_due()                    # report evenings: the after-the-close pulse watch
            pulse.score_due()                  # the morning after: its orders replayed on 1-minute bars
        except Exception as e:  # noqa: BLE001
            print(f"[markets] pulse check failed: {type(e).__name__}: {e}")
        try:
            from . import briefs
            briefs.run_due()
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001 - the next tick tries again
            print(f"[markets] briefs check failed: {type(e).__name__}: {e}")
        try:
            from . import catalysts
            catalysts.run_due()
        except Exception as e:  # noqa: BLE001
            print(f"[markets] catalysts check failed: {type(e).__name__}: {e}")
        try:
            from . import hint_scores
            hint_scores.run_due()              # after the close: the buy gate probed and logged; due rows scored
        except Exception as e:  # noqa: BLE001
            print(f"[markets] hint scoring failed: {type(e).__name__}: {e}")
        try:
            from . import devils_advocate
            devils_advocate.watch_due()        # market hours: a held stock down >= 2 x ATR gets the case against adding
        except Exception as e:  # noqa: BLE001
            print(f"[markets] devil's advocate check failed: {type(e).__name__}: {e}")
        time.sleep(TICK)


def start() -> None:
    if _started.is_set():
        return
    _started.set()
    threading.Thread(target=_loop, name="markets-background", daemon=True).start()
    # analysts' ratings: its own thread, since a pass over every followed stock takes a minute or more
    from . import analysts
    analysts.start()
    # cases against adding left queued or running by a restart
    from . import devils_advocate
    devils_advocate.resume()
