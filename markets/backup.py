"""Copy markets.db to Google Drive a few seconds after a change (one copy per burst of saves)."""

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .config import BACKUP_PATH, DB_PATH

DELAY = 5.0
_timer: Optional[threading.Timer] = None
_lock = threading.Lock()


def backup_now(source: Path = DB_PATH, target: Path = BACKUP_PATH) -> bool:
    if not source.exists() or not target.parent.exists():
        return False
    tmp = target.with_name(target.name + ".tmp")
    src, dst = sqlite3.connect(str(source), timeout=30), sqlite3.connect(str(tmp))
    try:
        src.backup(dst)                  # a consistent snapshot even while the server writes
    finally:
        dst.close()
        src.close()
    tmp.replace(target)
    return True


def schedule_backup() -> None:
    global _timer
    with _lock:
        if _timer:
            _timer.cancel()
        _timer = threading.Timer(DELAY, lambda: _safe(backup_now))
        _timer.daemon = True
        _timer.start()


def _safe(fn) -> None:
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - a missed backup must never break a save
        print(f"[markets] backup failed: {type(e).__name__}: {e}")
