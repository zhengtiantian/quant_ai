"""Copy markets.db to Google Drive a few seconds after a change (one copy per burst of saves)."""

import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .config import BACKUP_PATH, DB_PATH, connect_db

DELAY = 5.0
_timer: Optional[threading.Timer] = None
_lock = threading.Lock()


def backup_now(source: Path = DB_PATH, target: Path = BACKUP_PATH) -> bool:
    if not source.exists() or not target.parent.exists():
        return False
    # Snapshot locally in small stepped chunks first so Google Drive FUSE I/O never holds a SQLite lock.
    local_tmp = source.with_name(source.name + ".backup.tmp")
    gdrive_tmp = target.with_name(target.name + ".tmp")
    src = connect_db(source)
    dst = sqlite3.connect(str(local_tmp))
    try:
        try:
            src.backup(dst, pages=200, sleep=0.02)
        finally:
            dst.close()
            src.close()
        shutil.copyfile(local_tmp, gdrive_tmp)
        gdrive_tmp.replace(target)
        return True
    finally:
        local_tmp.unlink(missing_ok=True)
        gdrive_tmp.unlink(missing_ok=True)


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
