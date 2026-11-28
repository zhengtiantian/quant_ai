"""Where the markets data lives and what it talks to. All overridable by environment."""

import os
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Optional

DATA_DIR = Path(os.getenv("MARKETS_DATA_DIR", "~/Library/Application Support/quant_ai")).expanduser()
DB_PATH = DATA_DIR / "markets.db"
# A copy after every change, in Google Drive (written by Python: launchd's bash cannot read
# ~/Library/CloudStorage). Skipped quietly when the folder does not exist.
BACKUP_PATH = Path(os.getenv(
    "MARKETS_BACKUP_PATH",
    "~/Library/CloudStorage/GoogleDrive-biaszh00001@gmail.com/My Drive/quant_ai_backup/markets.db")).expanduser()
# job_agent's dashboard owns the budget: the USD rate is read from it.
JOB_AGENT_URL = os.getenv("JOB_AGENT_URL", "http://127.0.0.1:5050").rstrip("/")

DATA_DIR.mkdir(parents=True, exist_ok=True)

_WAL_LOCK = threading.Lock()
_WAL_DONE: set[tuple[str, int]] = set()
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_DONE: set[tuple[str, str, int]] = set()


def _db_key(db_path: Path) -> Optional[tuple[str, int]]:
    """Return (resolved_path, inode) once the SQLite file exists and is non-empty."""
    if str(db_path) == ":memory:" or not db_path.exists():
        return None
    try:
        st = db_path.stat()
    except OSError:
        return None
    if st.st_size <= 0:
        return None
    return (str(db_path.resolve()), int(st.st_ino))


def run_schema_once(
    tag: str,
    path: Optional[Path],
    conn: sqlite3.Connection,
    init_fn: Callable[[sqlite3.Connection], None],
) -> None:
    """Run a schema/migration callback at most once per (tag, resolved_path, inode)."""
    db_path = Path(path) if path is not None else DB_PATH
    key_base = _db_key(db_path)
    if key_base is not None:
        full_key = (tag, key_base[0], key_base[1])
        with _SCHEMA_LOCK:
            if full_key in _SCHEMA_DONE:
                return
    with _SCHEMA_LOCK:
        key_base = _db_key(db_path)
        if key_base is not None:
            full_key = (tag, key_base[0], key_base[1])
            if full_key in _SCHEMA_DONE:
                return
        init_fn(conn)
        conn.commit()
        key_after = _db_key(db_path)
        if key_after is not None:
            _SCHEMA_DONE.add((tag, key_after[0], key_after[1]))


def connect_db(
    path: Optional[Path] = None,
    *,
    timeout: float = 30.0,
    immediate: bool = False,
    schema_tag: Optional[str] = None,
    schema_fn: Optional[Callable[[sqlite3.Connection], None]] = None,
) -> sqlite3.Connection:
    """Open a SQLite connection configured for WAL + busy_timeout, optionally initializing schema and BEGIN IMMEDIATE."""
    db_path = Path(path) if path is not None else DB_PATH
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    conn.execute("PRAGMA synchronous = NORMAL")

    if str(db_path) != ":memory:":
        key = _db_key(db_path)
        if key is None or key not in _WAL_DONE:
            with _WAL_LOCK:
                key = _db_key(db_path)
                if key is None or key not in _WAL_DONE:
                    conn.execute("PRAGMA journal_mode = WAL")
                    key_after = _db_key(db_path)
                    if key_after is not None:
                        _WAL_DONE.add(key_after)

    if schema_tag and schema_fn is not None:
        run_schema_once(schema_tag, db_path, conn, schema_fn)

    if immediate:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")

    return conn
