"""Regression tests for SQLite WAL concurrency and backup decoupling on markets.db."""
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from markets import backup, portfolio as pf


class MarketsSqliteConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "markets.db"
        self.backup_dst = Path(self.tmp.name) / "backup" / "markets.db"
        self.backup_dst.parent.mkdir(parents=True, exist_ok=True)
        pf.init_db(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_wal_enabled_and_concurrent_writers_never_lock(self):
        conn = pf._conn(self.db)
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(str(mode).lower(), "wal")

        errors: list[BaseException] = []
        barrier = threading.Barrier(6)

        def watch_worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5)
                for j in range(15):
                    sym = f"SYM{idx}{chr(65 + (j % 5))}"
                    pf.add_watch(sym, note=f"w-{j}", path=self.db)
                    if j % 3 == 0:
                        pf.remove_watch(sym, path=self.db)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def cash_and_backup_worker() -> None:
            try:
                barrier.wait(timeout=5)
                for j in range(12):
                    pf.set_cash(1000 + j, path=self.db)
                    if j % 4 == 0:
                        backup.backup_now(source=self.db, target=self.backup_dst)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def reader_worker() -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(25):
                    c = pf._conn(self.db)
                    try:
                        c.execute("SELECT COUNT(*) FROM watchlist").fetchone()
                    finally:
                        c.close()
                    pf.get_cash(path=self.db)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = (
            [threading.Thread(target=watch_worker, args=(i,)) for i in range(4)]
            + [threading.Thread(target=cash_and_backup_worker)]
            + [threading.Thread(target=reader_worker)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
            self.assertFalse(t.is_alive(), "worker thread timed out")

        self.assertEqual(errors, [])
        self.assertTrue(self.backup_dst.exists())
        bconn = sqlite3.connect(str(self.backup_dst))
        try:
            cnt = bconn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='watchlist'").fetchone()[0]
            self.assertEqual(cnt, 1)
        finally:
            bconn.close()
