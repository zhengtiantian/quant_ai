"""One-off: move the portfolio from job_agent's dashboard into quant_ai (markets/), 2026-10-22.

Copies transactions (ids kept), the watchlist and the quote / chart caches, then checks that
holdings, average costs and realised P&L come out identical from both databases. Sales whose
profit job_agent had already written to the budget are adopted by quant_ai (linked, not written
again), via job_agent's own budget module.

    .venv/bin/python tools/migrate_portfolio_from_job_agent.py            # refuses if data exists
    .venv/bin/python tools/migrate_portfolio_from_job_agent.py --replace  # re-run at cutover
"""

import argparse
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import budget_link, portfolio  # noqa: E402
from markets.config import DB_PATH  # noqa: E402

SOURCE = Path.home() / "Library/Application Support/job_agent/portfolio.db"
JOB_AGENT_PY = Path.home() / ".venvs/job_agent/bin/python"
JOB_AGENT_DIR = Path.home() / "Library/CloudStorage/GoogleDrive-biaszh00001@gmail.com/My Drive/17.找工作/job_agent"
TABLES = ("transactions", "watchlist", "quotes", "charts", "global_quotes")


def summary(path: Path):
    conn = portfolio._conn(path)
    try:
        positions, problems = portfolio.compute_holdings(portfolio._transactions(conn))
    finally:
        conn.close()
    keep = ("quantity", "cost", "avg_cost", "realized", "dividends", "fifo_cost", "fifo_realized")
    return {s: tuple(round(p.get(k) or 0, 6) for k in keep) for s, p in positions.items()}, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--target", type=Path, default=DB_PATH)
    ap.add_argument("--replace", action="store_true", help="overwrite data already in the target")
    args = ap.parse_args()

    portfolio.init_db(args.target)
    portfolio.init_db(args.source)          # same migrations on both, so the columns line up
    conn = sqlite3.connect(str(args.target))
    try:
        if conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] and not args.replace:
            print(f"{args.target} already has transactions; use --replace to overwrite")
            return 1
    finally:
        conn.close()
    if args.target.exists():
        backup = args.target.with_name(f"{args.target.stem}.{time.strftime('%Y-%m-%d-%H%M%S')}-before-migration.db")
        shutil.copy2(args.target, backup)
        print(f"target backed up to {backup}")

    conn = sqlite3.connect(str(args.target))
    try:
        conn.execute("ATTACH DATABASE ? AS src", (str(args.source),))
        for t in TABLES:
            cols = [r[1] for r in conn.execute(f"PRAGMA src.table_info({t})")]
            mine = {r[1] for r in conn.execute(f"PRAGMA main.table_info({t})")}
            shared = ", ".join(c for c in cols if c in mine)
            conn.execute(f"DELETE FROM main.{t}")
            conn.execute(f"INSERT INTO main.{t} ({shared}) SELECT {shared} FROM src.{t}")
            n = conn.execute(f"SELECT COUNT(*) FROM main.{t}").fetchone()[0]
            print(f"  {t}: {n} rows")
        conn.commit()
        conn.execute("DETACH DATABASE src")
    finally:
        conn.close()

    old, new = summary(args.source), summary(args.target)
    if old != new:
        print("MISMATCH between source and target holdings:", old, new)
        return 2
    print(f"holdings identical: {len(new[0])} symbols, problems {new[1] or 'none'}")

    conn = portfolio._conn(args.target)
    try:
        linked = [(r["id"], r["budget_entry_id"]) for r in conn.execute(
            "SELECT id, budget_entry_id FROM transactions WHERE budget_entry_id IS NOT NULL")]
    finally:
        conn.close()
    if linked:
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import budget\n"
                "for pair in sys.argv[2:]:\n"
                "    tid, eid = pair.split(':')\n"
                f"    budget.link_external_entry({budget_link.SOURCE!r}, 'tx:' + tid, int(eid))\n")
        subprocess.run([str(JOB_AGENT_PY), "-c", code, str(JOB_AGENT_DIR), *[f"{t}:{e}" for t, e in linked]], check=True)
        print(f"budget lines adopted: {len(linked)}")
    else:
        print("no sale was linked to the budget; nothing to adopt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
