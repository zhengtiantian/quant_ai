"""Daily market briefs, written by Claude Code in headless mode (`claude -p`) on the user's
subscription, in Chinese and English.

Two a day:
  asia_europe  14:00 Dublin, weekdays: Asia has closed and Europe is mid-session.
  us           20:05 New York, weekdays: after the after-hours session ends.

Each run takes a snapshot of the numbers (global exchanges, the macro board, the user's holdings
and watchlist), hands it to Claude with web search allowed, and stores the answer. The numbers in
the brief come from the snapshot; the reasons come from the news Claude finds and cites.

Missed runs (the Mac asleep) are caught up only while the snapshot would still describe that
session: an Asia/Europe brief until the end of that Dublin day, a US brief until the next US
pre-market. A failed run is retried a few times, then left failed with its error for the page.

No API key on purpose: with ANTHROPIC_API_KEY removed from the environment the CLI uses the
logged-in claude.ai subscription. That keeps this personal-use only.
"""

import glob
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from . import global_markets, portfolio
from .config import DATA_DIR, DB_PATH

DUBLIN = ZoneInfo("Europe/Dublin")
NEW_YORK = ZoneInfo("America/New_York")
KINDS = {
    "asia_europe": {"title": "Asia & Europe", "zone": DUBLIN, "at": dtime(14, 0)},
    "us": {"title": "US markets", "zone": NEW_YORK, "at": dtime(20, 5)},
}
MAX_ATTEMPTS = 3
RETRY_AFTER = 20 * 60          # seconds between attempts after a failure
STALE_RUNNING = 30 * 60        # a "running" row older than this was a crashed run
TIMEOUT = 15 * 60              # one claude run
WORK_DIR = DATA_DIR / "briefs_work"   # empty cwd, so no project CLAUDE.md is picked up
MODEL = os.getenv("BRIEFS_MODEL", "").strip()

# Extra US symbols for the US brief: sector ETFs and the largest companies.
US_SECTORS = [("XLK", "Technology"), ("SMH", "Semiconductors"), ("XLC", "Communication"), ("XLY", "Consumer discretionary"),
              ("XLP", "Consumer staples"), ("XLF", "Financials"), ("XLV", "Health care"), ("XLI", "Industrials"),
              ("XLE", "Energy"), ("XLU", "Utilities"), ("XLB", "Materials"), ("XLRE", "Real estate")]
US_MEGACAPS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "TSM", "JPM", "LLY", "XOM"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS briefs (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, session_date TEXT NOT NULL,
    status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, trigger TEXT,
    started_at REAL, finished_at REAL, duration_s REAL,
    headline_zh TEXT, headline_en TEXT, body_zh TEXT, body_en TEXT, sources TEXT,
    snapshot TEXT, error TEXT, raw TEXT, model TEXT, cost_usd REAL,
    UNIQUE (kind, session_date));
"""
_run_lock = threading.Lock()
_running: Dict[str, bool] = {}


def _conn(path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------- when

def _weekday_on_or_before(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def due_session(kind: str, now: Optional[datetime] = None) -> Optional[str]:
    """The session date a brief of this kind should exist for right now, or None."""
    k = KINDS[kind]
    local = (now or datetime.now(ZoneInfo("UTC"))).astimezone(k["zone"])
    if kind == "asia_europe":
        # From 14:00 to the end of that Dublin weekday; later the Asian session has moved on.
        return local.date().isoformat() if local.weekday() < 5 and local.time() >= k["at"] else None
    # US: the latest weekday whose 20:05 has passed, until the next weekday's 04:00 pre-market.
    d = local.date() if local.time() >= k["at"] else local.date() - timedelta(days=1)
    d = _weekday_on_or_before(d)
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    deadline = datetime.combine(nxt, dtime(4, 0), NEW_YORK)
    return d.isoformat() if local < deadline else None


def current_session(kind: str, now: Optional[datetime] = None) -> str:
    """For a run asked for by hand: the due session, else the market's own date today."""
    return due_session(kind, now) or (now or datetime.now(ZoneInfo("UTC"))).astimezone(KINDS[kind]["zone"]).date().isoformat()


def scheduled_at(kind: str, session: str) -> float:
    """When the scheduled brief of this session is written (UTC timestamp)."""
    k = KINDS[kind]
    return datetime.combine(date.fromisoformat(session), k["at"], k["zone"]).timestamp()


def _claim(kind: str, session: str, trigger: str, force: bool, path: Optional[Path]) -> bool:
    """Mark this brief as running unless it is done, running, or out of attempts. A brief
    written by hand before the scheduled time does not count as the scheduled one: at the
    scheduled time it is written again, with that later snapshot."""
    conn = _conn(path)
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM briefs WHERE kind = ? AND session_date = ?", (kind, session)).fetchone()
        now = time.time()
        if row:
            if row["status"] == "running" and now - (row["started_at"] or 0) < STALE_RUNNING:
                conn.execute("ROLLBACK")
                return False
            early = trigger == "schedule" and row["status"] == "done" and (row["finished_at"] or 0) < scheduled_at(kind, session)
            if early:
                pass
            elif not force and (row["status"] == "done" or row["attempts"] >= MAX_ATTEMPTS
                              or (row["status"] == "failed" and now - (row["finished_at"] or 0) < RETRY_AFTER)):
                conn.execute("ROLLBACK")
                return False
            attempts = 1 if force or early else row["attempts"] + 1
            conn.execute("UPDATE briefs SET status = 'running', attempts = ?, trigger = ?, started_at = ?, error = NULL "
                         "WHERE id = ?", (attempts, trigger, now, row["id"]))
        else:
            conn.execute("INSERT INTO briefs(kind, session_date, status, attempts, trigger, started_at) "
                         "VALUES (?, ?, 'running', 1, ?, ?)", (kind, session, trigger, now))
        conn.execute("COMMIT")
        return True
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------- the numbers

def _slim(q: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    return {k: (round(q[k], 4) if isinstance(q.get(k), float) else q.get(k)) for k in keys if q.get(k) is not None}


def _portfolio_snapshot(path: Optional[Path]) -> Dict[str, Any]:
    st = portfolio.get_state(path=path)             # fresh quotes (waits for Nasdaq; fine in the background)
    t = st["totals"]
    return {
        "us_session_now": st["session"],
        "totals": _slim(t, "market_value", "cost", "unrealized", "unrealized_pct", "day_pnl", "day_pnl_pct"),
        "holdings": [{"symbol": h["symbol"], "name": (h.get("quote") or {}).get("name"),
                      "weight_pct": round(h["weight"], 1) if h.get("weight") is not None else None,
                      **_slim(h, "quantity", "avg_cost", "day_change_pct", "unrealized_pct"),
                      "quote": _slim(h.get("quote") or {}, "price", "prev_close", "session", "regular_price",
                                     "session_change_pct")} for h in st["holdings"]],
        "watchlist": [{"symbol": w["symbol"], **_slim(w, "day_change_pct"),
                       "quote": _slim(w.get("quote") or {}, "price", "session", "regular_price", "session_change_pct")}
                      for w in st["watchlist"]],
    }


def _global_snapshot(path: Optional[Path]) -> List[Dict[str, Any]]:
    global_markets.refresh_global(path)
    g = global_markets.global_state(path, background=False)
    out = []
    for r in g["regions"]:
        for c in r["countries"]:
            for e in c["exchanges"]:
                out.append({"region": r["name"], "country": c["name"], "exchange": e["name"], "open_now": e["open"],
                            "local_time": e["local_time"], "hours": e["hours"],
                            "indices": [{"name": i["name"], **_slim(i, "price", "change_pct", "currency")} for i in e["indices"]],
                            "stocks": [{"name": i["name"], **_slim(i, "price", "change_pct", "currency")} for i in e["stocks"]]})
    return out


def _macro_snapshot(path: Optional[Path]) -> Dict[str, Any]:
    m = global_markets.macro_state(path, max_age=0)
    return {g["name"]: [{"name": i["name"], **_slim(i, "price", "change", "change_pct")} for i in g["items"]] for g in m["groups"]}


def snapshot(kind: str, path: Optional[Path] = None) -> Dict[str, Any]:
    now = datetime.now(ZoneInfo("UTC"))
    snap: Dict[str, Any] = {"taken_at_utc": now.strftime("%Y-%m-%d %H:%M"),
                            "dublin_time": now.astimezone(DUBLIN).strftime("%Y-%m-%d %H:%M %Z"),
                            "new_york_time": now.astimezone(NEW_YORK).strftime("%Y-%m-%d %H:%M %Z")}
    for key, fn in (("macro", _macro_snapshot), ("my_portfolio", _portfolio_snapshot)):
        try:
            snap[key] = fn(path)
        except Exception as e:  # noqa: BLE001 - a brief with a gap beats no brief
            snap[key] = {"unavailable": f"{type(e).__name__}: {e}"}
    if kind == "asia_europe":
        try:
            snap["exchanges"] = _global_snapshot(path)
        except Exception as e:  # noqa: BLE001
            snap["exchanges"] = {"unavailable": f"{type(e).__name__}: {e}"}
    else:
        try:
            q = global_markets.fetch_cnbc([s for s, _ in US_SECTORS] + US_MEGACAPS)
            snap["us_sectors"] = [{"etf": s, "sector": n, **_slim(q.get(s, {}), "price", "change_pct")} for s, n in US_SECTORS]
            snap["us_megacaps"] = [{"symbol": s, **_slim(q.get(s, {}), "price", "change_pct")} for s in US_MEGACAPS]
        except Exception as e:  # noqa: BLE001
            snap["us_sectors"] = {"unavailable": f"{type(e).__name__}: {e}"}
    return snap


# ---------------------------------------------------------------- the prompt

SECTIONS = ("HEADLINE_ZH", "HEADLINE_EN", "BODY_ZH", "BODY_EN", "SOURCES")

PROMPT = """You are writing the user's daily {title} market brief for {session} (a personal note, not advice
for anyone else). The user is in Dublin, holds the US stocks listed under my_portfolio and follows the
watchlist there.

Below is a snapshot of the numbers taken just now. Use these numbers as given; do not replace them with
numbers from the web. Use web search to find out WHY markets moved: the news, data releases, earnings,
policy decisions and flows behind the biggest moves. Search in a focused way (about 5 to 10 searches),
prefer reputable financial news from today, and keep track of the pages you rely on.

{focus}

Write the brief twice, the same content in Simplified Chinese and in English. Each version, in Markdown:
  ## Bottom line            (2-3 sentences: what happened and the main reason)
  ## What drove it          (the 2-4 main drivers, each tied to the moves it explains)
  ## Winners and losers     (sectors / regions / stocks that diverged, with the snapshot numbers)
  ## My holdings            (what today means for each holding and the watchlist that matters; be concrete)
  ## What to watch next     (events, data, earnings in the next sessions)
Use the Chinese section titles 一句话结论 / 主要驱动 / 分化：谁涨谁跌 / 对我的持仓 / 接下来关注 in the Chinese version.
Say plainly when a reason is your inference rather than something a source reports. 300-600 words per version.

Reply with exactly these five blocks and nothing before or after them:
=====HEADLINE_ZH=====
(one line, at most 40 Chinese characters)
=====HEADLINE_EN=====
(one line, at most 20 words)
=====BODY_ZH=====
(the Chinese Markdown)
=====BODY_EN=====
(the English Markdown)
=====SOURCES=====
(one per line: - [title](url))

SNAPSHOT (JSON):
{snapshot}
"""

FOCUS = {
    "asia_europe": "Focus: today's Asian sessions (closed) and European sessions (still trading), what they "
                   "imply for tonight's US open (see the US futures in macro), and the macro board.",
    "us": "Focus: today's US regular session and the after-hours session that just ended (sectors, megacaps, "
          "earnings released after the close), the macro board, and what it sets up for Asia tomorrow.",
}


def build_prompt(kind: str, session: str, snap: Dict[str, Any]) -> str:
    return PROMPT.format(title=KINDS[kind]["title"], session=session, focus=FOCUS[kind],
                         snapshot=json.dumps(snap, ensure_ascii=False, separators=(",", ":")))


def parse_reply(text: str) -> Dict[str, Any]:
    parts = re.split(r"^=====([A-Z_]+)=====\s*$", text or "", flags=re.M)
    blocks = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2) if parts[i] in SECTIONS}
    missing = [s for s in SECTIONS[:4] if not blocks.get(s)]
    if missing:
        raise ValueError(f"reply is missing {', '.join(missing)}")
    sources = [{"title": m.group(1).strip(), "url": m.group(2).strip()}
               for m in re.finditer(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", blocks.get("SOURCES", ""))]
    return {"headline_zh": blocks["HEADLINE_ZH"].splitlines()[0].strip(), "headline_en": blocks["HEADLINE_EN"].splitlines()[0].strip(),
            "body_zh": blocks["BODY_ZH"], "body_en": blocks["BODY_EN"], "sources": sources}


# ---------------------------------------------------------------- the CLI

def find_claude() -> Optional[str]:
    """CLAUDE_BIN, then PATH, then the usual install spots, then the newest VS Code extension's
    bundled binary (its folder name changes with every extension update)."""
    for cand in (os.getenv("CLAUDE_BIN", "").strip(), shutil.which("claude") or "",
                 str(Path.home() / ".local/bin/claude"), str(Path.home() / ".claude/local/claude")):
        if cand and os.access(cand, os.X_OK):
            return cand
    found = glob.glob(str(Path.home() / ".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"))
    ver = lambda p: [int(x) for x in re.findall(r"claude-code-(\d+)\.(\d+)\.(\d+)", p)[0]] if re.search(r"claude-code-\d+\.\d+\.\d+", p) else [0]
    found = [p for p in found if os.access(p, os.X_OK)]
    return max(found, key=ver) if found else None


def run_claude(prompt: str) -> Dict[str, Any]:
    binary = find_claude()
    if not binary:
        raise RuntimeError("claude CLI not found (set CLAUDE_BIN)")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    cmd = [binary, "-p", "--output-format", "json", "--no-session-persistence",
           "--tools", "WebSearch,WebFetch", "--allowedTools", "WebSearch,WebFetch"]
    if MODEL:
        cmd += ["--model", MODEL]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=str(WORK_DIR), env=env, timeout=TIMEOUT)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude exited {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}")
    if out.get("is_error") or proc.returncode != 0:
        raise RuntimeError(f"claude error: {str(out.get('result') or proc.stderr)[-500:]}")
    return out


def _notify(title: str, text: str) -> None:
    if not shutil.which("osascript"):
        return
    script = f'display notification {json.dumps(text[:180])} with title {json.dumps(title)}'
    subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)


# ---------------------------------------------------------------- one run

def generate(kind: str, session: Optional[str] = None, trigger: str = "schedule", force: bool = False,
             path: Optional[Path] = None) -> Optional[int]:
    """Write one brief. Returns its row id, or None when there was nothing to do."""
    if kind not in KINDS:
        raise ValueError(f"unknown brief kind {kind!r}")
    session = session or current_session(kind)
    if not _claim(kind, session, trigger, force, path):
        return None
    started = time.time()
    snap: Dict[str, Any] = {}
    raw = None
    try:
        snap = snapshot(kind, path)
        out = run_claude(build_prompt(kind, session, snap))
        raw = out.get("result") or ""
        parsed = parse_reply(raw)
        models = list((out.get("modelUsage") or {}).keys())
        fields = {**parsed, "sources": json.dumps(parsed["sources"], ensure_ascii=False), "status": "done", "error": None,
                  "model": ",".join(models) or None, "cost_usd": out.get("total_cost_usd")}
    except Exception as e:  # noqa: BLE001 - stored with the row, retried by run_due
        fields = {"status": "failed", "error": f"{type(e).__name__}: {e}"[:2000]}
    fields.update(snapshot=json.dumps(snap, ensure_ascii=False), raw=raw, finished_at=time.time(),
                  duration_s=round(time.time() - started, 1))
    conn = _conn(path)
    try:
        conn.execute(f"UPDATE briefs SET {', '.join(f'{k} = ?' for k in fields)} WHERE kind = ? AND session_date = ?",
                     (*fields.values(), kind, session))
        conn.commit()
        row = conn.execute("SELECT id, attempts FROM briefs WHERE kind = ? AND session_date = ?", (kind, session)).fetchone()
    finally:
        conn.close()
    title = f"{KINDS[kind]['title']} brief"
    if fields["status"] == "done":
        _notify(title, fields["headline_zh"])
    elif row["attempts"] >= MAX_ATTEMPTS or force:
        _notify(f"{title} failed", fields["error"])
    return row["id"]


def run_due(path: Optional[Path] = None) -> List[int]:
    """Called every minute by the background loop."""
    done = []
    for kind in KINDS:
        session = due_session(kind)
        if session:
            rid = generate(kind, session, "schedule", path=path)
            if rid:
                done.append(rid)
    return done


def generate_in_background(kind: str, path: Optional[Path] = None) -> bool:
    """For the page's "Generate now" button: regenerate the current session's brief."""
    if kind not in KINDS:
        raise ValueError(f"unknown brief kind {kind!r}")
    with _run_lock:
        if _running.get(kind):
            return False
        _running[kind] = True

    def run():
        try:
            generate(kind, trigger="manual", force=True, path=path)
        finally:
            with _run_lock:
                _running[kind] = False

    threading.Thread(target=run, name=f"brief-{kind}", daemon=True).start()
    return True


# ---------------------------------------------------------------- reading

_LIST_COLS = ("id", "kind", "session_date", "status", "attempts", "trigger", "started_at", "finished_at", "duration_s",
              "headline_zh", "headline_en", "error", "model", "cost_usd")


def list_briefs(kind: str = "", limit: int = 60, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    conn = _conn(path)
    try:
        sql = f"SELECT {', '.join(_LIST_COLS)} FROM briefs"
        args: List[Any] = []
        if kind:
            sql += " WHERE kind = ?"
            args.append(kind)
        rows = conn.execute(sql + " ORDER BY session_date DESC, kind LIMIT ?", (*args, int(limit))).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_brief(brief_id: int, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    conn = _conn(path)
    try:
        row = conn.execute("SELECT * FROM briefs WHERE id = ?", (int(brief_id),)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    d = dict(row)
    d["sources"] = json.loads(d["sources"]) if d.get("sources") else []
    d["snapshot"] = json.loads(d["snapshot"]) if d.get("snapshot") else None
    d.pop("raw", None)
    return d


def status() -> Dict[str, Any]:
    """Schedule facts for the page: which brief is due now, and whether one is being written."""
    return {"kinds": {k: {"title": v["title"], "due_session": due_session(k), "running": bool(_running.get(k))}
                      for k, v in KINDS.items()},
            "claude_found": bool(find_claude())}
