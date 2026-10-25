"""Daily market briefs, written by Claude Code in headless mode (`claude -p`) on the user's
subscription, in Chinese and English.

Summaries, two a day:
  asia_europe  14:00 Dublin, weekdays: Asia has closed and Europe is mid-session.
  us           20:05 New York, weekdays: after the after-hours session ends.
Forward-looking:
  outlook      13:30 Dublin, weekdays: earnings of the holdings, the watchlist and the sector
               bellwethers, and the main economic releases, over the next five trading days, with
               what the market expects, the implied move, and both scenarios for each holding.
  flash        whenever a bellwether moves 5% or more in pre-market or after hours (Lam Research
               on 2026-10-21): how the move should travel to Asia, Europe and the holdings.

Holdings advisor (PROJECT_PLAN Q.5, markets/advisor.py):
  advisor      13:00 Dublin, weekdays (US pre-market, 1.5 h before the open): what to do with each
               holding, as limit orders that are stored and scored; plus a focused update whenever
               a 15-minute check finds a reason (a move beyond the stock's threshold, an advised
               limit reached, earnings tomorrow or just out, important news). Only urgent updates
               notify; everything shows in My Holdings.

Every brief ends with checkable predictions (markets/predictions.py), scored against real closes,
so the briefs keep a track record instead of only sounding right.

Each run takes a snapshot of the numbers (global exchanges, the macro board, the user's holdings
and watchlist, and per kind the calendar or the trigger), hands it to Claude with web search
allowed, and stores the answer. The numbers come from the snapshot; the reasons from the news
Claude finds and cites.

Missed scheduled runs (the Mac asleep) are caught up only while the snapshot would still describe
that session. A failed run is retried a few times, then left failed with its error for the page.

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

from . import advisor, backup, events, global_markets, portfolio, predictions
from .config import DATA_DIR, DB_PATH

DUBLIN = ZoneInfo("Europe/Dublin")
NEW_YORK = ZoneInfo("America/New_York")
KINDS = {
    "asia_europe": {"title": "Asia & Europe", "zone": DUBLIN, "at": dtime(14, 0)},
    "us": {"title": "US markets", "zone": NEW_YORK, "at": dtime(20, 5)},
    "outlook": {"title": "Week ahead", "zone": DUBLIN, "at": dtime(13, 30)},
    "flash": {"title": "Flash", "zone": NEW_YORK, "at": None},          # event-driven, no schedule
    "advisor": {"title": "Holdings advisor", "zone": DUBLIN, "at": dtime(13, 0)},
}
MAX_ATTEMPTS = 3
RETRY_AFTER = 20 * 60          # seconds between attempts after a failure
STALE_RUNNING = 30 * 60        # a "running" row older than this was a crashed run
TIMEOUT = 15 * 60              # one claude run
SCORE_EVERY = 600              # seconds between prediction scoring passes
WORK_DIR = DATA_DIR / "briefs_work"   # empty cwd, so no project CLAUDE.md is picked up
MODEL = os.getenv("BRIEFS_MODEL", "").strip()

# Extra US symbols for the US brief: sector ETFs and the largest companies.
US_SECTORS = [("XLK", "Technology"), ("SMH", "Semiconductors"), ("XLC", "Communication"), ("XLY", "Consumer discretionary"),
              ("XLP", "Consumer staples"), ("XLF", "Financials"), ("XLV", "Health care"), ("XLI", "Industrials"),
              ("XLE", "Energy"), ("XLU", "Utilities"), ("XLB", "Materials"), ("XLRE", "Real estate")]
US_MEGACAPS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "TSM", "JPM", "LLY", "XOM"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS briefs (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, session_date TEXT NOT NULL, topic TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, trigger TEXT,
    started_at REAL, finished_at REAL, duration_s REAL,
    headline_zh TEXT, headline_en TEXT, body_zh TEXT, body_en TEXT, sources TEXT,
    snapshot TEXT, error TEXT, raw TEXT, model TEXT, cost_usd REAL, context TEXT,
    UNIQUE (kind, session_date, topic));
"""
_run_lock = threading.Lock()
_running: Dict[str, bool] = {}
_last = {"flash_check": 0.0, "score": 0.0, "advisor_check": 0.0}


def _conn(path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(briefs)")}
    if cols and "topic" not in cols:
        # 2026-10-22: flash briefs are one per company, so (kind, session_date) is no longer unique.
        shared = ", ".join(sorted(cols))
        conn.executescript(f"ALTER TABLE briefs RENAME TO briefs_v1; {SCHEMA}"
                           f"INSERT INTO briefs ({shared}) SELECT {shared} FROM briefs_v1; DROP TABLE briefs_v1;")
    else:
        conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------- when

def _weekday_on_or_before(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def due_session(kind: str, now: Optional[datetime] = None) -> Optional[str]:
    """The session date a scheduled brief of this kind should exist for right now, or None."""
    k = KINDS[kind]
    if k["at"] is None:
        return None
    local = (now or datetime.now(ZoneInfo("UTC"))).astimezone(k["zone"])
    if kind in ("asia_europe", "outlook", "advisor"):
        # From the scheduled time to the end of that Dublin weekday.
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
    """When the scheduled brief of this session is written (UTC timestamp); 0 for flash."""
    k = KINDS[kind]
    if k["at"] is None:
        return 0.0
    return datetime.combine(date.fromisoformat(session), k["at"], k["zone"]).timestamp()


def _claim(kind: str, session: str, topic: str, trigger: str, force: bool, path: Optional[Path]) -> bool:
    """Mark this brief as running unless it is done, running, or out of attempts. A brief
    written by hand before the scheduled time does not count as the scheduled one: at the
    scheduled time it is written again, with that later snapshot."""
    conn = _conn(path)
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM briefs WHERE kind = ? AND session_date = ? AND topic = ?",
                           (kind, session, topic)).fetchone()
        now = time.time()
        if row:
            if row["status"] == "running" and now - (row["started_at"] or 0) < STALE_RUNNING:
                conn.execute("ROLLBACK")
                return False
            early = trigger == "schedule" and row["status"] == "done" and (row["finished_at"] or 0) < scheduled_at(kind, session)
            if not early and not force and (row["status"] == "done" or row["attempts"] >= MAX_ATTEMPTS
                                            or (row["status"] == "failed" and now - (row["finished_at"] or 0) < RETRY_AFTER)):
                conn.execute("ROLLBACK")
                return False
            attempts = 1 if force or early else row["attempts"] + 1
            conn.execute("UPDATE briefs SET status = 'running', attempts = ?, trigger = ?, started_at = ?, error = NULL "
                         "WHERE id = ?", (attempts, trigger, now, row["id"]))
        else:
            conn.execute("INSERT INTO briefs(kind, session_date, topic, status, attempts, trigger, started_at) "
                         "VALUES (?, ?, ?, 'running', 1, ?, ?)", (kind, session, topic, trigger, now))
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
                            "indices": [{"symbol": i["symbol"], "name": i["name"], **_slim(i, "price", "change_pct", "currency")}
                                        for i in e["indices"]],
                            "stocks": [{"symbol": i["symbol"], "name": i["name"], **_slim(i, "price", "change_pct", "currency")}
                                       for i in e["stocks"]]})
    return out


def _macro_snapshot(path: Optional[Path]) -> Dict[str, Any]:
    m = global_markets.macro_state(path, max_age=0)
    return {g["name"]: [{"symbol": i["symbol"], "name": i["name"], **_slim(i, "price", "change", "change_pct")} for i in g["items"]]
            for g in m["groups"]}


def _us_symbols(snap: Dict[str, Any]) -> List[str]:
    p = snap.get("my_portfolio") or {}
    syms = [h["symbol"] for h in p.get("holdings") or []] + [w["symbol"] for w in p.get("watchlist") or []]
    return list(dict.fromkeys(syms + list(events.BELLWETHERS) + US_MEGACAPS + [s for s, _ in US_SECTORS]))


def snapshot(kind: str, path: Optional[Path] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = datetime.now(ZoneInfo("UTC"))
    snap: Dict[str, Any] = {"taken_at_utc": now.strftime("%Y-%m-%d %H:%M"),
                            "dublin_time": now.astimezone(DUBLIN).strftime("%Y-%m-%d %H:%M %Z (%a)"),
                            "new_york_time": now.astimezone(NEW_YORK).strftime("%Y-%m-%d %H:%M %Z (%a)")}
    if context:
        snap["trigger"] = context
    for key, fn in (("macro", _macro_snapshot), ("my_portfolio", _portfolio_snapshot)):
        try:
            snap[key] = fn(path)
        except Exception as e:  # noqa: BLE001 - a brief with a gap beats no brief
            snap[key] = {"unavailable": f"{type(e).__name__}: {e}"}
    if kind in ("asia_europe", "flash", "outlook"):
        try:
            snap["exchanges"] = _global_snapshot(path)
        except Exception as e:  # noqa: BLE001
            snap["exchanges"] = {"unavailable": f"{type(e).__name__}: {e}"}
    if kind in ("us", "flash"):
        try:
            q = global_markets.fetch_cnbc([s for s, _ in US_SECTORS] + US_MEGACAPS)
            snap["us_sectors"] = [{"etf": s, "sector": n, **_slim(q.get(s, {}), "price", "change_pct")} for s, n in US_SECTORS]
            snap["us_megacaps"] = [{"symbol": s, **_slim(q.get(s, {}), "price", "change_pct")} for s in US_MEGACAPS]
        except Exception as e:  # noqa: BLE001
            snap["us_sectors"] = {"unavailable": f"{type(e).__name__}: {e}"}
    if kind == "outlook":
        prices = {h["symbol"]: (h.get("quote") or {}).get("price") for h in (snap.get("my_portfolio") or {}).get("holdings") or []}
        try:
            bw = portfolio.get_quotes(list(events.BELLWETHERS), path=path, cached_only=False)
            prices.update({s: q.get("price") for s, q in bw.items() if q.get("price")})
        except Exception:  # noqa: BLE001
            pass
        try:
            snap["calendar"] = events.upcoming(_us_symbols(snap), path=path, prices=prices)
        except Exception as e:  # noqa: BLE001
            snap["calendar"] = {"unavailable": f"{type(e).__name__}: {e}"}
        snap["bellwethers"] = {s: {"name": n, "group": g} for s, (n, g) in events.BELLWETHERS.items()}
    if kind == "advisor":
        try:
            snap["advisor"] = advisor.build(path, focus=(context or {}).get("symbol"))
        except Exception as e:  # noqa: BLE001
            snap["advisor"] = {"unavailable": f"{type(e).__name__}: {e}"}
    if kind == "flash" and context:
        try:
            today = datetime.now(NEW_YORK).date()
            rows = events.earnings_on(today, path) + events.earnings_on(today - timedelta(days=1), path)
            snap["trigger_earnings"] = [r for r in rows if r["symbol"] == context.get("symbol")]
        except Exception:  # noqa: BLE001
            pass
    return snap


# ---------------------------------------------------------------- the prompt

SECTIONS = ("HEADLINE_ZH", "HEADLINE_EN", "BODY_ZH", "BODY_EN", "SOURCES", "PREDICTIONS", "ORDERS")

PROMPT = """You are writing the user's {title} market brief for {session} (a personal note, not advice
for anyone else). The user is in Dublin, holds the US stocks listed under my_portfolio and follows the
watchlist there.

Below is a snapshot of the numbers taken just now. Use these numbers as given; do not replace them with
numbers from the web. Use web search to find out WHY markets moved or what is expected: the news, data
releases, earnings, guidance, policy decisions, geopolitics and flows behind it. Search in a focused way
(about 5 to 10 searches), prefer reputable financial news, and keep track of the pages you rely on.

{focus}

Write the brief twice, the same content in Simplified Chinese and in English, in Markdown, with these
sections ({sections_note}):
{sections}
Say plainly when a reason is your inference rather than something a source reports. 300-700 words per
version.

Then make checkable predictions about the next sessions: {n_preds} of them, the ones you consider most
informative, each a JSON object on its own line:
{{"symbol": "<a symbol from the snapshot>", "date": "YYYY-MM-DD", "op": ">=|<=|abs>=|abs<=", "threshold": <percent>,
 "confidence": <0.5-0.95>, "zh": "<the claim in Chinese>", "en": "<the claim in English>"}}
meaning: that symbol's change on that trading date (its close against the previous close, in %, in the
exchange's own calendar) satisfies op threshold; abs>= / abs<= are about the size of the move either way.
Use only symbols that appear in the snapshot (exchanges indices/stocks by their "symbol", US tickers from
my_portfolio, the US indices .SPX .IXIC .DJI .SOX .RUT, or a ticker named in the snapshot); Korean
stocks are not scorable, use .KS11 / .KQ11. Pick dates that are trading days for that exchange. Prefer
thresholds that make the call informative (not "moves less than 20%"), and state an honest confidence:
the calls will be scored against real closes.
{extra_rules}
Reply with exactly these blocks and nothing before or after them:
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
=====PREDICTIONS=====
(one JSON object per line){extra_blocks}

SNAPSHOT (JSON):
{snapshot}
"""

SUMMARY_SECTIONS = """  ## Bottom line            (2-3 sentences: what happened and the main reason)
  ## What drove it          (the 2-4 main drivers, each tied to the moves it explains)
  ## Winners and losers     (sectors / regions / stocks that diverged, with the snapshot numbers)
  ## My holdings            (what today means for each holding and the watchlist that matters; be concrete)
  ## What to watch next     (events, data, earnings in the next sessions)"""
SUMMARY_ZH = "一句话结论 / 主要驱动 / 分化：谁涨谁跌 / 对我的持仓 / 接下来关注"

OUTLOOK_SECTIONS = """  ## Bottom line            (the 2-3 events of the next five trading days that matter most for this portfolio)
  ## Earnings calendar      (each relevant report from snapshot.calendar.earnings: date and timing, consensus EPS,
                             what the market focuses on (guidance, capex, margins...), the implied move when given)
  ## Scenarios              (for each important report: beat / miss, and what each would likely do to which holdings
                             and to which Asian / European stocks, through which chain)
  ## Macro and policy       (the main releases from snapshot.calendar.economic with consensus and previous, central
                             banks, and geopolitics such as oil or conflicts, and their channel to these holdings)
  ## Positioning notes      (concentration and exposure of the portfolio to these events; no buy/sell instructions)"""
OUTLOOK_ZH = "一句话结论 / 财报日历 / 情景推演 / 宏观与政策 / 持仓风险提示"

FLASH_SECTIONS = """  ## What happened          (the trigger: who moved, how much, in which session, and why, from the news)
  ## How it travels         (the chain: suppliers, customers, competitors, the sector; which Asian and European stocks
                             and indices should react in their next session, and by roughly how much, with reasons)
  ## My holdings            (for each holding and relevant watchlist name: likely direction at the next open, size, why)
  ## What would change this (what could stop or reverse the move: conference call details, peer results, macro)"""
FLASH_ZH = "发生了什么 / 传导链条 / 对我的持仓 / 什么会改变判断"

ADVISOR_SECTIONS = """  ## Bottom line            (the 2-3 actions that matter today, with prices; or why the best move is to wait)
  ## Holding by holding     (for each position in snapshot.advisor.positions: where it stands (cost, break-even, % to get
                             back to cost, weight), what is driving it now (news, earnings, sector, with sources), the plan
                             as concrete orders (limit, size, validity) and what happens if the call is wrong)
  ## Earnings plays         (for each holding reporting within the calendar: the implied move, the plan before and after,
                             sized as a part of the position; say plainly that selling before a gap up means buying back
                             higher or not at all)
  ## New ideas              (0-3 names, normally from snapshot.advisor.buy_candidates, preferring ones that reduce the
                             concentration in chips and memory; entry limit and size from the cash, or the switch that pays)
  ## Risks and rules        (concentration, the Irish four-week rule where a loss sale and re-buy is advised, what would
                             change this plan)"""
ADVISOR_ZH = "一句话结论 / 逐只持仓 / 财报应对 / 新的机会 / 风险与规则"

FOCUS = {
    "asia_europe": "Focus: today's Asian sessions (closed) and European sessions (still trading), what they "
                   "imply for tonight's US open (see the US futures in macro), and the macro board.",
    "us": "Focus: today's US regular session and the after-hours session that just ended (sectors, megacaps, "
          "earnings released after the close), the macro board, and what it sets up for Asia tomorrow.",
    "outlook": "Focus: the next five trading days. snapshot.calendar lists the earnings of the holdings, the "
               "watchlist and the sector bellwethers (snapshot.bellwethers), with consensus EPS and, when the option "
               "chain gave it, the implied move from the at-the-money straddle; it also lists the main economic "
               "releases. Search for what each report hinges on and for upcoming geopolitical or policy risks.",
    "flash": "Focus: snapshot.trigger is a sector bellwether that just moved sharply outside regular hours. Explain "
             "the move from the news, then trace how it should travel to the Asian and European exchanges in "
             "snapshot.exchanges (their next sessions) and to the user's holdings at the next US open.",
    "advisor": "You are the user's personal trading adviser for the positions in snapshot.advisor.positions. The user's "
               "priority is profit: first getting the losing positions back to cost (INTC is the largest loss and weight), "
               "then growing. Give concrete, sized, priced actions rather than commentary, and only actions you would "
               "defend: doing nothing is a valid answer. Lowering the cost basis by trimming into strength and re-buying "
               "lower is welcome when the volatility supports it, always on part of the position. Use the computed "
               "levels in each position (band from the options-implied move near earnings, else 2 x ATR; 20/60-day "
               "support and resistance; break-even) for prices. snapshot.advisor.quant_model and buy_candidates come from "
               "the platform's own ranking model; news_7d from its news store (search the web for anything newer and "
               "for SK hynix / Seagate, which the store does not cover). Respect cash_usd.",
}
ADVISOR_FOCUS_UPDATE = (" This is a focused update: snapshot.trigger says which holding and why it was written now. Put "
                        "that holding first and in depth, revise or cancel its open orders (listed in snapshot.trigger) "
                        "and give the other holdings one line each.")


def build_prompt(kind: str, session: str, snap: Dict[str, Any], topic: str = "") -> str:
    sections, zh, n = {"outlook": (OUTLOOK_SECTIONS, OUTLOOK_ZH, "4 to 8"), "flash": (FLASH_SECTIONS, FLASH_ZH, "4 to 8"),
                       "advisor": (ADVISOR_SECTIONS, ADVISOR_ZH, "2 to 4")}.get(kind, (SUMMARY_SECTIONS, SUMMARY_ZH, "2 to 4"))
    title = f"{KINDS[kind]['title']}: {topic}" if topic else KINDS[kind]["title"]
    focus = FOCUS[kind] + (ADVISOR_FOCUS_UPDATE if kind == "advisor" and topic else "")
    extra_rules, extra_blocks = "", ""
    if kind == "advisor":
        extra_rules = "\n" + advisor.ORDER_RULES.format(max_days=advisor.ORDER_MAX_DAYS) + "\n"
        extra_blocks = "\n=====ORDERS=====\n(one JSON object per line, or nothing)"
    return PROMPT.format(title=title, session=session, focus=focus, sections=sections, n_preds=n,
                         sections_note=f"in the Chinese version titled {zh}", extra_rules=extra_rules, extra_blocks=extra_blocks,
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
            "body_zh": blocks["BODY_ZH"], "body_en": blocks["BODY_EN"], "sources": sources,
            "predictions_text": blocks.get("PREDICTIONS", ""), "orders_text": blocks.get("ORDERS", "")}


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
             path: Optional[Path] = None, topic: str = "", context: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Write one brief. Returns its row id, or None when there was nothing to do."""
    if kind not in KINDS:
        raise ValueError(f"unknown brief kind {kind!r}")
    session = session or current_session(kind)
    if not _claim(kind, session, topic, trigger, force, path):
        return None
    started = time.time()
    snap: Dict[str, Any] = {}
    raw = None
    preds: List[Dict[str, Any]] = []
    orders: List[Dict[str, Any]] = []
    try:
        snap = snapshot(kind, path, context)
        out = run_claude(build_prompt(kind, session, snap, topic))
        raw = out.get("result") or ""
        parsed = parse_reply(raw)
        preds, dropped = predictions.parse_block(parsed.pop("predictions_text"), predictions.universe(_us_symbols(snap)),
                                                 date.fromisoformat(session), issued_ts=started)
        if dropped:
            print(f"[briefs] {kind} {session} {topic}: dropped predictions: {dropped}")
        orders_text = parsed.pop("orders_text")
        if kind == "advisor":
            adv = snap.get("advisor") or {}
            prices = {p["symbol"]: p.get("price") for p in adv.get("positions") or []}
            prices.update({c["symbol"]: c.get("price") for c in adv.get("buy_candidates") or [] if c.get("price")})
            orders, dropped_orders = advisor.parse_orders(orders_text, prices, date.fromisoformat(session))
            if dropped_orders:
                print(f"[briefs] advisor {session} {topic}: dropped orders: {dropped_orders}")
        models = list((out.get("modelUsage") or {}).keys())
        fields = {**parsed, "sources": json.dumps(parsed["sources"], ensure_ascii=False), "status": "done", "error": None,
                  "model": ",".join(models) or None, "cost_usd": out.get("total_cost_usd")}
    except Exception as e:  # noqa: BLE001 - stored with the row, retried by run_due
        fields = {"status": "failed", "error": f"{type(e).__name__}: {e}"[:2000]}
    fields.update(snapshot=json.dumps(snap, ensure_ascii=False), raw=raw, finished_at=time.time(),
                  duration_s=round(time.time() - started, 1), context=json.dumps(context) if context else None)
    conn = _conn(path)
    try:
        conn.execute(f"UPDATE briefs SET {', '.join(f'{k} = ?' for k in fields)} WHERE kind = ? AND session_date = ? AND topic = ?",
                     (*fields.values(), kind, session, topic))
        conn.commit()
        row = conn.execute("SELECT id, attempts FROM briefs WHERE kind = ? AND session_date = ? AND topic = ?",
                           (kind, session, topic)).fetchone()
    finally:
        conn.close()
    if fields["status"] == "done":
        predictions.save(row["id"], kind, preds, path)
        if kind == "advisor":
            advisor.save_orders(row["id"], orders, started, path)
    if path is None:
        backup.schedule_backup()
    title = f"{KINDS[kind]['title']}{f' · {topic}' if topic else ''} brief"
    if fields["status"] == "done":
        # The advisor shows in My Holdings; only an urgent update interrupts (Q.5).
        if kind != "advisor" or (context or {}).get("urgent"):
            _notify(title, fields["headline_zh"])
    elif row["attempts"] >= MAX_ATTEMPTS or force:
        _notify(f"{title} failed", fields["error"])
    return row["id"]


def _flash_count(us_date: str, path: Optional[Path]) -> int:
    conn = _conn(path)
    try:
        return conn.execute("SELECT COUNT(*) FROM briefs WHERE kind = 'flash' AND session_date = ?", (us_date,)).fetchone()[0]
    finally:
        conn.close()


def check_flash(path: Optional[Path] = None) -> List[int]:
    """During pre-market and after hours: a flash brief for each bellwether moving FLASH_THRESHOLD %
    or more, once per company and US date, at most FLASH_DAILY_CAP a day."""
    if portfolio.session_now() not in ("pre", "post"):
        return []
    us_date = datetime.now(NEW_YORK).date().isoformat()
    done = []
    for m in events.big_moves(path):
        if _flash_count(us_date, path) >= events.FLASH_DAILY_CAP:
            break
        rid = generate("flash", us_date, "schedule", path=path, topic=m["symbol"], context=m)
        if rid:
            done.append(rid)
    return done


def run_due(path: Optional[Path] = None) -> List[int]:
    """Called every minute by the background loop: scheduled briefs, flash triggers, scoring."""
    done = []
    for kind in ("advisor", "outlook", "asia_europe", "us"):
        session = due_session(kind)
        if session:
            rid = generate(kind, session, "schedule", path=path)
            if rid:
                done.append(rid)
    now = time.time()
    if now - _last["flash_check"] >= events.FLASH_CHECK_EVERY:
        _last["flash_check"] = now
        done += check_flash(path)
    if now - _last["advisor_check"] >= advisor.TRIGGER_EVERY:
        _last["advisor_check"] = now
        done += check_advisor(path)
    if now - _last["score"] >= SCORE_EVERY:
        _last["score"] = now
        if predictions.score_due(path) + advisor.score_due(path):
            if path is None:
                backup.schedule_backup()
    return done


def _advisor_topic(symbol: str) -> str:
    return f"{symbol} {datetime.now(DUBLIN):%H:%M}"


def check_advisor(path: Optional[Path] = None) -> List[int]:
    """Every 15 minutes: a focused advisor update for each holding with a reason to look again."""
    try:
        found = advisor.check_triggers(path)
    except Exception as e:  # noqa: BLE001 - the check runs again in 15 minutes
        print(f"[briefs] advisor trigger check failed: {type(e).__name__}: {e}")
        return []
    done = []
    session = datetime.now(DUBLIN).date().isoformat()
    for t in found:
        t = {**t, "open_orders": [{k: o[k] for k in ("id", "side", "tag", "pct", "limit_price", "valid_until", "status", "en")}
                                  for o in advisor.active_orders(path) if o["symbol"] == t["symbol"]]}
        rid = generate("advisor", session, "event", path=path, topic=_advisor_topic(t["symbol"]), context=t)
        if rid:
            done.append(rid)
    return done


def generate_in_background(kind: str, path: Optional[Path] = None, symbol: str = "") -> bool:
    """For the page's buttons: write the current session's brief now. A flash needs a symbol;
    its trigger is that symbol's latest move (whatever its size)."""
    if kind not in KINDS:
        raise ValueError(f"unknown brief kind {kind!r}")
    symbol = symbol.strip().upper()
    if kind == "flash" and not re.fullmatch(r"[A-Z][A-Z.\-]{0,9}", symbol):
        raise ValueError("a flash brief needs a US ticker, e.g. LRCX")
    if kind == "advisor" and symbol and not re.fullmatch(r"[A-Z][A-Z.\-]{0,9}", symbol):
        raise ValueError("bad ticker")
    key = f"{kind}:{symbol}"
    with _run_lock:
        if _running.get(key):
            return False
        _running[key] = True

    def run():
        try:
            context = None
            if kind == "flash":
                q = portfolio.get_quotes([symbol], path=path, force=True).get(symbol) or {}
                move = q.get("session_change_pct") if q.get("session") in ("pre", "post") else None
                if move is None and q.get("price") and q.get("prev_close"):
                    move = (q["price"] / q["prev_close"] - 1) * 100
                context = {"symbol": symbol, "name": events.BELLWETHERS.get(symbol, (q.get("name"), None))[0] or symbol,
                           "group": events.BELLWETHERS.get(symbol, (None, "user's pick"))[1], "session": q.get("session"),
                           "price": q.get("price"), "regular_close": q.get("regular_price") or q.get("prev_close"),
                           "move_pct": round(move, 2) if move is not None else None, "asked_by_hand": True}
            topic = symbol if kind == "flash" else ""
            if kind == "advisor" and symbol:
                topic = _advisor_topic(symbol)
                context = {"symbol": symbol, "urgent": False, "reasons": [{"kind": "manual", "text": "asked by hand", "urgent": False}],
                           "open_orders": [o for o in advisor.active_orders(path) if o["symbol"] == symbol]}
            generate(kind, session=datetime.now(DUBLIN).date().isoformat() if kind == "advisor" else None,
                     trigger="manual", force=True, path=path, topic=topic, context=context)
        finally:
            with _run_lock:
                _running[key] = False

    threading.Thread(target=run, name=f"brief-{key}", daemon=True).start()
    return True


# ---------------------------------------------------------------- reading

_LIST_COLS = ("id", "kind", "session_date", "topic", "status", "attempts", "trigger", "started_at", "finished_at", "duration_s",
              "headline_zh", "headline_en", "error", "model", "cost_usd")


def list_briefs(kind: str = "", limit: int = 60, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    conn = _conn(path)
    try:
        sql = f"SELECT {', '.join(_LIST_COLS)} FROM briefs"
        args: List[Any] = []
        if kind:
            sql += " WHERE kind = ?"
            args.append(kind)
        rows = conn.execute(sql + " ORDER BY session_date DESC, COALESCE(finished_at, started_at) DESC LIMIT ?",
                            (*args, int(limit))).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def list_advisor(limit: int = 30, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Advisor reports for My Holdings, newest first, with why each was written and whether urgent."""
    conn = _conn(path)
    try:
        rows = conn.execute(f"SELECT {', '.join(_LIST_COLS)}, context FROM briefs WHERE kind = 'advisor' "
                            "ORDER BY COALESCE(finished_at, started_at) DESC LIMIT ?", (int(limit),)).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        ctx = json.loads(d.pop("context") or "null") or {}
        d.update(symbol=ctx.get("symbol"), urgent=bool(ctx.get("urgent")), reasons=ctx.get("reasons") or [])
        out.append(d)
    return out


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
    d["context"] = json.loads(d["context"]) if d.get("context") else None
    d["predictions"] = predictions.for_brief(d["id"], path)
    if d["kind"] == "advisor":
        d["orders"] = advisor.orders_for_brief(d["id"], path)
    d.pop("raw", None)
    return d


def status() -> Dict[str, Any]:
    """Schedule facts for the page: which brief is due now, and whether one is being written."""
    return {"kinds": {k: {"title": v["title"], "due_session": due_session(k),
                          "running": any(r for key, r in _running.items() if key.split(":")[0] == k)}
                      for k, v in KINDS.items()},
            "claude_found": bool(find_claude()), "flash_threshold": events.FLASH_THRESHOLD,
            "bellwethers": {s: n for s, (n, _) in events.BELLWETHERS.items()}}
