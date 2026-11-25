"""The weekly keyword review (2026-11-04): quant_ui's Keywords tab reads and decides here.

quant_data's tools/keyword_refresh.py writes, each Saturday, the names it found in the week's headlines for
each company, with the local model's verdict and the co-mention share, to <dir>/runs/<date>.json. The user
decides each one on the tab: "add" (a keyword), "weak" (kept for the SLM pass, not matched), "reject", or
back to pending. Decisions go to <dir>/decisions.json; quant_data's hourly apply step writes them into the
rule files and logs each change to <dir>/applied.jsonl, which is read back here to show what is in effect.

2026-11-19: quant_data also screens each suggestion for collateral hits (`screen`: 24 months of headlines,
the words and companies next to the term, Claude on other meanings), may add the risk-free ones itself
(decisions with "by": "auto", only while <dir>/settings.json has "auto_apply": true; set_auto() is the
switch on the tab) and audits every added term daily (<dir>/audit.json: what it brought in, the share the
relevance model kept; failing auto-added terms are taken out again). tools/coverage_watch.py writes the
per-company coverage check to COVERAGE_DIR/latest.json; coverage() and summary() read it.

Files only: quant_ai does not import quant_data code.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

DIR = Path(os.getenv("KEYWORD_REFRESH_DIR", "/Volumes/Data4T/keyword_refresh"))
COVERAGE_DIR = Path(os.getenv("COVERAGE_WATCH_DIR", "/Volumes/Data4T/coverage_watch"))
DECISIONS = ("add", "weak", "reject")
HISTORY = ("full", "first_seen")
SCREEN_TIERS = ("add", "weak")
EXPIRE_DAYS = int(os.getenv("KEYWORD_EXPIRE_DAYS", "21"))
EXPIRE_KEEP_RECENT_RUNS = 2
# as quant_data's keyword_refresh.default_history: a product has no headlines before it exists, a
# subsidiary's or executive's past news belongs to someone else
FULL_HISTORY_KINDS = {"product", "brand", "service"}


def default_history(kind: str | None) -> str:
    return "full" if kind in FULL_HISTORY_KINDS else "first_seen"


def _read(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _runs() -> List[Path]:
    return sorted((DIR / "runs").glob("????-??-??.json"))


def _recent_weeks(run_paths: List[Path]) -> set[str]:
    return {p.stem for p in run_paths[-EXPIRE_KEEP_RECENT_RUNS:]}


def _is_expired(last_week: str | None, recent_weeks: set[str], today: date | None = None) -> bool:
    if not last_week or last_week in recent_weeks:
        return False
    try:
        week_d = date.fromisoformat(last_week[:10])
    except ValueError:
        return False
    return ((today or datetime.now().date()) - week_d).days >= EXPIRE_DAYS


def _screen_health(run_paths: List[Path]) -> Dict[str, Any]:
    if not run_paths:
        return {"week": None, "total": 0, "screened": 0, "claude_answered": 0, "claude_rate": None, "degraded": False}
    latest_path = run_paths[-1]
    latest = _read(latest_path, {})
    cands = [c for c in (latest.get("candidates") or []) if c.get("tier") in SCREEN_TIERS]
    total = len(cands)
    screened = sum(1 for c in cands if c.get("screen"))
    claude_answered = sum(1 for c in cands if bool((c.get("screen") or {}).get("claude")))
    claude_rate = round(claude_answered / total, 3) if total else None
    degraded = bool(total > 0 and (screened == 0 or claude_answered == 0))
    return {
        "week": latest_path.stem,
        "total": total,
        "screened": screened,
        "claude_answered": claude_answered,
        "claude_rate": claude_rate,
        "degraded": degraded,
    }


def _applied() -> Dict[tuple, Dict[str, Any]]:
    """(set, symbol, term) -> {"in_rules": bool, "at": time of the last change}."""
    out: Dict[tuple, Dict[str, Any]] = {}
    p = DIR / "applied.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        try:
            c = json.loads(line)
        except ValueError:
            continue
        for t in (c.get("added") or []) + (c.get("moved") or []):
            out[(c["set"], c["symbol"], t)] = {"in_rules": True, "at": c.get("at")}
        for t in c.get("removed") or []:
            out[(c["set"], c["symbol"], t)] = {"in_rules": False, "at": c.get("at")}
    return out


def review() -> Dict[str, Any]:
    props: Dict[str, Dict[str, Any]] = {}
    seen: Dict[str, List[str]] = {}
    runs = []
    run_paths = _runs()
    recent = _recent_weeks(run_paths)
    screen_health = _screen_health(run_paths)
    for p in run_paths:
        run = _read(p, {})
        runs.append({"week": p.stem, "created": run.get("created"), "judged": run.get("judged"),
                     "headlines": run.get("headlines"), "candidates": len(run.get("candidates") or [])})
        for c in run.get("candidates") or []:
            props[c["id"]] = {**c, "week": p.stem}
            seen.setdefault(c["id"], []).append(p.stem)
    decisions = _read(DIR / "decisions.json", {})
    applied = _applied()
    audit = _read(DIR / "audit.json", {})
    preview = _read(DIR / "auto_preview.json", {})
    eligible = set(preview.get("ids") or [])
    rescans: Dict[str, List[Dict[str, Any]]] = {}
    for v in _read(DIR / "rescan.json", {}).values():
        if isinstance(v, dict) and v.get("id"):
            rescans.setdefault(v["id"], []).append({k: v.get(k) for k in ("since", "at", "kept", "new_rows")})
    rows = []
    for pid, c in props.items():
        d = decisions.get(pid) or {}
        a = applied.get((c["set"], c["symbol"], c["term"])) or {}
        expired = (not d.get("decision")) and c.get("tier") in SCREEN_TIERS and _is_expired(c["week"], recent)
        rows.append({k: c.get(k) for k in ("id", "set", "symbol", "name", "term", "kind", "reason", "tier", "co_mention",
                                           "headlines_with_company", "headlines_total", "stories", "sites", "first_seen",
                                           "examples", "company_headlines", "current_keywords", "ordinary_word", "caveat",
                                           "caveat_reason", "week")}
                    | {"weeks_seen": seen[pid], "decision": d.get("decision"), "decided_at": d.get("at"),
                       "decided_by": d.get("by") or ("user" if d.get("decision") else None), "decision_reason": d.get("reason"),
                       "confirmed_at": d.get("confirmed"), "expired": expired, "screen": c.get("screen"),
                       "auto_eligible": pid in eligible, "audit": (audit.get("terms") or {}).get(pid),
                       "history": d.get("history") or default_history(c.get("kind")),
                       "rescans": sorted(rescans.get(pid, []), key=lambda r: r["since"] or ""),
                       "in_rules": bool(a.get("in_rules")), "applied_at": a.get("at")})
    tier_order = {"add": 0, "weak": 1, "unjudged": 2, "model_none": 3}
    rows.sort(key=lambda r: (-int((r["week"] or "0").replace("-", "")), tier_order.get(r["tier"], 9),
                             r["symbol"], -(r["co_mention"] or 0)))
    return {"dir": str(DIR), "runs": runs[::-1], "proposals": rows,
            "auto": {"enabled": bool(_read(DIR / "settings.json", {}).get("auto_apply")), "preview_at": preview.get("at"),
                     "eligible": len(eligible)},
            "screen_health": screen_health,
            "audit_at": audit.get("at")}


def coverage() -> Dict[str, Any]:
    """tools/coverage_watch.py's latest check, or {} before the first run."""
    return _read(COVERAGE_DIR / "latest.json", {})


def summary() -> Dict[str, Any]:
    """Counts for the tab's badge: suggestions waiting for the user, coverage alerts."""
    decisions = _read(DIR / "decisions.json", {})
    props: Dict[str, Dict[str, Any]] = {}
    run_paths = _runs()
    recent = _recent_weeks(run_paths)
    screen_health = _screen_health(run_paths)
    for p in run_paths:
        for c in _read(p, {}).get("candidates") or []:
            props[c["id"]] = {**c, "week": p.stem}
    expired = sum(
        1 for pid, c in props.items()
        if pid not in decisions and c.get("tier") in SCREEN_TIERS and _is_expired(c["week"], recent)
    )
    pending = sum(
        1 for pid, c in props.items()
        if pid not in decisions and c.get("tier") in SCREEN_TIERS and not _is_expired(c["week"], recent)
    )
    cov = coverage()
    return {"pending": pending, "expired": expired, "coverage_at": cov.get("at"),
            "coverage_alerts": len(cov.get("core_alerts") or []) + len(cov.get("universe_flagged") or []),
            "coverage_notices": len(set(cov.get("core_flagged") or []) - set(cov.get("core_alerts") or [])),
            "auto_apply": bool(_read(DIR / "settings.json", {}).get("auto_apply")),
            "screen_health": screen_health}


def set_auto(enabled: bool) -> Dict[str, Any]:
    """The tab's switch for quant_data's auto_decide (it acts at the next hourly apply)."""
    DIR.mkdir(parents=True, exist_ok=True)
    cur = _read(DIR / "settings.json", {})
    cur.update(auto_apply=bool(enabled), auto_apply_changed=datetime.now().isoformat(timespec="seconds"))
    tmp = DIR / "settings.json.tmp"
    tmp.write_text(json.dumps(cur, indent=1))
    tmp.rename(DIR / "settings.json")
    return {"auto_apply": cur["auto_apply"]}


def decide(ids: List[str], decision: str | None, history: str | None = None) -> Dict[str, Any]:
    """Sets (or with None / "pending" clears) the decision of each id. Takes effect at the next hourly apply.
    `history` alone (decision "keep") changes how far back an added keyword is searched."""
    if decision in ("", "pending"):
        decision = None
    if decision is not None and decision not in DECISIONS + ("keep",):
        raise ValueError(f"decision must be one of {DECISIONS}, keep or pending")
    if history is not None and history not in HISTORY:
        raise ValueError(f"history must be one of {HISTORY}")
    known = {c["id"] for p in _runs() for c in _read(p, {}).get("candidates") or []}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"unknown proposal ids: {unknown[:5]}")
    DIR.mkdir(parents=True, exist_ok=True)
    with open(DIR / "decisions.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cur = _read(DIR / "decisions.json", {})
        now = datetime.now().isoformat(timespec="seconds")
        for i in ids:
            if decision is None:
                cur.pop(i, None)
            elif decision == "keep":                   # history only; an undecided term keeps no decision
                if history:
                    cur[i] = {**cur.get(i, {}), "history": history}
            else:
                cur[i] = {"decision": decision, "at": now, **({"history": history} if history else
                                                             {"history": cur[i]["history"]} if cur.get(i, {}).get("history") else {})}
        tmp = DIR / "decisions.json.tmp"
        tmp.write_text(json.dumps(cur, indent=1, ensure_ascii=False))
        tmp.rename(DIR / "decisions.json")
    return {"updated": len(ids), "decision": decision}
