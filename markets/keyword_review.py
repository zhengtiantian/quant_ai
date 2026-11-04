"""The weekly keyword review (2026-11-04): quant_ui's Keywords tab reads and decides here.

quant_data's tools/keyword_refresh.py writes, each Saturday, the names it found in the week's headlines for
each company, with the local model's verdict and the co-mention share, to <dir>/runs/<date>.json. The user
decides each one on the tab: "add" (a keyword), "weak" (kept for the SLM pass, not matched), "reject", or
back to pending. Decisions go to <dir>/decisions.json; quant_data's hourly apply step writes them into the
rule files and logs each change to <dir>/applied.jsonl, which is read back here to show what is in effect.

Files only: quant_ai does not import quant_data code.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

DIR = Path(os.getenv("KEYWORD_REFRESH_DIR", "/Volumes/Data4T/keyword_refresh"))
DECISIONS = ("add", "weak", "reject")
HISTORY = ("full", "first_seen")
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
    for p in _runs():
        run = _read(p, {})
        runs.append({"week": p.stem, "created": run.get("created"), "judged": run.get("judged"),
                     "headlines": run.get("headlines"), "candidates": len(run.get("candidates") or [])})
        for c in run.get("candidates") or []:
            props[c["id"]] = {**c, "week": p.stem}
            seen.setdefault(c["id"], []).append(p.stem)
    decisions = _read(DIR / "decisions.json", {})
    applied = _applied()
    rescans: Dict[str, List[Dict[str, Any]]] = {}
    for v in _read(DIR / "rescan.json", {}).values():
        if isinstance(v, dict) and v.get("id"):
            rescans.setdefault(v["id"], []).append({k: v.get(k) for k in ("since", "at", "kept", "new_rows")})
    rows = []
    for pid, c in props.items():
        d = decisions.get(pid) or {}
        a = applied.get((c["set"], c["symbol"], c["term"])) or {}
        rows.append({k: c.get(k) for k in ("id", "set", "symbol", "name", "term", "kind", "reason", "tier", "co_mention",
                                           "headlines_with_company", "headlines_total", "stories", "sites", "first_seen",
                                           "examples", "company_headlines", "current_keywords", "ordinary_word", "caveat",
                                           "week")}
                    | {"weeks_seen": seen[pid], "decision": d.get("decision"), "decided_at": d.get("at"),
                       "history": d.get("history") or default_history(c.get("kind")),
                       "rescans": sorted(rescans.get(pid, []), key=lambda r: r["since"] or ""),
                       "in_rules": bool(a.get("in_rules")), "applied_at": a.get("at")})
    tier_order = {"add": 0, "weak": 1, "unjudged": 2, "model_none": 3}
    rows.sort(key=lambda r: (-int((r["week"] or "0").replace("-", "")), tier_order.get(r["tier"], 9),
                             r["symbol"], -(r["co_mention"] or 0)))
    return {"dir": str(DIR), "runs": runs[::-1], "proposals": rows}


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
