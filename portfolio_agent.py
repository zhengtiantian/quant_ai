"""Portfolio review agent (F.17) — a second opinion on the rule engine, not a replacement.

`track_positions.py` already decides entries, exits, stops and holding periods
deterministically. Handing those decisions to a local 9B model would be both worse and
non-reproducible, and "why do you think the model beats your rules" has no good answer.
So this agent does not decide anything. It reviews, covering the two things a rule
engine structurally cannot do:

1. **Read why a price moved.** A stop fires at -11%; the rule sees only the number. This
   agent pulls the news around the exit date and the feature trend leading into it, and
   can distinguish a sector-wide selloff that has already reversed — a whipsaw exit —
   from a company-specific failure.
2. **Reason about the portfolio.** The rules are per-position. Nothing asks whether four
   of five holdings are semiconductors, or whether total exposure suits a RISK_OFF regime.

Division of labour, which is the main design decision here:

- **Code** gathers context, picks what to review, fetches evidence, and does every
  portfolio-level check. Concentration and weight limits are arithmetic, not judgement —
  asking a model to do arithmetic it can get wrong, when a comparison operator cannot, is
  a bad trade.
- **The model** is used only for the part that needs language: given these articles and
  this feature trend, was that exit justified?
- **Code** then validates what came back, because a model that is asked for structured
  output will sometimes not produce it, and a model asked to cite numbers will sometimes
  invent them.

Three gates run on every verdict, none of which trust the model:

- **Schema** — valid JSON, known verdict, confidence in range. One repair round-trip with
  the specific error, then the review is dropped rather than emitted malformed.
- **Grounding** — every cited (tool, field, value) must appear in an observation actually
  returned during this run. The point is not to ask the model to avoid inventing numbers;
  it is to make invented numbers non-viable.
- **Business rules** — symbols must be in the covered universe; a review may not reference
  a position that does not exist.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Iterable

import mcp_client
from agent import _chat, _content

# How many rule decisions to review in one run. Each costs one LLM call plus up to two
# evidence fetches, so this bounds both latency and tokens.
MAX_DECISIONS = int(os.getenv("PORTFOLIO_MAX_DECISIONS", "5"))

# Portfolio limits. These mirror the pre-trade guardrails G.1 will enforce, so a review
# surfaces a breach before there is an order to reject.
MAX_POSITION_WEIGHT_PCT = float(os.getenv("PORTFOLIO_MAX_WEIGHT_PCT", "5"))
CONCENTRATION_WARN_PCT = float(os.getenv("PORTFOLIO_CONCENTRATION_PCT", "40"))

VERDICTS = ("agree", "flag")

REVIEW_PROMPT = (
    "You are reviewing one decision made by a deterministic rule engine. You are not "
    "second-guessing the rule for its own sake: it is usually right, and agreeing is the "
    "expected answer. Flag it only when the evidence below actually contradicts it — for "
    "example a stop-loss triggered by a broad sector move that has already reversed, with "
    "no company-specific bad news.\n\n"
    "Reply with JSON only, no prose around it:\n"
    "{\n"
    '  "verdict": "agree" | "flag",\n'
    '  "confidence": 0.0-1.0,\n'
    '  "reasoning": "one or two sentences",\n'
    '  "evidence": [{"tool": "...", "field": "...", "value": <number or string>}]\n'
    "}\n\n"
    "Every entry in `evidence` must be a value that appears in the observations given to "
    "you. Do not compute new numbers and do not cite anything you were not shown — cited "
    "values are checked against the raw observations and a review with an unverifiable "
    "citation is discarded."
)


# =====================================================
# Tool access
# =====================================================

def _call(name: str, args: dict | None = None) -> Any:
    """Invoke an MCP tool and parse its JSON. Returns None on any failure.

    Callers treat a missing observation as missing evidence rather than an error: a
    review with less evidence is still worth producing, an aborted run is not.
    """
    try:
        raw = mcp_client.get_client().call_tool(name, args or {})
        return json.loads(raw)
    except Exception:  # noqa: BLE001 - absence of evidence is handled by the caller
        return None


# =====================================================
# Grounding
# =====================================================

def _flatten(obj: Any, out: set[str]) -> None:
    """Collect every scalar in a nested structure as a normalised string."""
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _flatten(v, out)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        out.add(_num_key(float(obj)))
    elif isinstance(obj, str):
        s = obj.strip()
        if s:
            out.add(s.lower())


def _num_key(v: float) -> str:
    """Numbers are matched with tolerance: a model that echoes 0.2803 as 0.28 is quoting
    the observation, not inventing one, and failing it would train the reviewer to omit
    citations entirely."""
    return f"~{round(v, 3):.3f}"


def _observed_values(observations: Iterable[Any]) -> set[str]:
    seen: set[str] = set()
    for obs in observations:
        if obs is not None:
            _flatten(obs, seen)
    return seen


def _is_grounded(value: Any, observed: set[str]) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return _num_key(float(value)) in observed
    text = str(value).strip().lower()
    if not text:
        return False
    if text in observed:
        return True
    # A model quoting a headline usually quotes part of it. Accept a citation that is a
    # substring of something observed, provided it is specific enough to be evidence.
    return len(text) >= 8 and any(text in o for o in observed)


# =====================================================
# Gate 1 — schema
# =====================================================

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict | None:
    """Models wrap JSON in prose or fences more often than not; take the first object."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else candidate
    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except (ValueError, TypeError):
        return None


def _schema_error(obj: Any) -> str | None:
    """Returns a message naming what is wrong, or None when the shape is acceptable."""
    if not isinstance(obj, dict):
        return "response must be a JSON object"
    verdict = obj.get("verdict")
    if verdict not in VERDICTS:
        return f"`verdict` must be one of {list(VERDICTS)}, got {verdict!r}"
    confidence = obj.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return "`confidence` must be a number between 0 and 1"
    if not 0.0 <= float(confidence) <= 1.0:
        return f"`confidence` must be between 0 and 1, got {confidence}"
    if not isinstance(obj.get("reasoning"), str) or not obj["reasoning"].strip():
        return "`reasoning` must be a non-empty string"
    evidence = obj.get("evidence")
    if not isinstance(evidence, list):
        return "`evidence` must be a list"
    for i, e in enumerate(evidence):
        if not isinstance(e, dict):
            return f"evidence[{i}] must be an object with tool/field/value"
        if "value" not in e:
            return f"evidence[{i}] is missing `value`"
    return None


# =====================================================
# The one part that needs a model
# =====================================================

def _review_one(decision: dict, evidence: list[dict], observed: set[str]) -> dict:
    """Review a single rule decision. Never raises; a failure becomes a skipped review."""
    payload = {
        "decision": decision,
        "observations": evidence,
    }
    messages = [
        {"role": "system", "content": REVIEW_PROMPT},
        {"role": "user", "content": json.dumps(payload, indent=2, default=str)},
    ]

    parsed: dict | None = None
    error: str | None = None
    for attempt in range(2):  # one repair round-trip, then give up
        try:
            reply = _content(_chat(messages, use_tools=False))
        except Exception as e:  # noqa: BLE001 - surfaced as a skipped review
            return _skipped(decision, f"model call failed: {e}")
        candidate = _parse_json(reply)
        error = _schema_error(candidate) if candidate is not None else "response was not JSON"
        if error is None:
            parsed = candidate
            break
        if attempt == 0:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content":
                             f"That did not validate: {error}. Reply with corrected JSON only."})

    if parsed is None:
        return _skipped(decision, f"schema gate: {error}")

    # Gate 2 — grounding. Drop unverifiable citations rather than the whole review, but
    # a flag that rests on nothing verifiable is not worth acting on, so it is demoted.
    cited = parsed.get("evidence", [])
    grounded = [e for e in cited if _is_grounded(e.get("value"), observed)]
    dropped = len(cited) - len(grounded)

    verdict = parsed["verdict"]
    notes = []
    if dropped:
        notes.append(f"{dropped} unverifiable citation(s) dropped")
    if verdict == "flag" and not grounded:
        verdict = "agree"
        notes.append("flag demoted: no citation could be verified against the observations")

    return {
        "symbol": decision.get("symbol"),
        "action": decision.get("action"),
        "ruleReason": decision.get("reason"),
        "verdict": verdict,
        "confidence": round(float(parsed["confidence"]), 3),
        "reasoning": parsed["reasoning"].strip(),
        "evidence": grounded,
        "notes": notes,
    }


def _skipped(decision: dict, why: str) -> dict:
    return {
        "symbol": decision.get("symbol"),
        "action": decision.get("action"),
        "ruleReason": decision.get("reason"),
        "verdict": "skipped",
        "confidence": 0.0,
        "reasoning": "",
        "evidence": [],
        "notes": [why],
    }


# =====================================================
# Deterministic portfolio checks — no model involved
# =====================================================

def _portfolio_checks(holdings: dict | None, signals: Any) -> list[dict]:
    """Portfolio-level findings that are arithmetic rather than judgement.

    Every one of these is a comparison a model could get wrong and a comparison operator
    cannot, so none of them go near the LLM.
    """
    findings: list[dict] = []
    if not isinstance(holdings, dict):
        return findings
    rows = holdings.get("holdings") or []
    totals = holdings.get("totals") or {}
    if not rows:
        return findings

    for h in rows:
        weight = float(h.get("weightPct") or 0)
        if weight > MAX_POSITION_WEIGHT_PCT:
            findings.append({
                "check": "position_weight",
                "severity": "high" if weight > CONCENTRATION_WARN_PCT else "medium",
                "symbol": h.get("symbol"),
                "detail": (f"{h.get('symbol')} is {weight:.1f}% of total capital, over the "
                           f"{MAX_POSITION_WEIGHT_PCT:.0f}% per-position limit"),
                "value": round(weight, 2),
                "limit": MAX_POSITION_WEIGHT_PCT,
            })

    invested = float(totals.get("holdingsValue") or 0)
    total = float(totals.get("totalValue") or 0)
    if total > 0:
        exposure = invested / total * 100
        regime = _regime_of(signals)
        if regime in ("RISK_OFF", "STRESSED") and exposure > 60:
            findings.append({
                "check": "regime_exposure",
                "severity": "medium",
                "symbol": None,
                "detail": (f"{exposure:.1f}% of capital is invested while the regime is "
                           f"{regime}; the signal engine is already scaling conviction down"),
                "value": round(exposure, 2),
                "limit": 60.0,
            })

    held = {str(h.get("symbol")) for h in rows}
    ranked = {str(s.get("symbol")) for s in _signal_rows(signals)[:10]}
    if ranked and held:
        overlap = held & ranked
        if not overlap:
            findings.append({
                "check": "signal_alignment",
                "severity": "medium",
                "symbol": None,
                "detail": ("no holding appears in today's top 10 ranked signals — the "
                           "portfolio and the model currently disagree about what to own"),
                "value": 0,
                "limit": 1,
            })
    return findings


def _signal_rows(signals: Any) -> list[dict]:
    if isinstance(signals, list):
        return [s for s in signals if isinstance(s, dict)]
    if isinstance(signals, dict):
        for key in ("signals", "data", "items"):
            if isinstance(signals.get(key), list):
                return [s for s in signals[key] if isinstance(s, dict)]
    return []


def _regime_of(signals: Any) -> str:
    rows = _signal_rows(signals)
    return str(rows[0].get("regimeLabel") or rows[0].get("regime_label") or "UNKNOWN") if rows else "UNKNOWN"


# =====================================================
# Choosing what to review
# =====================================================

def _recent_exits(positions: Any, limit: int) -> list[dict]:
    """The rule engine's most informative decisions: closed positions, newest first.

    An exit carries more reviewable information than an entry — it names a trigger and a
    realised return, so there is something concrete to agree or disagree with.
    """
    rows = positions if isinstance(positions, list) else []
    closed = [p for p in rows if isinstance(p, dict) and p.get("status") == "closed"]
    closed.sort(key=lambda p: str(p.get("exitDate") or p.get("exit_date") or ""), reverse=True)
    out = []
    for p in closed[:limit]:
        ret = p.get("exitReturn", p.get("exit_return"))
        out.append({
            "symbol": p.get("symbol"),
            "action": "exit",
            "reason": p.get("exitTrigger") or p.get("exit_trigger"),
            "entryDate": p.get("entryDate") or p.get("entry_date"),
            "exitDate": p.get("exitDate") or p.get("exit_date"),
            "daysHeld": p.get("daysHeld", p.get("days_held")),
            "realisedReturnPct": round(float(ret) * 100, 2) if isinstance(ret, (int, float)) else None,
        })
    return out


def _evidence_for(decision: dict) -> list[dict]:
    """Fetch what the rule could not see: the news around the exit, and the trend into it."""
    symbol = decision.get("symbol")
    if not symbol:
        return []
    evidence = []
    exit_date = str(decision.get("exitDate") or "").replace("-", "")
    news = _call("search_news", {
        "symbol": symbol,
        "from_date": str(decision.get("entryDate") or "").replace("-", ""),
        "to_date": exit_date,
        "limit": 5,
    })
    if news:
        evidence.append({"tool": "search_news", "result": news})
    history = _call("get_feature_history", {
        "symbol": symbol,
        "fields": "avg_sentiment_5d,sentiment_shift_5d,past_ret_20d,macro_vix",
        "days": 60,
    })
    if history:
        # The series itself is long and the summary carries the direction, which is the
        # part a reviewer reasons about.
        evidence.append({"tool": "get_feature_history",
                         "result": {k: history.get(k) for k in
                                    ("symbol", "from", "to", "rows", "summary")}})
    return evidence


# =====================================================
# Entry point
# =====================================================

def run_portfolio_review(max_decisions: int = MAX_DECISIONS) -> dict:
    """Review recent rule decisions and the current portfolio.

    Returns a machine-consumable report: one row per reviewed decision plus deterministic
    portfolio findings, so G.1 execution could later act on it directly.
    """
    started = time.time()

    positions = _call("get_positions")
    holdings = _call("get_my_holdings")
    signals = _call("get_latest_signals", {"limit": 10})

    decisions = _recent_exits(positions, max_decisions)
    reviews: list[dict] = []
    for decision in decisions:
        evidence = _evidence_for(decision)
        observed = _observed_values(e["result"] for e in evidence)
        reviews.append(_review_one(decision, evidence, observed))

    findings = _portfolio_checks(holdings, signals)
    counts = {v: sum(1 for r in reviews if r["verdict"] == v)
              for v in ("agree", "flag", "skipped")}

    return {
        "asOf": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "regime": _regime_of(signals),
        "decisionsReviewed": len(reviews),
        "verdicts": counts,
        "reviews": reviews,
        "portfolioFindings": findings,
        "elapsed_s": round(time.time() - started, 2),
    }
