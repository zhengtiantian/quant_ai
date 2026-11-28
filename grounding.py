"""Shared structured output gates, per-tool Grounding, F09 sanitisation, and business rules.

Designed for `portfolio_agent.py` and reusable across any quant_ai module that asks a local
LLM for structured JSON verdicts backed by tool observations:
  - Gate 1 (Schema): extract JSON from prose/fences + error-guided 1-shot repair round-trip
  - Gate 2 (Grounding): per-tool attribution check + numeric tolerance + decimal/pct equivalence
  - Gate 3 (Business rules): universe/symbol membership, hallucinated ticker checks, flag demotion
  - F09 Sanitisation: screen external evidence text via `injection_guard` before prompting LLM
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import injection_guard

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_NUM_STR_RE = re.compile(r"^[\$€£]?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(%?)$")
_EXPLICIT_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")


@dataclass
class GroundingIndex:
    global_nums: list[float] = field(default_factory=list)
    global_strings: set[str] = field(default_factory=set)
    by_tool_nums: dict[str, list[float]] = field(default_factory=dict)
    by_tool_strings: dict[str, set[str]] = field(default_factory=dict)
    by_tool_fields: dict[tuple[str, str], list[Any]] = field(default_factory=dict)
    symbols_seen: set[str] = field(default_factory=set)


def num_key(v: float) -> str:
    return f"~{round(v, 3):.3f}"


def parse_numeric_token(value: Any) -> tuple[Optional[float], bool]:
    """Parses raw float or formatted number/percent strings like '-11.2%' or '$149.30'."""
    if isinstance(value, bool) or value is None:
        return None, False
    if isinstance(value, (int, float)):
        return float(value), False
    if isinstance(value, str):
        m = _NUM_STR_RE.match(value.strip())
        if m:
            try:
                num = float(m.group(1).replace(",", ""))
                is_pct = m.group(2) == "%"
                return num, is_pct
            except ValueError:
                return None, False
    return None, False


def _collect_scalars(
    obj: Any,
    nums: list[float],
    strings: set[str],
    tool_name: str,
    field_map: dict[tuple[str, str], list[Any]],
    symbols_seen: set[str],
    parent_key: str = "",
) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).strip().lower()
            if kl == "symbol" and isinstance(v, str) and v.strip():
                symbols_seen.add(v.strip().upper())
            if isinstance(v, (int, float, str)) and not isinstance(v, bool) and tool_name:
                field_map.setdefault((tool_name, kl), []).append(v)
            _collect_scalars(v, nums, strings, tool_name, field_map, symbols_seen, parent_key=kl)
    elif isinstance(obj, list):
        for v in obj:
            _collect_scalars(v, nums, strings, tool_name, field_map, symbols_seen, parent_key=parent_key)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        fv = float(obj)
        nums.append(fv)
        strings.add(num_key(fv))
    elif isinstance(obj, str):
        s = obj.strip()
        if s:
            strings.add(s.lower())


def build_grounding_index(observations: Iterable[Any]) -> GroundingIndex:
    idx = GroundingIndex()
    for item in observations or []:
        if item is None:
            continue
        if isinstance(item, dict) and "tool" in item and "result" in item:
            tname = str(item.get("tool") or "").strip().lower()
            t_nums = idx.by_tool_nums.setdefault(tname, [])
            t_strs = idx.by_tool_strings.setdefault(tname, set())
            _collect_scalars(
                item.get("result"),
                t_nums,
                t_strs,
                tname,
                idx.by_tool_fields,
                idx.symbols_seen,
            )
            idx.global_nums.extend(t_nums)
            idx.global_strings.update(t_strs)
        else:
            _collect_scalars(
                item,
                idx.global_nums,
                idx.global_strings,
                "",
                idx.by_tool_fields,
                idx.symbols_seen,
            )
    return idx


def flatten_observed_values(observations: Iterable[Any]) -> set[str]:
    """Legacy compatible scalar string set (including `~round(v, 3)` keys)."""
    idx = build_grounding_index(observations)
    return idx.global_strings


def number_matches(
    cited: float,
    observed_nums: Iterable[float],
    rel_tol: float = 0.02,
    abs_tol: float = 0.005,
    allow_pct_scale: bool = True,
) -> bool:
    candidates = [cited]
    if allow_pct_scale:
        candidates.extend([cited / 100.0, cited * 100.0])
    for obs in observed_nums:
        for cand in candidates:
            tol = max(abs_tol, rel_tol * abs(obs))
            if abs(cand - obs) <= tol:
                return True
    return False


def is_value_grounded(
    value: Any,
    observed_strings: set[str],
    observed_nums: Iterable[float] = (),
) -> bool:
    if isinstance(value, bool) or value is None:
        return False

    num_val, _ = parse_numeric_token(value)
    if num_val is not None:
        # 1. Exact/legacy ~num_key check (supports callers passing only `observed_strings`)
        for cand in (num_val, num_val / 100.0, num_val * 100.0):
            if num_key(cand) in observed_strings:
                return True
        # 2. Tolerance check against explicit numbers or `~float` tokens inside observed_strings
        if observed_nums and number_matches(num_val, observed_nums):
            return True
        parsed_obs: list[float] = []
        for s in observed_strings:
            if s.startswith("~"):
                try:
                    parsed_obs.append(float(s[1:]))
                except ValueError:
                    pass
        if parsed_obs and number_matches(num_val, parsed_obs):
            return True
        return False

    text = str(value).strip().lower()
    if not text:
        return False
    if text in observed_strings:
        return True
    return len(text) >= 8 and any(text in o for o in observed_strings)


def verify_evidence(cited: list[dict], index: GroundingIndex) -> tuple[list[dict], list[dict]]:
    """Verifies citations against per-tool observations (falls back to global if flat)."""
    grounded: list[dict] = []
    dropped: list[dict] = []

    has_tool_buckets = bool(index.by_tool_strings or index.by_tool_nums)
    for entry in cited or []:
        if not isinstance(entry, dict):
            dropped.append({"entry": entry, "reason": "non_object"})
            continue
        val = entry.get("value")
        tool = str(entry.get("tool") or "").strip().lower()

        if has_tool_buckets and tool:
            if tool not in index.by_tool_strings and tool not in index.by_tool_nums:
                dropped.append({"entry": entry, "reason": f"unknown_tool:{tool}"})
                continue
            ok = is_value_grounded(
                val,
                index.by_tool_strings.get(tool, set()),
                index.by_tool_nums.get(tool, ()),
            )
            if ok:
                grounded.append(entry)
            else:
                dropped.append({"entry": entry, "reason": f"value_not_in_tool:{tool}"})
        else:
            ok = is_value_grounded(val, index.global_strings, index.global_nums)
            if ok:
                grounded.append(entry)
            else:
                dropped.append({"entry": entry, "reason": "value_not_observed"})

    return grounded, dropped


def extract_json(text: str) -> dict | None:
    """Extracts the first JSON object from raw model output (handles fences & prose)."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else candidate
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def run_structured_gate(
    messages: list[dict],
    chat_fn: Callable[..., dict],
    content_fn: Callable[[dict], str],
    validator_fn: Callable[[Any], Optional[str]],
    max_repairs: int = 1,
    assistant_prefill: Optional[str] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Gate 1: runs LLM + JSON extraction + schema validation with bounded repair."""
    working = list(messages)
    last_error: Optional[str] = None
    for attempt in range(max_repairs + 1):
        call_msgs = (
            working + [{"role": "assistant", "content": assistant_prefill}]
            if assistant_prefill
            else working
        )
        reply = content_fn(chat_fn(call_msgs, use_tools=False))
        candidate = extract_json(reply)
        last_error = validator_fn(candidate) if candidate is not None else "response was not JSON"
        if last_error is None and candidate is not None:
            return candidate, None
        if attempt < max_repairs:
            working.append({"role": "assistant", "content": reply})
            working.append(
                {
                    "role": "user",
                    "content": f"That did not validate: {last_error}. Reply with corrected JSON only.",
                }
            )
    return None, last_error


def sanitise_evidence(evidence: Any) -> tuple[Any, list[str]]:
    """F09: screens untrusted external strings in evidence dicts/lists."""
    flagged_tools: list[str] = []

    def _walk(node: Any, tool_ctx: str) -> Any:
        if isinstance(node, dict):
            cur_tool = str(node.get("tool") or tool_ctx or "").strip()
            return {k: _walk(v, cur_tool) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v, tool_ctx) for v in node]
        if isinstance(node, str) and len(node) > 16:
            sc = injection_guard.screen(node)
            if sc.flags and tool_ctx and tool_ctx not in flagged_tools:
                flagged_tools.append(tool_ctx)
            return sc.text
        return node

    return _walk(evidence, ""), flagged_tools


def check_business_rules(
    decision_symbol: Optional[str],
    parsed: dict,
    grounded_evidence: list[dict],
    *,
    allowed_symbols: Optional[set[str]] = None,
    observed_symbols: Optional[set[str]] = None,
) -> tuple[str, list[str], bool]:
    """Gate 3: verifies symbol membership & ensures ungrounded flags cannot stand.

    Returns (verdict, notes, rejected_by_business_rule).
    """
    notes: list[str] = []
    verdict = str(parsed.get("verdict") or "agree")
    valid_syms = {s.upper() for s in (allowed_symbols or set()) if s} | {
        s.upper() for s in (observed_symbols or set()) if s
    }
    sym_u = (decision_symbol or "").strip().upper()

    if allowed_symbols and sym_u and sym_u not in {s.upper() for s in allowed_symbols}:
        return "skipped", [f"business gate: symbol {sym_u} is not in covered portfolio/universe"], True

    # Reject cited symbol evidence pointing to a non-existent symbol outside valid_syms.
    if valid_syms:
        for ev in grounded_evidence:
            if str(ev.get("field") or "").strip().lower() == "symbol":
                cited_sym = str(ev.get("value") or "").strip().upper()
                if cited_sym and cited_sym not in valid_syms:
                    return "skipped", [f"business gate: hallucinated symbol reference {cited_sym}"], True

    if verdict == "flag" and not grounded_evidence:
        verdict = "agree"
        notes.append("flag demoted: no citation could be verified against the observations")

    return verdict, notes, False
