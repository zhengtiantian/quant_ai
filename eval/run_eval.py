#!/usr/bin/env python3
"""Agent evaluation harness (F.22).

Answers "how do you know the agent is any good" with numbers instead of a demo.

The honest boundary first: **whether a recommendation makes money cannot be measured
here.** That needs the market to pass. What can be measured are the failure modes that
actually keep agents out of production, and none of them require knowing the future:

    schema validity   did the portfolio reviewer produce parseable, typed output
    grounding rate    what fraction of cited numbers exist in a real observation
    tool recall       did it call the tools the task requires
    tool precision    did it avoid tools the task forbids
    stability         how much does the answer change across repeated identical runs
    cost              steps, wall-clock

The existing unit tests mock the LLM, so they prove the loop is correct — not that the
decisions are. This closes that gap by running the real model against fixed cases.

Stability matters more than it looks. An agent that answers differently every time cannot
be trusted even when each individual answer is defensible, and it is invisible in a demo
because a demo is run once.

Usage:
    .venv/bin/python eval/run_eval.py                 # everything
    .venv/bin/python eval/run_eval.py --suite research
    .venv/bin/python eval/run_eval.py --json report.json

Needs LM Studio and a reachable quant_api; it drives the real agent, not a mock.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402


# =====================================================
# Metrics
# =====================================================

def jaccard(a: set, b: set) -> float:
    """Overlap between two runs' outputs. 1.0 means identical, 0.0 means nothing shared."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def stability(runs: list[set]) -> float:
    """Mean pairwise Jaccard across repeated runs of one case."""
    if len(runs) < 2:
        return 1.0
    scores = [jaccard(runs[i], runs[j])
              for i in range(len(runs)) for j in range(i + 1, len(runs))]
    return statistics.mean(scores) if scores else 1.0


# =====================================================
# Research agent cases
# =====================================================

def run_research_case(case: dict) -> dict:
    from agent import run_research_agent

    repeats = int(case.get("repeat", 1))
    must_call = set(case.get("must_call", []))
    should_call = set(case.get("should_call", []))
    must_not_call = set(case.get("must_not_call", []))
    must_mention = case.get("must_mention", [])
    must_mention_any = case.get("must_mention_any", [])

    runs = []
    for _ in range(repeats):
        started = time.time()
        try:
            result = run_research_agent(case["prompt"], max_steps=5)
        except Exception as e:  # noqa: BLE001 - a crashed run is a result, not an abort
            runs.append({"error": str(e), "tools": set(), "answer": "",
                         "steps": 0, "elapsed": time.time() - started})
            continue
        tools = {t["action"] for t in result.get("trace", [])}
        runs.append({
            "tools": tools,
            "answer": (result.get("answer") or ""),
            "steps": result.get("steps", 0),
            "elapsed": result.get("elapsed_s", time.time() - started),
        })

    ok_recall = sum(1 for r in runs if must_call <= r["tools"])
    ok_precision = sum(1 for r in runs if not (must_not_call & r["tools"]))
    ok_should = sum(1 for r in runs if not should_call or (should_call & r["tools"]))
    ok_mentions = sum(
        1 for r in runs
        if all(m.lower() in r["answer"].lower() for m in must_mention)
        and (not must_mention_any
             or any(m.lower() in r["answer"].lower() for m in must_mention_any))
    )
    errors = sum(1 for r in runs if r.get("error"))

    return {
        "id": case["id"],
        "runs": repeats,
        "errors": errors,
        "toolRecall": ok_recall / repeats,
        "toolPrecision": ok_precision / repeats,
        "reachedForContext": ok_should / repeats if should_call else None,
        "answerContent": ok_mentions / repeats,
        "toolStability": round(stability([r["tools"] for r in runs]), 3),
        "meanSteps": round(statistics.mean([r["steps"] for r in runs]), 2),
        "meanSeconds": round(statistics.mean([r["elapsed"] for r in runs]), 1),
        "toolsSeen": sorted({t for r in runs for t in r["tools"]}),
    }


# =====================================================
# Portfolio reviewer cases
# =====================================================

def run_portfolio_case(case: dict) -> dict:
    import portfolio_agent as pa

    repeats = int(case.get("repeat", 1))
    max_decisions = int(case.get("max_decisions", 2))

    schema_ok = grounded_cited = total_cited = 0
    reviews_total = 0
    verdict_sets: list[set] = []
    seconds: list[float] = []
    errors = 0

    for _ in range(repeats):
        try:
            report = pa.run_portfolio_review(max_decisions=max_decisions)
        except Exception:  # noqa: BLE001
            errors += 1
            continue
        seconds.append(report.get("elapsed_s", 0))
        verdicts = set()
        for review in report.get("reviews", []):
            reviews_total += 1
            if review["verdict"] != "skipped":
                schema_ok += 1
            verdicts.add(f"{review['symbol']}:{review['verdict']}")
            # A citation that survived the grounding gate is by construction grounded;
            # the dropped ones are recorded in notes, which is how the rate is recovered.
            kept = len(review.get("evidence", []))
            dropped = sum(int(n.split()[0]) for n in review.get("notes", [])
                          if n.split()[:1] and n.split()[0].isdigit()
                          and "unverifiable" in n)
            grounded_cited += kept
            total_cited += kept + dropped
        verdict_sets.append(verdicts)

    return {
        "id": case["id"],
        "runs": repeats,
        "errors": errors,
        "reviewsProduced": reviews_total,
        "schemaValidity": round(schema_ok / reviews_total, 3) if reviews_total else 0.0,
        "groundingRate": round(grounded_cited / total_cited, 3) if total_cited else None,
        "citationsKept": grounded_cited,
        "citationsDropped": total_cited - grounded_cited,
        "verdictStability": round(stability(verdict_sets), 3),
        "meanSeconds": round(statistics.mean(seconds), 1) if seconds else 0.0,
    }


# =====================================================
# Report
# =====================================================

def pct(v: Any) -> str:
    return "  n/a" if v is None else f"{v * 100:5.1f}%"


def print_report(results: dict) -> None:
    research = results.get("research", [])
    if research:
        print("\nResearch agent")
        print(f"  {'case':<36} {'recall':>7} {'prec':>7} {'content':>8} "
              f"{'stable':>7} {'steps':>6} {'sec':>6} {'err':>4}")
        for r in research:
            print(f"  {r['id']:<36} {pct(r['toolRecall']):>7} {pct(r['toolPrecision']):>7} "
                  f"{pct(r['answerContent']):>8} {r['toolStability']:>7.2f} "
                  f"{r['meanSteps']:>6.1f} {r['meanSeconds']:>6.0f} {r['errors']:>4}")

    portfolio = results.get("portfolio_review", [])
    if portfolio:
        print("\nPortfolio reviewer")
        for r in portfolio:
            print(f"  {r['id']}")
            print(f"    reviews produced   {r['reviewsProduced']}")
            print(f"    schema validity    {pct(r['schemaValidity'])}")
            print(f"    grounding rate     {pct(r['groundingRate'])} "
                  f"({r['citationsKept']} kept, {r['citationsDropped']} dropped)")
            print(f"    verdict stability  {r['verdictStability']:.2f}")
            print(f"    mean seconds       {r['meanSeconds']:.0f}")

    print("\nWhat this does not measure: whether any recommendation makes money.")
    print("That needs the market to pass; these are correctness and reliability only.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the agent evaluation suite")
    ap.add_argument("--suite", choices=["research", "portfolio_review"],
                    help="run only one suite")
    ap.add_argument("--cases", default=str(Path(__file__).parent / "cases.yaml"))
    ap.add_argument("--json", help="also write the full report here")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.cases).read_text())
    suites = [args.suite] if args.suite else list(spec.keys())

    results: dict[str, list] = {}
    for suite in suites:
        cases = spec.get(suite) or []
        runner = run_research_case if suite == "research" else run_portfolio_case
        results[suite] = []
        for case in cases:
            print(f"running {suite}/{case['id']} …", flush=True)
            results[suite].append(runner(case))

    print_report(results)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
