#!/usr/bin/env python3
"""R.11 — generation-side evaluation of /api/ask/news.

R.4 measured whether the right articles reach the context window. It says nothing about
what the model then writes: whether it used them, misread them, or produced a fluent
paragraph that cites them and says something they do not. This script measures that.

    faithfulness        share of factual sentences a retrieved excerpt actually supports
    citation precision  share of citations that point at an excerpt supporting the sentence
                        (a real source that does not say the thing is a hallucination with a
                        footnote)
    citation coverage   share of factual sentences that carry a citation at all
    numeric grounding   share of numbers in cited sentences that appear in the cited excerpts
                        (no model: the F.17 grounding gate's idea, with a rounding tolerance)
    answer relevance    does the answer address the question (0 / 1 / 2)
    refusal             on questions the corpus cannot answer, does it decline

Same discipline as R.4: the judge is an instrument, so it is measured before it is trusted.
`--reference` has Claude (headless `claude -p`, the user's subscription) label a sample of
sentences; `--kappa` compares the local judge with it. Only a judge that agrees well enough
reports a table.

    python eval/gen_eval.py --generate --run rerank_off --rerank off
    python eval/gen_eval.py --generate --run rerank_on  --rerank on
    python eval/gen_eval.py --judge                         # local judge, every pending sentence
    python eval/gen_eval.py --reference --sample 100        # Claude reference labels
    python eval/gen_eval.py --kappa reference_claude llm_gemma
    python eval/gen_eval.py --score --compare rerank_off rerank_on

Do not run the 27B judge, the reranker and a job_agent run at the same time: LM Studio
answers 500s under that load (2026-10-13).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

RUNS_COLL = "eval_generation_runs"        # one doc per (run, question)
SENT_COLL = "eval_generation_sentences"   # one doc per (run, question, sentence)
LABEL_COLL = "eval_generation_labels"     # one doc per (source, sentence) and per (source, answer)
POOL_JUDGE_COLL = "eval_retrieval_judgments"   # R.4 labels: is an excerpt an answer to the query

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1").rstrip("/")
GEN_MODEL = os.getenv("LOCAL_MODEL_NAME", "qwen3.5-9b-mlx")
JUDGE_MODEL = os.getenv("GEN_JUDGE_MODEL", "gemma-4-e4b-it-mlx")
JUDGE_SOURCE = os.getenv("GEN_JUDGE_SOURCE", "llm_gemma")
WORKERS = int(os.getenv("GEN_JUDGE_WORKERS", "3"))

# Questions the corpus cannot answer: future events, private facts, companies and topics it
# does not cover. A good answer declines; an answer here is invented by construction.
UNANSWERABLE = [
    ("u01", "What did Apple announce at its September 2031 product event?"),
    ("u02", "What is Jensen Huang's favourite restaurant?"),
    ("u03", "What was Tesla's exact revenue in Q3 2035?"),
    ("u04", "Why did the Bank of Tanzania raise interest rates last week?"),
    ("u05", "What colour is the carpet in Microsoft's Redmond boardroom?"),
    ("u06", "How many employees did a private bakery in Dublin hire in 2026?"),
    ("u07", "What will Nvidia's share price be on 1 January 2030?"),
    ("u08", "Which football team won the 2026 Irish under-12 cup final?"),
]


# ---------------------------------------------------------------- sentences and numbers

_CITE = re.compile(r"\[(\d+)\]")
# Models also cite in prose: "Excerpt 6 says...", "excerpts 2 and 5". Those are citations too, and
# their numbers are ids, not figures to check.
_PROSE_CITE = re.compile(r"\b(?:excerpts?|articles?|sources?)\s*#?\[?(\d+)\]?(?:\s*(?:,|and|&)\s*\[?(\d+)\]?)*", re.I)
# A sentence ends at . ! ? followed by space and a capital, quote, bracket or digit. "U.S." and
# "Inc." are protected first, since they end in a period followed by a space and a capital.
_ABBREV = re.compile(r"\b(U\.S|U\.K|Inc|Corp|Co|Ltd|vs|e\.g|i\.e|Mr|Ms|Dr|St|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.")


def split_sentences(answer: str) -> list[str]:
    """Answers are 3-4 sentences, often with bullets. Each sentence keeps its citations."""
    text = _ABBREV.sub(lambda m: m.group(0).replace(".", "․"), answer or "")
    parts: list[str] = []
    for block in re.split(r"\n+", text):
        block = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", block).strip()
        if not block:
            continue
        # Mark the boundaries first, then split, so citations that follow the period
        # ("...fell 7%. [2] Next") stay with the sentence they belong to.
        block = re.sub(r"([.!?])\s*((?:\[\d+\])+)\s+(?=[A-Z\"'(0-9$])", "\\1\\2\x00", block)
        block = re.sub(r"([.!?])\s+(?=[A-Z\"'(\[0-9$])", "\\1\x00", block)
        parts += [p.strip() for p in block.split("\x00") if p.strip()]
    # re-attach citations the split left at the start of the next part
    out: list[str] = []
    for p in parts:
        m = re.match(r"^((?:\[\d+\])+)\s*(.*)$", p)
        if m and out:
            out[-1] += " " + m.group(1)
            p = m.group(2)
        if p:
            out.append(p)
    return [s.replace("․", ".") for s in out]


def sentence_cites(sentence: str) -> list[int]:
    ids = {int(x) for x in _CITE.findall(sentence)}
    for m in _PROSE_CITE.finditer(sentence):
        ids |= {int(x) for x in re.findall(r"\d+", m.group(0))}
    return sorted(ids)


_NUM = re.compile(r"(?<![\w.])[-+]?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-+]?\$?\d+(?:\.\d+)?")
_YEARLIKE = re.compile(r"^(19|20)\d{2}$")


def numbers_in(text: str) -> list[tuple[float, int]]:
    """(value, decimals) for each number, citation markers and years removed. Years are left
    out: "2026" in an answer is a date, and dates are checked by the judge, not here."""
    clean = _PROSE_CITE.sub(" ", _CITE.sub(" ", text or ""))
    out = []
    for raw in _NUM.findall(clean):
        s = raw.replace("$", "").replace(",", "").lstrip("+")
        if _YEARLIKE.match(s):
            continue
        try:
            v = float(s)
        except ValueError:
            continue
        dec = len(s.split(".")[1]) if "." in s else 0
        out.append((v, dec))
    return out


def number_supported(value: float, decimals: int, source_values: Iterable[float]) -> bool:
    """A number counts as quoted when some source number rounds to it at the precision the
    answer used: 7% quotes 7.4%, 4.5 billion quotes 4.52 billion; 7% does not quote 7.6%.
    Same idea as F.17's grounding gate (a model echoing 0.2803 as 0.28 is quoting, not
    inventing), with the tolerance taken from the answer's own precision."""
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(abs(value) - abs(s)) <= tol for s in source_values)


def numeric_grounding(sentence: str, cited_texts: list[str]) -> tuple[int, int]:
    """(numbers found in the cited excerpts, numbers in the sentence)."""
    nums = numbers_in(sentence)
    if not nums:
        return 0, 0
    pool = [v for t in cited_texts for v, _ in numbers_in(t)]
    return sum(number_supported(v, d, pool) for v, d in nums), len(nums)


REFUSAL_MARKERS = re.compile(
    r"(do(?:es)? not (?:contain|mention|include|say|provide|address)|no (?:information|mention|excerpt)|"
    r"not (?:in|covered by|mentioned in) the (?:excerpts|articles)|could not find|cannot (?:answer|determine)|"
    r"nothing to answer|unable to (?:answer|find|determine))", re.I)


def is_refusal(answer: str, refused_flag: bool = False) -> bool:
    """Declined to answer: the endpoint's empty-retrieval refusal, or an answer whose every
    sentence says the excerpts do not cover it."""
    if refused_flag:
        return True
    sents = split_sentences(answer)
    return bool(sents) and all(REFUSAL_MARKERS.search(s) for s in sents)


# ---------------------------------------------------------------- metrics

def answer_metrics(sentences: list[dict], labels: dict[str, dict]) -> dict[str, Any]:
    """Per-answer numbers from its sentences and one label source. A sentence label is
    {"kind": claim|meta, "verdict": supported|partial|unsupported, "supported_by": [ids]}."""
    claims = [s for s in sentences if (labels.get(s["_key"]) or {}).get("kind") == "claim"]
    if not claims:
        return {"claims": 0}
    score = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
    faith = [score.get(labels[s["_key"]].get("verdict"), 0.0) for s in claims]
    cited = [s for s in claims if s["cites"]]
    cite_total = sum(len(s["cites"]) for s in cited)
    cite_good = sum(len(set(s["cites"]) & set(labels[s["_key"]].get("supported_by") or [])) for s in cited)
    correct_cited = sum(1 for s in cited if set(s["cites"]) & set(labels[s["_key"]].get("supported_by") or []))
    num_ok = sum(s["numbers_ok"] for s in claims)
    num_all = sum(s["numbers_total"] for s in claims)
    return {
        "claims": len(claims),
        "faithfulness": sum(faith) / len(claims),
        "unsupported_rate": sum(1 for s in claims if labels[s["_key"]].get("verdict") == "unsupported") / len(claims),
        "citation_coverage": len(cited) / len(claims),
        "citation_precision": cite_good / cite_total if cite_total else None,
        "correctly_cited": correct_cited / len(cited) if cited else None,
        "numbers_ok": num_ok, "numbers_total": num_all,
    }


def bootstrap_mean_ci(values: list[float], iters: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(iters))
    return sum(vals) / len(vals), means[int(0.025 * iters)], means[int(0.975 * iters)]


def paired_diff_ci(a: dict[str, float], b: dict[str, float], iters: int = 2000, seed: int = 7) -> tuple[float, float, float, int]:
    """Mean of b - a over the questions both have, with a bootstrap CI over questions."""
    keys = sorted(k for k in a if k in b and a[k] is not None and b[k] is not None)
    diffs = [b[k] - a[k] for k in keys]
    if not diffs:
        return float("nan"), float("nan"), float("nan"), 0
    m, lo, hi = bootstrap_mean_ci(diffs, iters, seed)
    return m, lo, hi, len(diffs)


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return float("nan")
    cats = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ---------------------------------------------------------------- database

def _db():
    from pymongo import MongoClient
    from hybrid_search import DB_NAME, MONGO_URI
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)[DB_NAME]


def load_questions() -> list[dict]:
    import yaml
    data = yaml.safe_load((ROOT / "eval" / "retrieval_queries.yaml").read_text())
    rows = data["queries"] if isinstance(data, dict) and "queries" in data else data
    qs = [{"id": q["id"], "question": q["query"], "challenge": q.get("challenge"), "symbol": q.get("symbol") or None,
           "since": q.get("since"), "until": q.get("until"), "answerable": None} for q in rows]
    qs += [{"id": i, "question": t, "challenge": "unanswerable", "symbol": None, "since": None, "until": None,
            "answerable": False} for i, t in UNANSWERABLE]
    return qs


# ---------------------------------------------------------------- 1. generate

def _date_int(v):
    if v in (None, ""):
        return None
    return int(str(v).replace("-", "")[:8])


def generate(run: str, rerank: bool, limit: int | None = None, reasoning: bool = True) -> None:
    """Answer every question exactly as /api/ask/news does (same function, same model and
    temperature) and store the answer with the excerpt text the model saw."""
    from langchain_openai import ChatOpenAI
    from pymongo import MongoClient
    import news_rag
    from hybrid_search import MONGO_URI

    db = _db()
    coll = MongoClient(MONGO_URI)["quant_data"]["news_articles_company_matched_v2"]
    # reasoning=True is what /api/ask/news does today: qwen3.5 thinks first, ~60-200 s an answer on
    # 2026-10-22 against ~9 s measured for R.10. LM Studio now ignores reasoning_effort "none",
    # chat_template_kwargs and "/no_think" (all 299 of 299 tokens went to reasoning); the one switch
    # that works is an empty think block pre-filled as the assistant turn, which the enrichment
    # pipeline already uses. reasoning=False does that.
    llm = ChatOpenAI(model=GEN_MODEL, openai_api_key="lm-studio", openai_api_base=LM_STUDIO_URL, temperature=0.2)
    if not reasoning:
        llm = NoThink(llm)
    done = {d["question_id"] for d in db[RUNS_COLL].find({"run": run, "error": None}, {"question_id": 1})}
    todo = [q for q in load_questions() if q["id"] not in done][:limit]
    print(f"[generate] run={run} rerank={rerank} reasoning={reasoning} model={GEN_MODEL}: {len(todo)} questions")
    for q in todo:
        try:
            out = news_rag.answer_from_news(q["question"], llm=llm, coll=coll, symbol=q["symbol"],
                                            since=_date_int(q["since"]), until=_date_int(q["until"]), rerank=rerank)
            err = out.get("error")
        except Exception as e:  # noqa: BLE001 - one question must not stop the run
            out, err = {}, f"{type(e).__name__}: {e}"[:300]
        excerpts = _excerpt_texts(coll, out.get("sources") or [])
        db[RUNS_COLL].update_one({"run": run, "question_id": q["id"]}, {"$set": {
            **{k: q[k] for k in ("question", "challenge", "symbol", "answerable")},
            "rerank": rerank, "reasoning": reasoning, "model": GEN_MODEL, "answer": out.get("answer"), "refused": out.get("refused", False),
            "cited": out.get("cited", []), "sources": out.get("sources", []), "excerpts": excerpts,
            "generation_ms": out.get("generation_ms"), "error": err, "generated_at": datetime.now(timezone.utc)}},
            upsert=True)
        print(f"  {q['id']}: {'ERROR ' + err if err else (out.get('answer') or '')[:90]!r}")
    _split_run(db, run)


class NoThink:
    """Wraps a chat model so every call ends with an empty, pre-filled think block: qwen3.5 then
    answers directly instead of reasoning for thousands of tokens."""

    PREFILL = "<think>\n\n</think>\n\n"

    def __init__(self, llm):
        self.llm = llm

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        return self.llm.invoke(list(messages) + [AIMessage(content=self.PREFILL)])


def _excerpt_texts(coll, sources: list[dict]) -> dict[str, str]:
    """id -> the text the model saw for that excerpt (title + body cut at news_rag.MAX_SNIPPET),
    rebuilt the same way build_context does, so the judge reads what the generator read."""
    import news_rag
    from injection_guard import screen
    from bson import ObjectId
    out = {}
    ids = [s for s in sources if s.get("mongo_id")]
    bodies = {}
    if ids:
        for d in coll.find({"_id": {"$in": [ObjectId(s["mongo_id"]) for s in ids]}}, {"content": 1}):
            bodies[str(d["_id"])] = d.get("content") or ""
    for s in sources:
        body = screen((bodies.get(s.get("mongo_id")) or "").strip()).text
        # The generator saw symbol and date in the excerpt tag; without them every dated
        # sentence looks invented (the first judge run marked "in 2018" as unsupported).
        out[str(s["id"])] = (f"symbol: {s.get('symbol')} | date: {s.get('date')}\n{s.get('title') or ''}\n"
                             f"{body[:news_rag.MAX_SNIPPET] if body else '(headline only)'}")
    return out


def refresh_excerpts(run: str | None = None) -> int:
    """Rebuild the stored excerpt text of existing runs (after a change to _excerpt_texts)."""
    from pymongo import MongoClient
    from hybrid_search import MONGO_URI
    db = _db()
    coll = MongoClient(MONGO_URI)["quant_data"]["news_articles_company_matched_v2"]
    n = 0
    for r in db[RUNS_COLL].find({"error": None} | ({"run": run} if run else {})):
        db[RUNS_COLL].update_one({"_id": r["_id"]}, {"$set": {"excerpts": _excerpt_texts(coll, r.get("sources") or [])}})
        n += 1
    return n


def _split_run(db, run: str) -> None:
    """One doc per sentence, with its citations and the deterministic number check."""
    from pymongo import UpdateOne
    ops = []
    for r in db[RUNS_COLL].find({"run": run, "error": None}):
        for i, s in enumerate(split_sentences(r.get("answer") or "")):
            cites = sentence_cites(s)
            ok, total = numeric_grounding(s, [r["excerpts"].get(str(c), "") for c in cites])
            ops.append(UpdateOne({"run": run, "question_id": r["question_id"], "idx": i}, {"$set": {
                "_key": f"{run}|{r['question_id']}|{i}", "text": s, "cites": cites, "numbers_ok": ok, "numbers_total": total,
                "challenge": r.get("challenge")}}, upsert=True))
    if ops:
        db[SENT_COLL].bulk_write(ops)


# ---------------------------------------------------------------- 2. judge

SENT_PROMPT = """You check ONE sentence from an AI-written answer against the news excerpts it was given.

Excerpts:
{excerpts}

Question that was asked: {question}

Sentence to check:
"{sentence}"

Decide three things.
kind:
  "claim" - it states something about the world: a fact, number, date, event, attribution or cause.
            A sentence that reports what the excerpts say ("Excerpt 2 says X", "the excerpts report X",
            "according to the April 21 excerpt, X") is a claim about X: check X.
  "meta"  - ONLY a statement about what the excerpts do NOT cover ("the excerpts do not mention ...",
            "no excerpt gives a figure"), or a transition with no factual content at all.
verdict (for a claim; for meta use "supported"):
  "supported"   - every fact in it is stated in at least one excerpt (the symbol and date lines count).
                  Paraphrase is fine; rounding a number is fine.
  "partial"     - part of it is stated and part is not. Typical cases: a generalisation the excerpts do
                  not make ("several carmakers" when only one is named); a stronger wording ("first time
                  ever" when the excerpt says "first time in a decade"); a date, figure or name that is not
                  in the excerpts; facts from different companies or dates merged into one statement.
  "unsupported" - none of it is in the excerpts, or it contradicts them.
supported_by: the ids of the excerpts that state it (empty if none). Judge only against the
excerpts, never against what you know yourself.

Reply with JSON only: {{"kind": "claim|meta", "verdict": "supported|partial|unsupported", "supported_by": [ids], "why": "<one short reason>"}}"""

REL_PROMPT = """Grade whether an answer addresses the question asked. Judge relevance only, not truth.

Question: {question}

Answer:
{answer}

grade 2 - it answers the question (or, when the material does not allow an answer, says so clearly).
grade 1 - related to the question but misses what was asked or answers a different question.
grade 0 - off-topic or empty.

Reply with JSON only: {{"grade": 0|1|2, "why": "<one short reason>"}}"""


def _excerpt_block(excerpts: dict[str, str]) -> str:
    return "\n\n".join(f'<excerpt id="{k}">\n{v}\n</excerpt>' for k, v in sorted(excerpts.items(), key=lambda kv: int(kv[0])))


def _chat(prompt: str, model: str, max_tokens: int = 250, tries: int = 5) -> str:
    """One judge call. LM Studio answers 500 while a JIT model is still loading and under load,
    so 5xx and connection errors are retried with backoff (same as the enrichment pipeline)."""
    import time
    import requests
    messages = [{"role": "user", "content": prompt}]
    if "qwen" in model:
        # reasoning_effort "none" is ignored now (2026-10-22); an empty pre-filled think block works.
        messages.append({"role": "assistant", "content": NoThink.PREFILL})
    payload = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": messages}
    for attempt in range(tries):
        try:
            r = requests.post(f"{LM_STUDIO_URL}/chat/completions", json=payload, timeout=300)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}", response=r)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning_content") or ""
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if attempt == tries - 1 or (status is not None and status < 500):
                raise
            time.sleep(min(30, 3 * 2 ** attempt))
    return ""


def warm_up(model: str) -> None:
    """Load the judge model before fanning out: concurrent first requests all get 500s."""
    _chat("Reply with OK.", model, max_tokens=3, tries=8)


def parse_sentence_label(text: str) -> dict:
    start, end = (text or "").find("{"), (text or "").rfind("}")
    if start < 0 or end < start:
        return {"error": f"unparseable: {(text or '')[:120]}"}
    try:
        out = json.loads(text[start:end + 1])
    except ValueError:
        return {"error": f"unparseable: {text[:120]}"}
    kind = str(out.get("kind", "")).lower().strip()
    verdict = str(out.get("verdict", "")).lower().strip()
    if kind not in ("claim", "meta") or verdict not in ("supported", "partial", "unsupported"):
        return {"error": f"bad label: {kind}/{verdict}"}
    ids = []
    for x in out.get("supported_by") or []:
        try:
            ids.append(int(str(x).strip("[] ")))
        except ValueError:
            pass
    return {"kind": kind, "verdict": verdict if kind == "claim" else "supported", "supported_by": sorted(set(ids)),
            "why": str(out.get("why", ""))[:300]}


def parse_relevance(text: str) -> dict:
    start, end = (text or "").find("{"), (text or "").rfind("}")
    try:
        out = json.loads(text[start:end + 1])
        g = int(out.get("grade"))
        assert g in (0, 1, 2)
        return {"grade": g, "why": str(out.get("why", ""))[:300]}
    except Exception:  # noqa: BLE001
        return {"error": f"unparseable: {(text or '')[:120]}"}


def _pending_sentences(db, source: str, run: str | None, keys: set[str] | None = None) -> list[dict]:
    done = {d["key"] for d in db[LABEL_COLL].find({"source": source, "level": "sentence", "error": None}, {"key": 1})}
    q = {"run": run} if run else {}
    return [s for s in db[SENT_COLL].find(q) if s["_key"] not in done and (keys is None or s["_key"] in keys)]


def judge(source: str = JUDGE_SOURCE, model: str = JUDGE_MODEL, run: str | None = None, limit: int | None = None,
          keys: set[str] | None = None) -> None:
    db = _db()
    runs = {(r["run"], r["question_id"]): r for r in db[RUNS_COLL].find({"error": None} | ({"run": run} if run else {}))}
    todo = _pending_sentences(db, source, run, keys)[:limit]
    print(f"[judge] {source} ({model}): {len(todo)} sentences")
    if todo:
        warm_up(model)

    def one(s):
        r = runs.get((s["run"], s["question_id"]))
        if not r:
            return s, {"error": "run missing"}
        try:
            text = _chat(SENT_PROMPT.format(excerpts=_excerpt_block(r["excerpts"]), question=r["question"], sentence=s["text"]), model)
            return s, parse_sentence_label(text)
        except Exception as e:  # noqa: BLE001
            return s, {"error": str(e)[:200]}

    with ThreadPoolExecutor(WORKERS) as ex:
        for s, lab in ex.map(one, todo):
            _save_label(db, source, "sentence", s["_key"], lab, model)
    if keys is None:
        _judge_relevance(db, source, model, run)


def _judge_relevance(db, source: str, model: str, run: str | None) -> None:
    done = {d["key"] for d in db[LABEL_COLL].find({"source": source, "level": "answer", "error": None}, {"key": 1})}
    todo = [r for r in db[RUNS_COLL].find({"error": None} | ({"run": run} if run else {}))
            if f"{r['run']}|{r['question_id']}" not in done]
    print(f"[judge] relevance: {len(todo)} answers")
    for r in todo:
        try:
            lab = parse_relevance(_chat(REL_PROMPT.format(question=r["question"], answer=r.get("answer") or ""), model, 150))
        except Exception as e:  # noqa: BLE001
            lab = {"error": str(e)[:200]}
        _save_label(db, source, "answer", f"{r['run']}|{r['question_id']}", lab, model)


def _save_label(db, source: str, level: str, key: str, label: dict, model: str) -> None:
    db[LABEL_COLL].update_one({"source": source, "level": level, "key": key},
                              {"$set": {**label, "error": label.get("error"), "model": model,
                                        "labelled_at": datetime.now(timezone.utc)}}, upsert=True)


# ---------------------------------------------------------------- 3. reference labels (Claude)

def reference(sample: int, seed: int = 11, source: str = "reference_claude") -> None:
    """Label a stratified sample of sentences with Claude through the claude CLI (the user's
    subscription), in batches, as the yardstick for the local judge. Stratified by run and by
    whether the sentence carries a citation, so uncited sentences are not drowned out."""
    sys.path.insert(0, str(ROOT))
    from markets.briefs import find_claude

    db = _db()
    runs = {(r["run"], r["question_id"]): r for r in db[RUNS_COLL].find({"error": None})}
    have = {d["key"] for d in db[LABEL_COLL].find({"source": source, "level": "sentence", "error": None}, {"key": 1})}
    rows = [s for s in db[SENT_COLL].find() if (s["run"], s["question_id"]) in runs]
    rng = random.Random(seed)
    groups: dict[tuple, list] = {}
    for s in rows:
        groups.setdefault((s["run"], bool(s["cites"])), []).append(s)
    pick: list[dict] = []
    for g in groups.values():
        rng.shuffle(g)
    while len(pick) < sample and any(groups.values()):
        for g in groups.values():
            if g and len(pick) < sample:
                pick.append(g.pop())
    pick = [s for s in pick if s["_key"] not in have]
    binary = find_claude()
    if not binary:
        raise SystemExit("claude CLI not found")
    print(f"[reference] {len(pick)} sentences via {binary}")
    for i in range(0, len(pick), 10):
        batch = pick[i:i + 10]
        items = []
        for j, s in enumerate(batch):
            r = runs[(s["run"], s["question_id"])]
            items.append(f"### ITEM {j}\n{SENT_PROMPT.format(excerpts=_excerpt_block(r['excerpts']), question=r['question'], sentence=s['text'])}")
        prompt = ("Label each ITEM below independently, exactly as its instructions say. Reply with one JSON object "
                  "per line, in order, each with an extra field \"item\": <number>. No other text.\n\n" + "\n\n".join(items))
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        proc = subprocess.run([binary, "-p", "--output-format", "json", "--no-session-persistence", "--tools", ""],
                              input=prompt, capture_output=True, text=True, env=env, timeout=900,
                              cwd=str(Path.home() / "Library/Application Support/quant_ai"))
        try:
            text = json.loads(proc.stdout).get("result") or ""
        except ValueError:
            print("  batch failed:", (proc.stderr or proc.stdout)[-300:])
            continue
        got = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    got[int(obj.get("item"))] = line
                except (ValueError, TypeError):
                    pass
        for j, s in enumerate(batch):
            _save_label(db, source, "sentence", s["_key"], parse_sentence_label(got.get(j, "")), "claude")
        print(f"  {i + len(batch)}/{len(pick)}")


# ---------------------------------------------------------------- 4. agreement and scores

def kappa(a: str, b: str) -> dict:
    db = _db()
    la = {d["key"]: d for d in db[LABEL_COLL].find({"source": a, "level": "sentence", "error": None})}
    lb = {d["key"]: d for d in db[LABEL_COLL].find({"source": b, "level": "sentence", "error": None})}
    keys = sorted(set(la) & set(lb))
    cls = lambda d: "meta" if d["kind"] == "meta" else d["verdict"]
    four = [(cls(la[k]), cls(lb[k])) for k in keys]
    binary = [("ok" if x in ("supported", "meta") else "bad", "ok" if y in ("supported", "meta") else "bad") for x, y in four]
    res = {"pairs": len(keys), "exact": sum(x == y for x, y in four) / len(keys) if keys else float("nan"),
           "kappa_4class": cohen_kappa(four), "kappa_supported_vs_not": cohen_kappa(binary)}
    print(json.dumps(res, indent=1))
    confusion: dict[str, dict[str, int]] = {}
    for x, y in four:
        confusion.setdefault(x, {}).setdefault(y, 0)
        confusion[x][y] += 1
    print(f"rows {a}, columns {b}:", json.dumps(confusion))
    return res


def score(source: str = JUDGE_SOURCE, compare: list[str] | None = None) -> dict:
    db = _db()
    labels = {d["key"]: d for d in db[LABEL_COLL].find({"source": source, "error": None})}
    rel = {k: v for k, v in labels.items() if v.get("level") == "answer"}
    sent = {k: v for k, v in labels.items() if v.get("level") == "sentence"}
    by_run: dict[str, dict[str, dict]] = {}
    for r in db[RUNS_COLL].find({"error": None}):
        sents = list(db[SENT_COLL].find({"run": r["run"], "question_id": r["question_id"]}).sort("idx", 1))
        m = answer_metrics(sents, sent)
        m["relevance"] = (rel.get(f"{r['run']}|{r['question_id']}") or {}).get("grade")
        m["refused"] = is_refusal(r.get("answer") or "", r.get("refused", False))
        m["answerable"] = r.get("answerable")
        by_run.setdefault(r["run"], {})[r["question_id"]] = m
    report = {}
    for run, qs in sorted(by_run.items()):
        ans = {k: v for k, v in qs.items() if v.get("answerable") is not False}
        una = {k: v for k, v in qs.items() if v.get("answerable") is False}
        row = {}
        for metric in ("faithfulness", "unsupported_rate", "citation_coverage", "citation_precision", "correctly_cited"):
            m, lo, hi = bootstrap_mean_ci([v.get(metric) for v in ans.values() if v.get("claims")])
            row[metric] = (round(m, 3), round(lo, 3), round(hi, 3))
        nok = sum(v.get("numbers_ok", 0) for v in ans.values())
        nall = sum(v.get("numbers_total", 0) for v in ans.values())
        row["numeric_grounding"] = f"{nok}/{nall}" + (f" = {nok / nall:.3f}" if nall else "")
        rels = [v["relevance"] for v in ans.values() if v.get("relevance") is not None]
        row["relevance_mean_0_2"] = round(sum(rels) / len(rels), 3) if rels else None
        row["refused_when_unanswerable"] = f"{sum(v['refused'] for v in una.values())}/{len(una)}"
        row["refused_when_answerable"] = f"{sum(v['refused'] for v in ans.values())}/{len(ans)}"
        row["answers_with_claims"] = sum(1 for v in ans.values() if v.get("claims"))
        report[run] = row
    print(json.dumps(report, indent=1))
    if compare and len(compare) == 2 and all(c in by_run for c in compare):
        a, b = (by_run[c] for c in compare)
        print(f"\npaired {compare[1]} - {compare[0]} (mean, 95% CI, n):")
        for metric in ("faithfulness", "citation_precision", "correctly_cited", "relevance"):
            print(f"  {metric:20}", tuple(round(x, 3) if isinstance(x, float) else x for x in
                                          paired_diff_ci({k: v.get(metric) for k, v in a.items()},
                                                         {k: v.get(metric) for k, v in b.items()})))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--rerank", choices=("on", "off"), default="off")
    ap.add_argument("--reasoning", choices=("on", "off"), default="on", help="--generate: qwen thinking (on = production today)")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--source", default=JUDGE_SOURCE)
    ap.add_argument("--subset", metavar="SOURCE", help="--judge: only sentences this source labelled")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--kappa", nargs=2, metavar=("A", "B"))
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.generate:
        if not a.run:
            ap.error("--generate needs --run")
        generate(a.run, a.rerank == "on", a.limit, a.reasoning == "on")
    if a.judge:
        keys = None
        if a.subset:
            keys = {d["key"] for d in _db()[LABEL_COLL].find({"source": a.subset, "level": "sentence"}, {"key": 1})}
        judge(a.source, a.model, a.run, a.limit, keys)
    if a.reference:
        reference(a.sample)
    if a.kappa:
        kappa(*a.kappa)
    if a.score:
        score(a.source, a.compare)


if __name__ == "__main__":
    main()
