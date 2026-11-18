"""Q.17.1's acceptance replay (PROJECT_PLAN Q.17.1 and its round 4 addendum, 2026-11-18): on historical
>= 2 x ATR drops, does the case against adding recall the guidance cuts and rating downgrades that were
in front of it, and does every citation stay inside the packet?

Each drop is replayed as of its close, with only what was known then: the drop itself, the articles of
the 7 days before the next open (news_articles_company_matched_v2 with production labels, newest 12,
the injection guard applied as live), the analyst changes of the 30 days before (quant_data's Q.17.3
history), and the same base rates and drifts as the live packet (devils_advocate.REBOUND / DROP_OUTCOMES /
NEWS_DRIFTS). Catalysts, the N.6 judgment and the buy gate are not reconstructed and say so. The answer
comes from the live prompt, schema, check and `claude -p` call (devils_advocate.SYSTEM / ask / check),
with the live single retry.

Sample: 20 drops from 2019 on, seed 7, ten with a guidance-cut headline on the drop or the next session
and ten with an analyst downgrade in the 30 days before (one per stock where possible). Scored on what
the packet held: a guidance cut is an article whose title matches news_events' GUIDE and CUT; a rating
cut is a downgrade (action "down"). A case is recalled when a point cites one of its cuts.
Accept: recall >= 95% of the cases holding a cut; 100% of answers pass the citation check.

Run with quant_data's Python (pyarrow, the data files):
  /Users/xiz/Quant_trade/quant_data/.venv311/bin/python eval/replay_devils_advocate.py [--dry]
Writes /Volumes/Data4T/q17/1/replay.json (and replay_packets.jsonl).
"""
from __future__ import annotations

import json
import re
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
QD = Path("/Users/xiz/Quant_trade/quant_data/research/models")
sys.path.insert(0, str(QD))

import markets.analysts as an  # noqa: E402
import markets.devils_advocate as da  # noqa: E402
from markets import daily_action  # noqa: E402
from injection_guard import screen  # noqa: E402

OUT = Path("/Volumes/Data4T/q17/1")
DROPS = Path("/Volumes/Data4T/q17/4/drops.parquet")
ACTIONS = Path("/Volumes/Data4T/q17/3/analyst_actions.parquet")
N_EACH, SEED = 10, 7


def _guide_cut(title: str) -> bool:
    import news_events as ne
    t = (title or "").lower()
    return bool(re.search(ne.GUIDE, t) and re.search(ne.CUT, t))


def _mongo():
    from pymongo import MongoClient
    return MongoClient("mongodb://root:root@127.0.0.1:37018/?authSource=admin", serverSelectionTimeoutMS=8000)["quant_data"]


def sample(acts: pd.DataFrame) -> pd.DataFrame:
    ev = pd.read_parquet(DROPS)
    ev = ev[ev.year >= 2019].copy()
    acts = acts.assign(t=pd.to_datetime(acts.graded_at))
    down = acts[acts.action == "down"]

    def had_down(r):
        d = down[(down.symbol == r.symbol) & (down.t < r.date + timedelta(hours=20)) & (down.t >= r.date - timedelta(days=30))]
        return len(d) > 0

    ev["had_down"] = ev.apply(had_down, axis=1)
    picks = []
    for flag in ("guide_cut", "had_down"):
        pool = ev[(ev[flag] > 0) & ~ev.index.isin([p for p in picks])].sample(frac=1, random_state=SEED)
        pool = pd.concat([pool.drop_duplicates("symbol"), pool[pool.duplicated("symbol")]])
        picks += list(pool.index[:N_EACH])
    return ev.loc[picks].assign(why=["guidance cut"] * N_EACH + ["downgrade"] * N_EACH)


def build(r: pd.Series, acts: pd.DataFrame, col) -> tuple[str, dict, dict]:
    ids, facts = {}, []

    def fact(text: str) -> None:
        k = f"F{len([i for i in ids if i.startswith('F')]) + 1}"
        ids[k] = {"text": text}
        facts.append(f"{k} {text}")

    day = pd.Timestamp(r.date)
    lines = [f"STOCK {r.symbol}; replay as of the close of {day:%Y-%m-%d} (historical evaluation)",
             f"TODAY: move vs the previous close {r.drop_pct:.2f}%, {r.drop_atr:.2f} x ATR14 down (ATR14 = {r.atr_pct:.2f}% of the price)",
             f"LARGE DROP: {r.drop_atr:.2f} x ATR on {day:%Y-%m-%d}; earnings report into it: {'yes' if r.earnings else 'no'}",
             "BUY GATE: not evaluated in this replay; catalysts and the news-storyline judgment are not available for past dates"]
    cause = daily_action.drop_cause({"earnings": int(r.earnings), "drop_atr": float(r.drop_atr)})
    rb = daily_action.REBOUND.get(cause)
    fact(f"Base rate after {rb['label']} ({rb['n']} drops, 102 stocks, 2016-2026): the high reached +1.5 / +2 x ATR within "
         f"15 sessions in {rb[1.5]}% / {rb[2.0]}% of cases; when +1.5 was not reached, the close 15 sessions later averaged "
         f"{rb['unfilled']}% against selling at the next open.")
    for o in da.DROP_OUTCOMES.values():
        if o["report"] == bool(r.earnings):
            fact(f"Shrunk base rate after a >= 2 x ATR drop {o['label']} ({o['drops']} drops of 101 stocks, 2018-2026, weighted "
                 f"by drops per day, empirical-Bayes shrunk): the lower barrier (1.5 x ATR under the next open) came first within "
                 f"5 sessions in {o['lower5']}%; the next 20 sessions trailed SPY by more than 3% in {o['bad20']}%.")
    for d in da.NEWS_DRIFTS:
        fact("Event drift, excess over all days (880k articles, 2018-2026), " + d + ".")
    t = pd.to_datetime(acts.graded_at)
    a = acts[(acts.symbol == r.symbol) & (t >= day - timedelta(days=da.ANALYST_DAYS)) & (t < day + timedelta(hours=20))]
    rating_cuts = []
    for _, x in a.sort_values("graded_at", ascending=False).head(12).iterrows():
        row = {k: (None if pd.isna(v) else v) for k, v in x.to_dict().items()}
        row["graded_at"] = str(row["graded_at"])
        fact(f"Analyst {row['graded_at'][:10]} {row['firm']}: {an.describe(row)}")
        if (row.get("action") or "") == "down":
            rating_cuts.append(f"F{len([i for i in ids if i.startswith('F')])}")
    lo = (day - timedelta(days=da.NEWS_DAYS)).strftime("%Y%m%d%H%M%S")
    hi = (day + timedelta(days=1, hours=13, minutes=30)).strftime("%Y%m%d%H%M%S")      # before the next open
    rows = list(col.find({"symbol": r.symbol, "date": {"$gte": lo, "$lt": hi}},
                         {"_id": 0, "title": 1, "url": 1, "date": 1, "llm_event_type_a": 1, "llm_sentiment_final": 1}).sort("date", -1))
    arts, seen, guide_cuts = [], set(), []
    for x in rows:
        title = (x.get("title") or "").strip()
        stamp = str(x.get("date") or "")
        if len(stamp) == 8 and stamp > f"{day:%Y%m%d}":
            continue                     # a date-only row of the next day may be from after the open
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        s = screen(title)
        if s.flags:
            continue
        k = f"A{len(arts) + 1}"
        ids[k] = {"title": s.text, "url": x.get("url"), "time": x.get("date")}
        tag = ", ".join(str(v) for v in (x.get("llm_event_type_a"), x.get("llm_sentiment_final")) if v not in (None, ""))
        arts.append(f"{k} {x.get('date')}: {s.text}" + (f" [{tag}]" if tag else ""))
        if _guide_cut(s.text):
            guide_cuts.append(k)
        if len(arts) >= da.NEWS_PER_SYMBOL:
            break
    text = "\n".join(lines + ["", "FACTS (cite by id)"] + facts + ["", "ARTICLES of the last week (cite by id)"] + (arts or ["(none)"]))
    return text, ids, {"guide_cuts": guide_cuts, "rating_cuts": rating_cuts}


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    acts = pd.read_parquet(ACTIONS)
    picks = sample(acts)
    col = _mongo()["news_articles_company_matched_v2"]
    dry = "--dry" in sys.argv
    cases = []
    with open(OUT / "replay_packets.jsonl", "w") as fh:
        for _, r in picks.iterrows():
            text, ids, truth = build(r, acts, col)
            case = {"symbol": r.symbol, "date": str(pd.Timestamp(r.date).date()), "why": r.why, **truth, "packet_chars": len(text)}
            if not dry:
                ans, tries = da._answer(da.ask(text)), 1
                problems = da.check(ans, ids)
                if problems:
                    ans, tries = da._answer(da.ask(text + "\n\nYour previous answer was rejected: " + "; ".join(problems)
                                                   + ". Answer again in the schema.")), 2
                    problems = da.check(ans, ids)
                cited = {c for p in ((ans or {}).get("points") or []) for c in (p.get("cites") or [])}
                case.update(tries=tries, problems=problems, strength=(ans or {}).get("strength"), cited=sorted(cited),
                            guide_recalled=bool(set(truth["guide_cuts"]) & cited) if truth["guide_cuts"] else None,
                            rating_recalled=bool(set(truth["rating_cuts"]) & cited) if truth["rating_cuts"] else None,
                            summary=((ans or {}).get("summary") or {}).get("en"))
                present = [x for x in (case["guide_recalled"], case["rating_recalled"]) if x is not None]
                case["recalled"] = any(present) if present else None
            cases.append(case)
            fh.write(json.dumps({**case, "packet": text}) + "\n")
            print(json.dumps({k: case.get(k) for k in ("symbol", "date", "why", "guide_cuts", "rating_cuts", "recalled", "problems")}), flush=True)
    with_cut = [c for c in cases if c["guide_cuts"] or c["rating_cuts"]]
    rep = {"cases": len(cases), "with_a_cut_in_the_packet": len(with_cut), "dry": dry}
    if not dry:
        rep.update({
            "recall": round(sum(bool(c["recalled"]) for c in with_cut) / max(1, len(with_cut)), 3),
            "guide_recall": f"{sum(bool(c['guide_recalled']) for c in cases if c['guide_cuts'])}/{sum(bool(c['guide_cuts']) for c in cases)}",
            "rating_recall": f"{sum(bool(c['rating_recalled']) for c in cases if c['rating_cuts'])}/{sum(bool(c['rating_cuts']) for c in cases)}",
            "citation_check_passed": f"{sum(not c['problems'] for c in cases)}/{len(cases)}",
            "strength": pd.Series([c["strength"] for c in cases]).value_counts().to_dict()})
        rep["passes"] = rep["recall"] >= 0.95 and all(not c["problems"] for c in cases)
    rep["cases_detail"] = cases
    (OUT / "replay.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "cases_detail"}, indent=1, default=str))
    return rep


if __name__ == "__main__":
    main()
