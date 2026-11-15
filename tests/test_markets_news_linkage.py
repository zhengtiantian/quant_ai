"""The holdings page's news hint (markets/news_linkage.py, N.6). No Mongo: the collection is faked."""
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from markets import news_linkage  # noqa: E402

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)


class _Cursor(list):
    def sort(self, key, order):
        return _Cursor(sorted(self, key=lambda d: d[key], reverse=order < 0))


class _Coll:
    def __init__(self, docs):
        self.docs = docs

    def find(self, q, proj=None):
        return _Cursor(d for d in self.docs if d["status"] == q["status"]
                       and ("at" not in q or d["at"] >= q["at"]["$gte"]) and ("score" not in q or "score" in d))


def judgment(sym, hours_ago, direction, status="ok"):
    return {"_id": f"{sym}:{hours_ago}", "symbol": sym, "at": NOW - timedelta(hours=hours_ago), "status": status,
            "answer": {"direction": direction, "confidence": "medium", "novelty": "new", "chain": ["a", "b"], "cites": ["A1"],
                       "watch": ["board statement"], "other_companies": [{"symbol": "AMD", "direction": 1, "why": "share gains"}],
                       "summary_en": "Leadership loss weighs on the stock.", "summary_zh": "失去领导人，股价承压。"},
            "cited": [{"id": "A1", "title": "CEO resigns", "url": "https://example.com/1"}],
            "storylines": [{"id": "L1", "reason": "leadership+escalation", "n24": 26, "head": "CEO resigns"}]}


class NewsLinkageTest(unittest.TestCase):
    def test_newest_valid_judgment_per_symbol_within_the_window(self):
        db = {"news_linkage_judgments": _Coll([
            judgment("NVDA", 2, -2), judgment("NVDA", 20, -1),           # the newer one wins
            judgment("MSFT", 1, 1, status="invalid"),                    # invalid answers never show
            judgment("META", news_linkage.SHOW_HOURS + 1, -1),           # too old
        ])}
        out = news_linkage.latest(now=NOW, db=db)
        self.assertEqual(set(out["hints"]), {"NVDA"})
        h = out["hints"]["NVDA"]
        self.assertEqual(h["direction"], -2)
        self.assertEqual(h["label"], {"en": "News: Strongly bearish", "zh": "新闻：强利空"})
        self.assertEqual(h["detail"]["zh"], "把握中 · 尚未反映")
        self.assertEqual(h["storylines"][0]["reason"]["en"], "a leadership event, escalating")
        self.assertEqual(h["others"][0]["label"]["zh"], "利好")
        self.assertEqual(h["sources"], [{"title": "CEO resigns", "url": "https://example.com/1"}])
        self.assertIn("hint", out["labels"])
        self.assertEqual(out["track"]["h5"]["calls"], 0)
        self.assertIn("还没有可打分的判断", out["track"]["text"]["zh"])

    def test_scores_after_the_call_and_the_record(self):
        a, b, c = judgment("NVDA", 30, -1), judgment("AMD", 200, 1), judgment("MSFT", 300, 0)
        a["score"] = {"h1": {"excess": -2.5, "right": True}, "complete": False}
        b["score"] = {"h1": {"excess": -8.4, "right": False}, "h5": {"excess": -6.0, "right": False}}
        c["score"] = {"h1": {"excess": 0.5, "right": None}, "h5": {"excess": 1.0, "right": None}}
        out = news_linkage.latest(now=NOW, db={"news_linkage_judgments": _Coll([a, b, c])})
        self.assertEqual(out["hints"]["NVDA"]["after"], [{"h": 1, "excess": -2.5, "right": True}])
        t = out["track"]
        self.assertEqual((t["h1"]["calls"], t["h1"]["right"]), (2, 1))          # MSFT's neutral call is not counted
        self.assertEqual(t["h5"]["mean_called_excess"], -6.0)
        self.assertIn("1 day: 1 of 2 right", t["text"]["en"])
        self.assertIn("Too few calls", t["text"]["en"])


if __name__ == "__main__":
    unittest.main()
