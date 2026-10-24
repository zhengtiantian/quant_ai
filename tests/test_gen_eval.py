"""R.11 pure logic: sentence splitting with citations, number grounding, refusal detection,
per-answer metrics, kappa and the paired bootstrap. No model, no database."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import gen_eval as ge  # noqa: E402


class SplitTest(unittest.TestCase):
    def test_sentences_keep_their_citations(self):
        a = ("Lam Research guided below expectations, sending shares down 7.4% after hours [1][3]. "
             "U.S. chip equipment peers also fell [2]. The excerpts do not mention Tokyo Electron.")
        s = ge.split_sentences(a)
        self.assertEqual(len(s), 3)
        self.assertEqual(ge.sentence_cites(s[0]), [1, 3])
        self.assertIn("U.S.", s[1])
        self.assertEqual(ge.sentence_cites(s[2]), [])

    def test_citation_after_the_period_stays_with_its_sentence(self):
        s = ge.split_sentences("Revenue rose 12%. [2] Margins fell [4].")
        self.assertEqual([ge.sentence_cites(x) for x in s], [[2], [4]])

    def test_prose_citations_count_and_are_not_figures(self):
        s = "Excerpt 6 says carmakers cut output by 12%, and excerpts 2 and 5 agree."
        self.assertEqual(ge.sentence_cites(s), [2, 5, 6])
        self.assertEqual(ge.numbers_in(s), [(12.0, 0)])

    def test_bullets(self):
        s = ge.split_sentences("- Nvidia rose [1].\n- AMD fell [2].")
        self.assertEqual(len(s), 2)


class NumbersTest(unittest.TestCase):
    def test_rounding_at_the_answers_precision(self):
        self.assertTrue(ge.number_supported(7, 0, [7.4]))
        self.assertFalse(ge.number_supported(7, 0, [7.6]))
        self.assertTrue(ge.number_supported(4.5, 1, [4.52]))
        self.assertFalse(ge.number_supported(4.5, 1, [4.62]))

    def test_years_and_citation_markers_are_not_numbers(self):
        self.assertEqual(ge.numbers_in("In 2026 shares fell 3% [12]."), [(3.0, 0)])
        self.assertEqual(ge.numbers_in("$1,250.5 million"), [(1250.5, 1)])

    def test_grounding_against_cited_excerpts(self):
        ok, total = ge.numeric_grounding("Shares fell 7% to $95 [1].", ["Shares dropped 7.2%, closing at $95.10"])
        self.assertEqual((ok, total), (2, 2))
        self.assertEqual(ge.numeric_grounding("Shares fell 9% [1].", ["dropped 7.2%"]), (0, 1))
        self.assertEqual(ge.numeric_grounding("No numbers here.", []), (0, 0))


class RefusalTest(unittest.TestCase):
    def test_refusal_detection(self):
        self.assertTrue(ge.is_refusal("The excerpts do not mention Apple's 2031 event."))
        self.assertTrue(ge.is_refusal("anything", refused_flag=True))
        self.assertFalse(ge.is_refusal("Apple launched a phone [1]. The excerpts do not mention prices."))


class MetricsTest(unittest.TestCase):
    def sents(self):
        return [{"_key": "a", "cites": [1], "numbers_ok": 1, "numbers_total": 1},
                {"_key": "b", "cites": [2, 3], "numbers_ok": 0, "numbers_total": 1},
                {"_key": "c", "cites": [], "numbers_ok": 0, "numbers_total": 0},
                {"_key": "m", "cites": [], "numbers_ok": 0, "numbers_total": 0}]

    def test_answer_metrics(self):
        labels = {"a": {"kind": "claim", "verdict": "supported", "supported_by": [1]},
                  "b": {"kind": "claim", "verdict": "partial", "supported_by": [3]},
                  "c": {"kind": "claim", "verdict": "unsupported", "supported_by": []},
                  "m": {"kind": "meta", "verdict": "supported", "supported_by": []}}
        m = ge.answer_metrics(self.sents(), labels)
        self.assertEqual(m["claims"], 3)
        self.assertAlmostEqual(m["faithfulness"], (1 + 0.5 + 0) / 3)
        self.assertAlmostEqual(m["unsupported_rate"], 1 / 3)
        self.assertAlmostEqual(m["citation_coverage"], 2 / 3)
        self.assertAlmostEqual(m["citation_precision"], 2 / 3)       # [1] good, [2] bad, [3] good
        self.assertAlmostEqual(m["correctly_cited"], 1.0)            # both cited sentences have a good cite
        self.assertEqual((m["numbers_ok"], m["numbers_total"]), (1, 2))   # claims only

    def test_parse_labels(self):
        lab = ge.parse_sentence_label('ok {"kind":"claim","verdict":"Supported","supported_by":["2", 5],"why":"x"}')
        self.assertEqual((lab["verdict"], lab["supported_by"]), ("supported", [2, 5]))
        self.assertIn("error", ge.parse_sentence_label("nope"))
        self.assertEqual(ge.parse_sentence_label('{"kind":"meta","verdict":"unsupported"}')["verdict"], "supported")
        self.assertEqual(ge.parse_relevance('{"grade": 2, "why": "y"}')["grade"], 2)


class StatsTest(unittest.TestCase):
    def test_kappa(self):
        self.assertAlmostEqual(ge.cohen_kappa([("a", "a"), ("b", "b")]), 1.0)
        self.assertAlmostEqual(ge.cohen_kappa([("a", "b"), ("b", "a")]), -1.0)

    def test_paired_diff(self):
        a = {"q1": 0.5, "q2": 0.5, "q3": None}
        b = {"q1": 1.0, "q2": 1.0, "q3": 1.0}
        m, lo, hi, n = ge.paired_diff_ci(a, b)
        self.assertEqual((round(m, 3), n), (0.5, 2))


class QuestionsTest(unittest.TestCase):
    def test_questions_include_the_unanswerable_probes(self):
        qs = ge.load_questions()
        self.assertEqual(sum(q["answerable"] is False for q in qs), len(ge.UNANSWERABLE))
        self.assertGreaterEqual(len(qs), 60 + 8)
        temporal = [q for q in qs if q["since"]]
        self.assertEqual(ge._date_int(temporal[0]["since"]), int(str(temporal[0]["since"]).replace("-", "")))


class DeclineTest(unittest.TestCase):
    def test_an_opening_decline_counts_even_with_context_after_it(self):
        u01 = ("The provided news excerpts do not contain any information regarding an Apple September 2031 "
               "product event. The most recent excerpt is from September 2023 [6], which discusses the iPhone 15.")
        self.assertTrue(ge.declines(u01))
        self.assertFalse(ge.is_refusal(u01))                       # the strict all-sentences rule
        self.assertTrue(ge.declines("The provided excerpts do not state what his favourite restaurant is. "
                                    "They mention Evvia in Palo Alto [3]."))
        self.assertTrue(ge.declines("Based on the provided excerpts, there is no specific share price predicted for 2030."))
        self.assertFalse(ge.declines("Tesla cut prices in China [1]. The excerpts do not mention Europe."))


class EvidenceTest(unittest.TestCase):
    EX = {"1": "symbol: TSLA | date: 20240421\nTesla cut prices of its \u201cModel Y\u201d in China\u2014by up to 20%.",
          "2": "Ford cut output because of the chip crunch."}

    def test_quote_must_be_in_the_named_excerpt_case_and_typography_aside(self):
        self.assertTrue(ge.quote_found('tesla cut prices of its "Model Y"  in China-by up to 20%', self.EX["1"]))
        self.assertTrue(ge.quote_found("date: 20240421", self.EX["1"]))
        self.assertFalse(ge.quote_found("Tesla cut prices by 30%", self.EX["1"]))     # not what it says
        self.assertFalse(ge.quote_found("Ford cut output", self.EX["1"]))             # wrong excerpt
        self.assertFalse(ge.quote_found("", self.EX["1"]))

    def test_evidence_verdict_counts_missing_and_invented_quotes(self):
        facts = [{"fact": "Tesla cut prices", "excerpt": 1, "quote": "Tesla cut prices"},
                 {"fact": "Ford output", "excerpt": 2, "quote": "Ford cut output because of the chip crunch"},
                 {"fact": "GM", "excerpt": None, "quote": None},
                 {"fact": "Up to 30%", "excerpt": 1, "quote": "by up to 30%"}]
        v = ge.evidence_verdict(facts, self.EX)
        self.assertEqual((v["verified"], v["missing"], v["bad_quote"], v["all_verified"]), (2, 1, 1, False))
        self.assertTrue(ge.evidence_verdict(facts[:2], self.EX)["all_verified"])
        overall = [{"fact": "no excerpt mentions a glut", "type": "overall", "excerpt": None, "quote": None}]
        v = ge.evidence_verdict(facts[:1] + overall, self.EX)
        self.assertEqual((v["facts"], v["overall"], v["missing"], v["all_verified"]), (1, 1, 0, True))


class ClaudeJudgePromptTest(unittest.TestCase):
    def test_answer_prompt_carries_the_same_rules_as_the_sentence_judge(self):
        self.assertIn('"partial"', ge._SENT_RULES)
        self.assertIn("never against what you know yourself", ge._SENT_RULES)
        self.assertNotIn("Reply with JSON", ge._SENT_RULES)
        self.assertTrue(ge._REL_RULES.startswith("grade 2"))
        prompt = ge.ANSWER_PROMPT.format(excerpts="<excerpt id=\"1\">x</excerpt>", question="q?",
                                         sentences='0. "a"\n1. "b"', rules=ge._SENT_RULES, rel_rules=ge._REL_RULES)
        self.assertIn('{"item": "answer", "grade": 0|1|2', prompt)
        self.assertEqual(prompt.count("<excerpt"), 1)             # excerpts once per answer


if __name__ == "__main__":
    unittest.main()
