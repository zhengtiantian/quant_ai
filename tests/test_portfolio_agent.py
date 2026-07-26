"""Unit tests for the portfolio review agent (F.17).

The LLM and the MCP tool surface are both mocked, so these need neither LM Studio nor a
database. What they cover is the part that has to hold when the model misbehaves: the
three gates, and the portfolio checks that deliberately never reach a model.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import portfolio_agent as pa  # noqa: E402


def reply(text):
    """Shape _chat returns and _content reads."""
    return {"content": text}


def good_verdict(verdict="agree", value=0.2803, confidence=0.8):
    return json.dumps({
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": "Sector-wide move, company sentiment still positive.",
        "evidence": [{"tool": "get_feature_history", "field": "last", "value": value}],
    })


DECISION = {"symbol": "AMD", "action": "exit", "reason": "stop_loss",
            "entryDate": "2026-07-10", "exitDate": "2026-07-17",
            "daysHeld": 5, "realisedReturnPct": -11.14}

EVIDENCE = [{"tool": "get_feature_history",
             "result": {"summary": {"avg_sentiment_5d": {"last": 0.2803, "change": 0.06}}}}]


class SchemaGateTests(unittest.TestCase):

    def test_clean_json_passes(self):
        with patch.object(pa, "_chat", return_value=reply(good_verdict())):
            out = pa._review_one(DECISION, EVIDENCE, pa._observed_values(
                e["result"] for e in EVIDENCE))
        self.assertEqual("agree", out["verdict"])
        self.assertEqual("AMD", out["symbol"])
        self.assertEqual(1, len(out["evidence"]))

    def test_json_wrapped_in_prose_is_recovered(self):
        """Small models narrate before answering; the JSON is still usable."""
        wrapped = "Here is my review:\n\n" + good_verdict() + "\n\nHope that helps."
        with patch.object(pa, "_chat", return_value=reply(wrapped)):
            out = pa._review_one(DECISION, EVIDENCE, pa._observed_values(
                e["result"] for e in EVIDENCE))
        self.assertEqual("agree", out["verdict"])

    def test_one_repair_round_trip_then_success(self):
        replies = [reply("I think the exit was fine."), reply(good_verdict())]
        with patch.object(pa, "_chat", side_effect=replies) as chat:
            out = pa._review_one(DECISION, EVIDENCE, pa._observed_values(
                e["result"] for e in EVIDENCE))
        self.assertEqual(2, chat.call_count, "should retry exactly once")
        self.assertEqual("agree", out["verdict"])

    def test_two_failures_is_skipped_not_emitted_malformed(self):
        with patch.object(pa, "_chat", return_value=reply("still not json")):
            out = pa._review_one(DECISION, EVIDENCE, set())
        self.assertEqual("skipped", out["verdict"])
        self.assertTrue(any("schema gate" in n for n in out["notes"]))

    def test_unknown_verdict_is_rejected(self):
        bad = json.dumps({"verdict": "maybe", "confidence": 0.5,
                          "reasoning": "unsure", "evidence": []})
        with patch.object(pa, "_chat", return_value=reply(bad)):
            out = pa._review_one(DECISION, EVIDENCE, set())
        self.assertEqual("skipped", out["verdict"])

    def test_confidence_out_of_range_is_rejected(self):
        bad = good_verdict(confidence=7)
        with patch.object(pa, "_chat", return_value=reply(bad)):
            out = pa._review_one(DECISION, EVIDENCE, set())
        self.assertEqual("skipped", out["verdict"])

    def test_model_exception_becomes_a_skipped_review(self):
        with patch.object(pa, "_chat", side_effect=RuntimeError("LM Studio down")):
            out = pa._review_one(DECISION, EVIDENCE, set())
        self.assertEqual("skipped", out["verdict"])
        self.assertTrue(any("model call failed" in n for n in out["notes"]))


class GroundingGateTests(unittest.TestCase):

    def setUp(self):
        self.observed = pa._observed_values(e["result"] for e in EVIDENCE)

    def test_cited_value_present_in_observations_is_kept(self):
        with patch.object(pa, "_chat", return_value=reply(good_verdict(value=0.2803))):
            out = pa._review_one(DECISION, EVIDENCE, self.observed)
        self.assertEqual(1, len(out["evidence"]))
        self.assertEqual([], out["notes"])

    def test_rounded_citation_still_counts_as_grounded(self):
        """0.28 for an observed 0.2803 is quoting, not inventing."""
        with patch.object(pa, "_chat", return_value=reply(good_verdict(value=0.28))):
            out = pa._review_one(DECISION, EVIDENCE, self.observed)
        self.assertEqual(1, len(out["evidence"]))

    def test_invented_number_is_dropped(self):
        with patch.object(pa, "_chat", return_value=reply(good_verdict(value=0.9999))):
            out = pa._review_one(DECISION, EVIDENCE, self.observed)
        self.assertEqual([], out["evidence"])
        self.assertTrue(any("unverifiable" in n for n in out["notes"]))

    def test_flag_with_no_verifiable_citation_is_demoted(self):
        """A flag is a call to action; one resting on nothing checkable must not stand."""
        with patch.object(pa, "_chat",
                          return_value=reply(good_verdict("flag", value=0.9999))):
            out = pa._review_one(DECISION, EVIDENCE, self.observed)
        self.assertEqual("agree", out["verdict"])
        self.assertTrue(any("demoted" in n for n in out["notes"]))

    def test_flag_with_a_verifiable_citation_stands(self):
        with patch.object(pa, "_chat",
                          return_value=reply(good_verdict("flag", value=0.2803))):
            out = pa._review_one(DECISION, EVIDENCE, self.observed)
        self.assertEqual("flag", out["verdict"])

    def test_headline_substring_is_grounded_but_a_short_word_is_not(self):
        observed = pa._observed_values([{"title": "Chip selloff drags the whole sector"}])
        self.assertTrue(pa._is_grounded("Chip selloff drags", observed))
        self.assertFalse(pa._is_grounded("chip", observed), "too short to be evidence")


class PortfolioCheckTests(unittest.TestCase):
    """These never call a model: they are comparisons, and code does not get them wrong."""

    HOLDINGS = {
        "holdings": [
            {"symbol": "AMD", "weightPct": 41.9},
            {"symbol": "AAPL", "weightPct": 8.0},
        ],
        "totals": {"holdingsValue": 24874.24, "totalValue": 49874.24},
    }

    def test_position_over_the_weight_limit_is_flagged(self):
        findings = pa._portfolio_checks(self.HOLDINGS, [])
        weights = [f for f in findings if f["check"] == "position_weight"]
        self.assertEqual(2, len(weights), "both exceed the 5% limit")
        self.assertEqual("high", weights[0]["severity"], "41.9% is past the concentration bar")
        self.assertEqual("AMD", weights[0]["symbol"])

    def test_holdings_within_the_limit_produce_no_weight_finding(self):
        small = {"holdings": [{"symbol": "AAPL", "weightPct": 3.0}],
                 "totals": {"holdingsValue": 1000, "totalValue": 50000}}
        findings = pa._portfolio_checks(small, [])
        self.assertEqual([], [f for f in findings if f["check"] == "position_weight"])

    # 85% invested. The class-level HOLDINGS is only 49.9% invested, which is below the
    # 60% bar on purpose — that portfolio should not trip this check.
    HEAVILY_INVESTED = {
        "holdings": [{"symbol": "AMD", "weightPct": 85.0}],
        "totals": {"holdingsValue": 42500, "totalValue": 50000},
    }

    def test_risk_off_regime_with_high_exposure_is_flagged(self):
        signals = [{"symbol": "MA", "regimeLabel": "RISK_OFF"}]
        findings = pa._portfolio_checks(self.HEAVILY_INVESTED, signals)
        self.assertTrue(any(f["check"] == "regime_exposure" for f in findings))

    def test_moderate_exposure_in_risk_off_is_not_flagged(self):
        signals = [{"symbol": "MA", "regimeLabel": "RISK_OFF"}]
        findings = pa._portfolio_checks(self.HOLDINGS, signals)  # 49.9% invested
        self.assertFalse(any(f["check"] == "regime_exposure" for f in findings))

    def test_risk_on_regime_does_not_flag_exposure(self):
        signals = [{"symbol": "MA", "regimeLabel": "RISK_ON"}]
        findings = pa._portfolio_checks(self.HEAVILY_INVESTED, signals)
        self.assertFalse(any(f["check"] == "regime_exposure" for f in findings))

    def test_no_overlap_with_ranked_signals_is_flagged(self):
        signals = [{"symbol": "MA"}, {"symbol": "LRCX"}]
        findings = pa._portfolio_checks(self.HOLDINGS, signals)
        self.assertTrue(any(f["check"] == "signal_alignment" for f in findings))

    def test_overlap_with_ranked_signals_is_not_flagged(self):
        signals = [{"symbol": "AMD"}, {"symbol": "LRCX"}]
        findings = pa._portfolio_checks(self.HOLDINGS, signals)
        self.assertFalse(any(f["check"] == "signal_alignment" for f in findings))

    def test_empty_portfolio_yields_no_findings(self):
        self.assertEqual([], pa._portfolio_checks({"holdings": [], "totals": {}}, []))
        self.assertEqual([], pa._portfolio_checks(None, []))


class DecisionSelectionTests(unittest.TestCase):

    POSITIONS = [
        {"symbol": "AMD", "status": "closed", "exit_date": "2026-07-17",
         "exit_trigger": "stop_loss", "exit_return": -0.1114, "days_held": 5},
        {"symbol": "NFLX", "status": "closed", "exit_date": "2026-07-16",
         "exit_trigger": "score_below_exit", "exit_return": 0.007, "days_held": 3},
        {"symbol": "MA", "status": "open", "entry_date": "2026-07-20"},
    ]

    def test_only_closed_positions_are_reviewed_newest_first(self):
        out = pa._recent_exits(self.POSITIONS, 5)
        self.assertEqual(["AMD", "NFLX"], [d["symbol"] for d in out])
        self.assertEqual("stop_loss", out[0]["reason"])
        self.assertEqual(-11.14, out[0]["realisedReturnPct"])

    def test_limit_is_respected(self):
        self.assertEqual(1, len(pa._recent_exits(self.POSITIONS, 1)))

    def test_camel_case_field_names_are_also_accepted(self):
        """The API serves camelCase; mongo stores snake_case. Both must parse."""
        camel = [{"symbol": "AMD", "status": "closed", "exitDate": "2026-07-17",
                  "exitTrigger": "stop_loss", "exitReturn": -0.1114, "daysHeld": 5}]
        out = pa._recent_exits(camel, 5)
        self.assertEqual("stop_loss", out[0]["reason"])
        self.assertEqual(-11.14, out[0]["realisedReturnPct"])

    def test_non_list_input_is_survivable(self):
        self.assertEqual([], pa._recent_exits(None, 5))
        self.assertEqual([], pa._recent_exits({"error": "unavailable"}, 5))


if __name__ == "__main__":
    unittest.main()
