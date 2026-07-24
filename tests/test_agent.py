"""Unit tests for the ReAct agent loop guardrails (no LLM / no mongo needed)."""

import json
import unittest
from unittest.mock import patch

import agent


def _tc(name: str, args: dict, call_id: str) -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class AgentLoopTests(unittest.TestCase):

    def test_final_answer_without_tools(self):
        """If the model answers directly, loop exits on step 1."""
        with patch.object(agent, "_chat", return_value={"content": "AAPL looks neutral."}):
            r = agent.run_research_agent("q")
        self.assertEqual(r["steps"], 1)
        self.assertEqual(r["answer"], "AAPL looks neutral.")
        self.assertEqual(r["trace"], [])

    def test_duplicate_tool_calls_in_one_step_execute_once(self):
        """27 identical tool_calls in one step -> tool runs once, all ids answered."""
        calls = {"n": 0}

        def fake_tool(args):
            calls["n"] += 1
            return json.dumps({"ok": True})

        dup = [_tc("get_features", {"symbol": "AAPL"}, f"id{i}") for i in range(27)]
        replies = iter([
            {"content": "", "tool_calls": dup},
            {"content": "done"},
        ])
        with patch.object(agent, "_chat", side_effect=lambda *a, **k: next(replies)), \
             patch.dict(agent.TOOL_IMPL, {"get_features": fake_tool}):
            r = agent.run_research_agent("q")
        self.assertEqual(calls["n"], 1)          # executed once
        self.assertEqual(len(r["trace"]), 1)     # traced once
        self.assertEqual(r["answer"], "done")

    def test_cross_step_repeat_uses_cache_with_nudge(self):
        """Same (tool,args) on a later step -> cached obs + stop-looping note, no re-execution."""
        calls = {"n": 0}

        def fake_tool(args):
            calls["n"] += 1
            return json.dumps({"ok": True})

        replies = iter([
            {"content": "", "tool_calls": [_tc("get_features", {"symbol": "AAPL"}, "a")]},
            {"content": "", "tool_calls": [_tc("get_features", {"symbol": "AAPL"}, "b")]},
            {"content": "final"},
        ])
        captured_messages = []

        def fake_chat(messages, use_tools=True):
            captured_messages[:] = messages
            return next(replies)

        with patch.object(agent, "_chat", side_effect=fake_chat), \
             patch.dict(agent.TOOL_IMPL, {"get_features": fake_tool}):
            r = agent.run_research_agent("q")
        self.assertEqual(calls["n"], 1)
        nudges = [m for m in captured_messages
                  if m.get("role") == "tool" and "do not call this tool again" in m.get("content", "")]
        self.assertTrue(nudges, "cached repeat should carry a stop-looping note")
        self.assertEqual(r["answer"], "final")

    def test_max_steps_forces_synthesis(self):
        """Model that never stops calling tools gets cut off and forced to answer."""
        def fake_chat(messages, use_tools=True):
            if not use_tools:
                return {"content": "forced summary"}
            return {"content": "", "tool_calls": [_tc("get_features", {"symbol": "AAPL"}, "x")]}

        with patch.object(agent, "_chat", side_effect=fake_chat), \
             patch.dict(agent.TOOL_IMPL, {"get_features": lambda a: "{}"}):
            r = agent.run_research_agent("q", max_steps=3)
        self.assertEqual(r["steps"], 3)
        self.assertEqual(r.get("note"), "hit max_steps")
        self.assertEqual(r["answer"], "forced summary")

    def test_reasoning_content_fallback(self):
        """Thinking models with empty content fall back to reasoning_content."""
        with patch.object(agent, "_chat",
                          return_value={"content": "", "reasoning_content": "from thinking"}):
            r = agent.run_research_agent("q")
        self.assertEqual(r["answer"], "from thinking")

    def test_unknown_tool_returns_error_observation(self):
        replies = iter([
            {"content": "", "tool_calls": [_tc("no_such_tool", {}, "z")]},
            {"content": "ok"},
        ])
        with patch.object(agent, "_chat", side_effect=lambda *a, **k: next(replies)):
            r = agent.run_research_agent("q")
        self.assertIn("unknown tool", r["trace"][0]["observation"])


if __name__ == "__main__":
    unittest.main()
