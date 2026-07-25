"""Unit tests for the ReAct agent loop guardrails.

The LLM and the MCP tool surface are both mocked, so these run with no LM Studio,
no MCP server subprocess, and no database.
"""

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

        def fake_exec(name, args):
            calls["n"] += 1
            return json.dumps({"ok": True})

        dup = [_tc("get_stock_features", {"symbol": "AAPL"}, f"id{i}") for i in range(27)]
        replies = iter([
            {"content": "", "tool_calls": dup},
            {"content": "done"},
        ])
        with patch.object(agent, "_chat", side_effect=lambda *a, **k: next(replies)), \
             patch.object(agent, "execute_tool", side_effect=fake_exec):
            r = agent.run_research_agent("q")
        self.assertEqual(calls["n"], 1)          # executed once
        self.assertEqual(len(r["trace"]), 1)     # traced once
        self.assertEqual(r["answer"], "done")

    def test_cross_step_repeat_uses_cache_with_nudge(self):
        """Same (tool,args) on a later step -> cached obs + stop-looping note, no re-execution."""
        calls = {"n": 0}

        def fake_exec(name, args):
            calls["n"] += 1
            return json.dumps({"ok": True})

        replies = iter([
            {"content": "", "tool_calls": [_tc("get_stock_features", {"symbol": "AAPL"}, "a")]},
            {"content": "", "tool_calls": [_tc("get_stock_features", {"symbol": "AAPL"}, "b")]},
            {"content": "final"},
        ])
        captured_messages = []

        def fake_chat(messages, use_tools=True):
            captured_messages[:] = messages
            return next(replies)

        with patch.object(agent, "_chat", side_effect=fake_chat), \
             patch.object(agent, "execute_tool", side_effect=fake_exec):
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
            return {"content": "", "tool_calls": [_tc("get_stock_features", {"symbol": "AAPL"}, "x")]}

        with patch.object(agent, "_chat", side_effect=fake_chat), \
             patch.object(agent, "execute_tool", return_value="{}"):
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


class ToolBridgeTests(unittest.TestCase):
    """execute_tool must turn any MCP failure into an observation, never an exception."""

    def test_tool_failure_becomes_error_observation(self):
        class Boom:
            def call_tool(self, name, args):
                raise RuntimeError("mcp server unreachable")

        with patch.object(agent.mcp_client, "get_client", return_value=Boom()):
            out = agent.execute_tool("get_positions", {})
        self.assertIn("error", out)
        self.assertIn("mcp server unreachable", out)

    def test_tool_specs_come_from_mcp(self):
        class Fake:
            def openai_tools(self):
                return [{"type": "function", "function": {"name": "get_positions"}}]

        with patch.object(agent.mcp_client, "get_client", return_value=Fake()):
            specs = agent.get_tool_specs()
        self.assertEqual(specs[0]["function"]["name"], "get_positions")


if __name__ == "__main__":
    unittest.main()
