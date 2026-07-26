"""End-to-end MCP protocol test for mcp_server.py.

Spawns the server over stdio and drives it exactly as its clients do (Claude
Desktop, and quant_ai's own agent via mcp_client) — handshake, tools/list,
tools/call — rather than just importing it.

It needs quant_api or mongo reachable, so it is skipped by default:

    RUN_MCP_E2E=1 .venv/bin/python -m unittest tests.test_mcp_server -v
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TOOLS = {
    "get_news_sentiment",
    "get_stock_features",
    "get_latest_signals",
    "get_positions",
    "get_my_holdings",
    "get_my_transactions",
    "get_performance",
    "list_symbols",
}

# (tool, arguments, substring the response must contain)
CALL_CHECKS = [
    ("get_news_sentiment", {"symbol": "AAPL", "days": 30}, "avgsentiment"),
    ("get_stock_features", {"symbol": "AAPL"}, "latestfeatures"),
    ("get_latest_signals", {"limit": 3}, "compositescore"),
    ("list_symbols", {}, "aapl"),
    ("get_positions", {}, "symbol"),
    ("get_performance", {}, "sharpe"),
    # Answers with totals even when the user has recorded no trades yet.
    ("get_my_holdings", {}, "totals"),
    ("get_my_transactions", {}, "["),
]


async def _drive_server():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from mcp.client.stdio import get_default_environment

    # stdio_client passes only a safe subset of the environment to the subprocess, so
    # QUANT_API and friends never reach the server unless forwarded explicitly. Without
    # this the test can only ever exercise the default endpoint.
    env = get_default_environment()
    for key in ("QUANT_API", "LOCAL_MONGO_URI", "FEATURE_DB_NAME", "QUANT_MCP_TIMEOUT"):
        if os.getenv(key):
            env[key] = os.environ[key]

    params = StdioServerParameters(
        command=sys.executable, args=[str(PROJECT_ROOT / "mcp_server.py")], env=env
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            results = {}
            for tool, args, _ in CALL_CHECKS:
                out = await session.call_tool(tool, args)
                results[tool] = "".join(
                    c.text for c in out.content if getattr(c, "type", "") == "text"
                )
            return names, results


@unittest.skipUnless(
    os.getenv("RUN_MCP_E2E") == "1",
    "needs a live quant_api/mongo; set RUN_MCP_E2E=1 to run",
)
class MCPServerEndToEndTests(unittest.TestCase):
    """Talks to the real server over the real protocol."""

    @classmethod
    def setUpClass(cls):
        cls.names, cls.results = asyncio.run(_drive_server())

    def test_exposes_expected_tools(self):
        self.assertEqual(EXPECTED_TOOLS, self.names)

    def test_every_tool_returns_expected_data(self):
        for tool, _, expected in CALL_CHECKS:
            with self.subTest(tool=tool):
                self.assertIn(expected, self.results[tool].lower())


if __name__ == "__main__":
    unittest.main()
