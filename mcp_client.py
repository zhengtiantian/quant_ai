"""Synchronous bridge to the quant MCP server (stdio transport).

quant_ai used to reimplement each platform data tool locally, duplicating logic
that already lives in quant_mcp. This module makes quant_ai an MCP *client*
instead: tools are discovered from the server at runtime, so registering a new
tool there is enough for the agent to gain it — no change needed here.

The MCP SDK is async while the ReAct loop is synchronous, so a single session is
kept alive on a background event loop and exposed through blocking wrappers.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import threading
from typing import Any

QUANT_MCP_PYTHON = os.getenv(
    "QUANT_MCP_PYTHON", "/Users/xiz/Quant_trade/quant_mcp/.venv/bin/python"
)
QUANT_MCP_SERVER = os.getenv(
    "QUANT_MCP_SERVER", "/Users/xiz/Quant_trade/quant_mcp/server.py"
)
STARTUP_TIMEOUT = float(os.getenv("QUANT_MCP_STARTUP_TIMEOUT", "30"))
CALL_TIMEOUT = float(os.getenv("QUANT_MCP_CALL_TIMEOUT", "60"))


class MCPToolClient:
    """Owns one long-lived MCP session on a private event loop thread."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._tools: list[dict] = []
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Spawn the server and complete the handshake. Idempotent."""
        with self._lock:
            if self._thread is not None:
                self._ready.wait(STARTUP_TIMEOUT)
                if self._error:
                    raise RuntimeError(f"MCP session failed: {self._error}")
                return
            self._thread = threading.Thread(
                target=self._run, name="mcp-client", daemon=True
            )
            self._thread.start()

        if not self._ready.wait(STARTUP_TIMEOUT):
            raise RuntimeError(
                f"MCP server did not become ready within {STARTUP_TIMEOUT}s "
                f"({QUANT_MCP_PYTHON} {QUANT_MCP_SERVER})"
            )
        if self._error:
            raise RuntimeError(f"MCP session failed: {self._error}")

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as e:  # noqa: BLE001 - surfaced via self._error
            self._error = e
            self._ready.set()
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=QUANT_MCP_PYTHON, args=[QUANT_MCP_SERVER]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                self._tools = [
                    {
                        "name": t.name,
                        "description": (t.description or "").strip(),
                        "input_schema": t.inputSchema or {"type": "object", "properties": {}},
                    }
                    for t in listed.tools
                ]
                self._session = session
                self._ready.set()
                # Hold the session open until shutdown.
                while not self._stop.is_set():
                    await asyncio.sleep(0.2)

    def close(self) -> None:
        self._stop.set()

    # -- tool surface -------------------------------------------------------

    def list_tools(self) -> list[dict]:
        self.start()
        return list(self._tools)

    def openai_tools(self) -> list[dict]:
        """MCP tool defs rendered as OpenAI-compatible function tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in self.list_tools()
        ]

    def call_tool(self, name: str, args: dict) -> str:
        """Invoke a tool and return its text content (blocking)."""
        self.start()
        if self._loop is None or self._session is None:
            raise RuntimeError("MCP session is not running")
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, args), self._loop
        )
        result = future.result(timeout=CALL_TIMEOUT)
        parts = [
            block.text
            for block in result.content
            if getattr(block, "type", None) == "text"
        ]
        return "\n".join(parts) if parts else ""


_client: MCPToolClient | None = None


def get_client() -> MCPToolClient:
    """Process-wide singleton — one server subprocess, not one per request."""
    global _client
    if _client is None:
        _client = MCPToolClient()
    return _client


@atexit.register
def _shutdown() -> None:
    if _client is not None:
        _client.close()
