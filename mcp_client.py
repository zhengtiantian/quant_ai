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
import sys
import threading
from pathlib import Path
from typing import Any

# The server ships alongside this client, so it runs on the same interpreter
# and needs no configured path. Both stay overridable for an out-of-tree server.
QUANT_MCP_PYTHON = os.getenv("QUANT_MCP_PYTHON", sys.executable)
QUANT_MCP_SERVER = os.getenv(
    "QUANT_MCP_SERVER", str(Path(__file__).resolve().parent / "mcp_server.py")
)
STARTUP_TIMEOUT = float(os.getenv("QUANT_MCP_STARTUP_TIMEOUT", "30"))
CALL_TIMEOUT = float(os.getenv("QUANT_MCP_CALL_TIMEOUT", "60"))

FORWARDED_ENV_KEYS: tuple[str, ...] = (
    "QUANT_API",
    "MARKETS_URL",
    "LOCAL_MONGO_URI",
    "FEATURE_DB_NAME",
    "QUANT_MCP_TIMEOUT",
    "QDRANT_URL",
    "LM_STUDIO_URL",
    "NEWS_RERANK",
    "KEYCLOAK_URL",
    "KEYCLOAK_REALM",
    "QUANT_AI_CLIENT_ID",
    "QUANT_AI_CLIENT_SECRET",
    "QUANT_AI_TOKEN_TIMEOUT",
    "QUANT_MCP_ALLOWED_JUDGES",
)

# Eval-only / write tools are hidden from autonomous research agents by default.
EVAL_ONLY_TOOLS: frozenset[str] = frozenset({"get_eval_batch", "submit_judgment"})


def _build_subprocess_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build subprocess environment forwarding platform & auth configuration."""
    if base_env is None:
        from mcp.client.stdio import get_default_environment
        env = get_default_environment()
    else:
        env = dict(base_env)
    for key in FORWARDED_ENV_KEYS:
        value = os.getenv(key)
        if value:
            env[key] = value
    return env


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

    def _reset_locked(self) -> None:
        self._stop.set()
        self._thread = None
        self._loop = None
        self._session = None
        self._error = None
        self._ready.clear()
        self._stop.clear()

    def _is_dead_locked(self) -> bool:
        if self._thread is None:
            return False
        if not self._thread.is_alive():
            return True
        if self._error is not None:
            return True
        if self._loop is not None and self._loop.is_closed():
            return True
        return False

    def start(self) -> None:
        """Spawn the server and complete the handshake. Idempotent & self-healing."""
        with self._lock:
            if self._is_dead_locked():
                self._reset_locked()
            if self._thread is not None:
                thread_ready = self._ready
                err = self._error
                t = self._thread
            else:
                self._thread = threading.Thread(
                    target=self._run, name="mcp-client", daemon=True
                )
                self._thread.start()
                thread_ready = self._ready
                t = self._thread
                err = None

        if not thread_ready.wait(STARTUP_TIMEOUT):
            with self._lock:
                self._reset_locked()
            raise RuntimeError(
                f"MCP server did not become ready within {STARTUP_TIMEOUT}s "
                f"({QUANT_MCP_PYTHON} {QUANT_MCP_SERVER})"
            )
        if err or self._error or not t.is_alive():
            detail = err or self._error or "subprocess exited unexpectedly"
            with self._lock:
                self._reset_locked()
            raise RuntimeError(f"MCP session failed: {detail}")

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

        env = _build_subprocess_env()
        params = StdioServerParameters(
            command=QUANT_MCP_PYTHON, args=[QUANT_MCP_SERVER], env=env
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                tools_out = []
                for t in listed.tools:
                    ann = getattr(t, "annotations", None)
                    ro_hint = getattr(ann, "readOnlyHint", None) if ann is not None else None
                    tools_out.append({
                        "name": t.name,
                        "description": (t.description or "").strip(),
                        "input_schema": t.inputSchema or {"type": "object", "properties": {}},
                        "read_only": True if ro_hint is None else bool(ro_hint),
                    })
                self._tools = tools_out
                self._session = session
                self._ready.set()
                while not self._stop.is_set():
                    await asyncio.sleep(0.2)

    def close(self) -> None:
        self._stop.set()

    # -- tool surface -------------------------------------------------------

    def list_tools(self) -> list[dict]:
        self.start()
        return list(self._tools)

    def openai_tools(self, include_eval: bool = False) -> list[dict]:
        """MCP tool defs rendered as OpenAI-compatible function tools.

        Excludes eval/write tools by default so autonomous research agents stay strictly read-only.
        """
        tools = []
        for t in self.list_tools():
            if not include_eval:
                if t["name"] in EVAL_ONLY_TOOLS or t.get("read_only") is False:
                    continue
            tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            })
        return tools

    def _invoke_once(self, name: str, args: dict) -> str:
        self.start()
        if self._loop is None or self._session is None or self._loop.is_closed():
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

    def call_tool(self, name: str, args: dict) -> str:
        """Invoke a tool and return its text content (blocking), reconnecting once on dead pipe."""
        try:
            return self._invoke_once(name, args)
        except Exception:  # noqa: BLE001 - retry once after clean reset if session died
            with self._lock:
                self._reset_locked()
            return self._invoke_once(name, args)


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
