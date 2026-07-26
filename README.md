# quant_ai

RAG + ReAct research agent service for the AI-Driven Equity Signal Platform.

## Overview

FastAPI service with two modes:

1. **RAG Q&A** — cosine similarity search over embedded knowledge documents,
   retrieved context injected into a single LLM prompt (one retrieval + one call)
2. **ReAct research agent** — a hand-written tool-calling loop: the LLM
   autonomously decides which read-only data tools to call
   (Thought → Action → Observation), then writes a research note grounded in
   the numbers it retrieved

```
Research question ("Compare NVDA and AMD news sentiment")
     │
     ▼
ReAct loop (agent.py, no framework)
     │  LLM (qwen3.5-9b, OpenAI-compatible tools API)
     │    │
     │    │  tools are discovered at runtime, not implemented here:
     │    └─ MCP client (mcp_client.py) ──stdio──> mcp_server.py
     │         get_news_sentiment · search_news · get_stock_features
     │         get_latest_signals
     │         get_positions · get_my_holdings · get_my_transactions
     │         get_performance · list_symbols
     │              └─ quant_api (mongo fallback inside mcp_server.py)
     │
     │    … repeats until the model answers or hits max_steps
     ▼
Grounded research note + full tool-call trace
(SSE stream: each tool call pushed live to the UI)
```

### Agent guardrails (controlled agency)

- tools are **read-only** — the MCP server exposes no mutating operation
- duplicate tool_calls within one step execute once
- repeated (tool, args) across steps return a cached observation plus a
  stop-looping nudge (small local models retry empty tools aggressively)
- hard `max_steps` cap with a forced final synthesis
- thinking-model fallback: empty `content` falls back to `reasoning_content`

Covered by 7 unit tests (`tests/test_agent.py`) that mock the LLM and the MCP
tool surface, so they need neither LM Studio nor a database.

### Portfolio review agent (F.17)

A second opinion on the rule engine, not a replacement. `track_positions.py` already
decides entries, exits and stops deterministically; a local 9B model would be worse and
non-reproducible at that job. This agent reviews those decisions instead, covering the
two things a per-position rule structurally cannot do — reading *why* a price moved, and
looking at the portfolio as a whole.

The division of labour is the design:

- **Code** gathers context, picks what to review, fetches evidence, and runs every
  portfolio-level check. Concentration and weight limits are arithmetic; asking a model
  to do arithmetic that a comparison operator cannot get wrong is a bad trade.
- **The model** handles only what needs language: given these articles and this feature
  trend, was that exit justified?
- **Code** then validates what came back, through three gates that trust nothing:

| Gate | Enforces | On failure |
|---|---|---|
| Schema | valid JSON, known verdict, confidence in 0–1 | one repair round-trip, then the review is dropped rather than emitted malformed |
| **Grounding** | every cited `(tool, field, value)` appears in a real observation from this run | citation dropped; a `flag` resting on nothing verifiable is demoted to `agree` |
| Business rules | symbol in the universe, position actually exists | rejected |

Grounding is the point: not asking the model to avoid inventing numbers, but making
invented numbers non-viable. Numeric citations match with tolerance — a model echoing
0.2803 as 0.28 is quoting, not fabricating, and failing that would teach it to stop
citing altogether.

Covered by 25 unit tests (`tests/test_portfolio_agent.py`) with the LLM and MCP mocked.

### Agent evaluation (F.22)

```bash
.venv/bin/python eval/run_eval.py --suite research
```

Whether a recommendation makes money cannot be scored without waiting for the market, and
claiming otherwise would be dishonest. What `eval/` does score are the failure modes that
actually keep agents out of production: schema validity, **grounding rate**, tool
recall/precision against the tools a case requires or forbids, run-to-run stability
(mean pairwise Jaccard over repeated identical runs), and cost in steps and seconds.

Stability matters more than it looks — an agent that answers differently every time cannot
be trusted even when each answer is defensible, and a demo run once cannot reveal it.

The unit tests mock the LLM, so they prove the loop is correct, not that the decisions
are. This closes that gap by driving the real model against fixed cases in
`eval/cases.yaml`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/agent/research` | ReAct agent run (blocking; returns answer + trace) |
| `POST` | `/api/agent/portfolio` | Portfolio review: rule decisions + deterministic portfolio checks |
| `POST` | `/api/agent/research/stream` | Same run as SSE: one event per tool call, then final |
| `POST` | `/api/ask` | Natural language Q&A with RAG context |
| `POST` | `/api/generate-script` | Generate a quant Python script from description |
| `GET` | `/health` | Service health + model connectivity |
| `GET` | `/docs` | FastAPI auto-generated docs |

## MCP Server

`mcp_server.py` exposes the platform over the Model Context Protocol (stdio), so
any MCP client can query it. It is not a service: each client spawns it as a
subprocess, so there is no port, container, or launchd entry to manage.

| Tool | Returns |
|---|---|
| `get_news_sentiment(symbol, days=90)` | Article count, average sentiment (-1..+1), model disagreement, recent headlines |
| `search_news(query, symbol, from_date, to_date, limit=20)` | The articles themselves — headline, date, excerpt, merged sentiment, model disagreement, event type, URL. Ranked by relevance (headline matches weighted 10x over body) |
| `get_stock_features(symbol)` | Latest engineered daily feature row |
| `get_latest_signals(limit=10)` | Ranked signals from the Ridge + LightGBM ensemble |
| `get_positions()` | Rule-generated paper positions with entry/current price and P&L |
| `get_my_holdings()` | The user's real portfolio — quantity, average cost, live price, unrealised/realised P&L, weight, plus totals and cash |
| `get_my_transactions(symbol="")` | The trade log behind those holdings, newest first |
| `get_performance()` | Backtest Sharpe, returns, hit rate, drawdown |
| `list_symbols()` | The covered universe |

`get_news_sentiment` and `search_news` are the aggregate and the evidence behind it: the
first says a symbol averaged +0.31 over 90 days, the second lets a client read the
articles that produced it. Backed by a weighted text index over 845K labeled articles.

`get_positions` and `get_my_holdings` answer different questions: the first returns the
synthetic positions the signal tracker opens mechanically from the daily top-5, the second
returns what the user actually owns.

Tools read through `quant_api` and fall back to mongo only when it is
unreachable. All are read-only — the server analyzes, it never trades.

Three clients use it today, none of which required a change to the server: this
service's own agent (above), Claude Code via `.mcp.json` in the platform root,
and Claude Desktop via
`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "quant": {
      "command": "/Users/xiz/Quant_trade/quant_ai/.venv/bin/python",
      "args": ["/Users/xiz/Quant_trade/quant_ai/mcp_server.py"],
      "env": { "QUANT_API": "http://localhost:18081" }
    }
  }
}
```

Absolute paths are required there — Claude Desktop launches from an arbitrary
working directory. Restart it after editing the config.

## Knowledge Documents

Stored in `knowledge/`:
- `quant_system_overview.md` — platform architecture and components
- `factor_reference.md` — all features, their sources and meanings
- `strategy_examples.md` — example prompts and expected responses
- `quant_sdk_quickref.md` — API quick reference

## Running

### As host process (recommended — LM Studio access)

```bash
# Load the launchd service
launchctl load ~/Library/LaunchAgents/com.quant.ai.plist

# Or run directly
bash run_host.sh
# or: python main.py
```

Runs on port 18000. Logs: `/tmp/quant-ai.log` / `/tmp/quant-ai-err.log`.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_MODEL_NAME` | `qwen3.5-9b-mlx` | LM Studio chat model |
| `EMBED_MODEL` | `text-embedding-nomic-embed-text-v1.5` | LM Studio embedding model |
| `LM_STUDIO_URL` | `http://127.0.0.1:1234/v1` | LM Studio base URL |
| `QUANT_API` | `http://localhost:18081` | quant_api for live signal data (RAG helpers) |
| `KNOWLEDGE_PATHS` | `./knowledge` | Comma-separated knowledge dirs |
| `PORT` | `18000` | Listen port |
| `QUANT_MCP_PYTHON` | current interpreter | Override only to run an out-of-tree MCP server |
| `QUANT_MCP_SERVER` | `./mcp_server.py` | Override only to run an out-of-tree MCP server |

### Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## LLM Fallback Chain

1. **LM Studio** (`LOCAL_MODEL_NAME`) — primary, fully local
2. **Anthropic** (`ANTHROPIC_API_KEY`) — if LM Studio unavailable
3. **OpenAI** (`OPENAI_API_KEY`) — final fallback

## Why Host Process (not Docker)

Under VPN, Docker's VPNKit TCP stack cannot reach the Mac host's `host.docker.internal`. Since LM Studio serves on `127.0.0.1:1234`, the container cannot call it. Running quant_ai directly on the host avoids this entirely.

## Docker (reference only)

A Docker image is published to `xiz001/quant_ai` via `.github/workflows/deploy.yml` for environments where the above networking issue does not apply.

```bash
docker build -f Dockerfile.runtime -t xiz001/quant_ai:local .
```
