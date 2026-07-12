# quant_ai

RAG + Local LLM assistant service for the AI-Driven Equity Signal Platform.

## Overview

FastAPI service that answers natural language questions about stocks by combining:
1. **RAG** — cosine similarity search over embedded knowledge documents
2. **Direct LLM call** — injects retrieved context into a single prompt sent to LM Studio

No agent loop. No LangGraph. Each request is one retrieval + one LLM call.

```
User question
     │
     ▼
Embed query (nomic-embed-text via LM Studio)
     │
     ▼
Cosine similarity search over knowledge store
(fallback: keyword search if embedding model unavailable)
     │
     ▼
Build prompt: [system] + [retrieved docs] + [live data] + [question]
     │
     ▼
LM Studio (qwen3.5-9b-mlx) → Anthropic → OpenAI  (fallback chain)
     │
     ▼
Structured JSON response → quant_ui
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Natural language Q&A with RAG context |
| `POST` | `/api/generate-spec` | Generate signal spec JSON from description |
| `GET` | `/api/health` | Service health + model connectivity |
| `GET` | `/docs` | FastAPI auto-generated docs |

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
| `QUANT_API` | `http://localhost:18081` | quant_api for live signal data |
| `KNOWLEDGE_PATHS` | `./knowledge` | Comma-separated knowledge dirs |
| `PORT` | `18000` | Listen port |

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
