#!/bin/bash
cd "$(dirname "$0")"
export LOCAL_MODEL_NAME="qwen3.5-9b-mlx"
export EMBED_MODEL="text-embedding-nomic-embed-text-v1.5"
export LM_STUDIO_URL="http://127.0.0.1:1234/v1"
export QUANT_API="http://localhost:18081"
export KNOWLEDGE_PATHS="$(pwd)/knowledge"
export PORT=18000

# R.5 phase 1b — service credentials for quant_api, kept out of this file on purpose.
# .env is gitignored; without it the service simply calls quant_api anonymously, which
# still works while the API permits all requests.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

exec .venv/bin/python3 main.py
