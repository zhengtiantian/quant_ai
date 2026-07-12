#!/bin/bash
cd "$(dirname "$0")"
export LOCAL_MODEL_NAME="qwen3.5-9b-mlx"
export EMBED_MODEL="text-embedding-nomic-embed-text-v1.5"
export LM_STUDIO_URL="http://127.0.0.1:1234/v1"
export QUANT_API="http://localhost:18081"
export KNOWLEDGE_PATHS="$(pwd)/knowledge"
export PORT=18000

exec .venv/bin/python3 main.py
