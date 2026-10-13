#!/bin/bash
# R.3 follow-up -- how many candidates (and how many tokens each) does the reranker need?
# hybrid_k60_rerank (top-100, 512 tokens) lifted nDCG@10 0.784 -> 0.879 but costs ~5 s/query.
# Each variant reranks hybrid_k60's fused top-N; the pool is extended with anything new and
# judged by the same blind judge, then all variants are scored together.
#   bash eval/rerank_sweep.sh
set -u
cd "$(dirname "$0")/.."
SRC=${SRC:-llm_v3_qwen27}
MODEL=${MODEL:-qwen/qwen3.8-27b}
LMS="$HOME/.lmstudio/bin/lms"
"$LMS" unload "$MODEL" >/dev/null 2>&1   # the reranker gets the GPU to itself while timed

run() {  # candidates max_len
  echo "[$(date +%H:%M)] rerank top-$1, $2 tokens"
  RERANK_CANDIDATES=$1 RERANK_MAX_LEN=$2 RERANK_SYSTEM="hybrid_k60_rerank_n$1_t$2" \
    .venv/bin/python eval/rerank_runs.py 2>&1 | grep -E "latency|entered|extended"
}
run 50 512
run 30 512
run 20 512
run 30 256
run 100 256

echo "[$(date +%H:%M)] judging pairs the variants added"
"$LMS" load "$MODEL" -y --ttl 3600 >/dev/null 2>&1
until curl -s -m 120 http://127.0.0.1:1234/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"reasoning_effort\":\"none\",\"max_tokens\":5,\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}]}" \
  | grep -q '"choices"'; do sleep 10; done
for i in 1 2; do JUDGE_WORKERS=1 .venv/bin/python eval/judge_pool.py --llm --rubric v3 --model "$MODEL" --source "$SRC" 2>&1 | tail -1; done
"$LMS" unload "$MODEL" >/dev/null 2>&1

echo "[$(date +%H:%M)] scoring"
.venv/bin/python eval/score_runs.py --source "$SRC" | sed -n '1,14p'
.venv/bin/python - <<'PY'
from pymongo import MongoClient
import statistics
db = MongoClient("mongodb://root:root@localhost:37018/?authSource=admin")["quant_data"]
lat = {}
for r in db.eval_retrieval_runs.find({"system": {"$regex": "^hybrid_k60_rerank"}}):
    lat.setdefault(r["system"], []).append(r.get("rerank_seconds") or 0)
for s, v in sorted(lat.items()):
    print(f"{s:<32} latency median {statistics.median(v):.2f}s  p90 {sorted(v)[int(len(v)*0.9)]:.2f}s")
PY
echo "[$(date +%H:%M)] done"
