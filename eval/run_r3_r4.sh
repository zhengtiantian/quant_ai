#!/bin/bash
# R.4 strict re-judge + R.3 rerank evaluation, end to end. Resumable: judge_pool.py skips
# pairs already graded and retries errored ones, so killing and rerunning loses nothing.
#
# Lessons from the first attempt (2026-10-12), built in here:
#  - load qwen 27B explicitly and wait: requests sent while LM Studio JIT-loads it all
#    return 500 (800 of them did)
#  - nothing else should use the GPU meanwhile (the job_agent run and the reranker
#    loading at the same time made LM Studio unload models and fail both)
#  - Docker: start only mongodb + qdrant; Docker Desktop otherwise restarts every container
#    with a restart policy (quant_data, kafka, airflow...) and the VM eats ~75% CPU
#
#   bash eval/run_r3_r4.sh            # ~2-3 h, fans will run
set -u
cd "$(dirname "$0")/.."
SRC=${SRC:-llm_v3_qwen27}
MODEL=${MODEL:-qwen/qwen3.8-27b}
LMS="$HOME/.lmstudio/bin/lms"

if pgrep -f "main.py --hours" >/dev/null; then
  echo "job_agent run in progress -- it uses LM Studio too; run this afterwards"; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  open -a Docker
  for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 3; done
fi
# `docker info` answers before the daemon can serve images: on 2026-10-13 the first
# `compose up` got "500 Internal Server Error" and every later step ran without a database.
# Retry until both are reachable, and stop rather than run blind.
for i in $(seq 1 20); do
  (cd ../quant_docker && docker compose up -d mongodb qdrant) >/dev/null 2>&1
  if curl -s -m 5 http://localhost:26333/collections | grep -q '"ok"' && \
     .venv/bin/python -c "from pymongo import MongoClient; MongoClient('mongodb://root:root@localhost:37018/?authSource=admin', serverSelectionTimeoutMS=5000).admin.command('ping')" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done
if ! curl -s -m 5 http://localhost:26333/collections | grep -q '"ok"'; then echo "qdrant not reachable -- stopping"; exit 1; fi
.venv/bin/python -c "from pymongo import MongoClient; MongoClient('mongodb://root:root@localhost:37018/?authSource=admin', serverSelectionTimeoutMS=5000).admin.command('ping')" \
  || { echo "mongo not reachable -- stopping"; exit 1; }
echo "[$(date +%H:%M)] mongo + qdrant ready"

"$LMS" load "$MODEL" -y --ttl 3600 >/dev/null 2>&1
until curl -s -m 120 http://127.0.0.1:1234/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"reasoning_effort\":\"none\",\"max_tokens\":5,\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}]}" \
  | grep -q '"choices"'; do sleep 10; done
echo "[$(date +%H:%M)] $MODEL ready"

judge() { JUDGE_WORKERS=1 .venv/bin/python eval/judge_pool.py --llm --rubric v3 --model "$MODEL" --source "$SRC" 2>&1 | tail -1; }
judge; judge                                   # second pass retries any LM Studio hiccups

echo "[$(date +%H:%M)] reranking (unloading the judge first so the reranker has the GPU)"
"$LMS" unload "$MODEL" >/dev/null 2>&1
.venv/bin/python eval/rerank_runs.py 2>&1 | grep -v Warning | tail -6

echo "[$(date +%H:%M)] judging the pairs the reranker added"
"$LMS" load "$MODEL" -y --ttl 3600 >/dev/null 2>&1
judge; judge

echo "[$(date +%H:%M)] scoring"
.venv/bin/python eval/judge_pool.py --kappa reference_claude "$SRC"
[ -n "$(.venv/bin/python eval/judge_pool.py --kappa human "$SRC" 2>/dev/null | grep overlapping)" ] && \
  .venv/bin/python eval/judge_pool.py --kappa human "$SRC"
.venv/bin/python eval/score_runs.py --source "$SRC"
.venv/bin/python eval/score_runs.py --source "$SRC" --compare hybrid_k60 hybrid_k60_rerank | tail -20
"$LMS" unload "$MODEL" >/dev/null 2>&1
echo "[$(date +%H:%M)] done"
