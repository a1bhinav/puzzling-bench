#!/bin/bash
# puzzling-bench: serve Llama-3.1-8B-Instruct with vLLM on one GPU, then run the 255-puzzle pilot
# and the full text-only v1 set. GENERATION ONLY — the judge runs later from the login node
# (see slurm/README.md). Submit from the repo root:  sbatch slurm/run_vllm.sbatch [RESULTS_DIR]
#
#SBATCH -J puzzling-vllm
#SBATCH -p gpu,gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=[a100|l40s|h100]
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH -e slurm/logs/%x-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load conda/latest
conda activate vllm

# ---- settings (override with env vars: VAR=value sbatch ... ) ---------------------------------------
SERVED_NAME=${SERVED_NAME:-meta-llama/Llama-3.1-8B-Instruct}
MODEL=${MODEL:-$SERVED_NAME}       # a local model directory or a Hugging Face model id
V1=${V1:-data/hf/puzzles.jsonl}   # the released dataset (see README.md)
PILOT_IDS=${PILOT_IDS:-data/hf/results/references/pilot255.locked_ref.jsonl}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-64}          # concurrent requests from run_eval.py
MAX_TOKENS=${MAX_TOKENS:-4096}  # per solve call (recorded in every result row)
TIMEOUT=${TIMEOUT:-900}         # seconds per request; 64 streams x 4k tokens on one GPU can take minutes
EXTRA_BODY=${EXTRA_BODY:-}      # JSON merged into every request, e.g. '{"chat_template_kwargs": {"enable_thinking": false}}'
# Results dir: pass it as $1 to RESUME a killed/preempted run (rows already written are skipped);
# the default embeds today's date, so a resubmission on a later day would start a fresh dir.
RESULTS=${1:-data/eval/$(echo "$SERVED_NAME" | tr '/' '_')_$(date +%Y%m%d)}
mkdir -p "$RESULTS"

echo "job $SLURM_JOB_ID on $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "model dir: $MODEL   served as: $SERVED_NAME   results: $RESULTS"

# ---- 1. vLLM server in the background -------------------------------------------------------------
vllm serve "$MODEL" --served-model-name "$SERVED_NAME" --dtype bfloat16 --max-model-len 8192 --port "$PORT" \
    > "$RESULTS/vllm_server_$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'echo "stopping vLLM ($VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true' EXIT

# poll /v1/models until the server answers (model load + CUDA graphs usually take 2-5 min)
for i in $(seq 1 120); do
    if curl -sf --noproxy '*' "http://localhost:$PORT/v1/models" > /dev/null; then
        echo "vLLM ready after ~$((i * 10))s"; break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited during startup; see $RESULTS/vllm_server_$SLURM_JOB_ID.log" >&2; exit 1
    fi
    sleep 10
done
curl -sf --noproxy '*' "http://localhost:$PORT/v1/models" > /dev/null || { echo "vLLM not ready after 20 min" >&2; exit 1; }

# ---- 2. runs (generation only: --no-judge is REQUIRED here, otherwise the judge fallback would
#         send judge calls to the same local server) ------------------------------------------------
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY=dummy
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
COMMON=(--v1 "$V1" --model "$SERVED_NAME" --no-judge --workers "$WORKERS" --max-tokens "$MAX_TOKENS" --timeout "$TIMEOUT")
[ -n "$EXTRA_BODY" ] && COMMON+=(--extra-body "$EXTRA_BODY")

# 2a. pilot: the 255 Pilot-255 puzzles every model in the paper saw (seed 0, --n 200 --max-short 100 --post-cutoff-all)
echo "=== pilot (255 puzzles) ==="
if [ -f "$PILOT_IDS" ]; then
    python3 scripts/run_eval.py "${COMMON[@]}" --ids-from "$PILOT_IDS" \
        --out "$RESULTS/pilot255.jsonl"
else
    echo "WARNING: pilot ids file missing; falling back to filter recreation (may not match pilot255 exactly)"
    python3 scripts/run_eval.py "${COMMON[@]}" --n 200 --max-short 100 --post-cutoff-all \
        --out "$RESULTS/pilot255.jsonl"
fi

# 2b. full text-only v1 (14,375 puzzles). Resumable: resubmit with the same RESULTS dir to continue.
echo "=== full text-only v1 ==="
python3 scripts/run_eval.py "${COMMON[@]}" --all-text-only --n 100000 \
    --out "$RESULTS/full_text_only.jsonl"

echo "done: $RESULTS"
