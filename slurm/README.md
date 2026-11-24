# Running open-weight models on a Slurm cluster

`run_vllm.sbatch` serves a model with vLLM on one GPU and runs the evaluation against it, **generation only**.
Judging happens afterwards, from any machine that can reach the judge endpoint (compute nodes often have no
internet access).

## One-time setup

```bash
git clone https://github.com/a1bhinav/puzzling-bench && cd puzzling-bench
mkdir -p slurm/logs                     # Slurm writes job logs here; it does not create the folder itself
# the released dataset, including the frozen Pilot-255 id list (see the top-level README)
hf download puzzling-bench/puzzling-bench --repo-type dataset --local-dir data/hf --exclude "images/*"
# vLLM environment (once); adapt the module line to your cluster
module load conda/latest && conda create -n vllm python=3.11 -y && conda activate vllm && pip install vllm
```

The runner (`scripts/run_eval.py`, `scripts/score.py`) uses the standard library only, so the `vllm`
environment's Python is all it needs.

## Submit

```bash
sbatch slurm/run_vllm.sbatch                         # results -> data/eval/meta-llama_Llama-3.1-8B-Instruct_<YYYYMMDD>/
sbatch slurm/run_vllm.sbatch data/eval/my_run_dir    # explicit results folder (use this to resume, see below)
squeue -u $USER ; tail -f slurm/logs/puzzling-vllm-<jobid>.out
```

What the job does, in order:

1. `vllm serve` the model (bf16, max-model-len 8192, port 8000) in the background, polling
   `localhost:8000/v1/models` until it answers.
2. **Pilot-255 first**: the 255 puzzles of the paper's main evaluation, read from
   `data/hf/results/references/pilot255.locked_ref.jsonl` → `pilot255.jsonl`.
3. **Then the full text-only set** (14,375 puzzles, `--all-text-only`) → `full_text_only.jsonl`.

Both runs use `--no-judge --workers 64 --max-tokens 4096 --timeout 900` against the local server. Override any
setting with an environment variable, e.g. `MODEL=/path/to/weights WORKERS=32 sbatch slurm/run_vllm.sbatch`.
Unlike the paper's CLI harnesses, vLLM enforces `--max-tokens`.

Each results folder gets `pilot255.jsonl` and `full_text_only.jsonl` (one row per puzzle, written as they
finish), their `*.summary.txt` (tier-1-only accuracy and a failure-mode table; no judge yet) and the vLLM server log.

### Preemption and time limits

Rows are appended as they complete, and **a run resumes when re-submitted with the same results folder**:
puzzles already in the `.jsonl` are skipped. For long runs, name the folder yourself so that a resubmission
continues the same run:

```bash
sbatch slurm/run_vllm.sbatch data/eval/meta-llama_Llama-3.1-8B-Instruct_run1   # and again after a kill
```

## Judge afterwards

`--judge-file` runs tier 2 on the rows tier 1 did not already accept, writes `<file>.judged.jsonl` (resumable;
the generation file is never modified) and prints the final tables. Any OpenAI-compatible judge endpoint works;
the paper's main tables use Claude Opus 4.8, so a different judge gives numbers that are not directly comparable.

```bash
set -a; source .env; set +a              # JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL (see .env.example)
R=data/eval/meta-llama_Llama-3.1-8B-Instruct_run1
python3 scripts/run_eval.py --v1 data/hf/puzzles.jsonl --judge-file $R/pilot255.jsonl --workers 6
python3 scripts/run_eval.py --v1 data/hf/puzzles.jsonl --judge-file $R/full_text_only.jsonl --workers 6
```

Rows whose judge call failed get a `judge_error` field and are retried on the next `--judge-file` run. To
re-print the tables without judging: `python3 scripts/run_eval.py --summarize $R/full_text_only.judged.jsonl`.
Rate-limited endpoints return `HTTP 429` rows; lower `--workers` if that happens.

## Qwen3 and other thinking models

Serve with thinking **disabled**; otherwise the 4,096-token budget goes to reasoning and the `ANSWER:` line is
truncated away:

```bash
MODEL=/path/to/Qwen3-8B SERVED_NAME=Qwen/Qwen3-8B EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}' \
    sbatch slurm/run_vllm.sbatch data/eval/Qwen_Qwen3-8B_run1
```

`EXTRA_BODY` is passed to `run_eval.py --extra-body`, which merges the JSON into every chat request; vLLM forwards
`chat_template_kwargs` to Qwen3's chat template, which is the documented off switch. Adding
`--reasoning-parser qwen3` to the `vllm serve` line moves any remaining thinking into `reasoning_content`.

## Sanity checks before a long run

```bash
# is the base URL respected? (should fail fast with a connection error)
OPENAI_BASE_URL=http://localhost:9/v1 OPENAI_API_KEY=x python3 scripts/run_eval.py --v1 data/hf/puzzles.jsonl \
    --model x --n 1 --no-judge --timeout 5 --out /tmp/probe.jsonl
# 3 puzzles against a live server
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=dummy python3 scripts/run_eval.py --v1 data/hf/puzzles.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct --n 3 --no-judge --out /tmp/three.jsonl
```
