# Puzzling-Bench

Code for **Puzzling-Bench: Evaluating Language Models on 23,777 Human-Written Puzzles with Community-Verified
Answers** ([arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)).

Puzzling-Bench is a benchmark of 23,777 puzzles from [Puzzling Stack Exchange](https://puzzling.stackexchange.com),
each paired with the answer its own author accepted: riddles, ciphers, logic grids, lateral-thinking problems,
wordplay and recreational mathematics posted between May 2014 and June 2026. Every puzzle carries the question
(with and without the asker's spoiler-tagged hints), the accepted answer, a short answer taken from it
(`short_answer`: the first spoiler block of the accepted answer, else its first paragraph; all 23,777 puzzles),
community tags, a creation date and full attribution metadata. For the 355 puzzles of Pilot-255 and Img-100 the
release also has LLM-extracted reference answers (`results/references/`), checked on a 50-puzzle human review.

- **Dataset:** <https://huggingface.co/datasets/puzzling-bench/puzzling-bench> (CC BY-SA 4.0). It contains the
  puzzles, the referenced images, and every per-row model output and judge verdict behind the paper.
- **Code:** this repository (MIT License).

## What is in the benchmark

| | Puzzles |
|---|---:|
| All puzzles (questions not closed, with an accepted answer present in the dump) | 23,777 |
| Strict text-only (no image in the question or the accepted answer) | 14,375 |
| With at least one asker hint | 4,825 |
| Posted on or after 2026-05-01 (the post-cutoff slice) | 254 |

The puzzles come from the Stack Exchange data dump of 2026-06-30 (snapshot
[`stackexchange_20260630_sakura`](https://archive.org/details/stackexchange_20260630_sakura) on the Internet Archive,
content through 2026-06-30 23:58 UTC). The paper evaluates on three sets:

- **Pilot-255**: 255 strict text-only puzzles: 200 drawn at random (seed 0) plus all 56 eligible post-cutoff
  puzzles, one of which the draw already contained. The id list is `results/references/pilot255.locked_ref.jsonl`.
- **Img-100**: 100 puzzles with an image in the question (`results/references/imgpilot100.ids.jsonl`).
- **Full text-only**: all 14,375 strict text-only puzzles.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/parse_dump.py`, `scripts/fetch_images.py`, `scripts/build_v1.py` | dump XML to JSONL, image download, the dataset build |
| `scripts/extract_answers.py` | LLM extraction of the reference answers for the 355 Pilot-255 and Img-100 puzzles |
| `scripts/run_eval.py`, `scripts/score.py` | solver runs, `ANSWER:` extraction, tier-1 matching and the LLM judge, `--summarize` reports |
| `scripts/results_table.py`, `scripts/compare_judges.py`, `scripts/failure_overlap.py`, `scripts/hint_ablation.py`, `scripts/paired_accuracy.py`, `scripts/bootstrap_ci.py`, `scripts/dataset_stats.py`, `scripts/credits_used.py` | analyses behind the tables |
| `scripts/figures/` | the three figures |
| `scripts/release/` | how the released `images/` and `results/` folders were built from the private run files |
| `docs/answer_matching.md` | the scoring protocol (tier-1 normalisation rules and the judge contract) |
| `slurm/` | serving open-weight models with vLLM on a Slurm cluster |

Requirements: Python 3.9+. The scripts use the standard library only, except `scripts/figures/` (matplotlib).

## Get the data

```bash
pip install -U huggingface_hub    # provides the `hf` command
hf download puzzling-bench/puzzling-bench --repo-type dataset --local-dir data/hf --exclude "images/*"
```

Leave out `--exclude` to also download the images (2.8 GB in six tar shards; `images/index.jsonl` maps each post
and image URL to its file). All commands below assume the dataset is in `data/hf/`.

## Reproduce the tables and figures

Every number below is computed from the released per-row results; no model or API is called. Accuracy is the
share of rows scored correct among rows without an `error`.

```bash
R=data/hf/results

# Table 1 (filtering funnel)
sed -n '/FUNNEL/,/kept/p' data/hf/stats.txt
# Table 2 (dataset statistics)
python3 scripts/dataset_stats.py data/hf/puzzles.jsonl

# Table 3 (hint ablation)
python3 scripts/hint_ablation.py --v1 data/hf/puzzles.jsonl --pilot-ids $R/references/pilot255.locked_ref.jsonl \
    --with-pattern "$R/pilot255/{model}.judged.locked_ref.opus48.jsonl" \
    --without-pattern "$R/pilot255_no_hints/{model}.judged.locked_ref.opus48.jsonl" --out hint_ablation.txt

# Table 4 (main results): accuracy under the three judges, judge spread and errored rows ...
python3 scripts/compare_judges.py "$R/pilot255/*.judged.locked_ref.*.jsonl"
python3 scripts/results_table.py $R/pilot255/*.judged.locked_ref.opus48.jsonl
# ... and the 95% bootstrap intervals of the Opus 4.8 column
python3 scripts/bootstrap_ci.py $R/pilot255/*.judged.locked_ref.opus48.jsonl

# Table 5 (image puzzles: attached vs. withheld)
for m in claude-opus-5.5 gpt-6.1-sol; do
  python3 scripts/paired_accuracy.py $R/imgpilot100/$m.image.judged.locked_ref.opus48.jsonl \
                                     $R/imgpilot100/$m.no_image.judged.locked_ref.opus48.jsonl
done

# Table 6 (greedy ensemble) and Appendix D (the hard core)
python3 scripts/failure_overlap.py $R/pilot255/*.judged.locked_ref.opus48.jsonl --puzzles data/hf/puzzles.jsonl

# Table 7 (per-tag accuracy) and the period breakdown behind Figure 2: per-model summaries
for f in $R/pilot255/*.judged.locked_ref.opus48.jsonl; do python3 scripts/run_eval.py --summarize $f; done
#   -> each <file>.summary.txt has a BY TAG block (Table 7) and a BY YEAR BUCKET block

# Figures 1-3 (need matplotlib)
python3 scripts/figures/make_tag_heatmap.py --tex PATH/TO/acl_latex_arxiv.tex   # draws Table 7 from the paper source
python3 scripts/figures/make_by_date.py          # reads the *.summary.txt files written above
python3 scripts/figures/make_acc_vs_tokens.py
```

The paper's bootstrap draws were not seeded, so `bootstrap_ci.py` (seed 0) gives interval bounds within about one
point of those in Table 4.

Results in the text:

```bash
# Sec. 4.3 (tier-1-only accuracy) and Sec. 6.4 (errored rows before and after re-running timed-out rows)
python3 scripts/results_table.py $R/pilot255/*.judged.locked_ref.opus48.jsonl
python3 scripts/results_table.py $R/pilot255_errored_rows/*.jsonl \
    $R/pilot255_reference_ablation/{kimi-k3,grok-4.7}.judged.spoiler_ref.opus48.jsonl
# Sec. 5.2 and 5.3 (full text-only runs; their Pilot-255 counterparts with the same reference)
python3 scripts/results_table.py $R/full_text_only/*.jsonl $R/pilot255_reference_ablation/*.judged.spoiler_ref.opus48.jsonl
python3 scripts/run_eval.py --summarize $R/full_text_only/claude-sonnet-5.judged.spoiler_ref.jsonl
# Sec. 6.1 (judge agreement, Cohen's kappa, own-family bonus; then under the spoiler-based reference)
python3 scripts/compare_judges.py "$R/pilot255/*.judged.locked_ref.*.jsonl"
python3 scripts/compare_judges.py "$R/pilot255_reference_ablation/*.judged.spoiler_ref.*.jsonl"
# Sec. 6.1 (weak judges)
python3 scripts/compare_judges.py "$R/weak_judges/*.jsonl" --judge-map \
    '.spoiler_ref.llama-3.1-8b.jsonl=llama-3.1-8b,.spoiler_ref.qwen3-14b.jsonl=qwen3-14b,.spoiler_ref.opus48.jsonl=claude-opus-4.8'
# Sec. 6.2 (effect of the reference answer)
python3 scripts/results_table.py $R/pilot255_reference_ablation/*.opus48.jsonl
# Appendix B (cost per judge verdict and per puzzle; 100 credits = 1 US dollar)
python3 scripts/credits_used.py $R/pilot255/*.judged.locked_ref.opus48.jsonl
```

## Evaluate a new model

`run_eval.py` talks to any OpenAI-compatible chat-completions endpoint: a provider API, a vLLM server
(`slurm/`), or a local proxy. Configure the solver and the judge through environment variables (see
`.env.example`) or the matching command-line flags, then run Pilot-255 and score it:

```bash
cp .env.example .env    # fill in LLM_* (the model under test) and JUDGE_* (the judge)
set -a; source .env; set +a
python3 scripts/run_eval.py --v1 data/hf/puzzles.jsonl --ids-from data/hf/results/references/pilot255.locked_ref.jsonl \
    --out data/eval/mymodel.pilot255.jsonl                       # add --reasoning-effort LEVEL if the model has one
python3 scripts/run_eval.py --summarize data/eval/mymodel.pilot255.jsonl
```

Runs resume where they stopped when restarted with the same `--out`. Use `--all-text-only --n 100000` for the
full text-only set, `--hints without` for the no-hint condition, and `--no-judge` followed later by
`--judge-file` to separate generation from judging. The paper's main tables use Claude Opus 4.8 as the judge;
scores from another judge are not directly comparable. Note that the paper's solve calls request
`--max-tokens 4096`, which the paper's harnesses recorded but did not enforce; most servers do enforce it.

For Img-100, unpack the image shards and write the manifest that `--images` expects:

```bash
hf download puzzling-bench/puzzling-bench --repo-type dataset --local-dir data/hf --include "images/*"
mkdir -p data/images && for t in data/hf/images/*.tar; do tar -xf $t -C data/images; done
python3 -c 'import json; [print(json.dumps({"url": r["url"], "path": r["file"], "status": "ok"})) for r in map(json.loads, open("data/hf/images/index.jsonl")) if r["file"]]' > data/images/manifest.jsonl
python3 scripts/run_eval.py --v1 data/hf/puzzles.jsonl --images --ids-from data/hf/results/references/imgpilot100.ids.jsonl \
    --out data/eval/mymodel.imgpilot100.jsonl
```

### How the models in the paper were reached

Every model, solver or judge, was called through an OpenAI-compatible endpoint. The frontier-model calls in the
paper, solvers and judges, went through local proxies in front of two agentic command-line harnesses, referred
to as *harness A* (the 2026-10 runs) and *harness B* (the earlier Claude runs) in the paper and in the `run_meta.harness` field of the result
files. The proxies are not part of this repository. They disabled tool use and passed the reasoning effort
through; harness A's proxy also attached an `x_run_meta` object (harness label, requested and served model,
effort, token usage, cost, latency) to every reply, which `run_eval.py` stores with each row as `run_meta`.
Harness B rows and errored rows carry only the harness label, and the open-model rows in `weak_judges/` (served
with vLLM) have no `run_meta`.

## Rebuild the dataset from the dump

Requirements: Python 3.9+ and `7zz` (`brew install sevenzip`). The paper uses `puzzling.stackexchange.com.7z` from
the Internet Archive item [`stackexchange_20260630_sakura`](https://archive.org/details/stackexchange_20260630_sakura)
(Stack Exchange Data Dump 2026-06-30):

```bash
mkdir -p data/raw
curl -L -o data/raw/puzzling.stackexchange.com.7z \
    https://archive.org/download/stackexchange_20260630_sakura/stackexchange_20260630/puzzling.stackexchange.com.7z
md5 data/raw/puzzling.stackexchange.com.7z    # 7caa2147543343474bea7c761660d701 (md5sum on Linux)
7zz x data/raw/puzzling.stackexchange.com.7z -odata/raw/puzzling
python3 scripts/parse_dump.py --dump data/raw/puzzling --out data/processed     # -> questions/answers/comments JSONL
python3 scripts/fetch_images.py --rps 6 --workers 6                            # -> data/images/ + manifest.jsonl
python3 scripts/build_v1.py --proc data/processed --out data/v1 --images data/images   # -> puzzles.jsonl, stats.txt
```

The image host enforces a per-IP limit of roughly 7 requests per second; `fetch_images.py` respects it and
resumes after an interruption. Stack Exchange-hosted images posted before 2023-12-08 are also archived in the
[Stack Exchange Image Dataset](https://github.com/Franck-Dernoncourt/Stack-Exchange-Image-Dataset) (Dernoncourt, 2023;
doi:10.5281/zenodo.10465913).

## License and attribution

The code in this repository is released under the MIT License (see `LICENSE`). The dataset on Hugging Face is
released under CC BY-SA 4.0, like the Stack Exchange content it is derived from. Anyone who redistributes or
displays puzzle text must attribute it as Stack Exchange requires: link to the original question, show the
author name of every question and answer, and link each name to the author's profile. Every record carries the
fields for this (`url`, `answer_url`, `owner_display_name`, `owner_profile_url`, `answer_owner_display_name`,
`answer_owner_profile_url`); see the dataset card.

## Citation

```bibtex
@misc{sharma2026puzzlingbench,
  title         = {Puzzling-Bench: Evaluating Language Models on 23,777 Human-Written Puzzles with Community-Verified Answers},
  author        = {Sharma, Abhinav and Li, Li and Nguyen, Chien Van and Navuluru, Sai Karthik and Ni, Bo and Gao, Xiangbo and Huang, Xiaolei and Zhao, Yue and Ahmed, Nesreen K. and Derr, Tyler and Wang, Yu and Chen, Hongjie and Tu, Zhengzhong and Nguyen, Thien Huu and Rossi, Ryan A. and Dernoncourt, Franck},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
