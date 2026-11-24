#!/usr/bin/env python3
"""
Build-time answer extraction (protocol v1, "LLM-assisted, once, at build time").

A cheap model reads each puzzle and its accepted answer and returns a canonical one-line answer
plus an answer type, so scoring no longer depends on the first spoiler block being the answer.

Usage:
    python3 scripts/extract_answers.py --mock --limit 20                # plumbing test, no API
    python3 scripts/extract_answers.py --limit 30                       # eyeball 30 real extractions first
    python3 scripts/extract_answers.py                                  # full pass over data/v1/puzzles.jsonl (resumable)
    python3 scripts/extract_answers.py --merge                          # write data/v1/puzzles_canon.jsonl

Model: JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL (falls back to LLM_*), or --model/--base-url/--api-key.

Writes:  data/v1/canonical.jsonl        one row per puzzle: post_id, canonical_answer, answer_type, parts, notes, raw, error
         data/v1/puzzles_canon.jsonl    (--merge) v1 records with short_answer replaced by canonical_answer,
                                        short_answer_raw kept, answer_type/parts added, answer_source="llm"

answer_type:  single       one definite answer (word, number, phrase, expression)
              multi_part   several sub-answers that must all be given
              strategy     the answer is a procedure / strategy / proof sketch
              construction the answer is an arrangement, grid, list, or other object too long for a line
              unclear      the accepted answer does not state a definite final answer

Then: python3 scripts/run_eval.py --v1 data/v1/puzzles_canon.jsonl --answer-types single ...
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import chat, judge_config  # noqa: E402

SYSTEM = """You extract final answers from puzzle solutions. You will see a puzzle and the accepted human-written answer to it. The accepted answer usually contains the solution plus reasoning, hints, or worked steps.

Return ONLY a JSON object with these fields:
  "canonical_answer": the final answer as one concise line, exactly as a grader should compare against (a word, number, phrase, expression, or short list). No reasoning. If the puzzle has several parts, join the parts with " | ".
  "answer_type": one of "single", "multi_part", "strategy", "construction", "unclear".
      single = one definite answer; multi_part = several sub-answers all required; strategy = a procedure or proof; construction = a grid, arrangement, or list too long for one line; unclear = no definite final answer stated.
  "parts": for multi_part, a list of the sub-answers in order; otherwise [].
  "notes": one short sentence if anything is ambiguous, else "".

The canonical answer must contain ONLY the final answer(s) the question explicitly asks for, copied as faithfully as possible from the accepted answer. No justification, no explanation, no restatement of the method. Use multi_part ONLY when the question itself asks for several things; never treat an explanation as a part. Do not correct, expand, or complete words; copy them exactly.

Do not solve the puzzle yourself; extract what the accepted answer says. No markdown, no code fences."""

TYPES = {"single", "multi_part", "strategy", "construction", "unclear"}


def _fix_escapes(s):
    """Double backslashes that are not valid JSON escapes (LaTeX: \\frac, \\(, ...)."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)


def parse_json(raw):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.S)
    m = re.search(r"\{.*\}", s, re.S)  # first {...} block, in case of chatter around it
    for cand in (s, m.group(0) if m else None):
        if not cand:
            continue
        # a lone \f or \b followed by a letter is LaTeX (\frac, \begin, \binom), never a control char
        cand = re.sub(r"\\([bf])(?=[A-Za-z])", r"\\\\\1", cand)
        for txt in (cand, _fix_escapes(cand)):
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                pass
    return None


def mock_extract(r):
    first = r["short_answer"].split("\n")[0].strip().rstrip(",")
    t = "multi_part" if r["n_question_marks"] >= 3 else "single"
    return json.dumps({"canonical_answer": first, "answer_type": t, "parts": [], "notes": ""})


def run_one(r, args):
    row = {"post_id": r["post_id"], "canonical_answer": None, "answer_type": None, "parts": [],
           "notes": None, "raw": None, "error": None}
    try:
        if args.mock:
            raw = mock_extract(r)
        else:
            user = (f"PUZZLE TITLE: {r['title']}\n\nPUZZLE:\n{r['question_text'][:5000]}\n\n"
                    f"ACCEPTED ANSWER:\n{r['full_answer'][:5000]}")
            raw = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                       temperature=0.0, max_tokens=400,
                       **judge_config(args.model, args.base_url, args.api_key))
        row["raw"] = raw
        obj = parse_json(raw)
        if not isinstance(obj, dict) or not str(obj.get("canonical_answer", "")).strip():
            raise ValueError("no canonical_answer in model output")
        t = str(obj.get("answer_type", "")).strip().lower()
        row.update(canonical_answer=str(obj["canonical_answer"]).strip(),
                   answer_type=t if t in TYPES else "unclear",
                   parts=[str(p) for p in (obj.get("parts") or [])],
                   notes=str(obj.get("notes") or "")[:300])
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:500]
    return row


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def merge(v1_path, canon_path, out_path):
    canon = {}
    for x in read_jsonl(canon_path):          # last good row per post_id wins
        if x.get("error") is None and x.get("canonical_answer"):
            canon[x["post_id"]] = x
    n, m, types = 0, 0, Counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for r in read_jsonl(v1_path):
            n += 1
            c = canon.get(r["post_id"])
            if c:
                m += 1
                r["short_answer_raw"] = r["short_answer"]
                r["short_answer"] = c["canonical_answer"]
                r["answer_type"] = c["answer_type"]
                r["answer_parts"] = c["parts"]
                r["answer_source"] = "llm"
                types[c["answer_type"]] += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"merged {m:,}/{n:,} records -> {out_path}")
    for t, k in types.most_common():
        print(f"  {t:<13} {k:>7,}  ({100 * k / max(m, 1):.1f}%)")
    if m < n:
        print(f"  {n - m:,} records still use the spoiler-based short_answer (no extraction yet)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", default="data/v1/puzzles.jsonl")
    ap.add_argument("--out", default="data/v1/canonical.jsonl")
    ap.add_argument("--merged", default="data/v1/puzzles_canon.jsonl")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only process the first N not-yet-done puzzles")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retry-errors", action="store_true", help="re-run rows that errored last time")
    ap.add_argument("--mock", action="store_true", help="no API; canonical = first line of the spoiler")
    ap.add_argument("--merge", action="store_true", help="only merge canonical.jsonl into the v1 records")
    args = ap.parse_args()

    if args.merge:
        merge(args.v1, args.out, args.merged)
        return
    if not args.mock and not judge_config(args.model, args.base_url, args.api_key)["model"]:
        sys.exit("need a model: --model, or JUDGE_MODEL / LLM_MODEL in the environment (or --mock)")

    recs = read_jsonl(args.v1)
    done = {x["post_id"] for x in read_jsonl(args.out) if x.get("error") is None or not args.retry_errors}
    todo = [r for r in recs if r["post_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"v1 records {len(recs):,}  done {len(done):,}  to run {len(todo):,}  -> {args.out}", file=sys.stderr)
    if not todo:
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lock, n_done, t0 = threading.Lock(), 0, time.time()
    errs = 0
    with open(args.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, r, args) for r in todo]
        for fut in as_completed(futs):
            row = fut.result()
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
                errs += row["error"] is not None
                if n_done % 50 == 0 or n_done == len(todo):
                    rate = n_done / max(time.time() - t0, 1e-6)
                    eta = (len(todo) - n_done) / max(rate, 1e-6)
                    print(f"  {n_done:,}/{len(todo):,}  errors {errs}  {rate:.1f}/s  eta {eta / 60:.0f} min",
                          file=sys.stderr)

    if args.limit:  # show what came back so it can be eyeballed before the full run
        for row in read_jsonl(args.out)[-len(todo):]:
            print(f"{row['post_id']:>6} | {row['answer_type'] or 'ERR':<12} | {str(row['canonical_answer'])[:90]!r}"
                  + (f"   <- {row['error']}" if row["error"] else ""))
    print("next: python3 scripts/extract_answers.py --merge", file=sys.stderr)


if __name__ == "__main__":
    main()
