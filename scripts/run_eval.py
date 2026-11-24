#!/usr/bin/env python3
"""
Run a model on puzzling-bench v1 and score it (docs/answer_matching.md, protocol v1).

Usage:
    python3 scripts/run_eval.py --mock --n 50 --no-judge                # pipeline smoke test, no API, no cost
    python3 scripts/run_eval.py --model MODEL --n 200                   # with hints (default condition)
    python3 scripts/run_eval.py --model MODEL --n 200 --hints both      # same puzzles, with & without hints
    python3 scripts/run_eval.py --model MODEL --n 200 --post-cutoff-all # + every eligible May-June puzzle
    python3 scripts/run_eval.py --model MODEL --ids-from data/eval/claude-sonnet-5.pilot255.jsonl --no-judge
                                                                         # exactly the puzzles of an earlier run
    python3 scripts/run_eval.py --summarize data/eval/MODEL.jsonl       # recompute tables from a results file
    python3 scripts/run_eval.py --model M --all-text-only --n 100000 --no-judge --workers 64 --out F   # generation only
    python3 scripts/run_eval.py --judge-file F                           # judge a --no-judge results file -> F.judged.jsonl
    python3 scripts/run_eval.py --judge-file F --rejudge --judge-model M2 --out F.judged.M2.jsonl
                                                                         # re-judge with another judge (keeps the old verdicts in prev_judgments)
    python3 scripts/run_eval.py --model M --images --ids-from data/eval/imgpilot100.ids.jsonl --no-judge --out F
                                                                         # image pilot: question images sent as image_url parts
    python3 scripts/run_eval.py --model M --no-images-control --ids-from data/eval/imgpilot100.ids.jsonl --no-judge --out F2
                                                                         # control: same puzzles/prompt, images omitted (condition 'noimg')

Model under test:  LLM_BASE_URL / LLM_API_KEY / LLM_MODEL   (or --base-url / --api-key / --model)
                   OPENAI_BASE_URL / OPENAI_API_KEY are accepted as fallbacks (e.g. a local vLLM server)
                   --reasoning-effort LEVEL  is shorthand for --extra-body '{"reasoning_effort": "LEVEL"}' (if the endpoint accepts it)
Judge:             JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL (or --judge-*); all three required unless
                   --no-judge -- there is no fallback to the model's settings
                   --judge-reasoning-effort LEVEL / --judge-extra-body JSON (or JUDGE_EXTRA_BODY) for the judge's request
Every row records what was actually run: model, served_model, reasoning_effort, run_meta (harness version, credits)
and, when the judge was used, judge_model / judge_served_model / judge_meta -- quote these in the paper.

Pilot selection: text-only puzzles (no images in question or accepted answer), short answer taken
from a spoiler block (or, after extract_answers.py --merge, canonical answers of --answer-types),
short answer <= --max-short chars, question <= --max-q chars.
--hints without / both additionally restrict to puzzles that have a hint (spoiler block in the question).
Sampling is seeded (--seed) so reruns and other models get the same puzzles.
Resumable: (post_id, condition) rows already in --out are skipped.

Writes:  <out>              one JSON row per (puzzle, condition)
         <out>.summary.txt  accuracy overall / by condition / by year bucket / by tag
"""
import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import chat_full, judge_config, score_answer  # noqa: E402

# ----------------------------------------------------------------------------- images (--images / --no-images-control)
IMAGE_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}


def load_manifest(path):
    """url -> local file name for every image fetch_images.py marked ok. Resolving through the manifest fixes the
    v1 rows whose question_image_paths are empty because puzzles.jsonl was built before the fetch finished."""
    m = {}
    if path and os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            if ln.strip():
                x = json.loads(ln)
                if x.get("status") == "ok" and x.get("path"):
                    m[x["url"]] = x["path"]
    return m


def question_images(r, manifest, images_dir):
    """Local paths of the question's images (manifest lookup by URL, else the row's own paths); missing files dropped."""
    paths = [manifest.get(u) for u in (r.get("question_image_urls") or [])]
    if not paths or any(p is None for p in paths):
        paths = list(r.get("question_image_paths") or [])
    full = [os.path.join(images_dir, p) for p in paths if p]
    return [p for p in full if os.path.exists(p)]


def image_parts(paths):
    """OpenAI image_url content parts (base64 data URLs) for the given files, plus the total bytes encoded."""
    parts, nbytes = [], 0
    for p in paths:
        mime = IMAGE_MIME.get(p.rsplit(".", 1)[-1].lower())
        if not mime:
            continue
        data = open(p, "rb").read()
        nbytes += len(data)
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}})
    return parts, nbytes


def text_of(content):
    """Text of a user message whose content is either a string or a list of content parts."""
    if isinstance(content, str):
        return content
    return "\n".join((b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") == "text")

SOLVE_SYSTEM = ("You are solving a puzzle from Puzzling Stack Exchange. Think it through, but keep the reasoning "
                "compact. Your reply MUST end with exactly one line of the form:\n"
                "ANSWER: <your answer>\n"
                "Put only the answer on that line, with no explanation. Never end without the ANSWER line.")
FORCE_MSG = ("\n\n---\nYour earlier attempt at this puzzle was cut off. The tail of your notes:\n\n{notes}\n\n---\n"
             "Do NOT continue the notes and do NOT reason further. Commit to your single best final answer now, "
             "guessing if necessary, and reply with exactly one line:\nANSWER: <answer>")
ANS_RE = re.compile(r"^\W{0,6}(?:final\s+)?answer\W{0,6}:\s*(.+?)\s*$", re.I | re.M)


def extract_answer(text):
    """Last 'ANSWER: ...' line; else the last non-empty line (flagged)."""
    ms = ANS_RE.findall(text or "")
    if ms:
        return ms[-1].strip("* `"), True
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return (lines[-1].strip("* `") if lines else ""), False


def eligible(r, args, need_hint):
    if getattr(args, "images", False) or getattr(args, "no_images_control", False):
        # image pilot: puzzles WITH question images whose ground truth is text (non-empty short_answer);
        # an image in the accepted answer is fine -- it is explanation, the short_answer is what we score
        if not r["has_image"] or not (r.get("short_answer") or "").strip():
            return False
    elif r["has_image"] or r["answer_has_image"]:
        return False
    if args.all_text_only:                      # every text-only puzzle, whatever the answer source/length
        return not need_hint or r["has_hint"]
    if r.get("answer_source") == "llm":        # canonical answers present (extract_answers.py --merge)
        if args.answer_types and r.get("answer_type") not in args.answer_types:
            return False
    elif r["short_answer_source"] != "spoiler":
        return False
    if len(r["short_answer"]) > args.max_short or len(r["question_text"]) > args.max_q:
        return False
    if need_hint and not r["has_hint"]:
        return False
    return True


def build_prompt(r, condition):
    body = r["question_text_no_hints"] if condition == "without" else r["question_text"]   # 'with' and 'noimg': full text
    return f"Title: {r['title']}\n\n{body}\n\nEnd your reply with one line: ANSWER: <your answer>"


def mock_answer(r):
    """Deterministic fake model for pipeline tests: right 2/3 of the time, varied phrasing,
    and every 5th output is 'truncated' with no ANSWER line to exercise the forcing path."""
    if r["post_id"] % 5 == 0:
        return "Let me think about this at length and never quite get to"
    if r["post_id"] % 3 == 0:
        return "I think it is something else.\nANSWER: forty-two"
    if r["post_id"] % 3 == 1:
        return f"Reasoning here.\nANSWER: {r['short_answer']}"
    return f"**Answer:** The answer is {r['short_answer']}"


def year_bucket(r, cutoff):
    d = str(r.get("creation_date") or "")
    y = r.get("year") or 0
    if d >= cutoff:
        return "post-cutoff"
    if y >= 2026:
        return "2026 pre-cutoff"
    if y == 2025:
        return "2025"
    if y >= 2021:
        return "2021-2024"
    return "<=2020"


def _add_usage(row, usage):
    if usage:
        row["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        row["completion_tokens"] += int(usage.get("completion_tokens") or 0)


def _call_record(kind, res, answer_line_found):
    """Per-call metadata kept in row['calls'] (the row-level token fields are sums over calls)."""
    u = res.get("usage") or {}
    return {"call": kind, "finish_reason": res.get("finish_reason"),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "answer_line_found": answer_line_found,
            "served_model": res.get("model"), "run_meta": res.get("run_meta")}


SCORE_KEYS = ("tier1", "judge_used", "verdict", "reason", "correct",
              "judge_model", "judge_served_model", "judge_meta", "judge_usage")


def run_one(r, condition, args):
    t0 = time.time()
    row = {"post_id": r["post_id"], "condition": condition, "model": args.model, "tags": r["tags"],
           "year": r["year"], "creation_date": r["creation_date"], "has_hint": r["has_hint"],
           "short_answer": r["short_answer"], "raw_output": None, "finish_reason": None,
           "prompt_tokens": 0, "completion_tokens": 0, "max_tokens": args.max_tokens,
           "extra_body": args.extra_body,
           "reasoning_effort": (args.extra_body or {}).get("reasoning_effort"),   # what we asked for
           "served_model": None, "run_meta": None,                                # what the server says it ran
           "calls": [], "extracted_answer": None,
           "answer_line_found": None, "answer_source": None, "forced": False, "forced_output": None,
           "tier1": None,
           "judge_used": False, "verdict": None, "reason": None, "correct": None,
           "judge_model": None, "judge_served_model": None, "judge_meta": None, "judge_usage": None,
           "latency_s": None, "error": None}
    kw = dict(model=args.model, base_url=args.base_url, api_key=args.api_key,
              temperature=None if args.no_temperature else 0.0, timeout=args.timeout,
              extra_body=args.extra_body)
    try:
        prompt = build_prompt(r, condition)
        img_mode = getattr(args, "images", False) or getattr(args, "no_images_control", False)
        img_paths = question_images(r, args.manifest_map, args.images_dir) if img_mode else []
        parts, img_bytes = image_parts(img_paths) if getattr(args, "images", False) else ([], 0)
        content = [{"type": "text", "text": prompt}] + parts if parts else prompt
        msgs = [{"role": "system", "content": SOLVE_SYSTEM},
                {"role": "user", "content": content}]
        if args.mock:
            res = {"content": mock_answer(r), "finish_reason": "length" if r["post_id"] % 5 == 0 else "stop",
                   "usage": None}
        else:
            res = chat_full(msgs, max_tokens=args.max_tokens, **kw)
        raw = res["content"]
        row["finish_reason"] = res.get("finish_reason")
        row["served_model"], row["run_meta"] = res.get("model"), res.get("run_meta")
        if img_mode:   # what was sent (image_count/bytes) vs what the puzzle has (images_available)
            row["run_meta"] = dict(row["run_meta"] or {}, image_count=len(parts), image_bytes=img_bytes,
                                   images_available=len(img_paths),
                                   images_omitted=len(img_paths) if getattr(args, "no_images_control", False) else 0)
        _add_usage(row, res.get("usage"))
        ans, found = extract_answer(raw)
        row["calls"].append(_call_record("solve", res, found))
        truncated = res.get("finish_reason") == "length"
        # finished without the prefix -> its last line is probably the answer; cut off mid-reasoning -> no answer
        source = "answer_line" if found else ("last_line" if not truncated else "none")
        if not found and not args.no_force:
            # second, cheap call: fresh user turn with the tail of the model's notes, instruction last
            notes = (raw or "(no output)")[-1500:]
            msgs2 = [msgs[0], {"role": "user", "content": text_of(msgs[1]["content"]) + FORCE_MSG.format(notes=notes)}]
            if args.mock:
                fc = "I keep thinking\nand thinking" if r["post_id"] % 10 == 0 else f"ANSWER: {r['short_answer']}"
                res2 = {"content": fc, "finish_reason": "stop", "usage": None}
            else:
                res2 = chat_full(msgs2, max_tokens=100, **kw)
            _add_usage(row, res2.get("usage"))
            forced = (res2["content"] or "").strip()
            row["forced_output"] = forced
            ans2, found2 = extract_answer(forced)
            row["calls"].append(_call_record("force", res2, found2))
            if found2:
                ans, source, row["forced"] = ans2, "forced_line", True
            elif forced and "\n" not in forced and len(forced) <= 150:
                ans, source, row["forced"] = forced.strip("* `"), "forced_bare", True   # a bare short reply
            # otherwise keep "last_line" (model had finished) or "none" (it was cut off and would not commit)
        if source == "none":
            ans = ""
        row.update(raw_output=raw, extracted_answer=ans, answer_line_found=found, answer_source=source)
        sc = score_answer(r, ans, use_judge=not args.no_judge and source != "none",
                          model=args.judge_model, base_url=args.judge_base_url, api_key=args.judge_api_key,
                          extra_body=args.judge_extra_body)
        row.update({k: sc.get(k) for k in SCORE_KEYS})
        if "judge_raw" in sc:
            row["judge_raw"] = sc["judge_raw"]
    except Exception as e:  # keep going; the row records the failure
        row["error"] = f"{type(e).__name__}: {e}"[:500]
    row["latency_s"] = round(time.time() - t0, 2)
    return row


# ----------------------------------------------------------------------------- summary
def acc_line(rows):
    n = len(rows)
    ok = [x for x in rows if x["error"] is None]
    if not ok:
        return f"n={n:<5} (all errored)"
    t1 = sum(bool(x["tier1"]) for x in ok)
    corr = sum(bool(x["correct"]) for x in ok)
    ju = sum(bool(x["judge_used"]) for x in ok)
    none = sum(1 for x in ok if x.get("answer_source") == "none")
    return (f"n={n:<5} tier1-only {100 * t1 / len(ok):5.1f}%   tier1+judge {100 * corr / len(ok):5.1f}%   "
            f"no-answer {100 * none / len(ok):5.1f}%   judge used {100 * ju / len(ok):5.1f}%   errors {n - len(ok)}")


OUTCOMES = ["correct", "wrong", "no_answer_tag", "hit_max_tokens", "api_error"]


def outcome(x):
    """One exclusive failure-mode class per row (so the classes sum to n)."""
    if x.get("error") is not None:
        return "api_error"
    if x.get("answer_source") == "none" or not str(x.get("extracted_answer") or "").strip():
        return "hit_max_tokens" if x.get("finish_reason") == "length" else "no_answer_tag"
    return "correct" if x.get("correct") else "wrong"


def summarize(rows, cutoff, path=None, price_in=None, price_out=None):
    L = []
    by_cond = defaultdict(list)
    for x in rows:
        by_cond[x["condition"]].append(x)
    models = sorted({x["model"] for x in rows if x["model"]})
    L.append("=" * 88)
    L.append(f"PUZZLING-BENCH v1 pilot   model(s): {', '.join(models) or 'mock'}   rows: {len(rows):,}")
    # provenance: what was actually run (served model, effort, harness) and who judged
    served = sorted({str(x.get("served_model")) for x in rows if x.get("served_model")})
    efforts = sorted({str(x.get("reasoning_effort")) for x in rows if x.get("reasoning_effort")})
    harness = sorted({str((x.get("run_meta") or {}).get("harness")) for x in rows if (x.get("run_meta") or {}).get("harness")})
    judges = sorted({str(x.get("judge_model")) for x in rows if x.get("judge_model")})
    jeff = sorted({str(((x.get("judge_meta") or {}).get("reasoning_effort"))) for x in rows
                   if (x.get("judge_meta") or {}).get("reasoning_effort")})
    L.append(f"  served model(s): {', '.join(served) or 'n/a'}   effort: {', '.join(efforts) or 'n/a (default)'}"
             f"   harness: {', '.join(harness) or 'n/a'}")
    L.append(f"  judge(s): {', '.join(judges) or 'n/a (not recorded -- pre-2026-10 rows or tier-1 only)'}"
             f"   judge effort: {', '.join(jeff) or 'n/a'}")
    L.append("=" * 88)
    for cond, xs in sorted(by_cond.items()):
        L.append(f"[{cond} hints]  {acc_line(xs)}")
    L.append("")
    L.append("BY YEAR BUCKET")
    order = ["<=2020", "2021-2024", "2025", "2026 pre-cutoff", "post-cutoff"]
    for cond, xs in sorted(by_cond.items()):
        L.append(f"  [{cond}]")
        b = defaultdict(list)
        for x in xs:
            b[year_bucket(x, cutoff)].append(x)
        for k in order:
            if b.get(k):
                L.append(f"    {k:<16} {acc_line(b[k])}")
    L.append("")
    L.append("BY TAG (top 12 in sample; a puzzle counts under each of its tags)")
    for cond, xs in sorted(by_cond.items()):
        L.append(f"  [{cond}]")
        b = defaultdict(list)
        for x in xs:
            for t in x["tags"]:
                b[t].append(x)
        for t, ys in sorted(b.items(), key=lambda kv: -len(kv[1]))[:12]:
            L.append(f"    {t:<22} {acc_line(ys)}")
    ok = [x for x in rows if x["error"] is None]
    L.append("")
    L.append("HARNESS")
    L.append(f"  truncated (finish_reason=length):        {sum(1 for x in ok if x.get('finish_reason') == 'length')}")
    src = Counter(x.get("answer_source") or ("answer_line" if x["answer_line_found"] else "last_line") for x in ok)
    L.append(f"  answer source: ANSWER line {src.get('answer_line', 0)}  |  follow-up ANSWER line {src.get('forced_line', 0)}"
             f"  |  follow-up bare reply {src.get('forced_bare', 0)}  |  finished w/o prefix, last line {src.get('last_line', 0)}"
             f"  |  NO ANSWER (cut off, would not commit) {src.get('none', 0)}")
    pt = sum(x.get("prompt_tokens") or 0 for x in ok)
    ct = sum(x.get("completion_tokens") or 0 for x in ok)
    line = f"  tokens: prompt {pt:,}  completion {ct:,}"
    if price_in is not None and price_out is not None:
        line += f"  ->  est. ${pt / 1e6 * price_in + ct / 1e6 * price_out:.2f} (model under test only; judge extra)"
    L.append(line)
    L.append("")
    L.append("FAILURE MODES (exclusive classes; % of ALL rows, errors included)")
    L.append(f"  {'model':<32} {'cond':<8} {'n':>5}  " + "  ".join(f"{k:>14}" for k in OUTCOMES) + "  max_tokens")
    for m in models:
        for cond, xs in sorted(by_cond.items()):
            ys = [x for x in xs if x["model"] == m]
            if not ys:
                continue
            c = Counter(outcome(x) for x in ys)
            mt = "/".join(sorted({str(x.get("max_tokens", "n/a")) for x in ys}))
            L.append(f"  {m:<32} {cond:<8} {len(ys):>5}  "
                     + "  ".join(f"{c[k]:>7} {100 * c[k] / len(ys):5.1f}%" for k in OUTCOMES) + f"  {mt}")
    miss = [x for x in ok if x.get("answer_line_found") is False]
    trunc = [x for x in ok if x.get("finish_reason") == "length"]
    scored = lambda xs: sum(1 for x in xs if outcome(x) in ("correct", "wrong"))
    L.append(f"  first reply had no ANSWER line: {len(miss)} (follow-up recovered an answer for {scored(miss)})"
             f"   |   first reply hit max_tokens: {len(trunc)} (recovered {scored(trunc)})")
    text = "\n".join(L)
    print(text)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")


# ----------------------------------------------------------------------------- judge an existing file
def judge_file(args):
    """Tier-2 pass over a generation-only (--no-judge) results file. Writes <file>.judged.jsonl (resumable);
    the input file is never modified. Rows that tier 1 already accepted, or that have no answer, are copied as-is.
    With --rejudge, rows that already carry a verdict are judged again (e.g. with a stronger or different-family
    judge); the earlier verdict is kept in prev_judgments so the judges can be compared row by row."""
    jkw = judge_config(args.judge_model, args.judge_base_url, args.judge_api_key, args.judge_extra_body)   # fail loudly before any work
    if args.rejudge and not args.out:
        sys.exit("--rejudge needs --out (e.g. F.judged.<judge-tag>.jsonl) so the first judge's file is not overwritten")
    recs = {}
    for ln in open(args.v1, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            recs[r["post_id"]] = r
    latest = {}
    for ln in open(args.judge_file, encoding="utf-8"):
        if ln.strip():
            x = json.loads(ln)
            latest[(x["post_id"], x["condition"])] = x
    out = args.out or re.sub(r"\.jsonl$", "", args.judge_file) + ".judged.jsonl"
    done = set()
    if os.path.exists(out):
        for ln in open(out, encoding="utf-8"):
            if ln.strip():
                x = json.loads(ln)
                if not x.get("judge_error"):
                    done.add((x["post_id"], x["condition"]))

    def needs_judge(x):
        return (x["error"] is None and (args.rejudge or not x.get("judge_used")) and not x.get("tier1")
                and x.get("answer_source") != "none" and bool(str(x.get("extracted_answer") or "").strip()))

    def do(x):
        try:
            if x.get("judge_used") or x.get("verdict"):      # keep the earlier judgment(s) for the comparison
                x.setdefault("prev_judgments", []).append(
                    {"judge_model": x.get("judge_model"), "judge_served_model": x.get("judge_served_model"),
                     "verdict": x.get("verdict"), "reason": x.get("reason"), "correct": x.get("correct"),
                     "judge_raw": x.get("judge_raw")})
            sc = score_answer(recs[x["post_id"]], x["extracted_answer"], use_judge=True, **jkw)
            x.update({k: sc.get(k) for k in SCORE_KEYS})
            if "judge_raw" in sc:
                x["judge_raw"] = sc["judge_raw"]
            x.pop("judge_error", None)
        except Exception as e:
            x["judge_error"] = f"{type(e).__name__}: {e}"[:500]
        return x

    todo = [x for k, x in latest.items() if k not in done]
    jobs = [x for x in todo if needs_judge(x)]
    print(f"rows {len(latest)}  already judged {len(done)}  to judge {len(jobs)}  judge {jkw['model']}"
          f"{' (re-judge)' if args.rejudge else ''}  -> {out}", file=sys.stderr)
    lock = threading.Lock()
    n_done = 0
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for x in todo:
            if not needs_judge(x):
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        f.flush()
        for fut in as_completed([ex.submit(do, x) for x in jobs]):
            x = fut.result()
            with lock:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
                if n_done % 10 == 0 or n_done == len(jobs):
                    print(f"  {n_done}/{len(jobs)}", file=sys.stderr)
    final = {}
    for ln in open(out, encoding="utf-8"):
        if ln.strip():
            x = json.loads(ln)
            final[(x["post_id"], x["condition"])] = x
    summarize(list(final.values()), args.cutoff, out + ".summary.txt", args.price_in, args.price_out)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", default="data/v1/puzzles.jsonl")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-api-key", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ids-from", default=None,
                    help="results file whose post_ids define the sample, so a new model gets exactly the same puzzles "
                         "as an earlier run (e.g. data/eval/claude-sonnet-5.pilot255.jsonl); "
                         "overrides --n/--seed/--post-cutoff-all and the eligibility filters")
    ap.add_argument("--hints", choices=["with", "without", "both"], default="with")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--post-cutoff-all", action="store_true",
                    help="add every eligible post-cutoff puzzle to the sample (for the leakage plot)")
    ap.add_argument("--max-short", type=int, default=200, help="max short_answer length (chars)")
    ap.add_argument("--answer-types", default="single",
                    help="comma-separated answer types to keep when canonical answers exist (default: single)")
    ap.add_argument("--max-q", type=int, default=4000, help="max question_text length (chars)")
    ap.add_argument("--all-text-only", action="store_true",
                    help="run every text-only puzzle (skip the answer-source and length filters)")
    ap.add_argument("--images", action="store_true",
                    help="image pilot: puzzles with question images (non-empty short_answer is the ground truth); each "
                         "question image is sent after the text as a base64 image_url part (an endpoint may forward "
                         "these to the model as file attachments). Defaults unchanged when absent.")
    ap.add_argument("--no-images-control", action="store_true",
                    help="control for --images: same puzzles and prompt, images deliberately omitted; rows get "
                         "condition 'noimg' so they never collide with --images rows")
    ap.add_argument("--images-dir", default="data/images")
    ap.add_argument("--manifest", default="data/images/manifest.jsonl",
                    help="fetch_images.py manifest used to resolve question image URLs to local files")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=180, help="seconds per request (raise for a busy local server)")
    ap.add_argument("--extra-body", default=None,
                    help='JSON object merged into every chat request body, e.g. '
                         '\'{"chat_template_kwargs": {"enable_thinking": false}}\' for Qwen3 on vLLM')
    ap.add_argument("--reasoning-effort", default=None,
                    help="thinking effort for the model under test (none|minimal|low|medium|high|xhigh|max on the "
                         "endpoints that accept it; OpenAI reasoning models take low|medium|high). Recorded in every row.")
    ap.add_argument("--judge-extra-body", default=None, help="JSON object merged into every judge request")
    ap.add_argument("--judge-reasoning-effort", default=None, help="thinking effort for the judge")
    ap.add_argument("--rejudge", action="store_true",
                    help="with --judge-file: judge rows again even if they already have a verdict (needs --out)")
    ap.add_argument("--no-temperature", action="store_true", help="omit temperature (some models reject it)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent requests (~64 against a local vLLM server)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-force", action="store_true", help="don't make the follow-up call when the ANSWER line is missing")
    ap.add_argument("--price-in", type=float, default=None, help="$ per 1M prompt tokens, for the cost estimate")
    ap.add_argument("--price-out", type=float, default=None, help="$ per 1M completion tokens")
    ap.add_argument("--mock", action="store_true", help="fake model, no API calls (implies --no-judge)")
    ap.add_argument("--cutoff", default="2026-05-01")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summarize", default=None, help="results file to summarize; nothing is run")
    ap.add_argument("--judge-file", default=None,
                    help="results file from a --no-judge run: judge its unjudged rows -> <file>.judged.jsonl")
    args = ap.parse_args()
    args.extra_body = json.loads(args.extra_body) if args.extra_body else None
    if args.extra_body is not None and not isinstance(args.extra_body, dict):
        sys.exit("--extra-body must be a JSON object")
    if args.reasoning_effort:
        args.extra_body = dict(args.extra_body or {}, reasoning_effort=args.reasoning_effort)
    args.judge_extra_body = json.loads(args.judge_extra_body) if args.judge_extra_body else None
    if args.judge_extra_body is not None and not isinstance(args.judge_extra_body, dict):
        sys.exit("--judge-extra-body must be a JSON object")
    if args.judge_reasoning_effort:
        args.judge_extra_body = dict(args.judge_extra_body or {}, reasoning_effort=args.judge_reasoning_effort)
    if args.images and args.no_images_control:
        sys.exit("--images and --no-images-control are mutually exclusive (run them as two passes)")
    args.manifest_map = load_manifest(args.manifest) if (args.images or args.no_images_control) else {}

    if args.summarize:
        rows = [json.loads(ln) for ln in open(args.summarize, encoding="utf-8") if ln.strip()]
        summarize(rows, args.cutoff, args.summarize + ".summary.txt", args.price_in, args.price_out)
        return
    if args.judge_file:
        judge_file(args)
        return
    if args.mock:
        args.model, args.no_judge = args.model or "mock", True
    if not args.model:
        sys.exit("need --model (or LLM_MODEL), or --mock")
    if not args.no_judge:
        judge_config(args.judge_model, args.judge_base_url, args.judge_api_key, args.judge_extra_body)   # fail now, not once per row
    out = args.out or os.path.join("data/eval", re.sub(r"[^\w.-]+", "_", args.model) + ".jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    recs = [json.loads(ln) for ln in open(args.v1, encoding="utf-8") if ln.strip()]
    args.answer_types = {t.strip() for t in args.answer_types.split(",") if t.strip()}
    need_hint = args.hints in ("without", "both")
    conds = ["with", "without"] if args.hints == "both" else [args.hints]
    if args.no_images_control:
        conds = ["noimg"]            # same prompt as 'with', images withheld; distinct label so rows never collide
    if args.ids_from:
        # exactly the puzzles of an earlier run (e.g. the shared pilot255 set), whatever filters produced them
        want, seen = [], set()
        for ln in open(args.ids_from, encoding="utf-8"):
            if ln.strip():
                pid = json.loads(ln)["post_id"]
                if pid not in seen:
                    seen.add(pid)
                    want.append(pid)
        by_id = {r["post_id"]: r for r in recs}
        sample = [by_id[p] for p in want if p in by_id]
        print(f"v1 records {len(recs):,}  ids-from {args.ids_from}: {len(want)} puzzles, {len(sample)} found in v1"
              f"  conditions {conds}", file=sys.stderr)
    else:
        pool = [r for r in recs if eligible(r, args, need_hint)]
        sample = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
        if args.post_cutoff_all:
            have = {r["post_id"] for r in sample}
            sample += [r for r in pool if str(r["creation_date"]) >= args.cutoff and r["post_id"] not in have]
        print(f"v1 records {len(recs):,}  eligible {len(pool):,}  sampled {len(sample)}  conditions {conds}",
              file=sys.stderr)

    done = set()
    if os.path.exists(out):
        for ln in open(out, encoding="utf-8"):
            if ln.strip():
                x = json.loads(ln)
                if x.get("error") is None:
                    done.add((x["post_id"], x["condition"]))
    jobs = [(r, c) for r in sample for c in conds if (r["post_id"], c) not in done]
    print(f"already done {len(done)}  to run {len(jobs)}  -> {out}", file=sys.stderr)

    lock = threading.Lock()
    n_done = 0
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, r, c, args) for r, c in jobs]
        for fut in as_completed(futs):
            row = fut.result()
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
                if n_done % 10 == 0 or n_done == len(jobs):
                    print(f"  {n_done}/{len(jobs)}", file=sys.stderr)

    rows = [json.loads(ln) for ln in open(out, encoding="utf-8") if ln.strip()]
    # keep only the latest row per (post_id, condition) so retried errors don't double count
    latest = {}
    for x in rows:
        latest[(x["post_id"], x["condition"])] = x
    summarize(list(latest.values()), args.cutoff, out + ".summary.txt", args.price_in, args.price_out)


if __name__ == "__main__":
    main()
