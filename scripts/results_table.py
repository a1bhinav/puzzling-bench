#!/usr/bin/env python3
"""
One table over every judged results file: a leaderboard of the runs.

    python3 scripts/results_table.py                       # data/eval/*.judged*.jsonl
    python3 scripts/results_table.py data/eval/claude-*.judged*.jsonl --md > results.md
    python3 scripts/results_table.py --common               # also: accuracy on the puzzles ALL listed files share

Per file: rows, API errors, tier1-only, tier1+judge (the headline accuracy), no-answer, judge used, judge errors,
tokens, and the judge that produced the verdicts (from judge_model when recorded; otherwise guessed from the
file-name suffix, marked '?'). Standard library only.
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

SUFFIX_JUDGE = {".judged.llama.jsonl": "llama (local)?", ".judged.qwen14b.jsonl": "qwen3-14b (local)?",
                ".judged.jsonl": "unrecorded (Opus 4.8[1m] per 2026-09-28 call)?"}


def load_latest(path):
    latest = {}
    for ln in open(path, encoding="utf-8"):
        if ln.strip():
            x = json.loads(ln)
            latest[(x["post_id"], x["condition"])] = x
    return list(latest.values())


def judge_label(path, rows):
    rec = sorted({str(x.get("judge_model")) for x in rows if x.get("judge_model")})
    if rec:
        return ", ".join(rec)
    for suf, lab in SUFFIX_JUDGE.items():
        if path.endswith(suf):
            m = re.search(r"\.judged\.([^.]+)\.jsonl$", path)
            return lab if suf != ".judged.jsonl" or not m else f"{m.group(1)}?"
    m = re.search(r"\.judged\.([^.]+)\.jsonl$", path)
    return f"{m.group(1)}?" if m else "?"


def stats(rows):
    n = len(rows)
    ok = [x for x in rows if x.get("error") is None]
    jerr = sum(1 for x in rows if x.get("judge_error"))
    if not ok:
        return dict(n=n, err=n, t1=0.0, acc=0.0, none=0.0, ju=0.0, jerr=jerr, pt=0, ct=0, scored=0)
    t1 = sum(bool(x.get("tier1")) for x in ok)
    corr = sum(bool(x.get("correct")) for x in ok)
    ju = sum(bool(x.get("judge_used")) for x in ok)
    none = sum(1 for x in ok if x.get("answer_source") == "none")
    return dict(n=n, err=n - len(ok), t1=100 * t1 / len(ok), acc=100 * corr / len(ok), none=100 * none / len(ok),
                ju=100 * ju / len(ok), jerr=jerr, scored=len(ok),
                pt=sum(x.get("prompt_tokens") or 0 for x in ok), ct=sum(x.get("completion_tokens") or 0 for x in ok))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--md", action="store_true", help="markdown table instead of fixed-width text")
    ap.add_argument("--common", action="store_true", help="add a table restricted to puzzles present in every file")
    ap.add_argument("--min-rows", type=int, default=50, help="skip files with fewer rows (probes, smoke tests)")
    args = ap.parse_args()
    files = args.files or sorted(glob.glob("data/eval/*.judged*.jsonl"))
    data = []
    for f in files:
        rows = load_latest(f)
        if len(rows) < args.min_rows:
            continue
        model = ", ".join(sorted({str(x.get("model")) for x in rows if x.get("model")})) or "?"
        effort = ", ".join(sorted({str(x.get("reasoning_effort")) for x in rows if x.get("reasoning_effort")})) or "-"
        data.append((os.path.basename(f), model, effort, judge_label(f, rows), stats(rows), rows))

    cols = ["file", "model", "effort", "judge", "n", "api err", "tier1 %", "tier1+judge %", "no-ans %", "judge used %", "judge err", "compl. tokens"]

    def emit(rowsets, title):
        print(f"\n{title}")
        lines = []
        for f, model, effort, judge, s, _ in rowsets:
            lines.append([f, model, effort, judge, s["n"], s["err"], f"{s['t1']:.1f}", f"{s['acc']:.1f}",
                          f"{s['none']:.1f}", f"{s['ju']:.1f}", s["jerr"], f"{s['ct']:,}"])
        if args.md:
            print("| " + " | ".join(cols) + " |")
            print("|" + "|".join("---" for _ in cols) + "|")
            for ln in lines:
                print("| " + " | ".join(str(v) for v in ln) + " |")
        else:
            w = [max(len(str(c)), *(len(str(ln[i])) for ln in lines)) for i, c in enumerate(cols)]
            print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(cols)))
            for ln in lines:
                print("  ".join(str(v).ljust(w[i]) for i, v in enumerate(ln)))

    emit(data, "ALL ROWS PER FILE (tier1+judge % = headline accuracy over non-errored rows)")

    if args.common and len(data) > 1:
        common = set.intersection(*[{(x["post_id"], x["condition"]) for x in d[5]} for d in data])
        sub = [(f, m, e, j, stats([x for x in rows if (x["post_id"], x["condition"]) in common]), rows)
               for f, m, e, j, _, rows in data]
        emit(sub, f"SAME {len(common)} PUZZLES IN EVERY FILE (directly comparable)")

    # which puzzle sets are these? group files by their post_id set
    sets = Counter(frozenset(x["post_id"] for x in d[5]) for d in data)
    print("\nPUZZLE SETS (files sharing an identical post_id set are directly comparable):")
    for i, (s, c) in enumerate(sorted(sets.items(), key=lambda kv: -len(kv[0]))):
        members = [d[0] for d in data if frozenset(x["post_id"] for x in d[5]) == s]
        print(f"  set {i + 1}: {len(s):,} puzzles  <- " + ", ".join(members))


if __name__ == "__main__":
    main()
