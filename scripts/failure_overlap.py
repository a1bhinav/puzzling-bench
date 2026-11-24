#!/usr/bin/env python3
"""
Failure overlap / complementarity across solvers judged on the same puzzle set: do the models fail on the SAME
puzzles (a shared hard core) or on different ones (complementary strengths an ensemble could exploit)?

Inputs are judged results files, one solver per file, e.g.

    python3 scripts/failure_overlap.py 'data/eval/*.pilot255*.judged*.jsonl' --md > data/eval/failure_overlap.md
    python3 scripts/failure_overlap.py data/eval/claude-opus-5.5.pilot255.judged.opus48.jsonl \
                                       data/eval/gpt-5.6-sol.pilot255.judged.opus48.jsonl --puzzles data/v1/puzzles.jsonl

The solver label is derived from the file name (everything before '.pilot255' / '.judged' / '.jsonl'); override
with --labels 'file-prefix=label,...'. Per puzzle, correctness is the row's `correct` field; rows with `error`
set are dropped. ALL statistics are restricted to the post_ids present (non-errored) in EVERY file, so the
models are compared on exactly the same puzzles.

Reports
  1. common-set size and each model's accuracy on it
  2. conditional failure matrix: cell (A,B) = P(B fails | A fails), with each model's base failure rate alongside
     -> a cell far above B's base rate means A's failures predict B's (shared hard core)
  3. pairwise failure-set overlap: Jaccard and the phi coefficient of the two failure indicators
  4. oracle (union) accuracy, solved-by-all %, and the puzzles NO model solved (with tags)
  5. unique solves per model (puzzles only that model got right)
  6. greedy ensemble curve: best single model, then the model that adds the most union coverage, and so on
Standard library only.
"""
import argparse
import glob
import json
import math
import os
import re
from itertools import combinations

LABEL_CUT = re.compile(r"\.(pilot255|judged|jsonl)")


def label_of(path, overrides):
    base = os.path.basename(path)
    for pre, lab in overrides.items():
        if base.startswith(pre):
            return lab
    m = LABEL_CUT.search(base)
    return base[:m.start()] if m else base


def load_latest(path, condition=None):
    """{post_id: row} keeping the last row per post_id (files are appended on resume); errored rows dropped."""
    latest = {}
    for ln in open(path, encoding="utf-8"):
        if ln.strip():
            x = json.loads(ln)
            if condition and x.get("condition") != condition:
                continue
            latest[x["post_id"]] = x
    return {pid: x for pid, x in latest.items() if x.get("error") is None and not x.get("judge_error")}


def load_tags(puzzles_path, rows_by_model):
    tags = {}
    for rows in rows_by_model.values():
        for pid, x in rows.items():
            if x.get("tags") and pid not in tags:
                tags[pid] = list(x["tags"])
    if puzzles_path and os.path.exists(puzzles_path):
        for ln in open(puzzles_path, encoding="utf-8"):
            if ln.strip():
                x = json.loads(ln)
                tags.setdefault(x["post_id"], list(x.get("tags") or []))
    return tags


def phi(a, b):
    """Phi coefficient between two equal-length boolean lists."""
    n11 = sum(x and y for x, y in zip(a, b))
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n00 = len(a) - n11 - n10 - n01
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    return (n11 * n00 - n10 * n01) / math.sqrt(den) if den else float("nan")


def jaccard(s, t):
    u = s | t
    return len(s & t) / len(u) if u else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="judged files or globs (one solver per file)")
    ap.add_argument("--labels", default=None,
                    help="file-prefix=label pairs, comma-separated, to override the file-name-derived labels")
    ap.add_argument("--condition", default=None, help="keep only rows with this `condition` (default: all)")
    ap.add_argument("--puzzles", default="data/v1/puzzles.jsonl",
                    help="puzzles file used to fill in tags for puzzles whose rows carry none")
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    overrides = {}
    if args.labels:
        for kv in args.labels.split(","):
            k, v = kv.split("=", 1)
            overrides[k.strip()] = v.strip()
    files = []
    for f in args.files:
        files += sorted(glob.glob(f)) or [f]

    rows_by_model, dropped = {}, {}
    for f in files:
        lab = label_of(f, overrides)
        rows = load_latest(f, args.condition)
        if not rows:
            continue
        if lab in rows_by_model:
            lab = lab + "@" + os.path.basename(f)
        rows_by_model[lab] = rows
        dropped[lab] = sum(1 for _ in open(f, encoding="utf-8")) - len(rows)
    if len(rows_by_model) < 2:
        raise SystemExit("need at least two judged files")

    models = sorted(rows_by_model)
    common = set.intersection(*(set(r) for r in rows_by_model.values()))
    ids = sorted(common)
    n = len(ids)
    if n == 0:
        raise SystemExit("no post_id is present (non-errored) in every file")
    tags = load_tags(args.puzzles, rows_by_model)

    correct = {m: {pid: bool(rows_by_model[m][pid].get("correct")) for pid in ids} for m in models}
    solved = {m: {pid for pid in ids if correct[m][pid]} for m in models}
    failed = {m: common - solved[m] for m in models}
    acc = {m: 100 * len(solved[m]) / n for m in models}
    by_acc = sorted(models, key=lambda m: -acc[m])

    def table(header, lines):
        if args.md:
            print("| " + " | ".join(str(c) for c in header) + " |")
            print("|" + "|".join("---" for _ in header) + "|")
            for ln in lines:
                print("| " + " | ".join(str(v) for v in ln) + " |")
        else:
            w = [max(len(str(c)), *(len(str(ln[i])) for ln in lines)) for i, c in enumerate(header)]
            print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(header)))
            for ln in lines:
                print("  ".join(str(v).ljust(w[i]) for i, v in enumerate(ln)))
        print()

    def h(title):
        print(("## " if args.md else "") + title + "\n")

    def fmt_ids(pids):
        return ", ".join(str(p) for p in pids) if pids else "(none)"

    h("1. COMMON SET AND PER-MODEL ACCURACY")
    print(f"{len(models)} models; common set (post_ids non-errored in every file): **{n}** puzzles"
          if args.md else f"{len(models)} models; common set (post_ids non-errored in every file): {n} puzzles")
    print()
    lines = []
    for m in by_acc:
        lines.append([m, len(rows_by_model[m]), dropped[m], len(solved[m]), f"{acc[m]:.1f}",
                      f"{100 - acc[m]:.1f}"])
    table(["model", "rows (non-errored)", "rows dropped (error)", "solved on common", "acc %", "fail %"], lines)

    h("2. CONDITIONAL FAILURE MATRIX  cell(A,B) = P(B fails | A fails) %")
    lines = []
    for a in by_acc:
        fa = failed[a]
        row = [a, f"{100 * len(fa) / n:.1f}"]
        for b in by_acc:
            if a == b:
                row.append("-")
            elif not fa:
                row.append("n/a")
            else:
                row.append(f"{100 * len(fa & failed[b]) / len(fa):.1f}")
        lines.append(row)
    table(["A (given A fails)", "A base fail %"] + [f"-> {b}" for b in by_acc], lines)
    print("Reading: compare each cell with the column model's base fail rate (first column of its own row). "
          "A cell well above that base rate means A's failures predict B's; a cell near the base rate means the "
          "two models fail independently.\n")

    h("3. PAIRWISE FAILURE-SET OVERLAP (Jaccard of failure sets, phi of failure indicators)")
    lines = []
    for a, b in combinations(by_acc, 2):
        fa, fb = failed[a], failed[b]
        ia = [not correct[a][p] for p in ids]
        ib = [not correct[b][p] for p in ids]
        lines.append([a, b, len(fa), len(fb), len(fa & fb), len(fa | fb), f"{jaccard(fa, fb):.3f}",
                      f"{phi(ia, ib):.3f}"])
    table(["model A", "model B", "A fails", "B fails", "both fail", "either fails", "Jaccard", "phi"], lines)

    h("4. ORACLE / UNION ACCURACY AND THE HARD CORE")
    union = set.union(*solved.values())
    inter = set.intersection(*solved.values())
    nobody = sorted(common - union)
    lines = [["solved by >=1 model (oracle union)", len(union), f"{100 * len(union) / n:.1f}"],
             ["solved by ALL models", len(inter), f"{100 * len(inter) / n:.1f}"],
             ["solved by NO model (hard core)", len(nobody), f"{100 * len(nobody) / n:.1f}"]]
    # distribution of how many models solve each puzzle
    k_count = {}
    for p in ids:
        k = sum(correct[m][p] for m in models)
        k_count[k] = k_count.get(k, 0) + 1
    for k in range(len(models) + 1):
        lines.append([f"solved by exactly {k} model(s)", k_count.get(k, 0), f"{100 * k_count.get(k, 0) / n:.1f}"])
    table(["set", "n", "% of common"], lines)
    print(("**Hard core**" if args.md else "HARD CORE") + f" ({len(nobody)} puzzles solved by no model), post_id and tags:\n")
    if nobody:
        table(["post_id", "tags"], [[p, ", ".join(tags.get(p, [])) or "-"] for p in nobody])
        # tag census of the hard core vs the common set
        tc, tall = {}, {}
        for p in ids:
            for t in tags.get(p, []):
                tall[t] = tall.get(t, 0) + 1
                if p in set(nobody):
                    tc[t] = tc.get(t, 0) + 1
        top = sorted(tc.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
        print("Hard-core tag census (count in hard core / count in common set):\n")
        table(["tag", "hard core", "common set", "hard-core share %"],
              [[t, c, tall[t], f"{100 * c / tall[t]:.0f}"] for t, c in top])
    else:
        print("  (none)\n")

    h("5. UNIQUE SOLVES PER MODEL (solved by this model only)")
    lines = []
    for m in by_acc:
        others = set.union(*(solved[o] for o in models if o != m))
        uniq = sorted(solved[m] - others)
        lines.append([m, len(uniq), f"{100 * len(uniq) / n:.1f}", fmt_ids(uniq)])
    table(["model", "unique solves", "% of common", "post_ids"], lines)

    h("6. GREEDY ENSEMBLE CURVE (add the model that most increases union coverage)")
    chosen, covered, lines = [], set(), []
    remaining = list(models)
    while remaining:
        best = max(remaining, key=lambda m: (len(covered | solved[m]), acc[m], m))
        gain = len((covered | solved[best]) - covered)
        covered |= solved[best]
        chosen.append(best)
        remaining.remove(best)
        lines.append([len(chosen), best, f"{acc[best]:.1f}", f"+{gain}", len(covered), f"{100 * len(covered) / n:.1f}"])
    table(["step", "model added", "solo acc %", "new puzzles", "union solved", "union acc %"], lines)
    print("Reading: a steep curve (big gains after step 1) means the models are complementary; a flat one means "
          "they share the same failures and ensembling would not help.")


if __name__ == "__main__":
    main()
