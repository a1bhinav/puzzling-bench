#!/usr/bin/env python3
"""
Compare LLM judges on the same model outputs: does a stronger judge change the verdicts (has judge quality
plateaued?), and does a judge favor solvers from its own model family?

Inputs are judged results files for the SAME solver outputs, one per judge, e.g.

    python3 scripts/compare_judges.py data/eval/claude-sonnet-5.judged.jsonl \
                                      data/eval/claude-sonnet-5.judged.opus55.jsonl \
                                      data/eval/claude-sonnet-5.judged.gpt56.jsonl

or many solvers at once (the solver is read from each row's `model`, the judge from `judge_model`):

    python3 scripts/compare_judges.py 'data/eval/*.judged*.jsonl' --md > judges.md

Files written before judge_model was recorded get their judge from the file-name suffix via --judge-map
(default maps '.judged.jsonl' -> 'claude-opus-4-8[1m]', '.judged.llama.jsonl' -> 'llama-3.1-8b',
'.judged.qwen14b.jsonl' -> 'qwen3-14b'; adjust if that is wrong).

Reports
  1. accuracy of every solver under every judge (solver x judge matrix, tier1+judge %)
  2. pairwise judge agreement on the rows BOTH judged: % agree, Cohen's kappa, and the flips each way
     -> if Opus 4.8 vs Opus 5.5 agree on ~all rows, the 4.8 judge had already plateaued
  3. family bias: for each judge, MATCH rate on solvers of its own family vs. other families, compared with
     what the OTHER judges say about the same rows (a judge that is generous only to its own family shows a
     positive 'own-family bonus' here while the others do not)
Standard library only.
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from itertools import combinations

DEFAULT_JUDGE_MAP = {".judged.llama.jsonl": "llama-3.1-8b", ".judged.qwen14b.jsonl": "qwen3-14b",
                     ".judged.jsonl": "claude-opus-4-8[1m]"}
FAMILY = [("claude", "anthropic"), ("opus", "anthropic"), ("sonnet", "anthropic"), ("haiku", "anthropic"),
          ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),
          ("gemini", "google"), ("flash", "google"), ("llama", "meta"), ("qwen", "alibaba"),
          ("kimi", "moonshot"), ("deepseek", "deepseek"), ("mistral", "mistral"), ("grok", "xai")]


def family(name):
    n = (name or "").lower()
    for k, f in FAMILY:
        if k in n:
            return f
    return "other"


def load_latest(path):
    latest = {}
    for ln in open(path, encoding="utf-8"):
        if ln.strip():
            x = json.loads(ln)
            latest[(x["post_id"], x["condition"])] = x
    return latest


def judge_of(path, rows, jmap):
    rec = sorted({str(x.get("judge_model")) for x in rows if x.get("judge_model")})
    if rec:
        return rec[0] if len(rec) == 1 else "+".join(rec)
    for suf, lab in sorted(jmap.items(), key=lambda kv: -len(kv[0])):   # longest suffix first
        if path.endswith(suf):
            return lab
    m = re.search(r"\.judged\.([^.]+)\.jsonl$", path)
    return m.group(1) if m else path


def kappa(a, b):
    """Cohen's kappa for two lists of booleans."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="judged files or globs")
    ap.add_argument("--judge-map", default=None,
                    help="suffix=judge pairs for files without judge_model, comma-separated, "
                         "e.g. '.judged.jsonl=claude-opus-4-8[1m],.judged.o55.jsonl=claude-opus-5.5'")
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    jmap = dict(DEFAULT_JUDGE_MAP)
    if args.judge_map:
        for kv in args.judge_map.split(","):
            k, v = kv.split("=", 1)
            jmap[k.strip()] = v.strip()
    files = []
    for f in args.files:
        files += sorted(glob.glob(f)) or [f]

    # verdicts[solver][judge][(post_id, cond)] = correct (bool)   -- only rows the judge actually judged
    verdicts = defaultdict(lambda: defaultdict(dict))
    accs = defaultdict(dict)        # accs[solver][judge] = (acc %, n)
    for f in files:
        rows = load_latest(f)
        if not rows:
            continue
        judge = judge_of(f, rows.values(), jmap)
        by_solver = defaultdict(list)
        for k, x in rows.items():
            by_solver[str(x.get("model"))].append((k, x))
        for solver, items in by_solver.items():
            ok = [(k, x) for k, x in items if x.get("error") is None and not x.get("judge_error")]
            if not ok:
                continue
            accs[solver][judge] = (100 * sum(bool(x.get("correct")) for _, x in ok) / len(ok), len(ok))
            for k, x in ok:
                if x.get("judge_used"):
                    verdicts[solver][judge][k] = bool(x.get("correct"))

    judges = sorted({j for s in accs for j in accs[s]})
    sep = "|" if args.md else "  "

    def table(header, lines):
        if args.md:
            print("| " + " | ".join(header) + " |")
            print("|" + "|".join("---" for _ in header) + "|")
            for ln in lines:
                print("| " + " | ".join(str(v) for v in ln) + " |")
        else:
            w = [max(len(str(c)), *(len(str(ln[i])) for ln in lines)) for i, c in enumerate(header)]
            print(sep.join(str(c).ljust(w[i]) for i, c in enumerate(header)))
            for ln in lines:
                print(sep.join(str(v).ljust(w[i]) for i, v in enumerate(ln)))
        print()

    print("1. ACCURACY (tier1+judge %, n) OF EACH SOLVER UNDER EACH JUDGE\n")
    lines = []
    for s in sorted(accs):
        lines.append([s, family(s)] + [f"{accs[s][j][0]:.1f} (n={accs[s][j][1]})" if j in accs[s] else "-" for j in judges])
    table(["solver", "family"] + judges, lines)

    print("2. PAIRWISE JUDGE AGREEMENT on rows both judges graded (tier-1 misses only)\n")
    lines = []
    for s in sorted(verdicts):
        for ja, jb in combinations(sorted(verdicts[s]), 2):
            common = sorted(set(verdicts[s][ja]) & set(verdicts[s][jb]))
            if not common:
                continue
            a = [verdicts[s][ja][k] for k in common]
            b = [verdicts[s][jb][k] for k in common]
            agree = sum(x == y for x, y in zip(a, b))
            a_only = sum(x and not y for x, y in zip(a, b))
            b_only = sum(y and not x for x, y in zip(a, b))
            lines.append([s, ja, jb, len(common), f"{100 * agree / len(common):.1f}", f"{kappa(a, b):.3f}",
                          f"{100 * sum(a) / len(a):.1f}", f"{100 * sum(b) / len(b):.1f}", a_only, b_only])
    if lines:
        table(["solver", "judge A", "judge B", "n both", "agree %", "kappa", "A MATCH %", "B MATCH %",
               "A yes / B no", "B yes / A no"], lines)
    else:
        print("  (need two judges on the same solver outputs)\n")

    print("3. FAMILY BIAS: each judge's MATCH rate on its own family's solvers vs. the other judges' MATCH rate "
          "on the SAME rows\n   (own-family bonus = this judge's MATCH % minus the mean of the other judges' "
          "MATCH %, on rows all of them judged; positive only for its own family = bias)\n")
    lines = []
    for j in judges:
        jf = family(j)
        for grp, pred in (("own family (" + jf + ")", lambda s: family(s) == jf),
                          ("other families", lambda s: family(s) != jf)):
            mine, others = [], []
            for s in verdicts:
                if not pred(s) or j not in verdicts[s]:
                    continue
                other_js = [o for o in verdicts[s] if o != j]
                if not other_js:
                    continue
                for k, v in verdicts[s][j].items():
                    ov = [verdicts[s][o][k] for o in other_js if k in verdicts[s][o]]
                    if len(ov) == len(other_js):
                        mine.append(v)
                        others.append(sum(ov) / len(ov))
            if mine:
                lines.append([j, grp, len(mine), f"{100 * sum(mine) / len(mine):.1f}",
                              f"{100 * sum(others) / len(others):.1f}",
                              f"{100 * (sum(mine) / len(mine) - sum(others) / len(others)):+.1f}"])
            else:
                lines.append([j, grp, 0, "-", "-", "-"])
    table(["judge", "solvers", "rows", "this judge MATCH %", "other judges MATCH %", "own-family bonus (pp)"], lines)
    print("Reading: a judge is biased toward its own family if its bonus is clearly positive for 'own family' and "
          "~0 for 'other families'. If every judge's bonus is ~0 everywhere, the judges are interchangeable; "
          "if a stronger judge (e.g. Opus 5.5) agrees with Opus 4.8 on ~all rows (section 2), judge quality has plateaued.")


if __name__ == "__main__":
    main()
