#!/usr/bin/env python3
"""
Hint ablation on the PILOT-255 set: the same puzzles solved WITH the asker's hints (the main runs) and WITHOUT
them (--hints without), both judged by the same judge against the same locked reference.

    python3 scripts/hint_ablation.py                       # gpt-6.1-sol and claude-opus-5.5, Opus 4.8 judge, canon2 reference
    python3 scripts/hint_ablation.py --models gpt-6.1-sol --tag canon2.opus48
    python3 scripts/hint_ablation.py --v1 data/hf/puzzles.jsonl \
        --pilot-ids data/hf/results/references/pilot255.locked_ref.jsonl \
        --with-pattern 'data/hf/results/pilot255/{model}.judged.locked_ref.opus48.jsonl' \
        --without-pattern 'data/hf/results/pilot255_no_hints/{model}.judged.locked_ref.opus48.jsonl' --out hint_ablation.txt

Inputs per model (latest row per puzzle; retried rows supersede their errored first attempt):
    data/eval/<model>.pilot255.judged.<tag>.jsonl          condition 'with'     (hints included, the main result files)
    data/eval/<model>.pilot255.nohints.judged.<tag>.jsonl  condition 'without'  (spoiler blocks removed from the question)
Hint blocks are counted from --v1 (question_hints). Puzzles with no hint block get the identical prompt in both
conditions, so for them the comparison is a same-prompt re-run and measures run-to-run noise, not the hints; the
ablation proper is the subset with >= 1 hint block. Everything is paired on puzzles that are non-errored in BOTH
conditions. Prints the report and writes it to --out (default data/eval/hint_ablation.summary.txt).
Standard library only.
"""
import argparse
import json
import math
import os
import sys
from datetime import date

DEFAULT_MODELS = ["gpt-6.1-sol", "claude-opus-5.5"]
LABELS = {"gpt-6.1-sol": "GPT-6.1 Sol", "claude-opus-5.5": "Claude Opus 5.5", "gpt-5.6-sol": "GPT-5.6 Sol",
          "claude-opus-4-8": "Claude Opus 4.8", "claude-sonnet-5.5": "Claude Sonnet 5.5", "claude-sonnet-5": "Claude Sonnet 5",
          "claude-haiku-4-5": "Claude Haiku 4.5", "kimi-k3": "Kimi K3", "grok-4.7": "Grok 4.7"}
PERIODS = ["<=2020", "2021-Apr 2026", "post-cutoff"]


def latest_rows(path, condition):
    """{post_id: row} keeping the last row written per puzzle (a retry supersedes an errored attempt)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                x = json.loads(ln)
                if x["condition"] == condition:
                    out[x["post_id"]] = x
    return out


def period(rec, cutoff):
    d = str(rec.get("creation_date") or "")
    if d >= cutoff:
        return "post-cutoff"
    if (rec.get("year") or 0) >= 2021:
        return "2021-Apr 2026"
    return "<=2020"


def binom_two_sided(k, n):
    """Exact two-sided sign test on the discordant pairs (only-with vs only-without), p = 0.5."""
    if n == 0:
        return float("nan")
    pk = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pk if p <= pk[k] + 1e-12))


def cell(ids, w, wo):
    """Paired statistics on a set of puzzle ids."""
    n = len(ids)
    cw = sum(1 for p in ids if w[p]["correct"])
    cwo = sum(1 for p in ids if wo[p]["correct"])
    both = sum(1 for p in ids if w[p]["correct"] and wo[p]["correct"])
    only_w = sum(1 for p in ids if w[p]["correct"] and not wo[p]["correct"])
    only_wo = sum(1 for p in ids if wo[p]["correct"] and not w[p]["correct"])
    neither = n - both - only_w - only_wo
    aw = 100.0 * cw / n if n else float("nan")
    awo = 100.0 * cwo / n if n else float("nan")
    tw = sum(w[p].get("completion_tokens") or 0 for p in ids) / n if n else float("nan")
    two = sum(wo[p].get("completion_tokens") or 0 for p in ids) / n if n else float("nan")
    return {"n": n, "with": aw, "without": awo, "delta": awo - aw, "both": both, "only_with": only_w,
            "only_without": only_wo, "neither": neither, "p": binom_two_sided(only_w, only_w + only_wo),
            "tok_with": tw, "tok_without": two}


def fmt(c, name):
    if c["n"] == 0:
        return f"  {name:<28} n=0"
    return (f"  {name:<28} n={c['n']:<4} with {c['with']:5.1f}%  without {c['without']:5.1f}%  delta {c['delta']:+5.1f} pts   "
            f"both {c['both']:<3} only-with {c['only_with']:<3} only-without {c['only_without']:<3} neither {c['neither']:<3}"
            f"  sign-test p={'n/a' if math.isnan(c['p']) else format(c['p'], '.2f')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--tag", default="canon2.opus48", help="judge/reference tag of the judged files")
    ap.add_argument("--v1", default="data/v1/puzzles_canon.v2.pilot255.jsonl")
    ap.add_argument("--cutoff", default="2026-05-01")
    ap.add_argument("--out", default="data/eval/hint_ablation.summary.txt")
    ap.add_argument("--with-pattern", default="data/eval/{model}.pilot255.judged.{tag}.jsonl")
    ap.add_argument("--without-pattern", default="data/eval/{model}.pilot255.nohints.judged.{tag}.jsonl")
    ap.add_argument("--pilot-ids", default=None, help="JSONL with post_id per line: restrict --v1 to these puzzles")
    args = ap.parse_args()

    keep = None
    if args.pilot_ids:
        keep = {json.loads(ln)["post_id"] for ln in open(args.pilot_ids, encoding="utf-8") if ln.strip()}
    recs = {}
    with open(args.v1, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                if keep is None or r["post_id"] in keep:
                    recs[r["post_id"]] = r
    n_hint = {p: len(r.get("question_hints") or []) for p, r in recs.items()}

    L = []
    P = L.append
    P(f"HINT ABLATION on PILOT-255   judge/reference tag: {args.tag}   generated {date.today().isoformat()}")
    P(f"with    = {args.with_pattern.format(model='<model>', tag=args.tag)}  (condition 'with')")
    P(f"without = {args.without_pattern.format(model='<model>', tag=args.tag)}  (condition 'without')")
    P(f"hint blocks from {args.v1} (question_hints); periods: <=2020 | 2021-Apr 2026 | post-cutoff (creation_date >= {args.cutoff})")
    hinted = sorted(p for p in recs if n_hint[p] >= 1)
    P(f"pilot puzzles: {len(recs)}; with >=1 hint block: {len(hinted)} (1 block: {sum(1 for p in hinted if n_hint[p] == 1)}, "
      f">=2 blocks: {sum(1 for p in hinted if n_hint[p] >= 2)}); the other {len(recs) - len(hinted)} get the identical prompt in both "
      f"conditions and serve as a same-prompt re-run (noise) baseline")
    P("paired on puzzles with error == null in BOTH conditions; accuracy = tier-1 match or judge MATCH; delta = without - with")
    P("")

    latex_rows, latex_rows_rerun = [], []
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        fw = args.with_pattern.format(model=m, tag=args.tag)
        fwo = args.without_pattern.format(model=m, tag=args.tag)
        for fp in (fw, fwo):
            if not os.path.exists(fp):
                sys.exit(f"missing {fp}")
        w, wo = latest_rows(fw, "with"), latest_rows(fwo, "without")
        err_w = sum(1 for x in w.values() if x.get("error"))
        err_wo = sum(1 for x in wo.values() if x.get("error"))
        paired = sorted(p for p in w if p in wo and w[p].get("error") is None and wo[p].get("error") is None)
        sets = {
            "ALL PAIRED": paired,
            "HINTED (>=1 block)": [p for p in paired if n_hint[p] >= 1],
            "  1 hint block": [p for p in paired if n_hint[p] == 1],
            "  >=2 hint blocks": [p for p in paired if n_hint[p] >= 2],
            "NO HINT (same-prompt re-run)": [p for p in paired if n_hint[p] == 0],
        }
        P(f"== {m} ({LABELS.get(m, m)}) ==")
        P(f"  rows: with {len(w)} puzzles ({err_w} errored)   without {len(wo)} puzzles ({err_wo} errored)   paired n={len(paired)}")
        cells = {k: cell(v, w, wo) for k, v in sets.items()}
        for k in sets:
            P(fmt(cells[k], k))
        P("  by period, hinted puzzles:")
        for per in PERIODS:
            ids = [p for p in sets["HINTED (>=1 block)"] if period(recs[p], args.cutoff) == per]
            P(fmt(cell(ids, w, wo), "    " + per))
        P("  by period, all paired puzzles:")
        for per in PERIODS:
            ids = [p for p in paired if period(recs[p], args.cutoff) == per]
            P(fmt(cell(ids, w, wo), "    " + per))
        c_all, c_h = cells["ALL PAIRED"], cells["HINTED (>=1 block)"]
        P(f"  mean completion tokens (paired rows): all  with {c_all['tok_with']:,.0f}  without {c_all['tok_without']:,.0f};"
          f"   hinted  with {c_h['tok_with']:,.0f}  without {c_h['tok_without']:,.0f}")
        P(f"  errored rows: with {err_w}, without {err_wo} (latest attempt per puzzle)")
        P("")
        c_r = cells["NO HINT (same-prompt re-run)"]
        latex_rows.append(f"{LABELS.get(m, m)} & {c_h['n']} & {c_h['with']:.1f} & {c_h['without']:.1f} & ${c_h['delta']:+.1f}$ & "
                          f"{c_h['only_with']} & {c_h['only_without']} \\\\")
        latex_rows_rerun.append(f"{LABELS.get(m, m)} & {c_r['n']} & {c_r['with']:.1f} & {c_r['without']:.1f} & ${c_r['delta']:+.1f}$ & "
                                f"{c_r['only_with']} & {c_r['only_without']} \\\\")

    P("LaTeX (booktabs):")
    P("\\begin{tabular}{@{}lrrrrrr@{}}")
    P("\\toprule")
    P("Model & $n$ & With hints & Without & $\\Delta$ & Only with & Only without \\\\")
    P("\\midrule")
    P("\\multicolumn{7}{@{}l}{\\emph{Puzzles with at least one hint block}} \\\\")
    L.extend(latex_rows)
    P("\\midrule")
    P("\\multicolumn{7}{@{}l}{\\emph{Puzzles without hints (identical prompt; run-to-run noise)}} \\\\")
    L.extend(latex_rows_rerun)
    P("\\bottomrule")
    P("\\end{tabular}")

    text = "\n".join(L) + "\n"
    print(text, end="")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
