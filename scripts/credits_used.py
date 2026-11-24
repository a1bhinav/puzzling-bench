#!/usr/bin/env python3
"""
How many credits / tokens did a results file cost? Sums the per-call usage that the endpoint records in
run_meta (solver calls, including the forced follow-up) and judge_meta (judge calls).

    python3 scripts/credits_used.py results/pilot255/claude-opus-5.5.judged.locked_ref.opus48.jsonl [more files]

Harness A reports cost as `credit_units` (1e-9 of a credit) and `billed_requests` (requests weighted by the
per-request multiplier `request_cost`).
Rows without run_meta (harness B runs, mock) contribute tokens only. Standard library only.
"""
import json
import sys


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    grand = {"rows": 0, "solver_calls": 0, "judge_calls": 0, "solver_units": 0, "judge_units": 0,
             "solver_in": 0, "solver_out": 0, "judge_in": 0, "judge_out": 0, "billed": 0}
    for path in sys.argv[1:]:
        t = {k: 0 for k in grand}
        latest = {}
        for ln in open(path, encoding="utf-8"):
            if ln.strip():
                x = json.loads(ln)
                latest[(x["post_id"], x["condition"])] = x
        for x in latest.values():
            t["rows"] += 1
            metas = [c.get("run_meta") for c in (x.get("calls") or []) if c.get("run_meta")] or ([x["run_meta"]] if x.get("run_meta") else [])
            for m in metas:
                u = m.get("usage") or {}
                t["solver_calls"] += 1
                t["solver_units"] += u.get("credit_units") or 0
                t["solver_in"] += (u.get("input_tokens") or 0) + (u.get("cache_read_tokens") or 0) + (u.get("cache_write_tokens") or 0)
                t["solver_out"] += u.get("output_tokens") or 0
                t["billed"] += u.get("billed_requests") or 0
            jm = x.get("judge_meta")
            if jm:
                u = jm.get("usage") or {}
                t["judge_calls"] += 1
                t["judge_units"] += u.get("credit_units") or 0
                t["judge_in"] += (u.get("input_tokens") or 0) + (u.get("cache_read_tokens") or 0) + (u.get("cache_write_tokens") or 0)
                t["judge_out"] += u.get("output_tokens") or 0
                t["billed"] += u.get("billed_requests") or 0
        for k in grand:
            grand[k] += t[k]
        report(path, t)
    if len(sys.argv) > 2:
        report("TOTAL", grand)


def report(name, t):
    s_credits, j_credits = t["solver_units"] / 1e9, t["judge_units"] / 1e9
    per_row = (s_credits + j_credits) / t["rows"] if t["rows"] else 0
    print(f"{name}")
    print(f"  rows {t['rows']:,}   solver calls {t['solver_calls']:,}  ({t['solver_in']:,} in / {t['solver_out']:,} out tokens)   "
          f"judge calls {t['judge_calls']:,}  ({t['judge_in']:,} in / {t['judge_out']:,} out)")
    print(f"  credits: solver {s_credits:,.2f}  judge {j_credits:,.2f}  total {s_credits + j_credits:,.2f}   "
          f"= {per_row:.4f} per puzzle   (billed requests: {t['billed']})")
    if t["solver_calls"] and not t["solver_units"]:
        print("  (no credit figures recorded -- the endpoint reported none)")


if __name__ == "__main__":
    main()
