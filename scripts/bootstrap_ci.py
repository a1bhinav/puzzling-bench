#!/usr/bin/env python3
"""Accuracy over non-errored rows with a percentile bootstrap 95% interval (1,000 resamples of the puzzles).

    python3 scripts/bootstrap_ci.py data/hf/results/pilot255/*.judged.locked_ref.opus48.jsonl [--seed 0]

Standard library only.
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for path in args.files:
        ok = [bool(r["correct"]) for r in map(json.loads, open(path, encoding="utf-8")) if not r.get("error")]
        rng = random.Random(args.seed)
        accs = sorted(100 * sum(rng.choices(ok, k=len(ok))) / len(ok) for _ in range(args.resamples))
        lo, hi = accs[int(0.025 * args.resamples)], accs[int(0.975 * args.resamples) - 1]
        print(f"{100 * sum(ok) / len(ok):5.1f}%  95% CI {lo:.1f}-{hi:.1f}  n={len(ok)}  {path}")


if __name__ == "__main__":
    main()
