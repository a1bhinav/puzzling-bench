#!/usr/bin/env python3
"""Accuracy of two runs on the same puzzles (e.g. image attached vs. withheld) and their paired difference.

    python3 scripts/paired_accuracy.py A.jsonl B.jsonl

Each accuracy is over that file's non-errored rows; delta = A - B on the puzzles non-errored in both.
Standard library only.
"""
import json
import sys


def latest(path):
    return {r["post_id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def acc(rows):
    return 100 * sum(bool(r["correct"]) for r in rows) / len(rows) if rows else float("nan")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = latest(sys.argv[1]), latest(sys.argv[2])
    ok_a = [r for r in a.values() if not r.get("error")]
    ok_b = [r for r in b.values() if not r.get("error")]
    both = [p for p in a if p in b and not a[p].get("error") and not b[p].get("error")]
    print(f"A {acc(ok_a):.1f}% (n={len(ok_a)})   B {acc(ok_b):.1f}% (n={len(ok_b)})   "
          f"paired on {len(both)}: delta {acc([a[p] for p in both]) - acc([b[p] for p in both]):+.1f} pts")
