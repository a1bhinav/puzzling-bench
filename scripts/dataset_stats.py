#!/usr/bin/env python3
"""Dataset statistics of the paper's Table 2 from puzzles.jsonl.

    python3 scripts/dataset_stats.py data/hf/puzzles.jsonl [--cutoff 2026-05-01]

Standard library only.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("puzzles")
    ap.add_argument("--cutoff", default="2026-05-01")
    a = ap.parse_args()
    R = [json.loads(line) for line in open(a.puzzles, encoding="utf-8") if line.strip()]
    n, post = len(R), lambda r: r["creation_date"] >= a.cutoff
    rows = [("Puzzles", n), ("  text-only question", sum(not r["has_image"] for r in R)),
            ("  strict text-only (Q and A)", sum(not r["has_image"] and not r["answer_has_image"] for r in R)),
            ("  image in question", sum(r["has_image"] for r in R)), ("  image in accepted answer", sum(r["answer_has_image"] for r in R)),
            ("With asker's hint(s)", sum(r["has_hint"] for r in R)), ("Self-answered", sum(r["self_answered"] for r in R)),
            ("Posted <=2020", sum(r["year"] <= 2020 for r in R)), ("Posted 2021-2024", sum(2021 <= r["year"] <= 2024 for r in R)),
            ("Posted 2025", sum(r["year"] == 2025 for r in R)), ("Posted 2026, before cutoff", sum(r["year"] == 2026 and not post(r) for r in R)),
            (f"Posted after cutoff ({a.cutoff})", sum(post(r) for r in R)), ("Puzzles with >1 tag", sum(len(r["tags"]) > 1 for r in R))]
    for name, k in rows:
        print(f"{name:<36} {k:>7,}  {100 * k / n:5.1f}%")
    print(f"{'Distinct tags':<36} {len({t for r in R for t in r['tags']}):>7,}")
    print(f"{'Mean tags per puzzle':<36} {sum(len(r['tags']) for r in R) / n:>7.2f}")


if __name__ == "__main__":
    main()
