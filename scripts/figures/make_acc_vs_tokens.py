#!/usr/bin/env python3
"""
Figure: Pilot-255 accuracy (Opus 4.8 judge, locked canonical-v2 reference) against mean completion
length per non-errored puzzle, one labelled point per model.

Reads <model>.judged.locked_ref.opus48.jsonl (--pattern) from the eval directory; x is the mean of the per-row
`completion_tokens` field (solve call plus any forced follow-up) over non-errored rows, y the share of
those rows scored correct. Writes acc_vs_tokens.pdf (vector, ACL column width) next to this script.

    python3 make_acc_vs_tokens.py [--eval-dir DIR] [--pattern NAME] [--png]

Same encoding as by_date.pdf: colour = developer, marker = model line, hollow marker = previous
generation. Requires matplotlib (`uv run --with matplotlib python make_acc_vs_tokens.py`).
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

# stem, label, colour, marker, current generation?, label offset (points) and alignment
MODELS = [
    ("gpt-6.1-sol",       "GPT-6.1 Sol", BLUE,   "o", True,  (5, 3),   "left"),
    ("gpt-5.6-sol",       "GPT-5.6 Sol", BLUE,   "o", False, (-5, -4), "right"),
    ("kimi-k3",           "Kimi K3",     AQUA,   "P", True,  (5, 3),   "left"),
    ("claude-opus-5.5",   "Opus 5.5",    ORANGE, "^", True,  (5, 2),   "left"),
    ("claude-opus-4-8",   "Opus 4.8",    ORANGE, "^", False, (-5, 3),  "right"),
    ("grok-4.7",          "Grok 4.7",    VIOLET, "X", True,  (-5, 4),  "right"),
    ("claude-sonnet-5.5", "Sonnet 5.5",  ORANGE, "D", True,  (5, -2),  "left"),
    ("claude-sonnet-5",   "Sonnet 5",    ORANGE, "D", False, (2, -8),  "left"),
    ("claude-haiku-4-5",  "Haiku 4.5",   ORANGE, "s", True,  (5, 2),   "left"),
]


def stats(path):
    rows = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    ok = [r for r in rows if not r.get("error")]
    toks = [r.get("completion_tokens") or 0 for r in ok]
    return sum(toks) / len(ok), 100.0 * sum(1 for r in ok if r["correct"]) / len(ok), len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=os.path.join(HERE, "..", "..", "data", "hf", "results", "pilot255"))
    ap.add_argument("--pattern", default="{model}.judged.locked_ref.opus48.jsonl")
    ap.add_argument("--out", default=os.path.join(HERE, "acc_vs_tokens.pdf"))
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
    })
    fig, ax = plt.subplots(figsize=(3.03, 2.0))
    for stem, label, colour, marker, current, (dx, dy), ha in MODELS:
        x, y, n = stats(os.path.join(args.eval_dir, args.pattern.format(model=stem)))
        print(f"{label:<12} n={n:3d}  mean completion tokens {x:7.0f}  accuracy {y:5.1f}")
        size = {"o": 5.2, "^": 6.0, "D": 4.9, "s": 4.9, "P": 6.0, "X": 6.0}[marker]
        ax.plot([x], [y], linestyle="none", marker=marker, markersize=size if current else size - 0.4,
                markerfacecolor=colour if current else "white", markeredgecolor="white" if current else colour,
                markeredgewidth=0.6 if current else 1.0, zorder=3, clip_on=False)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=7, color=INK, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(1000, 12000)
    ax.set_xticks([1000, 2000, 5000, 10000])
    ax.set_xticklabels(["1K", "2K", "5K", "10K"])
    ax.set_ylim(20, 85)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_xlabel("Mean completion tokens per puzzle (log scale)", color=INK, fontsize=8, labelpad=2)
    ax.set_ylabel("Accuracy (%)", color=INK, fontsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0, colors=MUTED, labelsize=7.5)
    ax.tick_params(axis="x", which="both", length=2.5, color=AXIS, labelsize=7.5)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xticks([], minor=True)
    fig.subplots_adjust(left=0.135, right=0.975, top=0.96, bottom=0.19)
    fig.savefig(args.out, metadata={"Title": "Accuracy vs completion length", "Author": "", "Creator": "matplotlib"})
    if args.png:
        fig.savefig(re.sub(r"\.pdf$", ".png", args.out), dpi=300)
    print("wrote", os.path.relpath(args.out))


if __name__ == "__main__":
    main()
