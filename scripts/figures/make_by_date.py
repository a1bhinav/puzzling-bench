#!/usr/bin/env python3
"""
Figure: accuracy by posting period on Pilot-255 (Opus 4.8 judge, locked canonical-v2 reference).

Reads the BY YEAR BUCKET block of each
    <model>.judged.locked_ref.opus48.jsonl.summary.txt   (written by: scripts/run_eval.py --summarize FILE)
and writes by_date.pdf (vector, ACL single-column width) next to this script.

The 2021-2024 (n=28), 2025 (n=5) and 2026 pre-cutoff (n=3) buckets are merged into one
"2021 - Apr 2026" bucket (n=36). Counts are reconstructed from the summary lines:
non-errored = n - errors, correct = round(pct * non-errored / 100).

    python3 make_by_date.py [--eval-dir DIR] [--pattern NAME] [--png]

Colour = developer (4 hues; every pair stays separated under simulated protanopia and
deuteranopia on a white page); marker shape = model line; hollow marker and thinner line =
previous generation of the same line, so no series relies on colour alone. Requires matplotlib
(e.g. `uv run --with matplotlib python make_by_date.py`).
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUCKETS = ["<=2020", "2021-2024", "2025", "2026 pre-cutoff", "post-cutoff"]

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#f4f3f0"

# file stem, legend label, colour (developer), marker (model line), current generation?
# Listed in legend order (column-wise, three per column): one model line per row pair.
MODELS = [
    ("gpt-6.1-sol",       "GPT-6.1 Sol", BLUE,   "o", True),
    ("gpt-5.6-sol",       "GPT-5.6 Sol", BLUE,   "o", False),
    ("kimi-k3",           "Kimi K3",     AQUA,   "P", True),
    ("claude-opus-5.5",   "Opus 5.5",    ORANGE, "^", True),
    ("claude-opus-4-8",   "Opus 4.8",    ORANGE, "^", False),
    ("grok-4.7",          "Grok 4.7",    VIOLET, "X", True),
    ("claude-sonnet-5.5", "Sonnet 5.5",  ORANGE, "D", True),
    ("claude-sonnet-5",   "Sonnet 5",    ORANGE, "D", False),
    ("claude-haiku-4-5",  "Haiku 4.5",   ORANGE, "s", True),
]

LINE = re.compile(r"^\s{4}(\S.*?)\s+n=(\d+)\s+tier1-only\s+[\d.]+%\s+tier1\+judge\s+([\d.]+)%.*errors (\d+)")


def read_buckets(path):
    """bucket -> (n_rows, n_non_errored, n_correct) from the BY YEAR BUCKET block of a summary file."""
    out, on = {}, False
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("BY YEAR BUCKET"):
            on = True
            continue
        if ln.startswith("BY TAG"):
            break
        m = LINE.match(ln) if on else None
        if m:
            n, pct, err = int(m.group(2)), float(m.group(3)), int(m.group(4))
            ok = n - err
            out[m.group(1).strip()] = (n, ok, round(pct * ok / 100))
    missing = [b for b in BUCKETS if b not in out]
    if missing:
        raise SystemExit(f"{path}: missing buckets {missing}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=os.path.join(HERE, "..", "..", "data", "hf", "results", "pilot255"))
    ap.add_argument("--pattern", default="{model}.judged.locked_ref.opus48.jsonl.summary.txt")
    ap.add_argument("--out", default=os.path.join(HERE, "by_date.pdf"))
    ap.add_argument("--png", action="store_true", help="also write a 300-dpi PNG preview next to the PDF")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # embed TrueType, no Type-3 fonts
        "axes.linewidth": 0.5,
    })

    series, n_rows = [], None
    for stem, label, colour, marker, current in MODELS:
        b = read_buckets(os.path.join(args.eval_dir, args.pattern.format(model=stem)))
        groups = [["<=2020"], ["2021-2024", "2025", "2026 pre-cutoff"], ["post-cutoff"]]
        ns = [sum(b[k][0] for k in g) for g in groups]
        ok = [sum(b[k][1] for k in g) for g in groups]
        corr = [sum(b[k][2] for k in g) for g in groups]
        if n_rows is None:
            n_rows = ns
        assert ns == n_rows, (stem, ns, n_rows)
        acc = [100.0 * c / o for c, o in zip(corr, ok)]
        series.append((label, colour, marker, current, acc, corr, ok))
        print(f"{label:<12} " + "   ".join(f"{c:>3}/{o:<3} = {a:5.1f}%" for c, o, a in zip(corr, ok, acc)))

    fig, ax = plt.subplots(figsize=(3.03, 2.62))
    x = [0, 1, 2]
    ax.axvspan(1.5, 2.34, color=BAND, linewidth=0, zorder=0)          # the held-out (post-cutoff) period
    ax.text(1.92, 88.5, "post-cutoff", ha="center", va="top", fontsize=7, color=MUTED)
    handles = []
    for label, colour, marker, current, acc, _, _ in series:
        size = {"o": 5.0, "^": 5.9, "D": 4.8, "s": 4.8, "P": 5.8, "X": 5.8}[marker]
        style = dict(marker=marker,
                     markersize=size if current else size - 0.5,
                     markerfacecolor=colour if current else "white",
                     markeredgecolor="white" if current else colour,      # white ring keeps overlapping marks apart
                     markeredgewidth=0.6 if current else 1.0)
        lw = 1.25 if current else 0.85                                  # thinner line + hollow marker = previous generation
        ax.plot(x, acc, color=colour, linewidth=lw, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        ax.plot(x, acc, linestyle="none", clip_on=False,                # hollow below filled below the thin + and x marks
                zorder=5 if marker in "PX" else (4 if current else 3), **style)
        handles.append(Line2D([0], [0], color=colour, linewidth=lw, label=label, **style))

    ax.set_xlim(-0.34, 2.34)
    ax.set_ylim(10, 90)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_xticks(x)
    ax.set_xticklabels([f"\u22642020\n$n={n_rows[0]}$",
                        f"2021\u2013Apr. 2026\n$n={n_rows[1]}$",
                        f"May\u2013Jun. 2026\n$n={n_rows[2]}$"], color=INK)
    ax.set_ylabel("Accuracy (%)", color=INK, fontsize=8)
    ax.set_xlabel("Posting period", color=INK, fontsize=8, labelpad=2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0, colors=MUTED, labelsize=7.5)
    ax.tick_params(axis="x", length=2.5, color=AXIS, labelsize=7.5)

    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5, handlelength=2.1,
                     columnspacing=1.3, handletextpad=0.5, labelspacing=0.35, bbox_to_anchor=(0.53, 0.0))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.975, bottom=0.375)

    fig.savefig(args.out, metadata={"Title": "Accuracy by posting period", "Author": "", "Creator": "matplotlib"})
    if args.png:
        fig.savefig(re.sub(r"\.pdf$", ".png", args.out), dpi=300)
    print("wrote", os.path.relpath(args.out))


if __name__ == "__main__":
    main()
