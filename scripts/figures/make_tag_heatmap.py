#!/usr/bin/env python3
"""
Figure: per-tag accuracy heatmap on Pilot-255 (Opus 4.8 judge, locked canonical-v2 reference).

Reads the twelve tag rows of the per-tag table (Table 6, \\label{tab:tags}) directly from acl_latex.tex,
so the cells are the published table numbers, and writes tag_heatmap.pdf (vector, ACL column width)
next to this script. Columns follow the Table 1 model order.

    python3 make_tag_heatmap.py [--tex PATH] [--png]

Sequential single-hue (blue) ramp, light = low accuracy; every cell also prints its value, so the
figure does not depend on colour. Requires matplotlib (`uv run --with matplotlib python make_tag_heatmap.py`).
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["GPT-6.1", "Opus 5.5", "Opus 4.8", "Kimi K3", "GPT-5.6", "Grok 4.7", "Sonnet 5", "Sonnet 5.5", "Haiku 4.5"]
# one-hue sequential ramp (steps 100 -> 700)
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
ROW = re.compile(r"^\\textit\{([a-z-]+)\}\s*&\s*(\d+)\s*&(.*)\\\\\s*$")


def read_table(tex):
    rows, in_tab = [], False
    for ln in open(tex, encoding="utf-8"):
        if r"\label{tab:tags}" in ln:
            break
        if ln.startswith(r"\begin{tabular}{@{}lrrrrrrrrrr@{}}"):
            in_tab = True
            continue
        m = ROW.match(ln.strip()) if in_tab else None
        if m:
            cells = [c.strip() for c in m.group(3).split("&")]
            vals = [float(re.sub(r"\\textbf\{([\d.]+)\}", r"\1", c)) for c in cells]
            assert len(vals) == len(MODELS), (m.group(1), vals)
            rows.append((m.group(1), int(m.group(2)), vals))
    assert len(rows) == 12, len(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=os.path.join(HERE, "..", "acl_latex.tex"))
    ap.add_argument("--out", default=os.path.join(HERE, "tag_heatmap.pdf"))
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    rows = read_table(args.tex)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
    })
    cmap = LinearSegmentedColormap.from_list("blues", RAMP)
    data = [vals for _, _, vals in rows]
    vmin, vmax = 10, 95

    fig, ax = plt.subplots(figsize=(3.03, 2.9))
    ax.pcolormesh(data, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="white", linewidth=1.2)
    ax.invert_yaxis()
    for i, (_, _, vals) in enumerate(rows):
        for j, v in enumerate(vals):
            ax.text(j + 0.5, i + 0.5, f"{v:.0f}", ha="center", va="center", fontsize=6.3,
                    color="white" if v >= 58 else INK)
    ax.set_xticks([j + 0.5 for j in range(len(MODELS))])
    ax.set_xticklabels(MODELS, rotation=45, ha="left", rotation_mode="anchor", fontsize=6.6, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([f"{tag} ({n})" for tag, n, _ in rows], fontsize=6.6, color=INK)
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.subplots_adjust(left=0.355, right=0.935, top=0.80, bottom=0.01)
    fig.savefig(args.out, metadata={"Title": "Per-tag accuracy heatmap", "Author": "", "Creator": "matplotlib"})
    if args.png:
        fig.savefig(re.sub(r"\.pdf$", ".png", args.out), dpi=300)
    for tag, n, vals in rows:
        print(f"{tag:20s} n={n:3d} " + " ".join(f"{v:5.1f}" for v in vals))
    print("wrote", os.path.relpath(args.out))


if __name__ == "__main__":
    main()
