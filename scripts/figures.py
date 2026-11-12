"""Figures for the scorer-minutes paper, from docs/paper/aggregates/ only (no run folders needed).

  .venv/bin/python scripts/paper/figures.py [--out docs/paper/figures]

fig2_control.png: the section 4.2 control from docs/paper/aggregates/control_43.csv: solved tasks and sessions cut by
the cap per setup (same agent, same 55 tasks), the scorer row highlighted, the unconverted run in a contrasting color.
fig1_speed.png: per-session decode speed of every full 31B run (>= 20 sessions), one dot per run, grouped by setup,
against the scorer anchor (logged sequential run on Kaggle 4x L4). The point: concurrency and serving changes move
speed as much as the GPU does, so a wall-clock cap means a different budget on each setup.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "docs/paper/aggregates"
INK, MUTED, GRID, SURFACE, MARK, ANCHOR = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb", "#2a78d6", "#eb6834"

GROUPS = [  # (label shown, test on the run label)
    ("Kaggle 4x L4, 1 session\n(scorer setup)", lambda l: "4xL4, TP4, 1 workers, 31B" in l),
    ("H100 SXM, 5 sessions", lambda l: "H100" in l and "5 workers" in l),
    ("RTX PRO 6000, 5 sessions", lambda l: "RTX PRO 6000" in l and "5 workers" in l and "LoRA" not in l and "attempts" not in l),
    ("RTX PRO 6000, 5 sessions\n+ LoRA adapter", lambda l: "RTX PRO 6000" in l and "LoRA" in l),
    ("A100 80 GB, 5 sessions", lambda l: "A100" in l and "5 workers" in l and "LoRA" not in l),
    ("A100 80 GB, 5 sessions\n+ LoRA adapter", lambda l: "A100" in l and "LoRA" in l),
    ("L40S, 5 sessions", lambda l: "L40S" in l and "5 workers" in l),
    ("Kaggle 4x L4, 10 sessions", lambda l: "4xL4, TP4, 10 workers, 31B" in l),
]


def fig2_label(r: dict) -> str:
    """Plain-language row label: machine, sessions in parallel, and the time limit with its factor."""
    gpu, sessions = r["setup"].split(", ")[:2]
    gpu = "4 L4 GPUs" if "4x L4" in gpu else gpu.replace(" SXM", "") + " GPU"
    cap = f"{float(r['wall_cap_min']):g} min"
    if "scorer" in r["setup"]:
        return f"Scorer setup\n{gpu}, 1 session at a time\ntime limit {cap}"
    kind = "unconverted" if "nominal" in r["setup"] else "converted"
    return f"{gpu}, {sessions} at once\n{kind} limit {cap} (f = {float(r['factor']):.2f})"


def fig2(out_dir: Path) -> Path:
    from matplotlib.patches import Patch
    rows = [r for r in csv.DictReader(open(AGG / "control_43.csv")) if "earlier reference" not in r["setup"]]
    labels = [fig2_label(r) for r in rows]
    colors = [ANCHOR if "nominal" in r["setup"] else (INK if "scorer" in r["setup"] else MARK) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), dpi=200, facecolor=SURFACE, sharey=True)
    panels = ((axes[0], "solved", "Tasks solved", "Number of tasks solved (out of 55)", 25),
              (axes[1], "timed_out", "Sessions that ran out of time", "Number of sessions stopped by the time limit (out of 55)", 55))
    for ax, key, title, xlabel, xmax in panels:
        ax.set_facecolor(SURFACE)
        vals = [int(r[key]) for r in rows]
        ys = list(range(len(rows)))[::-1]
        ax.barh(ys, vals, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
        for y, v in zip(ys, vals):
            ax.text(v + xmax * 0.015, y, str(v), va="center", color=INK, fontsize=9.5)
        ax.set_xlim(0, xmax)
        ax.set_title(title, color=INK, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlabel(xlabel, color=INK, fontsize=9.5, labelpad=6)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(range(len(rows)))[::-1], labels, fontsize=8.5, color=INK)
    fig.suptitle("Same agent, same 55 tasks, six setups: converted time limits match the scorer, the unconverted limit does not",
                 color=INK, x=0.01, ha="left", fontsize=11)
    fig.legend(handles=[Patch(color=INK, label="Scorer setup (reference)"),
                        Patch(color=MARK, label="Other machine, time limit converted with the speed factor f"),
                        Patch(color=ANCHOR, label="Other machine, scorer's 7-minute limit kept unconverted")],
               loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=3, frameon=False, fontsize=8.5, labelcolor=INK,
               handlelength=1.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = out_dir / "fig2_control.png"
    fig.savefig(out, facecolor=SURFACE)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/paper/figures")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(AGG / "speed_table.csv")))
    anchor = next(float(r["median_tok_s"]) for r in rows if "v11m-seq5" in r["run"])
    a.out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    labels, medians = [], []
    for i, (name, test) in enumerate(GROUPS):
        pts = [float(r["median_tok_s"]) for r in rows
               if test(r["label"]) and int(r["sessions"]) >= 20]
        y = len(GROUPS) - 1 - i
        jit = [((k % 3) - 1) * 0.12 for k in range(len(pts))]  # small vertical offsets so overlapping runs stay visible
        ax.scatter(pts, [y + j for j in jit], s=40, color=MARK, alpha=0.85, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        if pts:
            medians.append(statistics.median(pts))
            ax.text(57, y, f"{statistics.median(pts):.1f} tok/s (n={len(pts)})", color=INK, fontsize=8.5,
                    va="center")
        labels.append((y, re.sub(r"\s+$", "", name)))
    ax.axvline(anchor, color=ANCHOR, lw=1.5, ls=(0, (4, 3)), zorder=2)
    ax.text(anchor, len(GROUPS) - 0.45, f"scorer reference {anchor:.1f}", color=ANCHOR, fontsize=8.5, ha="center")
    ax.set_yticks([y for y, _ in labels], [n for _, n in labels], fontsize=8.5, color=INK)
    ax.set_xlim(0, 56)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_ylim(-0.6, len(GROUPS) - 0.2)
    ax.set_xlabel("generation speed per session, tokens/s (one dot per run: median over its sessions)", color=MUTED, fontsize=9)
    ax.set_title(f"Gemma 4 31B: the same model, a {max(medians) / min(medians):.1f}x spread in per-session speed", color=INK, loc="left", fontsize=11)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    out = a.out / "fig1_speed.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out.relative_to(ROOT))
    print(fig2(a.out).relative_to(ROOT))


if __name__ == "__main__":
    main()
