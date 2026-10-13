"""Draw the README architecture diagram of the formal v1 three-tower model (documentation only).

Run:  uv run python scripts/draw_architecture.py
Output: report/figures/arch_threetower.png

The diagram follows src/kuairand_rec/models/towers.py and models/loss.py (formal v1):
  * MusicTower: music ID (NO_MUSIC for music_id 0) + music type -> MLP + LN -> m_base (128-d).
    ONE instance; ThreeTowerModel._music_forward runs it once per catalog forward and the video
    tower gathers its music input from that same output (no second music tower).
  * VideoTower: video ID + author ID + gathered m_base + metadata (type, upload type/date, music
    type, tags, log duration/width/height + missing flags) -> MLP + LN -> v_base (128-d).
  * UserTower: user ID + HistoryEncoder (2-layer Transformer + attention pooling over v_base of the
    last 100 strictly-past events with feedback / play-ratio / recency / position) [+ 21 U1 snapshot
    fields] -> MLP + LN -> u_base (128-d).
  * Heads: Linear(128 -> 64) + L2 per task; user 4, video 2, music 2. Score = cosine.
  * Loss: per task, full softmax over the train-vocab candidates (tau = 0.05) with same-user
    false-negative masking; equal-weight sum of the 4 task losses.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "report" / "figures" / "arch_threetower.png"

BLUE, BLUE_E = "#e8f0fc", "#2f5fb3"
GREEN, GREEN_E = "#e9f5ec", "#3b8a52"
PURP, PURP_E = "#f1ebfa", "#6b4fa0"
GREY, GREY_E = "#f4f4f4", "#7a7a7a"
OPT, OPT_E = "#fff7e6", "#c27c0e"


def box(ax, x, y, w, h, text, fc, ec, ls="-", bold_first=True, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                fc=fc, ec=ec, lw=1.2, ls=ls))
    lines = text.split("\n")
    n = len(lines)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h / 2 + (n - 1 - 2 * i) * 0.0125, ln, ha="center", va="center", fontsize=fs,
                fontweight="bold" if (bold_first and i == 0) else "normal", color="#1a1a1a")


def arrow(ax, p, q, color="#444", ls="-", rad=0.0, text=None, tpos=0.5, toff=(0, 0), fs=7.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.1, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if text:
        x = p[0] + (q[0] - p[0]) * tpos + toff[0]
        y = p[1] + (q[1] - p[1]) * tpos + toff[1]
        ax.text(x, y, text, fontsize=fs, color=color, ha="center", va="center", style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6))


def main():
    fig = plt.figure(figsize=(11, 6.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column x positions: music | video | user
    W = 0.25
    xm, xv, xu = 0.04, 0.375, 0.71

    # Inputs (top row)
    box(ax, xm, 0.80, W, 0.13, "Music inputs\nmusic ID (NO_MUSIC if id = 0)\nmusic type", GREY, GREY_E)
    box(ax, xv, 0.80, W, 0.13, "Video inputs\nvideo ID  ·  author ID\ntype · upload · tags · duration · size",
        GREY, GREY_E)
    box(ax, xu, 0.80, W, 0.13, "Strict-past history\nlast 100 events before t_pred\nvideo + feedback + recency",
        GREY, GREY_E)

    # Towers
    box(ax, xm, 0.56, W, 0.12, "Music tower\none shared instance\nMLP + LayerNorm", GREEN, GREEN_E)
    box(ax, xv, 0.56, W, 0.12, "Video tower\nMLP + LayerNorm", GREEN, GREEN_E)
    box(ax, xu, 0.62, W, 0.11, "History encoder\n2-layer Transformer\n+ attention pooling", GREEN, GREEN_E, fs=7.8)
    box(ax, xu + W * 0.30, 0.49, W * 0.28, 0.085, "User ID", GREY, GREY_E)
    box(ax, xu + W * 0.62, 0.49, W * 0.38, 0.085, "U1 only\n21 snapshot fields\n(unverified)",
        OPT, OPT_E, ls="--", fs=6.8)
    box(ax, xu, 0.355, W, 0.10, "User tower\nMLP + LayerNorm", GREEN, GREEN_E)

    # Base embeddings
    box(ax, xm, 0.40, W, 0.075, "m_base  (128-d)", BLUE, BLUE_E)
    box(ax, xv, 0.40, W, 0.075, "v_base  (128-d)", BLUE, BLUE_E)
    box(ax, xu, 0.245, W, 0.075, "u_base  (128-d)", BLUE, BLUE_E)

    # Heads
    box(ax, xm, 0.215, W, 0.11, "Music heads\nlong view · like\nLinear 128→64 + L2", PURP, PURP_E, fs=7.8)
    box(ax, xv, 0.215, W, 0.11, "Video heads\nlong view · like\nLinear 128→64 + L2", PURP, PURP_E, fs=7.8)
    box(ax, xu, 0.085, W, 0.125, "User heads (4)\nvideo LV · video like\nmusic LV · music like\nLinear 128→64 + L2",
        PURP, PURP_E, fs=7.6)

    # Scoring
    box(ax, 0.17, 0.03, 0.48, 0.115,
        "Score = cosine(user head_task, item head_task), per task\nfull-catalog top-k: video LV · video like · music LV · music like\n"
        "training: full softmax over train-vocab items (τ = 0.05), same-user FN mask, 4 losses summed",
        "#ffffff", "#333333", fs=7.6)

    # Arrows: inputs -> towers -> bases -> heads
    for x in (xm, xv):
        arrow(ax, (x + W / 2, 0.80), (x + W / 2, 0.682))
        arrow(ax, (x + W / 2, 0.56), (x + W / 2, 0.477))
        arrow(ax, (x + W / 2, 0.40), (x + W / 2, 0.327))
    # shared music output reused by the video tower
    arrow(ax, (xm + W, 0.44), (xv, 0.60), color=GREEN_E, rad=-0.25,
          text="same m_base reused\nas video input", tpos=0.45, toff=(0.0, 0.035))
    # history tokens come from v_base
    arrow(ax, (xv + W, 0.455), (xu, 0.665), color=BLUE_E, rad=0.2, text="history tokens\n= v_base", tpos=0.45,
          toff=(0.0, -0.05))
    # user side
    arrow(ax, (xu + W / 2, 0.80), (xu + W / 2, 0.732))
    arrow(ax, (xu + W * 0.13, 0.62), (xu + W * 0.13, 0.457), color="#444")
    arrow(ax, (xu + W * 0.44, 0.49), (xu + W * 0.44, 0.457))
    arrow(ax, (xu + W * 0.81, 0.49), (xu + W * 0.81, 0.457), color=OPT_E, ls="--")
    arrow(ax, (xu + W / 2, 0.355), (xu + W / 2, 0.322))
    arrow(ax, (xu + W / 2, 0.245), (xu + W / 2, 0.212))
    # heads -> score
    arrow(ax, (xm + W / 2, 0.215), (0.30, 0.147))
    arrow(ax, (xv + W / 2, 0.215), (0.48, 0.147))
    arrow(ax, (xu, 0.12), (0.652, 0.10))

    ax.text(0.5, 0.975, "Formal v1 three-tower model (U0 = without the dashed U1 input)", ha="center", va="center",
            fontsize=10.5, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
