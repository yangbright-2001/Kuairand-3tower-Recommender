"""Figures for S9 Wave 3 (validation only, post-hoc exploratory) from results/explore/wave3/*.csv.

  uv run python scripts/explore/plot_wave3.py
Writes results/explore/wave3/figures/{fig1_u0_vs_nomusic_r20.png, fig2_nomusic_minus_u0.png,
fig3_video_lv_epochs.png}. Inputs are the val-only CSVs written by analyze_wave3.py; no test data.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_explore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kuairand_rec import paths

D = paths.PROJECT_ROOT / "results" / "explore" / "wave3"
TASKS = ["video_long_view", "music_long_view", "video_is_like", "music_is_like"]
LABEL = {"video_long_view": "video / long view", "music_long_view": "music / long view",
         "video_is_like": "video / like", "music_is_like": "music / like"}
TAG = "VALIDATION ONLY, post-hoc exploratory (normal val masked frames at T_val)"


def main() -> int:
    df = pd.read_csv(D / "validation_comparison.csv")
    pdf = pd.read_csv(D / "paired_nomusic_minus_u0.csv")
    ep = pd.read_csv(D / "validation_by_epoch.csv")
    boot = pd.read_csv(D / "validation_bootstrap.csv")
    out = D / "figures"
    out.mkdir(exist_ok=True)

    # Fig 1: val R@20 per task, 3 seeds per architecture (dots) + mean (bar); ALS reference line.
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharey=False)
    als = boot[(boot.metric == "level_recall@20") & (boot.model_a == "als (pure)")].set_index("task").mean_a
    for ax, t in zip(axes, TASKS + ["mean4"]):
        col = "mean_val_recall@20" if t == "mean4" else f"{t}/val_recall@20"
        for i, arch in enumerate(["U0", "no-music"]):
            v = df[df.arch == arch].sort_values("seed")[col].to_numpy()
            ax.bar(i, v.mean(), color=["#8c9fb5", "#d98c5f"][i], alpha=0.55, width=0.6)
            ax.scatter(np.full(3, i) + np.array([-0.12, 0, 0.12]), v, color="k", s=14, zorder=3)
        if t in als.index and "long_view" in t:
            ax.axhline(als[t], color="#2a7d2a", ls="--", lw=1, label="pure ALS (val)")
            ax.legend(fontsize=7, loc="lower right")
        lo = df[col].min()
        ax.set_ylim(lo - 0.012, df[col].max() + 0.006)
        ax.set_xticks([0, 1], ["U0", "no-music"])
        ax.set_title("4-task mean" if t == "mean4" else LABEL[t], fontsize=9)
    axes[0].set_ylabel("val Recall@20 (dots = seeds 0/1/2, bar = mean)")
    fig.suptitle(f"U0 vs no-music, 3 seeds each. {TAG}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig1_u0_vs_nomusic_r20.png", dpi=150)
    plt.close(fig)

    # Fig 2: same-seed difference no-music - U0 per task (3 seeds + mean). Same seed number is not a
    # paired initialisation (different first-layer width).
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cols = TASKS + ["mean4"]
    x = np.arange(len(cols))
    seeds = pdf[pdf.seed.isin(["0", "1", "2", 0, 1, 2])]
    for j, s in enumerate(seeds.seed):
        vals = [seeds.iloc[j][f"delta/{'mean_val_recall@20' if c == 'mean4' else c + '/val_recall@20'}"] for c in cols]
        ax.scatter(x + (j - 1) * 0.12, vals, s=22, label=f"seed {s}", zorder=3)
    mean = pdf[pdf.seed == "mean"].iloc[0]
    mv = [float(mean[f"delta/{'mean_val_recall@20' if c == 'mean4' else c + '/val_recall@20'}"]) for c in cols]
    ax.bar(x, mv, color=["#d98c5f" if v > 0 else "#7f9cc2" for v in mv], alpha=0.45, width=0.6, label="mean of 3")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [LABEL.get(c, "4-task mean") for c in cols], fontsize=8)
    ax.set_ylabel("val R@20: no-music - U0")
    ax.legend(fontsize=7, ncol=4)
    ax.set_title(f"Same-seed differences (not paired inits). {TAG}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_nomusic_minus_u0.png", dpi=150)
    plt.close(fig)

    # Fig 3: video long-view val R@20 per epoch, all 6 runs.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, t in zip(axes, LV := ["video_long_view", "music_long_view"]):
        for (arch, seed), g in ep.groupby(["arch", "seed"]):
            g = g.sort_values("epoch")
            ax.plot(g.epoch, g[t], marker="o", ms=3, lw=1.2, color="#d98c5f" if arch == "no-music" else "#5a7393",
                    ls=["-", "--", ":"][seed], label=f"{arch} s{seed}")
        if t in als.index:
            ax.axhline(als[t], color="#2a7d2a", ls="--", lw=1, label="pure ALS")
        ax.set_title(LABEL[t], fontsize=9)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("val Recall@20")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Long-view val R@20 by epoch. {TAG}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig3_longview_epochs.png", dpi=150)
    plt.close(fig)
    print(sorted(p.name for p in out.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
