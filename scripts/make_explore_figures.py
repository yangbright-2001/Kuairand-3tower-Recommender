"""Figures for the post-hoc exploratory (S9) study. VALIDATION metrics only (read-only).

Run:  uv run python scripts/make_explore_figures.py
Inputs (repo-relative, copied from branch explore/s9-opt, see results/explore/PROVENANCE.md):
  results/explore/wave2/validation_comparison.csv   best checkpoint per run, 3 seeds x 2 learning rates
  results/explore/wave2/paired_lr_diffs.csv         same-seed differences lr 5e-4 - lr 1e-3
  results/explore/wave1/validation_comparison.csv   tau 0.03 / 0.1 vs the formal U0 (seed 0)
  results/explore/wave1/validation_bootstrap.csv    paired val user-bootstrap CIs vs U0 seed 0
Outputs: report/figures/explore_fig_a_lr_mean.png, explore_fig_b_lr_task_deltas.png,
         explore_fig_c_tau_tradeoff.png
Nothing is re-scored; every plotted value is a cell of those CSVs. No test metric is read.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "results" / "explore"
OUT = ROOT / "report" / "figures"
TASKS = ["video_long_view", "video_is_like", "music_long_view", "music_is_like"]
TNAME = {"video_long_view": "video LV", "video_is_like": "video like", "music_long_view": "music LV",
         "music_is_like": "music like"}
C_LR1, C_LR5 = "#2f5fb3", "#e69138"
NOTE = "Post-hoc exploratory study · VALIDATION metrics (M3 val masked frames) · not test results"


def fig_a(v2: pd.DataFrame):
    """Mean val R@20 over the 4 tasks, per seed, lr 1e-3 vs lr 5e-4 (dots joined by seed)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    seeds = [0, 1, 2]
    for lr, c, off, lab in ((0.001, C_LR1, -0.08, "lr 1e-3 (formal setting)"), (0.0005, C_LR5, 0.08, "lr 5e-4")):
        g = v2[v2.lr == lr].set_index("seed").loc[seeds]
        ax.scatter(np.array(seeds) + off, g["mean_val_recall@20"], s=42, color=c, label=lab, zorder=3)
        m = g["mean_val_recall@20"].mean()
        ax.hlines(m, -0.35, 2.35, color=c, lw=1, ls="--", alpha=0.8)
        ax.text(2.4, m, f"mean {m:.4f}", color=c, va="center", fontsize=7.5)
    for s in seeds:
        a = v2[(v2.lr == 0.001) & (v2.seed == s)]["mean_val_recall@20"].iloc[0]
        b = v2[(v2.lr == 0.0005) & (v2.seed == s)]["mean_val_recall@20"].iloc[0]
        ax.plot([s - 0.08, s + 0.08], [a, b], color="#999", lw=0.8, zorder=2)
        ax.text(s, max(a, b) + 0.0008, f"{b - a:+.4f}", ha="center", fontsize=7, color="#333")
    ax.set_xticks(seeds)
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=8)
    ax.set_xlim(-0.4, 2.9)
    ax.set_ylabel("mean val Recall@20 (4 tasks)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", lw=0.3, alpha=0.6)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    ax.set_title("Learning rate: mean val R@20 per seed (best checkpoint, D1-8 selection)", fontsize=9)
    fig.text(0.5, 0.005, NOTE + " · y-axis zoomed; differences are ~1 seed std", ha="center", fontsize=6.3,
             color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "explore_fig_a_lr_mean.png", dpi=200)
    plt.close(fig)


def fig_b(d: pd.DataFrame):
    """Same-seed lr 5e-4 - lr 1e-3 per task (val R@20); dots = seeds, bar = seed mean."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    xs = np.arange(len(TASKS))
    marks = {0: "o", 1: "s", 2: "^"}
    means = []
    for i, t in enumerate(TASKS):
        col = f"delta/{t}/val_recall@20"
        means.append(d[col].mean())
        for s in (0, 1, 2):
            v = d.loc[d.seed == s, col].iloc[0]
            ax.scatter(i + (s - 1) * 0.12, v, marker=marks[s], s=30, color="#333", zorder=3,
                       label=f"seed {s}" if i == 0 else None)
    ax.bar(xs, means, width=0.55, color=["#c9d7f0" if "long" in t else "#f6d7b0" for t in TASKS],
           edgecolor="#888", lw=0.6, label="3-seed mean", zorder=1)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([TNAME[t] for t in TASKS], fontsize=8)
    ax.set_ylabel("Δ val Recall@20 (lr 5e-4 − lr 1e-3)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", lw=0.3, alpha=0.6)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_title("Learning rate: same-seed per-task differences (long view is seed-dependent)", fontsize=9)
    fig.text(0.5, 0.005, NOTE, ha="center", fontsize=6.3, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "explore_fig_b_lr_task_deltas.png", dpi=200)
    plt.close(fig)


def fig_c(bs: pd.DataFrame):
    """tau 0.03 / 0.1 minus U0 (tau 0.05), seed 0: per-task val R@20 difference with paired user CI."""
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    xs = np.arange(len(TASKS) + 1)
    labels = [TNAME[t] for t in TASKS] + ["4-task mean"]
    for k, (run, c, lab) in enumerate((("x_tau003", "#8fb3e8", "τ 0.03 − τ 0.05"),
                                       ("x_tau010", "#c0504d", "τ 0.10 − τ 0.05"))):
        g = bs[(bs.model_a == run) & (bs.model_b == "u0_s0_formal") & (bs.metric == "recall@20")].set_index("task")
        g = g.loc[TASKS + ["mean4"]]
        off = (k - 0.5) * 0.25
        ax.errorbar(xs + off, g["diff"], yerr=[g["diff"] - g.ci_lo, g.ci_hi - g["diff"]], fmt="o", ms=4,
                    capsize=2, color=c, label=lab)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ val Recall@20 (95% user CI)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", lw=0.3, alpha=0.6)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    ax.set_title("Temperature (seed 0): τ 0.1 trades like for long view; the 4-task mean is unchanged", fontsize=9)
    fig.text(0.5, 0.005, NOTE + " · CIs = user sampling only, one seed per setting", ha="center", fontsize=6.3,
             color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "explore_fig_c_tau_tradeoff.png", dpi=200)
    plt.close(fig)


def main():
    v2 = pd.read_csv(EX / "wave2" / "validation_comparison.csv")
    # The file also lists the formal job 27 as a reference row; it is bitwise identical to its
    # same-SHA reproduction x_u0_repro (job 44), which is the lr 1e-3 seed-0 cell of the grid.
    v2 = v2[v2.run != "u0_formal_job27"]
    assert len(v2) == 6
    d = pd.read_csv(EX / "wave2" / "paired_lr_diffs.csv", dtype={"seed": str})
    d = d[d.seed.isin(["0", "1", "2"])].assign(seed=lambda x: x.seed.astype(int))   # drop the "mean" row
    bs1 = pd.read_csv(EX / "wave1" / "validation_bootstrap.csv")
    assert sorted(v2.seed.unique()) == [0, 1, 2] and sorted(v2.lr.unique()) == [0.0005, 0.001]
    fig_a(v2)
    fig_b(d)
    fig_c(bs1)
    print(sorted(p.name for p in OUT.glob("explore_fig_*.png")))


if __name__ == "__main__":
    main()
