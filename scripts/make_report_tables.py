"""Build the report's result tables and figures from the formal result files only (read-only).

Run:  uv run python scripts/make_report_tables.py
Inputs (all produced earlier; nothing is re-scored or re-trained):
  results/m5/main_formal/            main analysis of server job 37 (7 models, 1000 paired resamples)
  results/m5/cold_video/eval_671dde9_job33/   formal cold-video evaluation
  results/data_pipeline/data_report.json, results/baselines/frames_summary.json
Outputs: report/tables/*.md (supporting tables) and report/figures/fig{1..4}_*.png (embedded in
REPORT.md). Every plotted point / bar is a row of the formal bootstrap CSVs (mean + 95% CI).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (MaxNLocator for the ablation axes)
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results/m5/main_formal"
COLD = ROOT / "results/m5/cold_video/eval_671dde9_job33"
OUT_T, OUT_F = ROOT / "report/tables", ROOT / "report/figures"
TASKS = ["video_long_view", "video_is_like", "music_long_view", "music_is_like"]
TASK_NAME = {"video_long_view": "video / long view", "video_is_like": "video / like",
             "music_long_view": "music / long view", "music_is_like": "music / like"}
ORDER = ["pop", "als", "als_like_aux", "u0", "u1", "abl_nohist", "abl_nomusic"]
NAME = {"pop": "Pop", "als": "ALS (pure)", "als_like_aux": "ALS (aux)", "u0": "3T-U0", "u1": "3T-U1",
        "abl_nohist": "U0 no-history", "abl_nomusic": "U0 no-music"}


def ci(r, k="mean"):
    return f"{r[k]:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]"


def sig(r):
    return "**" if (r.ci_lo > 0 or r.ci_hi < 0) else ""


def main():
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)
    mo = pd.read_csv(MAIN / "bootstrap_models.csv")
    di = pd.read_csv(MAIN / "bootstrap_diffs.csv")
    mo_all = mo[(mo.stratum == "all") & (mo.frame_kind == "masked")]
    fs = json.loads((ROOT / "results/baselines/frames_summary.json").read_text())

    # Table 1: full-catalog test metrics, all six metrics.
    L = []
    for task in TASKS:
        L += [f"**{TASK_NAME[task]}** (users {fs[f'test/{task}/masked']['n_users']:,}, GT pairs "
              f"{fs[f'test/{task}/masked']['gt_pairs_effective']:,})", "",
              "| model | R@10 | R@20 | R@50 | N@10 | N@20 | N@50 |", "|---|---:|---:|---:|---:|---:|---:|"]
        g = mo_all[mo_all.task == task]
        for m in ORDER:
            x = g[g.model == m]
            if not len(x):
                continue
            v = {r.metric: r["mean"] for _, r in x.iterrows()}
            L.append(f"| {NAME[m]} | " + " | ".join(f"{v[k]:.4f}" for k in
                     ("recall@10", "recall@20", "recall@50", "ndcg@10", "ndcg@20", "ndcg@50")) + " |")
        L.append("")
    (OUT_T / "t1_main_metrics.md").write_text("\n".join(L))

    # Table 2: paired differences (R@20, N@20) with 95% CI.
    d = di[(di.stratum == "all") & (di.frame_kind == "masked")]
    pairs = [("u0", "pop"), ("u0", "als"), ("u0", "als_like_aux"), ("u1", "als"), ("u1", "u0"),
             ("abl_nohist", "u0"), ("abl_nomusic", "u0")]
    L = ["| comparison | metric | " + " | ".join(TASK_NAME[t] for t in TASKS) + " |", "|---|---|" + "---|" * 4]
    for a, b in pairs:
        for met in ("recall@20", "ndcg@20"):
            cells = []
            for t in TASKS:
                x = d[(d.task == t) & (d.model_a == a) & (d.model_b == b) & (d.metric == met)]
                cells.append(f"{sig(x.iloc[0])}{x.iloc[0]['diff']:+.4f} [{x.iloc[0].ci_lo:+.4f}, {x.iloc[0].ci_hi:+.4f}]{sig(x.iloc[0])}"
                             if len(x) else "n/a")
            L.append(f"| {NAME[a]} − {NAME[b]} | {met.replace('recall', 'R').replace('ndcg', 'N')} | " + " | ".join(cells) + " |")
    (OUT_T / "t2_paired_diffs.md").write_text("\n".join(L) + "\n")

    # Table 3: history buckets (R@20, users) for U0 / ALS / Pop / no-history.
    bc = pd.read_csv(MAIN / "bucket_counts.csv")
    b = mo[(mo.frame_kind == "masked") & (mo.stratum != "all") & (mo.metric == "recall@20")]
    L = ["| task | bucket | users | Pop | ALS (pure) | ALS (aux) | 3T-U0 | 3T-U1 | no-history | no-music |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in TASKS:
        for k in ("1-5", "6-20", ">20"):
            n = int(bc[(bc.task == t) & (bc.bucket == k)].n_users.iloc[0])
            g = b[(b.task == t) & (b.stratum == k)]
            cells = [f"{g[g.model == m]['mean'].iloc[0]:.4f}" if (g.model == m).any() else "–" for m in ORDER]
            L.append(f"| {TASK_NAME[t]} | {k} | {n:,} | " + " | ".join(cells) + " |")
    (OUT_T / "t3_history_buckets.md").write_text("\n".join(L) + "\n")
    bd = di[(di.frame_kind == "masked") & (di.stratum != "all") & (di.metric == "recall@20")]
    L = ["| task | bucket | 3T-U0 − ALS (pure) | no-history − U0 | no-music − U0 |", "|---|---|---|---|---|"]
    for t in TASKS:
        for k in ("1-5", "6-20", ">20"):
            cells = []
            for a, bb in (("u0", "als"), ("abl_nohist", "u0"), ("abl_nomusic", "u0")):
                x = bd[(bd.task == t) & (bd.stratum == k) & (bd.model_a == a) & (bd.model_b == bb)].iloc[0]
                cells.append(f"{sig(x)}{x['diff']:+.4f} [{x.ci_lo:+.4f}, {x.ci_hi:+.4f}]{sig(x)}")
            L.append(f"| {TASK_NAME[t]} | {k} | " + " | ".join(cells) + " |")
    (OUT_T / "t3b_bucket_diffs.md").write_text("\n".join(L) + "\n")

    # Table 4: cold users (no history before T_test), separate frame.
    c = mo[(mo.frame_kind == "cold") & (mo.metric == "recall@20")]
    L = ["| task | users | Pop | 3T-U0 | 3T-U1 | no-history | no-music |", "|---|---:|---:|---:|---:|---:|---:|"]
    for t in TASKS:
        g = c[c.task == t]
        L.append(f"| {TASK_NAME[t]} | {int(g.n_users.iloc[0])} | " + " | ".join(
            ci(g[g.model == m].iloc[0]) for m in ("pop", "u0", "u1", "abl_nohist", "abl_nomusic")) + " |")
    (OUT_T / "t4_cold_users.md").write_text("\n".join(L) + "\n")

    # Table 5: controlled cold-video experiment.
    cm = pd.read_csv(COLD / "cold_metrics.csv")
    cb = pd.read_csv(COLD / "cold_bootstrap_models.csv")
    cdf = pd.read_csv(COLD / "cold_bootstrap_diffs.csv")
    meta = json.loads((COLD / "cold_meta.json").read_text())
    cname = {"cold_threetower": "Cold 3T-U0 (H removed)", "cold_pop": "Cold Pop", "cold_als": "Cold ALS (pure)",
             "cold_als_like_aux": "Cold ALS (aux)", "random_expected": "Random (analytical)",
             "seen_threetower": "Seen 3T-U0 (reference)", "seen_pop": "Seen Pop (reference)",
             "seen_als": "Seen ALS pure (reference)", "seen_als_like_aux": "Seen ALS aux (reference)"}
    corder = ["cold_threetower", "cold_pop", "cold_als", "cold_als_like_aux", "random_expected",
              "seen_threetower", "seen_pop", "seen_als", "seen_als_like_aux"]
    L = []
    for t in ("video_long_view", "video_is_like"):
        fr = meta["frames"][f"{t}/cold_video_full"]
        L += [f"**{TASK_NAME[t]}** — users {fr['n_users']:,}, GT_H pairs {fr['gt_h_pairs']:,} on {fr['gt_h_videos']} H videos", "",
              "| group | model | Full R@20 [95% CI] | Full R@50 | Full N@10 | H-only R@20 [95% CI] | H-only R@50 | H-only N@10 |",
              "|---|---|---|---:|---:|---|---:|---:|"]
        for m in corder:
            rows = cm[(cm.task == t) & (cm.model == m)]
            if not len(rows):
                continue
            grp = rows.group.iloc[0].replace("_", " ")
            cell = []
            for f in ("cold_video_full", "cold_video_h"):
                r = rows[rows.frame == f].iloc[0]
                bb = cb[(cb.task == t) & (cb.frame == f) & (cb.model == m) & (cb.metric == "recall@20")].iloc[0]
                cell += [ci(bb), f"{r['recall@50']:.4f}", f"{r['ndcg@10']:.4f}"]
            nm = cname[m] + (" [a]" if m == "cold_pop" else "")
            L.append(f"| {grp} | {nm} | " + " | ".join(cell) + " |")
        L.append("")
    L += ["Paired differences, cold 3T-U0 minus each cold-condition baseline (95% CI; ** = CI excludes 0):", "",
          "| task | candidates | vs | ΔR@20 | ΔR@50 | ΔN@10 |", "|---|---|---|---|---|---|"]
    for t in ("video_long_view", "video_is_like"):
        for f, fn in (("cold_video_full", "full catalog"), ("cold_video_h", "H only")):
            for bname in ("cold_pop", "cold_als", "random_expected"):
                cells = []
                for met in ("recall@20", "recall@50", "ndcg@10"):
                    x = cdf[(cdf.task == t) & (cdf.frame == f) & (cdf.model_a == "cold_threetower")
                            & (cdf.model_b == bname) & (cdf.metric == met)].iloc[0]
                    cells.append(f"{sig(x)}{x['diff']:+.4f} [{x.ci_lo:+.4f}, {x.ci_hi:+.4f}]{sig(x)}")
                L.append(f"| {TASK_NAME[t]} | {fn} | {cname[bname]} | " + " | ".join(cells) + " |")
    (OUT_T / "t5_cold_video.md").write_text("\n".join(L) + "\n")

    figures(mo, di, b, cb, meta)
    print("tables:", sorted(p.name for p in OUT_T.iterdir()), "figures:", sorted(p.name for p in OUT_F.iterdir()))


# Fixed colours per model so the same model looks the same in every figure.
COLOR = {"pop": "#9e9e9e", "als": "#4c9a6a", "als_like_aux": "#a6d96a", "u0": "#2f5fb3", "u1": "#8fb3e8",
         "abl_nohist": "#c0504d", "abl_nomusic": "#e69138"}


def _bars(ax, groups, series, values, errs, colors, labels, width=0.8):
    """Grouped bars; `values[g][s]` may be None (bar omitted, e.g. ALS aux on long-view tasks)."""
    import numpy as np
    n = len(series)
    w = width / n
    for j, s in enumerate(series):
        xs = [i - width / 2 + w * (j + 0.5) for i in range(len(groups))]
        ok = [i for i in range(len(groups)) if values[i][j] is not None]
        ax.bar([xs[i] for i in ok], [values[i][j] for i in ok], w * 0.92,
               yerr=np.array([[errs[i][j][0] for i in ok], [errs[i][j][1] for i in ok]]),
               capsize=2, color=colors[j], label=labels[j], edgecolor="white", linewidth=0.4,
               error_kw={"elinewidth": 0.8})
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(bottom=0)                     # never truncate the axis (no visual exaggeration)
    ax.grid(axis="y", lw=0.3, alpha=0.6)
    ax.set_axisbelow(True)


def figures(mo, di, b, cb, meta):
    """Report figures. Every plotted value is a row of the formal bootstrap CSVs (point estimate =
    full-sample mean, error bar = 95% paired-bootstrap percentile CI); nothing is recomputed."""
    import numpy as np
    m20 = mo[(mo.stratum == "all") & (mo.frame_kind == "masked") & (mo.metric == "recall@20")]

    def get(task, model, frame=m20):
        x = frame[(frame.task == task) & (frame.model == model)]
        if not len(x):
            return None, (0, 0)
        r = x.iloc[0]
        return r["mean"], (r["mean"] - r.ci_lo, r.ci_hi - r["mean"])

    # Figure 1: four-task comparison of the headline models (same test masked frames).
    models = ["pop", "als", "als_like_aux", "u0", "u1"]
    labels = ["Popularity", "ALS (pure)", "ALS (aux, like tasks only)", "3T-U0 (main)", "3T-U1 (+ snapshot meta)"]
    groups = [TASK_NAME[t].replace(" / ", "\n") for t in TASKS]
    vals = [[get(t, m)[0] for m in models] for t in TASKS]
    errs = [[get(t, m)[1] for m in models] for t in TASKS]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    _bars(ax, groups, models, vals, errs, [COLOR[m] for m in models], labels)
    # Mark bars that do not exist (ALS aux is defined for the like tasks only) instead of leaving a gap.
    w = 0.8 / len(models)
    for i in range(len(TASKS)):
        for j in range(len(models)):
            if vals[i][j] is None:
                ax.text(i - 0.4 + w * (j + 0.5), 0.003, "n/a", ha="center", va="bottom", fontsize=7,
                        color="#333", rotation=90)
    ax.set_ylabel("test Recall@20 (95% CI)", fontsize=8)
    ax.set_title("Full-catalog, seen-masked test Recall@20 (same users per task)", fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_F / "fig1_baselines.png", dpi=200)
    plt.close(fig)

    # Figure 2: controlled cold-video experiment, R@20, two candidate sets in separate panels
    # (different scales). Only the cold-trained model and cold-condition baselines are drawn; the
    # seen-video reference is not a cold baseline and is given in the caption / table only.
    cm_models = ["cold_threetower", "cold_pop", "cold_als", "random_expected"]
    cm_labels = ["Cold 3T-U0 (H removed)", "Cold Pop", "Cold ALS (pure)", "Random (analytical expectation)"]
    cm_colors = [COLOR["u0"], COLOR["pop"], COLOR["als"], "#d9b3ff"]
    cgroups = ["video\nlong view", "video\nlike"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, (frame_kind, title) in zip(axes, (("cold_video_full", "(a) full catalog (7,583 videos, seen-masked)"),
                                              ("cold_video_h", "(b) H only (750 held-out videos, seen-masked)"))):
        fr = cb[(cb.frame == frame_kind) & (cb.metric == "recall@20")]
        v, e = [], []
        for t in ("video_long_view", "video_is_like"):
            row_v, row_e = [], []
            for m in cm_models:
                x = fr[(fr.task == t) & (fr.model == m)]
                r = x.iloc[0]
                row_v.append(r["mean"])
                row_e.append((r["mean"] - r.ci_lo, r.ci_hi - r["mean"]))
            v.append(row_v)
            e.append(row_e)
        _bars(ax, cgroups, cm_models, v, e, cm_colors, cm_labels)
        # A bar of height exactly 0 is invisible: label it so it is not mistaken for a missing bar.
        w = 0.8 / len(cm_models)
        for i in range(len(cgroups)):
            for j in range(len(cm_models)):
                if v[i][j] == 0:
                    ax.text(i - 0.4 + w * (j + 0.5), 0, "0", ha="center", va="bottom", fontsize=7, color="#333")
        ax.set_title(title, fontsize=8.5)
    axes[0].set_ylabel("test Recall@20 on GT_H (95% CI)", fontsize=8)
    n_lv = meta["frames"]["video_long_view/cold_video_full"]["n_users"]
    n_like = meta["frames"]["video_is_like/cold_video_full"]["n_users"]
    axes[1].text(0.98, 0.97, f"users: LV {n_lv:,}, like {n_like:,}", transform=axes[1].transAxes, ha="right",
                 va="top", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.045), frameon=False)
    fig.text(0.5, 0.012, "\"0\" = exactly zero. Cold ALS has no factors for the held-out videos (H is unscorable); "
             "cold Pop scores every H video 0, so on the full catalog H is always ranked last.",
             ha="center", va="bottom", fontsize=6.5, color="#444")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT_F / "fig2_cold_video.png", dpi=200)
    plt.close(fig)

    # Figure 3: R@20 by history-length bucket (all four tasks; capped length fed to the model).
    bucket_models = ["pop", "als", "als_like_aux", "u0", "u1", "abl_nohist"]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=False)
    for ax, t in zip(axes, TASKS):
        for m in bucket_models:
            g = b[(b.task == t) & (b.model == m)]
            if not len(g):
                continue
            g = g.set_index("stratum").loc[["1-5", "6-20", ">20"]]
            ax.errorbar([0, 1, 2], g["mean"], yerr=[g["mean"] - g.ci_lo, g.ci_hi - g["mean"]], marker="o", ms=3,
                        lw=1, capsize=1.5, label=NAME[m], color=COLOR[m])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["1-5", "6-20", ">20"], fontsize=7)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_title(TASK_NAME[t], fontsize=8)
        ax.grid(axis="y", lw=0.3, alpha=0.6)
    axes[0].set_ylabel("test Recall@20 (95% CI)", fontsize=7.5)
    fig.supxlabel("history length at T_test (events fed to the model, capped at 100)", fontsize=7.5, y=0.1)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, ncol=6, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT_F / "fig3_buckets.png", dpi=200)
    plt.close(fig)

    # Figure 4: ablations, paired differences vs U0 (R@20 and NDCG@20, 95% CI).
    d = di[(di.stratum == "all") & (di.frame_kind == "masked") & (di.model_b == "u0")
           & di.model_a.isin(["abl_nohist", "abl_nomusic"])]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6))           # stacked: each panel gets the full width
    ys = np.arange(len(TASKS))[::-1]
    for ax, met in zip(axes, ("recall@20", "ndcg@20")):
        for k, (m, off) in enumerate((("abl_nohist", 0.15), ("abl_nomusic", -0.15))):
            g = d[(d.model_a == m) & (d.metric == met)].set_index("task").loc[TASKS]
            ax.errorbar(g["diff"], ys + off, xerr=[g["diff"] - g.ci_lo, g.ci_hi - g["diff"]], fmt="o", ms=3.5,
                        capsize=2, color=COLOR[m], label={"abl_nohist": "no history − U0",
                                                         "abl_nomusic": "no music input − U0"}[m])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(ys)
        ax.set_yticklabels([TASK_NAME[t] for t in TASKS], fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel(f"Δ test {met.replace('recall', 'Recall').replace('ndcg', 'NDCG')} vs U0 (95% CI)",
                      fontsize=7.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=8))
        ax.grid(axis="x", lw=0.3, alpha=0.6)
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_F / "fig4_ablations.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
