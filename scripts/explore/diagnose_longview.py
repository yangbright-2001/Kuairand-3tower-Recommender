"""S9 Wave 3 gate: read-only long-view / music-fusion diagnostics (post-hoc exploratory).

No training. Reads only selection.json (val_history) of finished runs, validation per-user files
of the M3 baselines and the runs, and (for the init-scale probe) builds an *untrained* model the
same way `Trainer` does (seed 0). metrics.csv and per_user/test_* are never opened.

Outputs (results/explore/wave3_review/):
  epoch_task_curves.csv     every run x epoch x task val R@20 (from selection.json)
  selection_vs_task_best.csv  selected epoch vs each task's own best epoch (diagnostic only:
                              per-task oracle epochs are post-hoc val maxima, optimistic)
  nomusic_vs_u0_epochwise.json  no-music (job 35) vs each U0 seed per epoch, long-view tasks
  longview_vs_als.csv       val long-view R@20 of every run vs pure ALS, paired val bootstrap
  init_fusion_scale.json    first-layer input-block scale at initialisation (untrained U0 / gated)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kuairand_rec import paths
from kuairand_rec.experiments.bootstrap import diff_rows, paired_bootstrap

_spec = importlib.util.spec_from_file_location("aw1", Path(__file__).with_name("analyze_wave1.py"))
aw1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aw1)
TASKS, _guard, jload = aw1.TASKS, aw1._guard, aw1.jload
LV = ("video_long_view", "music_long_view")


def runs(a):
    f = Path(a.formal_root) / "artifacts"
    w1, w2 = Path(a.wave1_dir), Path(a.wave2_dir)
    return {
        "U0 lr1e-3 s0 (job27=44)": f / "u0/9218a76_seed0_job27",
        "U0 lr1e-3 s1 (39)": w1 / "x_u0_s1/1051f86_seed1_job39",
        "U0 lr1e-3 s2 (40)": w1 / "x_u0_s2/1051f86_seed2_job40",
        "lr5e-4 s0 (41)": w1 / "x_lr5e4/1051f86_seed0_job41",
        "lr5e-4 s1 (45)": w2 / "x_lr5e4_s1/a158e5e_seed1_job45",
        "lr5e-4 s2 (46)": w2 / "x_lr5e4_s2/a158e5e_seed2_job46",
        "tau0.03 s0 (42)": w1 / "x_tau003/1051f86_seed0_job42",
        "tau0.1 s0 (43)": w1 / "x_tau010/1051f86_seed0_job43",
        "no-music s0 (35, formal ablation)": f / "abl_nomusic/671dde9_seed0_job35",
        "no-history s0 (34, formal ablation)": f / "abl_nohist/671dde9_seed0_job34",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("wave1_dir", "wave2_dir", "formal_root"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out_dir", default=str(paths.PROJECT_ROOT / "results" / "explore" / "wave3_review"))
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    R = runs(a)

    # ------------------------------------------------ epoch x task curves and per-task best epochs
    curves, selrows = [], []
    for name, d in R.items():
        sel = jload(d / "selection.json")
        assert sel["test_used_for_selection"] is False
        H = sel["val_history"]
        for h in H:
            curves.append({"run": name, "epoch": h["epoch"], "mean4": h["score"],
                           **{t: h[f"{t}/recall@20"] for t in TASKS}})
        b = sel["best_epoch"]
        hb = next(h for h in H if h["epoch"] == b)
        row = {"run": name, "selected_epoch": b, "n_epochs": len(H), "mean4_at_selected": hb["score"]}
        for t in TASKS:
            best = max(H, key=lambda h: (h[f"{t}/recall@20"], -h["epoch"]))
            row[f"{t}/selected"] = hb[f"{t}/recall@20"]
            row[f"{t}/best_epoch"] = best["epoch"]
            row[f"{t}/best"] = best[f"{t}/recall@20"]
            row[f"{t}/left_on_table"] = best[f"{t}/recall@20"] - hb[f"{t}/recall@20"]
        selrows.append(row)
    cdf = pd.DataFrame(curves)
    cdf.to_csv(out / "epoch_task_curves.csv", index=False)
    sdf = pd.DataFrame(selrows)
    sdf.to_csv(out / "selection_vs_task_best.csv", index=False)

    # ------------------------------------------------ no-music vs every U0 seed, epoch-wise
    nm = cdf[cdf.run.str.startswith("no-music")].set_index("epoch")
    u0s = [r for r in R if r.startswith("U0 ")]
    em = {}
    for t in TASKS:
        per = {}
        for u in u0s:
            g = cdf[cdf.run == u].set_index("epoch")
            common = sorted(set(g.index) & set(nm.index))
            per[u] = {"epochs": len(common), "no_music_higher": int(sum(nm.loc[e, t] > g.loc[e, t] for e in common)),
                      "mean_diff": float(np.mean([nm.loc[e, t] - g.loc[e, t] for e in common]))}
        # against the per-epoch maximum over the 3 U0 seeds (a stricter, envelope comparison)
        ep = sorted(set(nm.index) & set.intersection(*[set(cdf[cdf.run == u].epoch) for u in u0s]))
        env = [max(cdf[(cdf.run == u) & (cdf.epoch == e)][t].iloc[0] for u in u0s) for e in ep]
        per["vs_max_of_3_U0_seeds"] = {"epochs": len(ep), "no_music_higher": int(sum(nm.loc[e, t] > m for e, m in zip(ep, env)))}
        em[t] = per
    sel_nm = sdf[sdf.run.str.startswith("no-music")].iloc[0]
    u0_sel = sdf[sdf.run.isin(u0s)]
    em["selected_epoch_values"] = {
        t: {"no_music": float(sel_nm[f"{t}/selected"]), "U0_seed_min": float(u0_sel[f"{t}/selected"].min()),
            "U0_seed_max": float(u0_sel[f"{t}/selected"].max()), "U0_seed_mean": float(u0_sel[f"{t}/selected"].mean())}
        for t in TASKS}
    em["caveat"] = ("no-music is ONE seed (job 35, sha 671dde9). It shares only the data order with U0 seed 0; its "
                    "first video-tower layer has 128 fewer inputs, so parameter initialisation and every later "
                    "random draw differ -> not a paired comparison. Epochs are autocorrelated.")
    (out / "nomusic_vs_u0_epochwise.json").write_text(json.dumps(em, indent=1))

    # ------------------------------------------------ long-view vs pure ALS (val, paired bootstrap)
    formal = Path(a.formal_root)
    lv_rows = []
    for t in LV:
        tab = {"als": pd.read_parquet(_guard(formal / "results/baselines/per_user/als" / f"val_{t}_masked.parquet"))}
        for name, d in R.items():
            tab[name] = pd.read_parquet(_guard(d / "per_user" / f"val_{t}_masked.parquet"))
        u = tab["als"].user_id.to_numpy()
        fid = tab["als"].frame_id.iloc[0]
        for k, x in tab.items():
            assert np.array_equal(x.user_id.to_numpy(), u) and (x.frame_id == fid).all(), k
        res = paired_bootstrap({k: v[["user_id", "recall@20", "ndcg@20"]] for k, v in tab.items()},
                               ["recall@20"], 1000, 0, tab["als"].population_id.iloc[0])
        for r in diff_rows(res, [(n, "als") for n in R], ["recall@20"]):
            r["numerically_above_als"] = r["diff"] > 0
            r["ci_supports_advantage"] = r["ci_lo"] > 0
            r["ci_supports_deficit"] = r["ci_hi"] < 0
            lv_rows.append({"task": t, **r})
    pd.DataFrame(lv_rows).to_csv(out / "longview_vs_als.csv", index=False)

    # ------------------------------------------------ input-block scale at initialisation
    from kuairand_rec.data.pipeline import load_processed
    from kuairand_rec.models import data as D
    from kuairand_rec.models import features as F
    from kuairand_rec.models.towers import ThreeTowerModel, TowerConfig
    from kuairand_rec.models.train import ItemInputs, configure_determinism

    P = load_processed(paths.PROJECT_ROOT / "data" / "processed")
    cond = D.normal_condition(P)
    items = ItemInputs.from_condition(cond, "cpu")
    rows = np.asarray(cond.cand_video_rows)
    init = {}
    for label, tc in (("U0 (use_music=True)", TowerConfig()), ("no-music", TowerConfig(use_music=False))):
        configure_determinism(0)
        m = ThreeTowerModel(F.FeatureSizes.from_vocabs(cond.vocabs), tc, F.HistoryConfig()).eval()
        vt, vf = m.video_tower, items.vf
        with torch.no_grad():
            enc = m.encode_items(vf, items.mf)
            parts = {"video_id": vt.video_emb(vf.video_idx), "author": vt.author_emb(vf.author_idx)}
            if tc.use_music:
                parts["music_base"] = enc.video_music_base
            for j, (c, e) in enumerate(zip(("video_type", "upload_type", "upload_dt", "music_type"), vt.cat_embs)):
                parts[c] = e(vf.cat[:, j])
            tv = (vf.tags != 0).float()
            parts["tags"] = (vt.tag_emb(vf.tags) * tv.unsqueeze(-1)).sum(1) / tv.sum(1, keepdim=True).clamp_min(1)
            parts["num"], parts["num_missing"] = vf.num, vf.num_missing
            W, off, C = vt.mlp[0].weight, 0, {}
            for k, x in parts.items():
                C[k] = (x @ W[:, off:off + x.shape[1]].T)[rows].double().numpy()
                off += x.shape[1]
            assert off == W.shape[1]
        tot = sum(C.values()).var(0).sum()
        init[label] = {k: {"input_norm": float(parts[k][rows].norm(dim=1).mean()),
                           "contrib_norm": float(np.linalg.norm(c, axis=1).mean()),
                           "var_share": float(c.var(0).sum() / tot)} for k, c in C.items()}
    init["note"] = ("Untrained model built like Trainer (configure_determinism(0)). music_base is a LayerNorm "
                    "output (RMS ~1 per dim, norm ~sqrt(128)); ID embeddings are N(0, 0.02^2) at init.")
    (out / "init_fusion_scale.json").write_text(json.dumps(init, indent=1))

    pd.set_option("display.width", 250)
    cols = ["run", "selected_epoch", "mean4_at_selected"] + [c for t in TASKS for c in (f"{t}/selected", f"{t}/best_epoch", f"{t}/left_on_table")]
    print(sdf[cols].round(4).to_string())
    print(json.dumps(em, indent=1)[:2500])
    print(pd.DataFrame(lv_rows)[["task", "model_a", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi"]].round(4).to_string())
    print(json.dumps({k: {kk: round(vv["var_share"], 3) for kk, vv in v.items()} for k, v in init.items() if k != "note"}, indent=1))
    print(json.dumps({k: {kk: round(vv["input_norm"], 3) for kk, vv in v.items()} for k, v in init.items() if k != "note"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
