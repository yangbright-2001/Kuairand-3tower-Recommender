"""S9 Wave 3: no-music cross-seed replication, val-only (post-hoc exploratory).

Verifies jobs 47-49, checks job 47 against the formal no-music ablation job 35 (bitwise
reproduction), and compares U0 {44, 39, 40} vs no-music {47, 48, 49} on the normal val masked
frames, plus the long-view comparison vs the M3 pure ALS val baseline.

Reads ONLY validation-side files (selection.json, run_meta.json, config.yaml, parity.json,
export/manifest.json, Slurm .rc, train_log.jsonl, per_user/val_*_masked.parquet, M3 val per-user
baseline files). metrics.csv and per_user/test_* are refused by the shared guard.

Run (explore worktree):
  uv run python scripts/explore/analyze_wave3.py --wave1_dir artifacts/explore/wave1 \
    --wave2_dir artifacts/explore/wave2 --wave3_dir artifacts/explore/wave3 \
    --meta1_dir artifacts/explore/wave1_meta --meta2_dir artifacts/explore/wave2_meta \
    --meta3_dir artifacts/explore/wave3_meta --formal_root /Users/yangliang/Desktop/kuairand-threetower-recsys
Outputs: results/explore/wave3/*.csv|json and figures/*.png (via plot_wave3.py)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from kuairand_rec import paths
from kuairand_rec.data.pipeline import load_processed
from kuairand_rec.experiments.bootstrap import diff_rows, paired_bootstrap
from kuairand_rec.models.evaluation import build_eval_frames

_spec = importlib.util.spec_from_file_location("aw1", Path(__file__).with_name("analyze_wave1.py"))
aw1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aw1)
TASKS, _guard, jload, best_record = aw1.TASKS, aw1._guard, aw1.jload, aw1.best_record

SHA = {1: "1051f86f373697de7eca600d1dc3cd674d7d05f6", 2: "a158e5e565534574c91fb56aa88a96ede3c6446a",
       3: "be5ceb96b5e10bf9754983de168d8c3542e0de85"}
# label -> (arch, seed, job, exp, wave, run dir)
RUNS = {
    "U0 s0": ("U0", 0, 44, "x_u0_repro", 2, "x_u0_repro/a158e5e_seed0_job44"),
    "U0 s1": ("U0", 1, 39, "x_u0_s1", 1, "x_u0_s1/1051f86_seed1_job39"),
    "U0 s2": ("U0", 2, 40, "x_u0_s2", 1, "x_u0_s2/1051f86_seed2_job40"),
    "no-music s0": ("no-music", 0, 47, "x_nomusic_repro", 3, "x_nomusic_repro/be5ceb9_seed0_job47"),
    "no-music s1": ("no-music", 1, 48, "x_nomusic_s1", 3, "x_nomusic_s1/be5ceb9_seed1_job48"),
    "no-music s2": ("no-music", 2, 49, "x_nomusic_s2", 3, "x_nomusic_s2/be5ceb9_seed2_job49"),
}
JOB35 = "abl_nomusic/671dde9_seed0_job35"
LV = ("video_long_view", "music_long_view")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("wave1_dir", "wave2_dir", "wave3_dir", "meta1_dir", "meta2_dir", "meta3_dir", "formal_root"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out_dir", default=str(paths.PROJECT_ROOT / "results" / "explore" / "wave3"))
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    formal = Path(a.formal_root)
    fs = {k: v["frame_id"] for k, v in json.loads((paths.PROJECT_ROOT / "results/baselines/frames_summary.json").read_text()).items()}
    wdir = {1: Path(a.wave1_dir), 2: Path(a.wave2_dir), 3: Path(a.wave3_dir)}
    mdir = {1: Path(a.meta1_dir), 2: Path(a.meta2_dir), 3: Path(a.meta3_dir)}
    nomusic_ref = yaml.safe_load((paths.PROJECT_ROOT / "configs/threetower_abl_nomusic.yaml").read_text())

    # ---------------------------------------------------------------- verification
    dirs, ver = {}, {}
    for lab, (arch, seed, job, exp, wave, rd) in RUNS.items():
        d = wdir[wave] / rd
        dirs[lab] = d
        v = aw1.verify(lab, job, exp, d, mdir[wave], fs)
        ok = (v["git_sha"] == SHA[wave] and not v["missing"] and v["exit_code"] == 0 and v["parity_ok"]
              and v["parity_rc"] == 0 and v["condition_id"] == aw1.CONDITION_ID and v["test_used_for_selection"] is False
              and v["best_pt_sha256_matches_selection"] and v["export_checkpoint_matches_selection"]
              and v["export_files_present"] and v["val_selection_frame_ids_match_m3"] and v["export_frame_ids_match_m3"])
        c = v["config"]
        want_model = {"use_music": False} if arch == "no-music" else {}
        ok &= (c["seed"] == seed and c["lr"] == 1e-3 and c["tau"] == 0.05 and c["model"] == want_model
               and c["history"] == {} and c["user_features"] == "u0" and c["condition"] == "normal"
               and (c["batch_size"], c["max_epochs"], c["patience"]) == (1024, 30, 3))
        if arch == "no-music":   # full effective config == formal no-music except exp / run_root / seed
            full = yaml.safe_load(_guard(d / "config.yaml").read_text())
            j35 = yaml.safe_load(_guard(formal / "artifacts" / JOB35 / "config.yaml").read_text())
            diff = {k for k in set(full) | set(j35) if k != "train" and full.get(k) != j35.get(k)}
            tdiff = {k for k in full["train"] if full["train"][k] != j35["train"][k]}
            v["config_diff_vs_job35"] = {"top": sorted(diff), "train": sorted(tdiff)}
            ok &= diff == {"exp", "run_root"} and tdiff == (set() if seed == 0 else {"seed"})
            ok &= nomusic_ref["model"] == {"use_music": False}
        v["valid_completed_run"] = bool(ok)
        ver[lab] = v
    dirs["job35"] = formal / "artifacts" / JOB35
    ver["job35 (formal no-music)"] = aw1.verify("u0_s0_formal", 35, "abl_nomusic", dirs["job35"], None, fs)
    fps = {json.dumps(v["data_fingerprint"], sort_keys=True) for v in ver.values()}

    # ---------------------------------------------------------------- job 47 vs job 35
    r47, r35 = dirs["no-music s0"], dirs["job35"]
    s47, s35 = jload(r47 / "selection.json"), jload(r35 / "selection.json")
    keys = ["score", "tie", *[f"{t}/{m}" for t in TASKS for m in ("recall@20", "ndcg@20")]]
    vh = [all(x[k] == y[k] for k in keys) for x, y in zip(s47["val_history"], s35["val_history"])]
    tl47 = [json.loads(x) for x in (r47 / "train_log.jsonl").read_text().splitlines()]
    tl35 = [json.loads(x) for x in (r35 / "train_log.jsonl").read_text().splitlines()]
    tk = ["loss", "grad_norm", *[f"loss_{t}" for t in TASKS]]
    tl_eq = len(tl47) == len(tl35) and all(x["step"] == y["step"] and all(x[k] == y[k] for k in tk) for x, y in zip(tl47, tl35))
    w47 = torch.load(r47 / "best.pt", map_location="cpu", weights_only=False)["model"]
    w35 = torch.load(r35 / "best.pt", map_location="cpu", weights_only=False)["model"]
    w_eq = sorted(w47) == sorted(w35) and all(torch.equal(w47[k], w35[k]) for k in w35)
    m47, m35 = jload(r47 / "export/manifest.json"), jload(r35 / "export/manifest.json")
    emb = {f: m47["files"][f] == m35["files"][f] for f in m35["files"] if f.startswith(("items_", "users_val_"))}
    pu_eq = {}
    for t in TASKS:
        x = pd.read_parquet(_guard(r47 / "per_user" / f"val_{t}_masked.parquet"))
        y = pd.read_parquet(_guard(r35 / "per_user" / f"val_{t}_masked.parquet"))
        pu_eq[t] = bool(np.array_equal(x[["user_id", "recall@20", "ndcg@20"]].to_numpy(), y[["user_id", "recall@20", "ndcg@20"]].to_numpy()))
    repro = {"job47_sha": ver["no-music s0"]["git_sha"], "job35_sha": ver["job35 (formal no-music)"]["git_sha"],
             "config_diff_vs_job35": ver["no-music s0"].get("config_diff_vs_job35"),
             "val_history_epochs": [len(s47["val_history"]), len(s35["val_history"])],
             "val_history_all_bitwise_equal": all(vh) and len(vh) == len(s35["val_history"]),
             "best_epoch": [s47["best_epoch"], s35["best_epoch"]],
             "best_score_equal": s47["best_score"] == s35["best_score"],
             "train_log_records": [len(tl47), len(tl35)], "train_log_bitwise_equal": tl_eq,
             "best_pt_state_dict_bitwise_equal": bool(w_eq),
             "export_item_and_val_user_embeddings_sha_equal": all(emb.values()), "n_embedding_files": len(emb),
             "val_per_user_metrics_equal": pu_eq}
    (out / "repro_job47_vs_job35.json").write_text(json.dumps(repro, indent=1))

    # ---------------------------------------------------------------- comparison
    rows, epochs = [], []
    for lab, (arch, seed, job, exp, wave, rd) in RUNS.items():
        sel = jload(dirs[lab] / "selection.json")
        b = best_record(sel)
        rows.append({"run": lab, "arch": arch, "seed": seed, "job": job, "exp": exp, "git_sha": ver[lab]["git_sha"][:7],
                     "best_epoch": sel["best_epoch"], "epochs_validated": len(sel["val_history"]),
                     **{f"{t}/val_{m}": b[f"{t}/{m}"] for t in TASKS for m in ("recall@20", "ndcg@20")},
                     "mean_val_recall@20": sel["best_score"], "mean_val_ndcg@20": sel["best_tie"],
                     "exit_code": ver[lab]["exit_code"], "parity_ok": ver[lab]["parity_ok"],
                     "valid_completed_run": ver[lab]["valid_completed_run"]})
        for h in sel["val_history"]:
            epochs.append({"run": lab, "arch": arch, "seed": seed, "epoch": h["epoch"], "mean4": h["score"],
                           **{t: h[f"{t}/recall@20"] for t in TASKS}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "validation_comparison.csv", index=False)
    ep = pd.DataFrame(epochs)
    ep.to_csv(out / "validation_by_epoch.csv", index=False)
    metrics = ["mean_val_recall@20", "mean_val_ndcg@20", *[f"{t}/val_{m}" for t in TASKS for m in ("recall@20", "ndcg@20")]]
    stats = []
    for arch, g in df.groupby("arch"):
        for m in metrics:
            x = g[m].to_numpy()
            stats.append({"arch": arch, "metric": m, "n_seeds": 3, "mean": x.mean(), "std_ddof1": x.std(ddof=1),
                          "min": x.min(), "max": x.max()})
    pdiff = []
    for s in (0, 1, 2):
        u, n = df[(df.arch == "U0") & (df.seed == s)].iloc[0], df[(df.arch == "no-music") & (df.seed == s)].iloc[0]
        pdiff.append({"seed": s, **{f"delta/{m}": n[m] - u[m] for m in metrics}})
    pdf = pd.DataFrame(pdiff)
    for agg in ("mean", "std", "min", "max"):
        row = {"seed": agg}
        for c in pdf.columns[1:]:
            col = pdf[c].iloc[:3].astype(float)
            row[c] = {"mean": col.mean(), "std": col.std(ddof=1), "min": col.min(), "max": col.max()}[agg]
        pdf = pd.concat([pdf, pd.DataFrame([row])], ignore_index=True)
    pdf.to_csv(out / "paired_nomusic_minus_u0.csv", index=False)
    pd.DataFrame(stats).to_csv(out / "seed_analysis.csv", index=False)

    # epoch-wise: same-seed no-music minus U0 on common epochs (descriptive; epochs are autocorrelated)
    epw = []
    for s in (0, 1, 2):
        u = ep[(ep.arch == "U0") & (ep.seed == s)].set_index("epoch")
        n = ep[(ep.arch == "no-music") & (ep.seed == s)].set_index("epoch")
        com = sorted(set(u.index) & set(n.index))
        for t in (*TASKS, "mean4"):
            d = np.array([n.loc[e, t] - u.loc[e, t] for e in com])
            epw.append({"seed": s, "task": t, "common_epochs": len(com), "no_music_higher": int((d > 0).sum()),
                        "mean_diff": float(d.mean())})
    pd.DataFrame(epw).to_csv(out / "epochwise_nomusic_minus_u0.csv", index=False)

    # ---------------------------------------------------------------- per-user val: bootstrap and ALS
    P = load_processed(paths.PROJECT_ROOT / "data" / "processed")
    frames = build_eval_frames(P, ("val",), ("masked",))           # asserts frame ids (= M3)
    boot = []
    for t in TASKS:
        f = frames[("val", t, "masked")]
        tab = {lab: pd.read_parquet(_guard(dirs[lab] / "per_user" / f"val_{t}_masked.parquet")) for lab in RUNS}
        tab["als (pure)"] = pd.read_parquet(_guard(formal / "results/baselines/per_user/als" / f"val_{t}_masked.parquet"))
        tab["pop"] = pd.read_parquet(_guard(formal / "results/baselines/per_user/pop" / f"val_{t}_masked.parquet"))
        if "is_like" in t:
            tab["als_like_aux"] = pd.read_parquet(_guard(formal / "results/baselines/per_user/als_like_aux" / f"val_{t}_masked.parquet"))
        for k, x in tab.items():
            if not (np.array_equal(x.user_id.to_numpy(), f.users) and (x.frame_id == f.frame_id).all()
                    and np.array_equal(x.n_gt.to_numpy(), np.diff(f.gt_indptr))):
                raise ValueError(f"{k}/{t}: not on the M3 val frame")
        res = paired_bootstrap({k: v[["user_id", "recall@20", "ndcg@20"]] for k, v in tab.items()},
                               ["recall@20", "ndcg@20"], 1000, 0, f.population_id)
        pairs = [(f"no-music s{s}", f"U0 s{s}") for s in (0, 1, 2)]
        pairs += [(lab, b) for lab in RUNS for b in ("als (pure)", "pop", "als_like_aux") if b in tab]
        for r in diff_rows(res, pairs, ["recall@20", "ndcg@20"]):
            r["numerically_above"] = r["diff"] > 0
            r["ci_supports_advantage"] = r["ci_lo"] > 0
            r["ci_supports_deficit"] = r["ci_hi"] < 0
            boot.append({"task": t, **r})
        for k, x in tab.items():
            boot.append({"task": t, "model_a": k, "model_b": None, "metric": "level_recall@20",
                         "mean_a": float(x["recall@20"].mean())})
    bdf = pd.DataFrame(boot)
    bdf.to_csv(out / "validation_bootstrap.csv", index=False)

    (out / "run_verification.json").write_text(json.dumps(
        {"study": "S9 Wave 3 (post-hoc exploratory): no-music cross-seed replication; validation only",
         "files_read": "validation-side files only; metrics.csv and per_user/test_* refused",
         "distinct_data_fingerprints_excl_data_git_sha": len(fps), "runs": ver}, indent=1, default=str))
    pd.set_option("display.width", 260)
    print(df[["run", "job", "git_sha", "best_epoch", "mean_val_recall@20", "mean_val_ndcg@20", "valid_completed_run"]].to_string())
    print(json.dumps(repro, indent=1))
    print("fingerprints", len(fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
