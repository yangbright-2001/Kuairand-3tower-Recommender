"""S9 Wave 2: run verification, 3 seeds x 2 learning rates, repro check vs formal U0 job 27, and
a validation-only comparison against the formal M3 Pop / ALS baselines (post-hoc exploratory).

Reads ONLY validation-side files: selection.json, run_meta.json, config.yaml, parity.json,
selection_vs_export.json, export/manifest.json, Slurm .rc files, train_log.jsonl,
per_user/val_*_masked.parquet of every run, the M3 baseline per_user/<model>/val_*_masked.parquet
and the M3 val frames. metrics.csv and every per_user/test_* file are refused (same guard as wave 1).

Frame identity: the M3 val masked frames are rebuilt from data/processed by `build_eval_frames`
(which asserts every frame_id == results/baselines/frames_summary.json, i.e. same split, catalog,
seen mask and ground truth), compared with the saved M3 frame parquet (users, GT rows, mask size,
history length), and every per-user file (models and baselines) must carry that frame_id and the
same user order.

Run (explore worktree):
  uv run python scripts/explore/analyze_wave2.py \
    --wave1_dir artifacts/explore/wave1 --wave2_dir artifacts/explore/wave2 \
    --meta1_dir artifacts/explore/wave1_meta --meta2_dir artifacts/explore/wave2_meta \
    --formal_root /Users/yangliang/Desktop/kuairand-threetower-recsys
Outputs: results/explore/wave2/{validation_comparison.csv, seed_analysis.csv, paired_lr_diffs.csv,
  validation_bootstrap.csv, baselines_val_comparison.csv, repro_vs_job27.json, run_verification.json}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kuairand_rec import paths
from kuairand_rec.data.pipeline import load_processed
from kuairand_rec.experiments.bootstrap import diff_rows, paired_bootstrap
from kuairand_rec.models.evaluation import build_eval_frames

import importlib.util

_spec = importlib.util.spec_from_file_location("aw1", Path(__file__).with_name("analyze_wave1.py"))
aw1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aw1)
TASKS, _guard, jload, best_record = aw1.TASKS, aw1._guard, aw1.jload, aw1.best_record

WAVE2_SHA = "a158e5e565534574c91fb56aa88a96ede3c6446a"
WAVE1_SHA = aw1.EXPECTED_SHA
# label -> (job, exp, wave, run dir under the wave dir, lr, seed)
RUNS = {
    "lr1e-3_s0 (x_u0_repro)": (44, "x_u0_repro", 2, "x_u0_repro/a158e5e_seed0_job44", 1e-3, 0),
    "lr1e-3_s1 (x_u0_s1)": (39, "x_u0_s1", 1, "x_u0_s1/1051f86_seed1_job39", 1e-3, 1),
    "lr1e-3_s2 (x_u0_s2)": (40, "x_u0_s2", 1, "x_u0_s2/1051f86_seed2_job40", 1e-3, 2),
    "lr5e-4_s0 (x_lr5e4)": (41, "x_lr5e4", 1, "x_lr5e4/1051f86_seed0_job41", 5e-4, 0),
    "lr5e-4_s1 (x_lr5e4_s1)": (45, "x_lr5e4_s1", 2, "x_lr5e4_s1/a158e5e_seed1_job45", 5e-4, 1),
    "lr5e-4_s2 (x_lr5e4_s2)": (46, "x_lr5e4_s2", 2, "x_lr5e4_s2/a158e5e_seed2_job46", 5e-4, 2),
}
FORMAL_U0 = "u0/9218a76_seed0_job27"
BASELINES = {"pop": "pop", "als (pure)": "als", "als_like_aux": "als_like_aux"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("wave1_dir", "wave2_dir", "meta1_dir", "meta2_dir", "formal_root"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out_dir", default=str(paths.PROJECT_ROOT / "results" / "explore" / "wave2"))
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fs = json.loads((paths.PROJECT_ROOT / "results" / "baselines" / "frames_summary.json").read_text())
    frames_summary = {k: v["frame_id"] for k, v in fs.items()}
    formal = Path(a.formal_root)

    # ---------------------------------------------------------------- run verification
    dirs, ver = {}, {}
    for lab, (job, exp, wave, rd, lr, seed) in RUNS.items():
        d = Path(a.wave1_dir if wave == 1 else a.wave2_dir) / rd
        dirs[lab] = d
        v = aw1.verify(lab, job, exp, d, Path(a.meta1_dir if wave == 1 else a.meta2_dir), frames_summary)
        want_sha = WAVE1_SHA if wave == 1 else WAVE2_SHA
        v["expected_sha"] = want_sha
        v["valid_completed_run"] = bool(v["valid_completed_run"] or (v["git_sha"] == want_sha and not v["missing"]
                                        and v["exit_code"] == 0 and v["parity_ok"] and v["parity_rc"] == 0
                                        and v["condition_id"] == aw1.CONDITION_ID
                                        and v["test_used_for_selection"] is False
                                        and v["best_pt_sha256_matches_selection"]
                                        and v["export_checkpoint_matches_selection"] and v["export_files_present"]
                                        and v["val_selection_frame_ids_match_m3"] and v["export_frame_ids_match_m3"]))
        v["valid_completed_run"] &= v["git_sha"] == want_sha
        assert v["config"]["lr"] == lr and v["config"]["seed"] == seed and v["config"]["tau"] == 0.05, lab
        ver[lab] = v
    dirs["u0_formal_job27"] = formal / "artifacts" / FORMAL_U0
    ver["u0_formal_job27"] = aw1.verify("u0_s0_formal", 27, "u0", dirs["u0_formal_job27"], None, frames_summary)
    fps = {json.dumps(v["data_fingerprint"], sort_keys=True) for v in ver.values()}
    proto = {json.dumps({k: v["config"][k] for k in ("tau", "batch_size", "max_epochs", "patience", "condition",
                                                     "user_features", "model", "history")}, sort_keys=True)
             for v in ver.values()}

    # ---------------------------------------------------------------- frames: rebuild + identity
    P = load_processed(paths.PROJECT_ROOT / "data" / "processed")
    frames = build_eval_frames(P, ("val",), ("masked",))        # asserts frame_id == frames_summary.json
    frame_check = {}
    for t in TASKS:
        f = frames[("val", t, "masked")]
        saved = pd.read_parquet(formal / "results" / "baselines" / "frames" / f"val_{t}_masked.parquet")
        gt = [list(f.gt_idx[f.gt_indptr[i]:f.gt_indptr[i + 1]]) for i in range(len(f.users))]
        frame_check[t] = {
            "frame_id": f.frame_id, "n_users": int(len(f.users)), "gt_pairs": int(len(f.gt_idx)),
            "mask_pairs": int(len(f.mask_idx)), "n_catalog_items": int(f.n_items),
            "saved_m3_frame_users_equal": bool(np.array_equal(saved.user_id.to_numpy(), f.users)),
            "saved_m3_frame_gt_equal": bool(all(list(x) == y for x, y in zip(saved.gt_rows, gt))),
            # n_masked_candidates = number of seen (masked) catalog rows per user (eval/frame.py)
            "saved_m3_frame_seen_mask_size_equal": bool(np.array_equal(saved.n_masked_candidates.to_numpy(),
                                                                       np.diff(f.mask_indptr))),
            "saved_m3_frame_hist_len_equal": bool(np.array_equal(saved.hist_len_capped.to_numpy(), f.hist_len_capped)),
        }

    # ---------------------------------------------------------------- comparison table
    rows, epochs = [], []
    for lab, d in dirs.items():
        sel = jload(d / "selection.json")
        b = best_record(sel)
        c = ver[lab]["config"]
        rows.append({"run": lab, "job": ver[lab]["job"], "git_sha": ver[lab]["git_sha"][:7], "lr": c["lr"],
                     "seed": c["seed"], "tau": c["tau"], "best_epoch": sel["best_epoch"],
                     "epochs_validated": len(sel["val_history"]),
                     **{f"{t}/val_recall@20": b[f"{t}/recall@20"] for t in TASKS},
                     **{f"{t}/val_ndcg@20": b[f"{t}/ndcg@20"] for t in TASKS},
                     "mean_val_recall@20": sel["best_score"], "mean_val_ndcg@20": sel["best_tie"],
                     "exit_code": ver[lab]["exit_code"], "parity_ok": ver[lab]["parity_ok"],
                     "valid_completed_run": ver[lab]["valid_completed_run"]})
        for h in sel["val_history"]:
            epochs.append({"run": lab, "epoch": h["epoch"], "score": h["score"], "tie": h["tie"],
                           **{f"{t}/val_recall@20": h[f"{t}/recall@20"] for t in TASKS}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "validation_comparison.csv", index=False)
    pd.DataFrame(epochs).to_csv(out / "validation_by_epoch.csv", index=False)
    grid = df[df.run != "u0_formal_job27"]
    metrics = ["mean_val_recall@20", "mean_val_ndcg@20", *[f"{t}/val_recall@20" for t in TASKS],
               *[f"{t}/val_ndcg@20" for t in TASKS]]

    seed_rows = []
    for lr, g in grid.groupby("lr"):
        for m in metrics:
            x = g[m].to_numpy()
            seed_rows.append({"lr": lr, "metric": m, "n_seeds": len(x), "mean": x.mean(), "std_ddof1": x.std(ddof=1),
                              "min": x.min(), "max": x.max(), "range": x.max() - x.min()})
    pd.DataFrame(seed_rows).to_csv(out / "seed_analysis.csv", index=False)

    pair_rows = []
    for s in (0, 1, 2):
        lo = grid[(grid.lr == 1e-3) & (grid.seed == s)].iloc[0]
        hi = grid[(grid.lr == 5e-4) & (grid.seed == s)].iloc[0]
        pair_rows.append({"seed": s, "lr1e-3_run": lo.run, "lr5e-4_run": hi.run,
                          **{f"delta/{m}": hi[m] - lo[m] for m in metrics}})
    pdf = pd.DataFrame(pair_rows)
    mean_row = {"seed": "mean", "lr1e-3_run": "", "lr5e-4_run": "",
                **{c: pdf[c].mean() for c in pdf.columns if c.startswith("delta/")}}
    pdf = pd.concat([pdf, pd.DataFrame([mean_row])], ignore_index=True)
    pdf.to_csv(out / "paired_lr_diffs.csv", index=False)

    # ---------------------------------------------------------------- per-user val: bootstrap + baselines
    pu, base_rows, boot_rows, mean4_reps = {}, [], [], {}
    for t in TASKS:
        f = frames[("val", t, "masked")]
        tab = {}
        for lab, d in dirs.items():
            tab[lab] = pd.read_parquet(_guard(d / "per_user" / f"val_{t}_masked.parquet"))
        for name, m in BASELINES.items():
            if name == "als_like_aux" and "is_like" not in t:
                continue
            p = formal / "results" / "baselines" / "per_user" / m / f"val_{t}_masked.parquet"
            tab[name] = pd.read_parquet(_guard(p))
        for name, x in tab.items():
            if not (np.array_equal(x.user_id.to_numpy(), f.users) and (x.frame_id == f.frame_id).all()
                    and np.array_equal(x.n_gt.to_numpy(), np.diff(f.gt_indptr))):
                raise ValueError(f"{name}/{t}: per-user file not on the rebuilt M3 val frame")
        pu[t] = tab
        for name, x in tab.items():
            base_rows.append({"task": t, "model": name, "n_users": len(x), "val_recall@20": x["recall@20"].mean(),
                              "val_ndcg@20": x["ndcg@20"].mean(),
                              "n_no_signal": int((~x.has_signal).sum()) if "has_signal" in x else None})
        res = paired_bootstrap({k: v[["user_id", "recall@20", "ndcg@20"]] for k, v in tab.items()},
                               ["recall@20", "ndcg@20"], a.n_boot, 0, f.population_id)
        pairs = [(f"lr5e-4_s{s} ({e})", f"lr1e-3_s{s} ({c})") for s, e, c in
                 ((0, "x_lr5e4", "x_u0_repro"), (1, "x_lr5e4_s1", "x_u0_s1"), (2, "x_lr5e4_s2", "x_u0_s2"))]
        pairs += [(f"lr5e-4_s{s} ({e})", b) for s, e in ((0, "x_lr5e4"), (1, "x_lr5e4_s1"), (2, "x_lr5e4_s2"))
                  for b in tab if b in BASELINES]
        pairs += [("lr1e-3_s0 (x_u0_repro)", b) for b in tab if b in BASELINES]
        for r in diff_rows(res, pairs, ["recall@20", "ndcg@20"]):
            boot_rows.append({"task": t, **r})
        for ra, rb in pairs[:3]:                       # same-seed lr pairs: keep replicates for the 4-task mean
            for k in ("recall@20", "ndcg@20"):
                mean4_reps.setdefault((ra, rb, k), []).append(res.boot[:, res.col(ra, k)] - res.boot[:, res.col(rb, k)])
    bdf = pd.DataFrame(base_rows)
    bdf.to_csv(out / "baselines_val_comparison.csv", index=False)
    # 4-task mean: the four task populations are bootstrapped independently; the mean of the per-task
    # replicate differences is one replicate of the 4-task mean difference.
    for (ra, rb, k), reps in mean4_reps.items():
        dd = np.mean(reps, axis=0)
        lo_, hi_ = np.quantile(dd, [0.025, 0.975])
        pt = float(df.set_index("run").loc[ra, f"mean_val_{k}"] - df.set_index("run").loc[rb, f"mean_val_{k}"])
        boot_rows.append({"task": "mean4", "model_a": ra, "model_b": rb, "metric": k, "diff": pt,
                          "boot_se": float(dd.std(ddof=1)), "ci_lo": float(lo_), "ci_hi": float(hi_),
                          "ci_excludes_0": bool(lo_ > 0 or hi_ < 0)})
    pd.DataFrame(boot_rows).to_csv(out / "validation_bootstrap.csv", index=False)
    # Sanity: per-user means reproduce selection.json best-epoch values (selection path vs export path).
    sel_vs_pu = {}
    for lab in dirs:
        r = df.set_index("run").loc[lab]
        sel_vs_pu[lab] = max(abs(pu[t][lab]["recall@20"].mean() - r[f"{t}/val_recall@20"]) for t in TASKS)

    # ---------------------------------------------------------------- repro vs formal job 27
    rep, f27 = dirs["lr1e-3_s0 (x_u0_repro)"], dirs["u0_formal_job27"]
    s44, s27 = jload(rep / "selection.json"), jload(f27 / "selection.json")
    keys = ["score", "tie", *[f"{t}/{m}" for t in TASKS for m in ("recall@20", "ndcg@20")]]
    vh = [{"epoch": h44["epoch"], "max_abs_diff": max(abs(h44[k] - h27[k]) for k in keys),
           "bitwise_equal": all(h44[k] == h27[k] for k in keys)} for h44, h27 in zip(s44["val_history"], s27["val_history"])]
    tl44 = [json.loads(x) for x in (rep / "train_log.jsonl").read_text().splitlines()]
    tl27 = [json.loads(x) for x in (f27 / "train_log.jsonl").read_text().splitlines()]
    tl_keys = ["loss", "grad_norm", *[f"loss_{t}" for t in TASKS]]
    tl_equal = len(tl44) == len(tl27) and all(x["step"] == y["step"] and all(x[k] == y[k] for k in tl_keys)
                                              for x, y in zip(tl44, tl27))
    pu_equal = {t: bool(np.array_equal(pu[t]["lr1e-3_s0 (x_u0_repro)"][["recall@20", "ndcg@20"]].to_numpy(),
                                       pu[t]["u0_formal_job27"][["recall@20", "ndcg@20"]].to_numpy())) for t in TASKS}
    import torch
    sd44 = torch.load(rep / "best.pt", map_location="cpu", weights_only=False)
    sd27 = torch.load(f27 / "best.pt", map_location="cpu", weights_only=False)
    w_equal = sorted(sd44["model"]) == sorted(sd27["model"]) and all(torch.equal(sd44["model"][k], sd27["model"][k])
                                                                    for k in sd27["model"])
    m44 = jload(rep / "export" / "manifest.json")
    m27 = jload(f27 / "export" / "manifest.json")
    emb_equal = {f: m44["files"][f] == m27["files"][f] for f in m27["files"] if f.startswith(("items_", "users_val_"))}
    repro = {"tier": "A (same cluster / GPU type / torch; different code SHA 9218a76 vs a158e5e)",
             "val_history_epochs": len(vh), "val_history_all_bitwise_equal": all(x["bitwise_equal"] for x in vh),
             "val_history_max_abs_diff": max(x["max_abs_diff"] for x in vh),
             "best_epoch": [s44["best_epoch"], s27["best_epoch"]],
             "train_log_records": [len(tl44), len(tl27)], "train_log_bitwise_equal": tl_equal,
             "best_pt_state_dict_bitwise_equal": bool(w_equal),
             "best_pt_file_sha_equal": s44["checkpoint_sha256"] == s27["checkpoint_sha256"],
             "export_val_and_item_embeddings_sha_equal": all(emb_equal.values()), "n_embedding_files_compared": len(emb_equal),
             "val_per_user_metrics_equal": pu_equal,
             "note": "best.pt file sha differs only if pickled metadata (fingerprint data_git_sha, train_config) differs; "
                     "weights compared tensor by tensor."}
    (out / "repro_vs_job27.json").write_text(json.dumps(repro, indent=1))

    (out / "run_verification.json").write_text(json.dumps(
        {"study": "S9 Wave 2 (post-hoc exploratory; designed after formal test results were analysed)",
         "files_read": "validation-side files only; metrics.csv and per_user/test_* refused",
         "frame_check_rebuilt_vs_saved_m3": frame_check,
         "distinct_data_fingerprints_excl_data_git_sha": len(fps), "distinct_protocol_configs": len(proto),
         "selection_vs_per_user_max_abs": sel_vs_pu, "runs": ver}, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(df[["run", "job", "git_sha", "lr", "seed", "best_epoch", "mean_val_recall@20", "mean_val_ndcg@20",
              "valid_completed_run"]].to_string())
    print(json.dumps(repro, indent=1))
    print("fingerprints", len(fps), "protocols", len(proto), "frame_check", json.dumps(frame_check)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
