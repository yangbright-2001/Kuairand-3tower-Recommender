"""S9 Wave 1 run verification + validation-only comparison (post-hoc exploratory study).

Reads ONLY:
  * selection.json (D1-8 val_history and the frozen selection), run_meta.json (minus nothing
    test-derived: run_meta carries config / sha / fingerprints / selection / timing / parity rc),
    config.yaml, parity.json (pass flags and counts only), export/manifest.json (file list,
    frame_ids keys are filtered to val), slurm status .rc files;
  * per_user/val_*_masked.parquet (validation per-user metrics, for the paired val bootstrap).
Never opens metrics.csv (it contains test rows) or any per_user/test_* file; the reader
functions refuse such paths.

Run (explore worktree):
  uv run python scripts/explore/analyze_wave1.py \
     --runs_dir artifacts/explore/wave1 --meta_dir artifacts/explore/wave1_meta \
     --u0_dir <formal_repo>/artifacts/u0/9218a76_seed0_job27
Outputs: results/explore/wave1/{validation_comparison.csv, validation_by_epoch.csv,
         validation_bootstrap.csv, run_verification.json}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kuairand_rec import paths
from kuairand_rec.experiments.bootstrap import diff_rows, paired_bootstrap

TASKS = ("video_long_view", "video_is_like", "music_long_view", "music_is_like")
EXPECTED_SHA = "1051f86f373697de7eca600d1dc3cd674d7d05f6"
CONDITION_ID = "133765eb12b75700"
RUNS = [  # (label, job, exp, run dir name)
    ("u0_s0_formal", 27, "u0", None),
    ("x_u0_s1", 39, "x_u0_s1", "x_u0_s1/1051f86_seed1_job39"),
    ("x_u0_s2", 40, "x_u0_s2", "x_u0_s2/1051f86_seed2_job40"),
    ("x_lr5e4", 41, "x_lr5e4", "x_lr5e4/1051f86_seed0_job41"),
    ("x_tau003", 42, "x_tau003", "x_tau003/1051f86_seed0_job42"),
    ("x_tau010", 43, "x_tau010", "x_tau010/1051f86_seed0_job43"),
]
REQUIRED = ["best.pt", "last.pt", "config.yaml", "selection.json", "selection_vs_export.json", "run_meta.json",
            "parity.json", "train_log.jsonl", "metrics.csv", "stdout.log", "stderr.log", "export/manifest.json",
            "export/video_ids.npy", "export/music_ids.npy", "export/user_ids_val.npy", "export/user_ids_test.npy",
            *[f"export/items_{t}.npy" for t in TASKS],
            *[f"export/users_{s}_{t}.npy" for s in ("val", "test") for t in TASKS],
            *[f"per_user/val_{t}_masked.parquet" for t in TASKS]]


def _guard(p: Path) -> Path:
    n = p.name
    if n == "metrics.csv" or (p.parent.name == "per_user" and not n.startswith("val_")):
        raise PermissionError(f"test-bearing file must not be read: {p}")
    return p


def jload(p: Path) -> dict:
    return json.loads(_guard(p).read_text())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def best_record(sel: dict) -> dict:
    return next(h for h in sel["val_history"] if h["epoch"] == sel["best_epoch"] and h.get("improved"))


def verify(label, job, exp, d: Path, meta_dir: Path | None, frames_summary: dict) -> dict:
    v = {"label": label, "job": job, "run_dir": str(d)}
    v["missing"] = [f for f in REQUIRED if not (d / f).exists()]
    sel, meta, par = jload(d / "selection.json"), jload(d / "run_meta.json"), jload(d / "parity.json")
    man = jload(d / "export" / "manifest.json")
    cfg = yaml.safe_load(_guard(d / "config.yaml").read_text())
    v["git_sha"] = meta["git_sha"]
    v["condition_id"] = meta["condition_id"]
    v["test_used_for_selection"] = sel["test_used_for_selection"]
    v["best_pt_sha256_matches_selection"] = sha256(d / "best.pt") == sel["checkpoint_sha256"]
    v["export_checkpoint_matches_selection"] = man.get("checkpoint_sha256") == sel["checkpoint_sha256"]
    v["export_files_listed"] = len(man["files"])
    v["export_files_present"] = all((d / "export" / f).exists() for f in man["files"])
    v["parity_ok"] = bool(par.get("ok"))
    c12 = par["check1_2_scores_topk"]["tasks"]
    v["parity"] = {"ok": par.get("ok"), "rebuilt_from_raw": par.get("rebuilt_from_raw"),
                   "check1_max_abs": max(x["max_abs"] for x in c12.values()),
                   "check1_all_bitwise_equal": all(x["bitwise_equal"] for x in c12.values()),
                   "check2_top50_mismatch_users": sum(int(x.get(k, 0)) for x in c12.values()
                                                      for k in x if k.startswith("top50") and k.endswith("mismatch_users")),
                   "check3_retrieve_ok": par["check3_retrieve"]["ok"], "check4_metrics_ok": par["check4_metrics"]["ok"],
                   "n_users": par["users"]["n_users"], "strata": par["users"]["strata"]}
    sve = jload(d / "selection_vs_export.json")
    v["selection_vs_export"] = {k: sve[k] for k in ("max_abs_diff", "near_tie_epochs", "selection_may_be_sensitive")}
    v["parity_rc"] = (meta.get("parity") or {}).get("returncode")
    v["data_fingerprint"] = {k: meta["data_fingerprint"][k] for k in sorted(meta["data_fingerprint"]) if k != "data_git_sha"}
    # Val frame ids used for selection must equal the M3 val masked frames (frames_summary.json).
    want = {t: frames_summary[f"val/{t}/masked"] for t in TASKS}
    got = best_record(sel)["frame_ids"]
    v["val_selection_frame_ids_match_m3"] = all(got.get(t) == want[t] for t in TASKS)
    v["export_frame_ids_match_m3"] = all(man["frame_ids"].get(k) == frames_summary[k] for k in frames_summary
                                         if k in man["frame_ids"]) and len(man["frame_ids"]) == 24
    v["config"] = {"seed": cfg["train"]["seed"], "lr": cfg["train"]["lr"], "tau": cfg["train"]["tau"],
                   "batch_size": cfg["train"]["batch_size"], "max_epochs": cfg["train"]["max_epochs"],
                   "patience": cfg["train"]["patience"], "condition": cfg["condition"],
                   "user_features": cfg["user_features"], "model": cfg["model"], "history": cfg["history"]}
    if meta_dir is not None:
        rc = meta_dir / "slurm" / "status" / f"{exp}_{job}.rc"
        v["slurm_rc_line"] = rc.read_text().strip() if rc.exists() else None
        v["exit_code"] = int(v["slurm_rc_line"].split()[0].split("=")[1]) if v["slurm_rc_line"] else None
        err = meta_dir / "slurm" / f"{exp}_{job}.err"
        v["stderr_bytes"] = err.stat().st_size if err.exists() else None
    else:
        v["exit_code"] = 0 if v["parity_rc"] == 0 else None   # formal run: verified in Track A
        v["slurm_rc_line"] = "formal M4 run (Track A verified; rc file not part of this check)"
    v["valid_completed_run"] = bool(not v["missing"] and v["exit_code"] == 0 and v["parity_ok"]
                                    and v["parity_rc"] == 0 and v["condition_id"] == CONDITION_ID
                                    and v["test_used_for_selection"] is False and v["best_pt_sha256_matches_selection"]
                                    and v["export_checkpoint_matches_selection"] and v["export_files_present"]
                                    and v["val_selection_frame_ids_match_m3"] and v["export_frame_ids_match_m3"]
                                    and (label == "u0_s0_formal" or v["git_sha"] == EXPECTED_SHA))
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--meta_dir", required=True)
    ap.add_argument("--u0_dir", required=True)
    ap.add_argument("--out_dir", default=str(paths.PROJECT_ROOT / "results" / "explore" / "wave1"))
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fs = json.loads((paths.PROJECT_ROOT / "results" / "baselines" / "frames_summary.json").read_text())
    frames_summary = {k: v["frame_id"] for k, v in fs.items()}       # "split/task/kind" -> frame_id (24)
    assert len(frames_summary) == 24

    dirs = {lab: (Path(a.u0_dir) if rd is None else Path(a.runs_dir) / rd) for lab, _, _, rd in RUNS}
    ver, rows, epochs = {}, [], []
    for lab, job, exp, rd in RUNS:
        d = dirs[lab]
        ver[lab] = verify(lab, job, exp, d, None if rd is None else Path(a.meta_dir), frames_summary)
        sel = jload(d / "selection.json")
        b = best_record(sel)
        c = ver[lab]["config"]
        row = {"experiment": lab, "job": job, "git_sha": ver[lab]["git_sha"][:7], "seed": c["seed"], "lr": c["lr"],
               "tau": c["tau"], "best_epoch": sel["best_epoch"], "epochs_validated": len(sel["val_history"]),
               **{f"{t}/val_recall@20": b[f"{t}/recall@20"] for t in TASKS},
               **{f"{t}/val_ndcg@20": b[f"{t}/ndcg@20"] for t in TASKS},
               "mean_val_recall@20": sel["best_score"], "mean_val_ndcg@20": sel["best_tie"],
               "parity_ok": ver[lab]["parity_ok"], "exit_code": ver[lab]["exit_code"],
               "valid_completed_run": ver[lab]["valid_completed_run"]}
        assert abs(np.mean([b[f"{t}/recall@20"] for t in TASKS]) - sel["best_score"]) < 1e-12
        rows.append(row)
        for h in sel["val_history"]:
            epochs.append({"experiment": lab, "epoch": h["epoch"], "score": h["score"], "tie": h["tie"],
                           **{f"{t}/val_recall@20": h[f"{t}/recall@20"] for t in TASKS}, "improved": h["improved"]})
    df = pd.DataFrame(rows)
    ref = df.set_index("experiment").loc["u0_s0_formal"]
    for col in ["mean_val_recall@20", "mean_val_ndcg@20", *[f"{t}/val_recall@20" for t in TASKS]]:
        df[f"delta_vs_u0s0/{col}"] = df[col] - ref[col]
    seeds = df[df.experiment.isin(["u0_s0_formal", "x_u0_s1", "x_u0_s2"])]
    seed_stats = {col: {"mean": float(seeds[col].mean()), "std_ddof1": float(seeds[col].std(ddof=1)),
                        "min": float(seeds[col].min()), "max": float(seeds[col].max()),
                        "range": float(seeds[col].max() - seeds[col].min())}
                  for col in ["mean_val_recall@20", "mean_val_ndcg@20", *[f"{t}/val_recall@20" for t in TASKS]]}
    df.to_csv(out / "validation_comparison.csv", index=False)
    pd.DataFrame(epochs).to_csv(out / "validation_by_epoch.csv", index=False)

    # Paired user bootstrap on VAL per-user files (best checkpoint of each run vs U0 s0), per task,
    # plus the 4-task mean computed per replicate on the same draws (one population per task).
    boot_rows = []
    labs = [lab for lab, *_ in RUNS]
    for t in TASKS:
        pu = {}
        for lab in labs:
            p = _guard(dirs[lab] / "per_user" / f"val_{t}_masked.parquet")
            pu[lab] = pd.read_parquet(p)[["user_id", "recall@20", "ndcg@20", "population_id", "frame_id"]]
        fid = {x.frame_id.iloc[0] for x in pu.values()}
        assert len(fid) == 1 and fid == {frames_summary[f"val/{t}/masked"]}
        res = paired_bootstrap(pu, ["recall@20", "ndcg@20"], a.n_boot, 0, pu[labs[0]].population_id.iloc[0])
        for r in diff_rows(res, [(lab, "u0_s0_formal") for lab in labs[1:]], ["recall@20", "ndcg@20"]):
            boot_rows.append({"task": t, **r})
        for lab in labs:
            for k in ("recall@20", "ndcg@20"):
                boot_rows.append({"task": t, "model_a": lab, "model_b": None, "metric": k, "_boot": res.boot[:, res.col(lab, k)]})
    # 4-task mean difference per replicate: the four task populations are bootstrapped independently
    # (different user sets), so the mean of per-task replicate differences is a valid replicate.
    reps = {}
    for r in boot_rows:
        if "_boot" in r:
            reps[(r["task"], r["model_a"], r["metric"])] = r["_boot"]
    mean_rows = []
    for lab in labs[1:]:
        for k in ("recall@20", "ndcg@20"):
            d = np.mean([reps[(t, lab, k)] - reps[(t, "u0_s0_formal", k)] for t in TASKS], axis=0)
            pt = float(df.set_index("experiment").loc[lab, f"mean_val_{k}"] - ref[f"mean_val_{k}"])
            lo, hi = np.quantile(d, [0.025, 0.975])
            mean_rows.append({"task": "mean4", "model_a": lab, "model_b": "u0_s0_formal", "metric": k, "diff": pt,
                              "boot_se": float(d.std(ddof=1)), "ci_lo": float(lo), "ci_hi": float(hi),
                              "ci_excludes_0": bool(lo > 0 or hi < 0)})
    bdf = pd.DataFrame([r for r in boot_rows if "_boot" not in r] + mean_rows)
    bdf.to_csv(out / "validation_bootstrap.csv", index=False)

    (out / "run_verification.json").write_text(json.dumps(
        {"study": "S9 Wave 1, post-hoc exploratory validation study (designed after the formal test results "
                  "had been analysed; not a pre-registered test-independent optimisation)",
         "expected_explore_sha": EXPECTED_SHA, "condition_id": CONDITION_ID,
         "files_read": "selection.json, run_meta.json, config.yaml, parity.json, export/manifest.json, slurm .rc, "
                       "per_user/val_*_masked.parquet; never metrics.csv or per_user/test_*",
         "runs": ver, "seed_stats_u0_s0_s1_s2": seed_stats}, indent=1, default=str))
    print(df[["experiment", "best_epoch", "mean_val_recall@20", "mean_val_ndcg@20", "valid_completed_run"]].to_string())
    print(json.dumps({k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in seed_stats.items()}, indent=1))
    print(bdf[bdf.task == "mean4"].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
