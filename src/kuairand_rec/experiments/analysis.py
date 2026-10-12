"""M5 main-comparison analysis: paired user bootstrap CIs + history-length buckets.

Run (CPU, reads finished results only; never trains or re-scores a model):
    uv run python -m kuairand_rec.experiments.analysis --config configs/m5_analysis.yaml
    # point at other finished runs without editing the config:
    uv run python -m kuairand_rec.experiments.analysis --config configs/m5_analysis.yaml \
        --run u0=/path/to/u0_run --run abl_nohist=/path/to/run

Inputs: the per-user parquet files written by the M3 baseline runner and by the three-tower
runner (`ExportedScorer` metrics, D1-7). Frames are rebuilt from the M2 artifacts with the shared
`build_eval_frames` (frame_ids asserted equal to M3 `frames_summary.json`), and every per-user
table is validated against its frame (`sources.load_per_user`) and against its own metrics.csv.

Outputs (`output_dir`, default results/m5/main):
  bootstrap_models.csv   per (task, frame, stratum, model, metric): n_users, mean, SE, 95% CI
  bootstrap_diffs.csv    per comparison a - b (paired on the same resampled users)
  bucket_counts.csv      users / GT pairs per history bucket and task
  user_sets/*.csv        evaluated user ids per frame (the bootstrap population)
  analysis_meta.json     seed, n_boot, CI method, metric definitions, frame ids, input sha256s,
                         run provenance, consistency checks
  MAIN_ANALYSIS.md       rendered tables
Strata: "all" (the frame population) and the history buckets 1-5 / 6-20 / >20 (capped history
length fed to the model at T_test, P1-2). The cold-user frame (no history) is analysed separately.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kuairand_rec.data.pipeline import load_processed
from kuairand_rec.experiments import bootstrap as B
from kuairand_rec.experiments import buckets as BK
from kuairand_rec.experiments import sources as S
from kuairand_rec.utils import git_sha

METRIC_DEFINITIONS = {
    "recall@K": "hits@K / |GT(u)| (P1-1); GT = deduped test positives minus seen / OOC / sentinel items",
    "ndcg@K": "binary-gain DCG@K / IDCG over min(|GT(u)|, K) relevant items",
    "ranking": "full catalog, score desc then catalog row asc; masked frames set seen items to -inf",
    "aggregate": "equal-weight mean over evaluated users (= eval.evaluator.summarize)",
    "bootstrap": "paired user-level resampling with replacement, n = population size, identical "
                 "replicates for every model of a (frame, stratum); stream = SeedSequence([seed, "
                 "sha256(population_id/stratum)])",
    "ci": B.CI_METHOD,
    "p_two_sided": "min(1, 2 min(P*(d<=0), P*(d>=0))) over replicates; descriptive, not corrected",
    "history_bucket": "min(#events with t < T_test, 100) from the unfiltered log (P1-2); 1-5 / 6-20 / >20",
}


def _strata(frame, bucketed: bool):
    if not bucketed:
        return [("all", np.ones(len(frame.users), bool))]
    b = BK.frame_buckets(frame)
    return [("all", np.ones(len(b), bool))] + [(k, b == k) for k in BK.BUCKETS]


def analyse_frame(frame, per_user: dict, comparisons, metrics, n_boot: int, seed: int, ci: float,
                  bucketed: bool) -> tuple[list, list, dict]:
    """Bootstrap all models on one frame (+ bucket strata). per_user: {model: aligned df}."""
    mrows, drows, checks = [], [], {}
    models = list(per_user)
    pairs = [(a, b) for a, b in comparisons if a in per_user and b in per_user]
    if bucketed:
        checks["bucket_partition_max_dev"] = max(BK.check_partition(frame, df, metrics) for df in per_user.values())
    for stratum, sel in _strata(frame, bucketed):
        if not sel.any():
            continue
        sub = {m: per_user[m][sel].reset_index(drop=True) for m in models}
        key = f"{frame.population_id}/{stratum}"
        res = B.paired_bootstrap(sub, metrics, n_boot, seed, key)
        common = {"split": frame.split, "task": frame.task, "frame_kind": frame.kind, "stratum": stratum,
                  "population_id": frame.population_id, "frame_id": frame.frame_id}
        mrows += [{**common, **r} for r in B.model_rows(res, ci)]
        drows += [{**common, **r} for r in B.diff_rows(res, pairs, metrics, ci)]
    return mrows, drows, checks


def run(cfg: dict, run_overrides: dict | None = None, P: dict | None = None, frames_summary="default") -> dict:
    """`P` / `frames_summary` are injectable for fixture tests; by default the M2 artifacts are
    loaded and every frame_id is asserted equal to the M3 `frames_summary.json`."""
    from kuairand_rec.models import artifacts as A
    from kuairand_rec.models.evaluation import FRAMES_SUMMARY, build_eval_frames

    t0 = time.time()
    frames_summary = FRAMES_SUMMARY if frames_summary == "default" else frames_summary
    out = S.resolve(cfg["output_dir"])
    (out / "user_sets").mkdir(parents=True, exist_ok=True)
    bc = cfg["bootstrap"]
    n_boot, seed, ci = int(bc["n_boot"]), int(bc["seed"]), float(bc["ci"])
    metrics = list(cfg.get("metrics", S.METRICS))
    tasks, split = list(cfg["tasks"]), cfg.get("split", "test")
    frame_kinds = list(cfg.get("frame_kinds", ["masked"]))

    P = load_processed(S.resolve(cfg.get("processed_dir", "data/processed"))) if P is None else P
    frames = build_eval_frames(P, splits=(split,), kinds=tuple(dict.fromkeys(frame_kinds + ["cold"])),
                               frames_summary=frames_summary)
    data_fp = A.data_fingerprint(P["events"], P["meta"])

    sources, pending, provenance = {}, [], {}
    for name, c in cfg["models"].items():
        c = dict(c)
        if run_overrides and name in run_overrides:
            c["run_dir"] = run_overrides[name]
        src = S.source_from_cfg(name, c)
        if src is None:
            pending.append(name)
            continue
        if src.kind == "threetower":
            provenance[name] = S.verify_threetower_run(src, data_fp)
        sources[name] = src
    comparisons = [tuple(x) for x in cfg["comparisons"]]

    mrows, drows, inputs, checks, user_sets, bucket_rows = [], [], {}, {}, {}, []

    def _collect(frame, kind_label, model_filter, bucketed):
        if len(frame.users) == 0:
            checks[f"{kind_label}/{frame.task}"] = {"skipped": "empty frame"}
            return
        per_user = {}
        for name, src in sources.items():
            if not src.has_task(frame.task) or not model_filter(name, src):
                continue
            df, prov = S.load_per_user(src, frame)
            chk = S.check_summary_consistency(src, df, frame)
            per_user[name] = df
            inputs[f"{kind_label}/{frame.task}/{name}"] = {**prov, **chk}
        if not per_user:
            return
        m, d, ch = analyse_frame(frame, per_user, comparisons, metrics, n_boot, seed, ci, bucketed)
        mrows.extend(m)
        drows.extend(d)
        checks[f"{kind_label}/{frame.task}"] = {**ch, "models": list(per_user)}

    for task in tasks:
        for kind in frame_kinds:
            f = frames[(split, task, kind)]
            _collect(f, kind, lambda n, s: True, bucketed=True)
            uid = S.user_set_id(f.users)
            user_sets[f"{split}/{task}/{kind}"] = {"n_users": int(len(f.users)), "population_id": f.population_id,
                                                  "frame_id": f.frame_id, "user_ids_sha256": uid,
                                                  "file": f"user_sets/{split}_{task}_{kind}.csv"}
            pd.DataFrame({"user_id": f.users}).to_csv(out / "user_sets" / f"{split}_{task}_{kind}.csv", index=False)
            bc_df = BK.bucket_counts(f)
            bc_df.insert(0, "task", task)
            bc_df.insert(1, "frame_kind", kind)
            bucket_rows.append(bc_df)
        if cfg.get("cold_user_frame", True):
            # Cold users (no history): only models evaluated there (Pop, three-tower; P1-1).
            f = frames[(split, task, "cold")]
            _collect(f, "cold", lambda n, s: s.per_user_path(f.split, f.task, "cold").exists(), bucketed=False)
            user_sets[f"{split}/{task}/cold"] = {"n_users": int(len(f.users)), "population_id": f.population_id,
                                                "frame_id": f.frame_id, "user_ids_sha256": S.user_set_id(f.users)}

    models_df = pd.DataFrame(mrows)
    diffs_df = pd.DataFrame(drows)
    buckets_df = pd.concat(bucket_rows, ignore_index=True)
    models_df.to_csv(out / "bootstrap_models.csv", index=False)
    diffs_df.to_csv(out / "bootstrap_diffs.csv", index=False)
    buckets_df.to_csv(out / "bucket_counts.csv", index=False)
    meta = {"config": cfg, "run_overrides": run_overrides or {}, "git_sha": git_sha(),
            "data_fingerprint": data_fp, "n_boot": n_boot, "seed": seed, "ci": ci, "metrics": metrics,
            "metric_definitions": METRIC_DEFINITIONS, "pending_models": pending,
            "comparisons": [list(c) for c in comparisons], "user_sets": user_sets, "inputs": inputs,
            "threetower_provenance": provenance, "checks": checks, "runtime_s": round(time.time() - t0, 1)}
    (out / "analysis_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "MAIN_ANALYSIS.md").write_text(render_markdown(models_df, diffs_df, buckets_df, meta))
    return {"models": models_df, "diffs": diffs_df, "buckets": buckets_df, "meta": meta}


# Display names for the markdown only; CSV / bootstrap rows keep the stable model keys.
DISPLAY = {"als": "als (pure)", "als_like_aux": "als_like_aux"}


def _d(name: str) -> str:
    return DISPLAY.get(name, name)


def _fmt_ci(r, k="mean"):
    return f"{r[k]:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]"


def render_markdown(mdf: pd.DataFrame, ddf: pd.DataFrame, bdf: pd.DataFrame, meta: dict) -> str:
    L = ["# M5 main comparison: paired user bootstrap and history buckets", "",
         f"Generated by `uv run python -m kuairand_rec.experiments.analysis`. git `{meta['git_sha']}`. "
         f"{meta['n_boot']} paired user resamples, seed {meta['seed']}, {int(meta['ci'] * 100)}% percentile CIs. "
         "Test split, masked frames (the M3 frames; frame_ids asserted). Values: mean [CI].",
         "`als (pure)` = M3 `als` (aux_w = 0; on like tasks = als_like_pure, the O1 headline); "
         "`als_like_aux` = like + 0.3 long_view signal (O2), a separate model. CSV files keep the keys "
         "`als` / `als_like_aux`.", ""]
    if meta["pending_models"]:
        L += [f"Pending (not produced yet, not analysed): {', '.join(meta['pending_models'])}.", ""]
    for m in ("recall@20", "ndcg@20"):
        if mdf.empty:
            break
        L += [f"## Models, {m} (stratum = all)", ""]
        sub = mdf[(mdf.metric == m) & (mdf.stratum == "all") & (mdf.frame_kind == "masked")]
        models = list(dict.fromkeys(sub.model))
        L += ["| task | users | " + " | ".join(map(_d, models)) + " |", "|---|---:|" + "---|" * len(models)]
        for task, g in sub.groupby("task", sort=False):
            cells = []
            for mo in models:
                r = g[g.model == mo]
                cells.append(_fmt_ci(r.iloc[0]) if len(r) else "n/a")
            L.append(f"| {task} | {int(g.n_users.iloc[0]):,} | " + " | ".join(cells) + " |")
        L.append("")
    if not ddf.empty:
        L += ["## Paired differences (a - b), stratum = all", "",
              "| task | a - b | metric | diff [95% CI] | excludes 0 |", "|---|---|---|---|---|"]
        sub = ddf[(ddf.stratum == "all") & (ddf.frame_kind == "masked") & ddf.metric.isin(["recall@20", "ndcg@20"])]
        for _, r in sub.iterrows():
            L.append(f"| {r.task} | {_d(r.model_a)} - {_d(r.model_b)} | {r.metric} | {_fmt_ci(r, 'diff')} | "
                     f"{'yes' if r.ci_excludes_0 else 'no'} |")
        L.append("")
    L += ["## History buckets (capped history length at T_test; main masked population)", "",
          "| task | bucket | users | share | GT pairs |", "|---|---|---:|---:|---:|"]
    for _, r in bdf.iterrows():
        L.append(f"| {r.task} | {r.bucket} | {r.n_users:,} | {r.share:.1%} | {r.gt_pairs:,} |")
    L.append("")
    if not mdf.empty:
        L += ["### Recall@20 by bucket", ""]
        sub = mdf[(mdf.metric == "recall@20") & (mdf.stratum != "all") & (mdf.frame_kind == "masked")]
        models = list(dict.fromkeys(sub.model))
        L += ["| task | bucket | " + " | ".join(map(_d, models)) + " |", "|---|---|" + "---|" * len(models)]
        for (task, st), g in sub.groupby(["task", "stratum"], sort=False):
            cells = [(_fmt_ci(g[g.model == mo].iloc[0]) if (g.model == mo).any() else "n/a") for mo in models]
            L.append(f"| {task} | {st} | " + " | ".join(cells) + " |")
        L.append("")
        cold = mdf[(mdf.frame_kind == "cold") & (mdf.metric == "recall@20")]
        if len(cold):
            L += ["## Cold users (no history before T_test; separate frame, P1-1), recall@20", "",
                  "| task | users | model | recall@20 [CI] |", "|---|---:|---|---|"]
            for _, r in cold.iterrows():
                L.append(f"| {r.task} | {r.n_users:,} | {_d(r.model)} | {_fmt_ci(r)} |")
            L.append("")
    L += ["## Definitions", ""] + [f"- **{k}**: {v}" for k, v in meta["metric_definitions"].items()]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", action="append", default=[], help="model=run_dir override (three-tower)")
    a = ap.parse_args(argv)
    ov = dict(x.split("=", 1) for x in a.run)
    r = run(yaml.safe_load(Path(a.config).read_text()), ov)
    print(json.dumps({"rows_models": len(r["models"]), "rows_diffs": len(r["diffs"]),
                      "pending": r["meta"]["pending_models"], "runtime_s": r["meta"]["runtime_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
