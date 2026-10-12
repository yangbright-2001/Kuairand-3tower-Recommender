"""Load already-produced per-user metric tables and prove they belong to the shared frames.

M5 never re-scores the main comparison: it reads the M3 baseline per-user parquet files and the
per-user parquet files that `models.train.run` wrote (from `ExportedScorer`, D1-7). Every table
is checked against the frame it claims to come from before any statistic is computed:

* frame_id / population_id columns equal the freshly rebuilt frame (which `build_eval_frames`
  already asserted equal to `results/baselines/frames_summary.json`);
* user_id column == frame users, same order; n_gt == |GT| of the frame;
* the per-user means reproduce the model's own `metrics.csv` row (|d| <= 1e-12);
* three-tower runs: selection did not use test, parity passed, condition_id is the expected
  one, and the export was produced from the same data (content fingerprint).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kuairand_rec import paths
from kuairand_rec.eval.evaluator import KS
from kuairand_rec.eval.frame import EvalFrame

METRICS = tuple(f"{p}@{k}" for p in ("recall", "ndcg") for k in KS)
REQUIRED_COLS = ("user_id", "n_gt", *METRICS, "frame_id", "population_id")
MEAN_TOL = 1e-12


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else paths.PROJECT_ROOT / p


def sha256_file(p) -> str:
    from kuairand_rec.models.artifacts import sha256_file as _s
    return _s(p)


@dataclass(frozen=True)
class ModelSource:
    """Where one model's per-user tables live.

    kind = "baseline": `<dir>/<m3_name>/<split>_<task>_<kind>.parquet` (M3 layout), summary rows
           in `<dir>/../metrics.csv` filtered by model = m3_name.
    kind = "threetower": `<run_dir>/per_user/<split>_<task>_<kind>.parquet`, summary in
           `<run_dir>/metrics.csv` (model = "threetower").
    """

    name: str
    kind: str
    path: Path
    m3_name: str | None = None
    tasks: tuple | None = None          # restrict to these tasks (e.g. als_like_aux: like tasks)
    expect_condition_id: str | None = None
    meta: dict = field(default_factory=dict)

    def has_task(self, task: str) -> bool:
        return self.tasks is None or task in self.tasks

    def per_user_path(self, split: str, task: str, kind: str) -> Path:
        if self.kind == "baseline":
            return self.path / "per_user" / self.m3_name / f"{split}_{task}_{kind}.parquet"
        return self.path / "per_user" / f"{split}_{task}_{kind}.parquet"

    def metrics_csv(self) -> Path:
        return self.path / "metrics.csv"

    def summary_model(self) -> str:
        return self.m3_name if self.kind == "baseline" else "threetower"


def source_from_cfg(name: str, c: dict) -> ModelSource | None:
    """None if the model is configured but not produced yet (run_dir: null)."""
    kind = c["kind"]
    if kind == "baseline":
        return ModelSource(name, "baseline", resolve(c.get("dir", "results/baselines")), c["m3_name"],
                           tuple(c["tasks"]) if c.get("tasks") else None)
    if kind == "threetower":
        if not c.get("run_dir"):
            return None
        return ModelSource(name, "threetower", resolve(c["run_dir"]), None,
                           tuple(c["tasks"]) if c.get("tasks") else None, c.get("expect_condition_id"))
    raise ValueError(f"unknown model kind {kind}")


def load_per_user(src: ModelSource, frame: EvalFrame, frame_kind: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Per-user table of `src` on `frame`, validated against the frame. Returns (df, provenance)."""
    kind = frame_kind or frame.kind
    p = src.per_user_path(frame.split, frame.task, kind)
    if not p.exists():
        raise FileNotFoundError(f"{src.name}: missing per-user file {p}")
    df = pd.read_parquet(p)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{src.name}: {p.name} lacks columns {missing}")
    if df.frame_id.nunique() != 1 or df.frame_id.iloc[0] != frame.frame_id:
        raise ValueError(f"{src.name}: {p.name} frame_id != shared frame {frame.frame_id}")
    if df.population_id.iloc[0] != frame.population_id:
        raise ValueError(f"{src.name}: {p.name} population_id differs")
    if not np.array_equal(df.user_id.to_numpy(np.int64), frame.users):
        raise ValueError(f"{src.name}: {p.name} users differ from the frame (set or order)")
    if not np.array_equal(df.n_gt.to_numpy(np.int64), np.diff(frame.gt_indptr)):
        raise ValueError(f"{src.name}: {p.name} n_gt differs from the frame GT")
    for m in METRICS:
        v = df[m].to_numpy(np.float64)
        if not (np.isfinite(v).all() and (v >= 0).all() and (v <= 1 + 1e-9).all()):
            raise ValueError(f"{src.name}: {p.name} {m} outside [0, 1]")
    return df.reset_index(drop=True), {"file": str(p), "sha256": sha256_file(p), "n_users": int(len(df))}


def check_summary_consistency(src: ModelSource, df: pd.DataFrame, frame: EvalFrame, frame_kind: str | None = None) -> dict:
    """The per-user means must reproduce the model's own reported summary row."""
    s = pd.read_csv(src.metrics_csv())
    kind = frame_kind or frame.kind
    r = s[(s.model == src.summary_model()) & (s.split == frame.split) & (s.target == frame.target)
          & (s.action == frame.action) & (s.kind == kind)]
    if len(r) != 1:
        raise ValueError(f"{src.name}: expected 1 summary row for {frame.split}/{frame.task}/{kind}, got {len(r)}")
    r = r.iloc[0]
    if r.frame_id != frame.frame_id or int(r.n_users) != len(df):
        raise ValueError(f"{src.name}: metrics.csv row does not match the frame")
    dev = max(abs(float(df[m].mean()) - float(r[m])) for m in METRICS)
    if dev > MEAN_TOL:
        raise ValueError(f"{src.name}: per-user means differ from metrics.csv by {dev}")
    return {"metrics_csv_max_abs_dev": dev}


def verify_threetower_run(src: ModelSource, data_fp: dict) -> dict:
    """Provenance of a finished three-tower run (read-only)."""
    d = src.path
    meta = json.loads((d / "run_meta.json").read_text())
    sel = json.loads((d / "selection.json").read_text())
    man = json.loads((d / "export" / "manifest.json").read_text())
    parity = json.loads((d / "parity.json").read_text()) if (d / "parity.json").exists() else None
    problems = []
    if sel.get("test_used_for_selection") is not False:
        problems.append("selection.json does not state test_used_for_selection = false")
    if parity is None or not parity.get("ok"):
        problems.append("parity.json missing or not ok")
    if man["condition_id"] != meta["condition_id"]:
        problems.append("export manifest condition_id != run_meta condition_id")
    if src.expect_condition_id and meta["condition_id"] != src.expect_condition_id:
        problems.append(f"condition_id {meta['condition_id']} != expected {src.expect_condition_id}")
    if man["checkpoint_sha256"] != sel["checkpoint_sha256"]:
        problems.append("export was not produced from the selected checkpoint")
    # Content identity of the evaluation data (data_git_sha may legitimately differ between
    # machines that built the same M2 artifacts from different commits).
    fp = {k: v for k, v in man["data_fingerprint"].items() if k != "data_git_sha"}
    want = {k: v for k, v in data_fp.items() if k != "data_git_sha"}
    if fp != want:
        problems.append("export data fingerprint differs from the local M2 artifacts")
    if problems:
        raise ValueError(f"{src.name} ({d}): " + "; ".join(problems))
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    return {"run_dir": str(d), "git_sha": meta.get("git_sha"), "condition_id": meta["condition_id"],
            "exp": cfg.get("exp"), "user_features": cfg.get("user_features"), "model_config": cfg.get("model"),
            "condition": cfg.get("condition"), "best_epoch": sel.get("best_epoch"),
            "best_val_score": sel.get("best_score"), "checkpoint_sha256": sel["checkpoint_sha256"],
            "parity_ok": bool(parity["ok"]), "data_git_sha_of_export": man["data_fingerprint"].get("data_git_sha")}


def user_set_id(users: np.ndarray) -> str:
    import hashlib
    a = np.ascontiguousarray(np.asarray(users, dtype="<i8"))
    return hashlib.sha256(a.tobytes()).hexdigest()
