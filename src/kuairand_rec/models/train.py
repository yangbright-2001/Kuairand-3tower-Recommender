"""Three-tower trainer (PLAN.md §3d D1-3, D1-5, D1-8, D1-11).

Run (CPU smoke on real data):
    uv run python -m kuairand_rec.models.train --config configs/threetower_smoke.yaml --max_steps 5

One training step:
    1. `encode_items` ONCE over the full video + music catalogs (train mode: one ID-dropout draw,
       one MusicTower call). The same `ItemEncodings` feed the video candidates, the music
       candidates and the history tokens, so all three share gradients into the item towers.
    2. User tower on the batch (rolling strict-past history built in `models/data.py`).
    3. Equal-weight sum of the 4 full-softmax task losses (`models/loss.py`).
    4. Fail fast on a non-finite loss / grad norm (debug dump, never skipped), clip, AdamW step.

`run()` (the CLI) then: validation after each epoch on the val masked frames only
(`evaluation.Validator`, D1-8) -> `best.pt` + `selection.json` (frozen) -> export
(`export.export_embeddings`) -> val + test metrics through `ExportedScorer` and the shared M3
evaluator -> strict Direct/Export parity in a separate process (`models.parity`).
The training loop itself never touches val/test labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml
from torch import nn

from kuairand_rec import paths
from kuairand_rec.data.pipeline import load_processed
from kuairand_rec.models import data as D
from kuairand_rec.models import features as F
from kuairand_rec.models.loss import multitask_loss
from kuairand_rec.models.towers import TASKS, MaskedEmbedding, ThreeTowerModel, TowerConfig
from kuairand_rec.utils import git_sha, set_seed


@dataclass(frozen=True)
class TrainConfig:
    tau: float = 0.05               # D1-3
    lr: float = 1e-3
    warmup_steps: int = 300         # linear warmup, then constant
    weight_decay: float = 0.01      # Linear / Transformer weights only
    clip_grad_norm: float = 1.0
    batch_size: int = 1024          # OPEN until the GPU benchmark (PLAN.md §4)
    max_epochs: int = 30
    patience: int = 3               # used only with a validate callback (D1-8)
    like_oversample: int = 1        # v1 default; k > 1 only as a targeted adjustment (D1-10)
    seed: int = 0
    max_steps: int | None = None    # stop after this many optimizer steps (smoke / benchmark)
    log_every: int = 1

    @classmethod
    def from_dict(cls, d: dict | None) -> "TrainConfig":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown train config keys: {sorted(unknown)}")
        return cls(**d)


# ----------------------------------------------------------------------------- setup helpers
def configure_determinism(seed: int) -> None:
    """Seeds + deterministic kernels + FP32 (TF32 off), D1-3 / D1-7."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay on nn.Linear weights only (incl. the Transformer projections); 0 on
    embeddings (decay would keep shrinking rows absent from the batch), biases, LayerNorm,
    positional embedding and pooling query (D1-3)."""
    decay = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
    emb = {id(m.weight) for m in model.modules() if isinstance(m, MaskedEmbedding)}
    assert not decay & emb
    d, nd = [], []
    for p in model.parameters():
        (d if id(p) in decay else nd).append(p)
    return [{"params": d, "weight_decay": weight_decay}, {"params": nd, "weight_decay": 0.0}]


def make_optimizer(model: nn.Module, tc: TrainConfig):
    opt = torch.optim.AdamW(param_groups(model, tc.weight_decay), lr=tc.lr)
    w = max(int(tc.warmup_steps), 0)
    # LambdaLR evaluates at step 0 before any update: the first step uses lr / w.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / w) if w > 0 else 1.0)
    return opt, sched


def tower_grad_norms(model: ThreeTowerModel) -> dict:
    out = {}
    for name in ("music_tower", "video_tower", "user_tower", "user_heads", "video_heads", "music_heads"):
        g = [p.grad.detach().pow(2).sum() for p in getattr(model, name).parameters() if p.grad is not None]
        out[name] = float(torch.stack(g).sum().sqrt()) if g else 0.0
    return out


# ----------------------------------------------------------------------------- one step
@dataclass
class ItemInputs:
    vf: F.VideoFeatures
    mf: F.MusicFeatures
    cand_rows: dict          # {"video": LongTensor [C_v], "music": LongTensor [C_m]}

    @classmethod
    def from_condition(cls, cond: D.TrainCondition, device=None) -> "ItemInputs":
        return cls(F.VideoFeatures.from_catalog(cond.video_catalog).to(device),
                   F.MusicFeatures.from_catalog(cond.music_catalog).to(device),
                   {"video": torch.tensor(np.array(cond.cand_video_rows), dtype=torch.long, device=device),
                    "music": torch.tensor(np.array(cond.cand_music_rows), dtype=torch.long, device=device)})


def forward_loss(model: ThreeTowerModel, items: ItemInputs, batch: D.TrainBatch, tau: float):
    """Loss of one batch. The item catalog is encoded exactly once (shared by the video
    candidates, the music candidates and the history tokens)."""
    enc = model.encode_items(items.vf, items.mf)
    item_embs = model.item_embeddings(enc)
    u = model.user_embeddings(model.encode_users(batch.user_idx, batch.history, enc, batch.user_feats))
    return multitask_loss(u, item_embs, items.cand_rows, batch.targets, tau)


class NonFiniteError(FloatingPointError):
    pass


def _dump(debug_dir: Path | None, payload: dict) -> Path | None:
    if debug_dir is None:
        return None
    debug_dir.mkdir(parents=True, exist_ok=True)
    p = debug_dir / f"nonfinite_step{payload['step']}.json"
    p.write_text(json.dumps(payload, indent=1, default=str))
    return p


def train_step(model, opt, sched, items: ItemInputs, batch: D.TrainBatch, tc: TrainConfig,
               step: int, debug_dir: Path | None = None) -> dict:
    model.train()
    opt.zero_grad(set_to_none=True)
    diag = {"step": step, "lr": opt.param_groups[0]["lr"], "batch_rows": batch.sample_idx.tolist(),
            "user_ids": batch.user_id.tolist()}
    try:
        loss, info = forward_loss(model, items, batch, tc.tau)
    except FloatingPointError as e:        # e.g. a non-finite label logit inside the loss
        diag["error"] = repr(e)
        path = _dump(debug_dir, diag)
        raise NonFiniteError(f"non-finite value in the loss at step {step} (debug: {path})") from e
    diag["tasks"] = info
    if loss is None:                       # no task has a valid row (cannot happen with real samples)
        raise ValueError(f"step {step}: batch has no valid row for any task")
    if not torch.isfinite(loss):
        diag["loss"] = float(loss.detach())
        path = _dump(debug_dir, diag)
        raise NonFiniteError(f"non-finite loss at step {step} (debug: {path})")
    loss.backward()
    # Pre-clip total norm. It is non-finite iff any gradient entry is; the check happens
    # before opt.step(), so a bad gradient never reaches the parameters.
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip_grad_norm)
    if not torch.isfinite(gn):
        diag.update(loss=float(loss.detach()), grad_norm=float(gn), tower_grad_norms=tower_grad_norms(model))
        path = _dump(debug_dir, diag)
        raise NonFiniteError(f"non-finite grad norm at step {step} (debug: {path})")
    tgn = tower_grad_norms(model)          # post-clip, per tower (diagnostic)
    opt.step()
    sched.step()
    return {"step": step, "loss": float(loss.detach()), "grad_norm": float(gn), "lr": diag["lr"],
            "tower_grad_norms_post_clip": tgn,
            **{f"loss_{t}": info[t].get("loss") for t in TASKS},
            **{f"n_{t}": info[t]["n_valid"] for t in TASKS},
            **{f"fn_{t}": info[t]["n_fn_masked"] for t in TASKS}}


# ----------------------------------------------------------------------------- trainer
def assert_outside_worktree(out_dir: Path) -> None:
    """D1-11: run outputs never go into the git worktree."""
    try:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=paths.PROJECT_ROOT,
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        top = str(paths.PROJECT_ROOT)
    out, top = Path(out_dir).resolve(), Path(top).resolve()
    if out == top or top in out.parents:
        raise ValueError(f"output dir {out} is inside the git worktree {top}; use RUN_ROOT outside it")


def assert_clean_worktree() -> None:
    st = subprocess.check_output(["git", "status", "--porcelain"], cwd=paths.PROJECT_ROOT, text=True)
    if st.strip():
        raise RuntimeError("require_clean_git: git worktree is dirty; commit first (D1-11)")


class Trainer:
    def __init__(self, cond: D.TrainCondition, tcfg: TowerConfig, hcfg: F.HistoryConfig, tc: TrainConfig,
                 out_dir: Path | None = None, device: str = "cpu",
                 user_vocab: F.UserFeatureVocab | None = None, user_table: F.UserFeatureTable | None = None):
        configure_determinism(tc.seed)
        self.cond, self.tcfg, self.hcfg, self.tc = cond, tcfg, hcfg, tc
        self.user_vocab = user_vocab
        self._fingerprint = None
        self.device = torch.device(device)
        self.out_dir = None if out_dir is None else Path(out_dir)
        base = D.build_samples(cond)
        self.pos = D.PosTrain.build(base, cond.vocabs.user.size)
        self.samples = D.oversample_likes(base, tc.like_oversample)
        self.builder = D.BatchBuilder(cond, self.samples, self.pos, hcfg, tcfg.use_history, user_table)
        self.items = ItemInputs.from_condition(cond, self.device)
        sizes = F.FeatureSizes.from_vocabs(cond.vocabs, user_vocab)
        self.model = ThreeTowerModel(sizes, tcfg, hcfg).to(self.device)
        self.opt, self.sched = make_optimizer(self.model, tc)
        self.step = 0
        self.log: list[dict] = []
        self.sample_counts = {"base": base.counts(), "expanded": self.samples.counts()}

    def _debug_dir(self):
        return None if self.out_dir is None else self.out_dir / "debug"

    def fit(self, validate: Callable[[ThreeTowerModel, int], dict] | None = None) -> dict:
        """Train until max_epochs / max_steps / early stop.

        `validate(model, epoch) -> {"score": float, "tie": float, ...}` is the D1-8 hook (val
        masked frames only). It runs after every epoch, and once more when `max_steps` cuts an
        epoch short. The best (score, tie) -- strictly greater, so the earliest wins exact
        ties -- is saved as `best.pt`; `patience` epochs without improvement stop training.
        Without the hook no selection happens and only the last state exists.
        """
        tc = self.tc
        best, bad, history = None, 0, []
        stop = False

        def _validate(epoch: int, partial: bool) -> bool:
            nonlocal best, bad
            v = validate(self.model, epoch)
            if not (math.isfinite(v["score"]) and math.isfinite(v["tie"])):
                raise NonFiniteError(f"non-finite validation score at epoch {epoch}")
            key = (v["score"], v["tie"])
            rec = {"epoch": epoch, "step": self.step, "partial_epoch": partial, **v}
            improved = best is None or key > best[0]
            rec["improved"] = improved
            history.append(rec)
            print(json.dumps({k: rec[k] for k in ("epoch", "step", "score", "tie", "improved", "val_s") if k in rec}),
                  flush=True)
            if improved:
                best, bad = (key, epoch, self.step), 0
                self.save("best.pt", extra={"epoch": epoch, "val": v, "selection": "val_masked_mean_recall@20"})
                return False
            bad += 1
            return bad >= tc.patience

        for epoch in range(tc.max_epochs):
            for idx in D.epoch_batches(len(self.samples), tc.batch_size, tc.seed, epoch):
                t0 = time.perf_counter()
                batch = self.builder(idx, self.device)
                t1 = time.perf_counter()
                rec = train_step(self.model, self.opt, self.sched, self.items, batch, tc, self.step,
                                 self._debug_dir())
                rec.update(epoch=epoch, data_s=t1 - t0, compute_s=time.perf_counter() - t1)
                self.step += 1
                if self.step % tc.log_every == 0 or self.step == 1:
                    self.log.append(rec)
                    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()
                                      if not isinstance(v, dict)}), flush=True)
                if tc.max_steps is not None and self.step >= tc.max_steps:
                    stop = True
                    break
            if validate is not None:
                if _validate(epoch, partial=stop):
                    break
            if stop:
                break
        out = {"steps": self.step, "val_history": history}
        if best is not None:
            out.update(best_score=best[0][0], best_tie=best[0][1], best_epoch=best[1], best_step=best[2])
        return out

    @property
    def fingerprint(self) -> dict:
        """Condition fingerprint (vocabs, scalers, feature tables, events); computed once."""
        if self._fingerprint is None:
            from kuairand_rec.models.artifacts import condition_fingerprint
            self._fingerprint = condition_fingerprint(self.cond, self.user_vocab)
        return self._fingerprint

    def checkpoint(self, extra: dict | None = None) -> dict:
        return {"model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "condition_id": self.cond.condition_id,
                "condition_name": self.cond.name, "exclude_video_ids": self.cond.exclude_video_ids.tolist(),
                "fingerprint": self.fingerprint, "num_stats": self.cond.num_stats,
                "tower_config": self.tcfg.to_dict(), "history_config": asdict(self.hcfg),
                "train_config": asdict(self.tc), "sizes": self.model.sizes.to_dict(), "step": self.step,
                **(extra or {})}

    def save(self, name: str, extra: dict | None = None) -> Path | None:
        if self.out_dir is None:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        p = self.out_dir / name
        torch.save(self.checkpoint(extra), p)
        return p


# ----------------------------------------------------------------------------- CLI
def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else paths.PROJECT_ROOT / p


def user_features_for(cfg: dict, cond: D.TrainCondition):
    """(user_vocab, user_table) for U1 (fitted on the condition's train users), (None, None) for U0."""
    uf_kind = cfg.get("user_features", "u0")
    if uf_kind == "u0":
        return None, None
    if uf_kind != "u1":
        raise ValueError(f"unknown user_features {uf_kind}")
    from kuairand_rec.data import io
    uf = io.load_user_features()
    voc = F.fit_user_feature_vocab(uf, cond.train_events(), min_count=int(cfg.get("u1_min_count", 5)))
    return voc, F.build_user_feature_table(uf, voc)


def compare_selection_with_export(selection: dict, summary) -> dict:
    """Compare the best epoch's validation-path metrics (batched user encoding, possibly on GPU)
    with the exported (canonical CPU batch-1) val masked metrics of the same checkpoint.

    Flags `selection_may_be_sensitive` if another validated epoch lies within the observed
    path difference of the best one (on the score, or on the tie-break when the scores tie),
    i.e. the choice of epoch could depend on the numeric path. Report-only: the frozen
    selection is never changed here.
    """
    hist = selection["val_history"]
    best = next(h for h in hist if h["epoch"] == selection["best_epoch"] and h.get("improved"))
    ex = summary[(summary.split == "val") & (summary.kind == "masked")]
    per, diffs = {}, []
    for _, r in ex.iterrows():
        t = f"{r.target}_{r.action}"
        for m in ("recall@20", "ndcg@20"):
            d = float(r[m]) - float(best[f"{t}/{m}"])
            per[f"{t}/{m}"] = {"validation": float(best[f"{t}/{m}"]), "export": float(r[m]), "diff": d}
            diffs.append(abs(d))
    ex_score, ex_tie = float(ex["recall@20"].mean()), float(ex["ndcg@20"].mean())
    d_score, d_tie = abs(ex_score - best["score"]), abs(ex_tie - best["tie"])
    near = []
    for h in hist:
        if h["epoch"] == best["epoch"] and h["step"] == best["step"]:
            continue
        gap_s = abs(h["score"] - best["score"])
        if gap_s <= 2 * d_score or (gap_s == 0 and abs(h["tie"] - best["tie"]) <= 2 * d_tie):
            near.append({"epoch": h["epoch"], "score": h["score"], "tie": h["tie"]})
    return {"best_epoch": best["epoch"], "per_task": per,
            "score": {"validation": best["score"], "export": ex_score, "abs_diff": d_score},
            "tie": {"validation": best["tie"], "export": ex_tie, "abs_diff": d_tie},
            "max_abs_diff": max(diffs) if diffs else 0.0, "near_tie_epochs": near,
            "selection_may_be_sensitive": bool(near)}


def run(cfg: dict, max_steps: int | None = None, out_dir: str | None = None, run_parity: bool = True) -> dict:
    """Train (val selection) -> best checkpoint -> export -> val+test evaluation via
    ExportedScorer -> parity in a separate process. Test is evaluated only after the
    checkpoint is frozen (D1-8)."""
    from kuairand_rec.models import artifacts as A
    from kuairand_rec.models.evaluation import Validator, build_eval_frames, contexts, evaluate_frames
    from kuairand_rec.models.export import ExportedScorer, export_embeddings
    from kuairand_rec.models.inference import InferenceSettings

    tc = TrainConfig.from_dict({**cfg.get("train", {}), **({"max_steps": max_steps} if max_steps else {})})
    tcfg = TowerConfig.from_dict(cfg.get("model"))
    hcfg = F.HistoryConfig.from_dict(cfg.get("history"))
    ecfg = dict(cfg.get("eval", {}))
    if cfg.get("require_clean_git", False):
        assert_clean_worktree()
    run_root = Path(os.environ.get("RUN_ROOT", cfg.get("run_root", str(paths.PROJECT_ROOT.parent / "kuairand_runs"))))
    out = Path(out_dir) if out_dir else run_root / "threetower" / cfg["exp"] / time.strftime("%Y%m%d-%H%M%S")
    assert_outside_worktree(out)
    out.mkdir(parents=True, exist_ok=True)
    # The exact config used (incl. CLI overrides) -- parity / reloads read this file.
    eff_cfg = {**cfg, "train": asdict(tc)}
    (out / "config.yaml").write_text(yaml.safe_dump(eff_cfg, sort_keys=False))

    t0 = time.time()
    P = load_processed(_resolve(cfg.get("processed_dir", "data/processed")))
    cond = D.build_condition(cfg.get("condition", "normal"), P)
    user_vocab, user_table = user_features_for(cfg, cond)
    device = cfg.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Trainer(cond, tcfg, hcfg, tc, out, device, user_vocab, user_table)
    validator = Validator(P, cond, user_table, threads=ecfg.get("threads", 1),
                          user_batch_size=int(ecfg.get("val_user_batch_size", 256)), device=device)
    t_setup = time.time() - t0
    n_params = sum(p.numel() for p in tr.model.parameters())
    print(json.dumps({"setup_s": round(t_setup, 2), "params": n_params, "samples": tr.sample_counts,
                      "candidates": {"video": len(cond.cand_video_rows), "music": len(cond.cand_music_rows)},
                      "condition_id": cond.condition_id, "device": str(tr.device)}), flush=True)
    t1 = time.time()
    res = tr.fit(validate=validator)
    t_train = time.time() - t1
    tr.save("last.pt")
    with open(out / "train_log.jsonl", "w") as f:
        for r in tr.log:
            f.write(json.dumps(r) + "\n")

    # ---- selection is frozen here; nothing below may influence it.
    selection = {"checkpoint": "best.pt", "rule": "max mean val masked recall@20 over 4 tasks; tie ndcg@20; "
                 "earliest wins exact ties", "best_epoch": res.get("best_epoch"), "best_step": res.get("best_step"),
                 "best_score": res.get("best_score"), "best_tie": res.get("best_tie"),
                 "val_history": res["val_history"], "test_used_for_selection": False,
                 "checkpoint_sha256": A.sha256_file(out / "best.pt"), "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    A.write_json(out / "selection.json", selection)

    t2 = time.time()
    # Export always runs on CPU (the device parity's DirectScorer uses): GPU kernels round
    # differently, and strict top-50 parity has no near-tie exemption.
    model, _ = A.load_model(out / "best.pt", cond, user_vocab, "cpu")
    frames = build_eval_frames(P)
    settings = InferenceSettings(threads=int(ecfg.get("threads", 1)), user_batch_size=1)
    manifest = export_embeddings(model, cond, P, out / "export", out / "best.pt", eff_cfg, user_vocab, user_table,
                                 {"/".join(k): f.frame_id for k, f in frames.items()}, settings=settings, device="cpu")
    t_export = time.time() - t2

    # ---- reported metrics: ExportedScorer only (D1-7), test after selection (D1-8).
    t3 = time.time()
    sc = ExportedScorer(out / "export", expect_condition_id=cond.condition_id)
    keys = [("val", t, "masked") for t in TASKS] + [("test", t, k) for t in TASKS for k in ("masked", "nomask", "cold")]
    pu, summ = evaluate_frames(sc, frames, contexts(P), keys, int(ecfg.get("chunk", 2048)))
    (out / "per_user").mkdir(exist_ok=True)
    for key, df in pu.items():
        df.to_parquet(out / "per_user" / f"{'_'.join(key)}.parquet", index=False)
    summ["checkpoint"] = "best.pt"
    summ.to_csv(out / "metrics.csv", index=False)
    t_eval = time.time() - t3
    # Report only (does not change the frozen selection): validation-path vs exported val metrics.
    sel_vs_export = compare_selection_with_export(selection, summ)
    A.write_json(out / "selection_vs_export.json", sel_vs_export)
    if sel_vs_export["selection_may_be_sensitive"]:
        print(json.dumps({"WARNING": "validation-path vs export differences could reorder near-tied epochs",
                          "near_tie_epochs": sel_vs_export["near_tie_epochs"]}), flush=True)

    meta = {"config": eff_cfg, "git_sha": git_sha(), "data_git_sha": P["meta"].get("git_sha"),
            "condition_id": cond.condition_id, "condition_fingerprint": tr.fingerprint,
            "data_fingerprint": A.data_fingerprint(P["events"], P["meta"]),
            "num_stats": cond.num_stats, "device": str(tr.device), "torch": torch.__version__, "params": n_params,
            "samples": tr.sample_counts, "candidates": {"video": len(cond.cand_video_rows),
                                                        "music": len(cond.cand_music_rows)},
            "selection": {k: v for k, v in selection.items() if k != "val_history"},
            "selection_vs_export": {k: sel_vs_export[k] for k in ("max_abs_diff", "selection_may_be_sensitive",
                                                                  "near_tie_epochs")},
            "export_manifest_files": len(manifest["files"]),
            "timing_s": {"setup": t_setup, "train_incl_val": t_train, "export": t_export, "eval": t_eval}}
    A.write_json(out / "run_meta.json", meta)
    print(json.dumps({"out_dir": str(out), "steps": res["steps"], "best_epoch": res.get("best_epoch"),
                      "best_val_score": res.get("best_score"), "train_s": round(t_train, 1),
                      "export_s": round(t_export, 1), "eval_s": round(t_eval, 1)}), flush=True)

    parity = None
    if run_parity:
        # Separate process (D1-7): reloads config, data, checkpoint and export from disk.
        t4 = time.time()
        proc = subprocess.run([sys.executable, "-m", "kuairand_rec.models.parity", "--run_dir", str(out),
                               "--n_users", str(int(ecfg.get("parity_users", 2048)))],
                              capture_output=True, text=True)
        print(proc.stdout.strip(), flush=True)
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr, flush=True)
        parity = {"returncode": proc.returncode, "parity_s": round(time.time() - t4, 1)}
        meta["parity"] = parity
        A.write_json(out / "run_meta.json", meta)
        if proc.returncode != 0:
            raise RuntimeError(f"parity failed (see {out / 'parity.json'})")
    return {"trainer": tr, "meta": meta, "out_dir": out, "metrics": summ, "parity": parity}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--skip_parity", action="store_true")
    a = ap.parse_args(argv)
    run(yaml.safe_load(Path(a.config).read_text()), a.max_steps, a.out_dir, run_parity=not a.skip_parity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
