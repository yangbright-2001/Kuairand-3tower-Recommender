"""Three-tower evaluation on the shared M3 frames (PLAN.md §3d D1-7, D1-8).

* Frames are built by the M3 `build_frames` from the *unfiltered* cleaned log (P1-9: the
  evaluation frame never depends on the training condition) and every frame_id is asserted
  equal to `results/baselines/frames_summary.json`, so the three-tower model is scored on
  exactly the users / GT / masks of Pop and ALS.
* Validation (checkpoint selection, D1-8) touches only the val masked frames at T_val with
  train-only history. Score = mean val masked Recall@20 over the 4 tasks, tie-break mean
  NDCG@20. Test frames are never built or read by the validator.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kuairand_rec import paths
from kuairand_rec.data.history import HistoryIndex
from kuairand_rec.eval import frame as Fr
from kuairand_rec.eval.evaluator import KS, EvalContext, evaluate, summarize
from kuairand_rec.eval.views import history_view, prediction_time
from kuairand_rec.models.inference import EmbeddingScorer, encode_items, encode_users, inference_settings
from kuairand_rec.models.towers import TASKS

FRAMES_SUMMARY = paths.RESULTS_DIR / "baselines" / "frames_summary.json"
SELECT_METRIC, TIE_METRIC = "recall@20", "ndcg@20"


def task_parts(task: str) -> tuple[str, str]:
    t, a = task.split("_", 1)
    return t, a


def build_eval_frames(P: dict, splits=("val", "test"), kinds=("masked", "nomask", "cold"),
                      frames_summary: Path | None = FRAMES_SUMMARY, hidx: HistoryIndex | None = None) -> dict:
    """{(split, task, kind): EvalFrame} from the unfiltered log; frame_ids checked against M3."""
    ev = P["events"]
    hidx = HistoryIndex(ev) if hidx is None else hidx
    ref = json.loads(Path(frames_summary).read_text()) if frames_summary is not None else None
    out = {}
    for s in splits:
        for task in TASKS:
            tgt, act = task_parts(task)
            fr = Fr.build_frames(ev, P["label_conflict_events"], P["catalogs"], P["meta"], hidx, s, tgt, act)
            for k in kinds:
                f = fr[k]
                if ref is not None:
                    want = ref[f"{s}/{task}/{k}"]["frame_id"]
                    if f.frame_id != want:
                        raise AssertionError(f"frame_id mismatch for {s}/{task}/{k}: {f.frame_id} != M3 {want}")
                out[(s, task, k)] = f
    return out


def contexts(P: dict, splits=("val", "test")) -> dict:
    return {s: EvalContext(s, prediction_time(P["meta"], s), history_view(P["events"], s, P["meta"])) for s in splits}


def evaluate_frames(scorer, frames: dict, ctxs: dict, keys, chunk: int = 2048, ks=KS):
    """Per-user tables + summary rows through the shared M3 evaluator."""
    per_user, rows = {}, []
    for key in keys:
        s, task, kind = key
        f = frames[key]
        pu = evaluate(scorer, f, ctxs[s], ks, chunk)
        per_user[key] = pu
        rows.append(summarize(pu, f, ks))
    return per_user, pd.DataFrame(rows)


def selection_score(summary: pd.DataFrame) -> dict:
    """D1-8 Q9a: mean over the 4 tasks of val masked R@20 (tie-break mean N@20)."""
    m = summary[(summary.split == "val") & (summary.kind == "masked")]
    if sorted(m.target + "_" + m.action) != sorted(TASKS):
        raise ValueError("selection needs exactly the 4 val masked frames")
    per = {}
    for _, r in m.iterrows():
        per[f"{r.target}_{r.action}/{SELECT_METRIC}"] = float(r[SELECT_METRIC])
        per[f"{r.target}_{r.action}/{TIE_METRIC}"] = float(r[TIE_METRIC])
    return {"score": float(m[SELECT_METRIC].mean()), "tie": float(m[TIE_METRIC].mean()), **per}


class Validator:
    """`validate(model, epoch)` hook for `Trainer.fit` (val masked frames only)."""

    def __init__(self, P: dict, cond, user_table=None, threads: int | None = 1, user_batch_size: int = 1,
                 frames_summary: Path | None = FRAMES_SUMMARY, device="cpu"):
        from kuairand_rec.models import features as F
        from kuairand_rec.models.train import ItemInputs

        self.frames = build_eval_frames(P, ("val",), ("masked",), frames_summary)
        self.ctx = contexts(P, ("val",))
        self.users = np.unique(np.concatenate([f.users for f in self.frames.values()]))
        # History of the model's own condition (cold: H removed), frozen at T_val (= history_view).
        self.hidx, self.t_val = F.frozen_history_index(cond.events, "val", P["meta"])
        self.cond, self.user_table = cond, user_table
        self.threads, self.user_batch_size, self.device = threads, user_batch_size, device
        self.items = ItemInputs.from_condition(cond, device)
        self.log: list[dict] = []

    def __call__(self, model, epoch: int) -> dict:
        t0 = time.time()
        was_training = model.training
        with inference_settings(self.threads):
            enc, I = encode_items(model, self.items.vf, self.items.mf)
            U = encode_users(model, enc, self.hidx, self.t_val, self.users, self.cond.vocabs, model.hcfg,
                             self.user_table, batch_size=self.user_batch_size, device=self.device)
        I = {t: v.detach().cpu().numpy() for t, v in I.items()}
        sc = EmbeddingScorer("threetower_val", {"val": self.users}, {"val": U}, I, {"val": self.t_val})
        _, summ = evaluate_frames(sc, self.frames, self.ctx, list(self.frames))
        model.train(was_training)
        v = {**selection_score(summ), "val_s": round(time.time() - t0, 2),
             "user_batch_size": self.user_batch_size,
             "frame_ids": {f"{k[1]}": f.frame_id for k, f in self.frames.items()}}
        self.log.append({"epoch": epoch, **v})
        return v
