"""Three-tower evaluation on the shared M3 frames (PLAN.md §3d D1-7, D1-8).

* Frames are built by the M3 `build_frames` from the *unfiltered* cleaned log (P1-9: the
  evaluation frame never depends on the training condition) and every frame_id is asserted
  equal to `results/baselines/frames_summary.json`, so the three-tower model is scored on
  exactly the users / GT / masks of Pop and ALS.
* Validation (checkpoint selection, D1-8) touches only the val masked frames at T_val with
  train-only history. Score = mean val masked Recall@20 over the 4 tasks, tie-break mean
  NDCG@20. Test frames are never built or read by the validator.
* Cold-video condition (M5 decision b1, PLAN.md §4): checkpoint selection uses a dedicated
  *cold validation selection frame* per task (`build_cold_selection_frames`) so that no
  validation label on a held-out video H influences the epoch choice. The selection objective,
  tie-break, patience and max epochs are unchanged. The normal condition is untouched. Reported
  val/test metrics of a cold run still use the standard M3 frames.
"""

from __future__ import annotations

import dataclasses
import hashlib
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


COLD_SELECTION_SCHEMA = "m5-cold-val-selection-v1"


def h_sha256(H) -> str:
    """sha256 of the sorted held-out ids (same formula as experiments.cold_video.h_sha256)."""
    return hashlib.sha256(json.dumps(sorted(int(x) for x in np.asarray(H).ravel())).encode()).hexdigest()


def _restrict_frame(f: Fr.EvalFrame, keep: np.ndarray) -> Fr.EvalFrame:
    """Rows `keep` (bool over users) of a frame; GT / mask CSRs, lengths and flags subset."""
    gp, gi = Fr._csr_take(f.gt_indptr, f.gt_idx, keep)
    mp, mi = Fr._csr_take(f.mask_indptr, f.mask_idx, keep)
    return dataclasses.replace(f, users=f.users[keep], gt_indptr=gp, gt_idx=gi, mask_indptr=mp, mask_idx=mi,
                               hist_len_full=f.hist_len_full[keep], hist_len_capped=f.hist_len_capped[keep],
                               is_train_oov_user=f.is_train_oov_user[keep])


def build_cold_selection_frames(P: dict, cond, hidx: HistoryIndex | None = None) -> tuple[dict, dict]:
    """Cold-condition validation selection frames (decision b1), one per task, at T_val.

    * GT   = val positives on non-H videos (`build_frames` on the log with H events removed). Music
             GT therefore comes from non-H video interactions only: a music shared with an H video
             keeps its legitimate non-H GT, and a music positive only via H videos is dropped.
    * mask = all seen interactions before T_val in the *unfiltered* log (incl. H videos), the
             same evaluation-side mask as the M3 val frames; candidates = the full catalog
             (H rows included, not masked unless seen).
    * population = |GT after mask| >= 1 (build_frames) and >= 1 non-H history event before T_val
             (= the cold model's actual frozen val input, from `cond.events`).
    * identity: population_id / frame_id are namespaced with COLD_SELECTION_SCHEMA, the H sha256
             and the condition_id, so they can never be confused with the M3 val frame ids.
    hist_len_* = the cold model's actual (non-H) input length; is_train_oov_user = cold-train OOV.
    counts: keys prefixed `base_` refer to the base population before the non-H-history filter;
    all other keys are recomputed on the final selection frame.
    Returns ({("val", task, "masked"): frame}, metadata).
    """
    from kuairand_rec.models import features as F

    H = np.asarray(cond.exclude_video_ids, dtype=np.int64)
    if len(H) == 0:
        raise ValueError("cold selection frames need a non-empty exclude_video_ids")
    if cond.events.video_id.isin(H).any():
        raise ValueError("condition events still contain held-out videos")
    ev = P["events"]
    hidx = HistoryIndex(ev) if hidx is None else hidx            # unfiltered: mask + raw history
    ev_non_h = ev[~ev.video_id.isin(H)]
    lc = P["label_conflict_events"]
    lc_non_h = lc[~lc.video_id.isin(H)]
    cold_hidx, t_val = F.frozen_history_index(cond.events, "val", P["meta"])
    hsha = h_sha256(H)
    frames, meta = {}, {"schema": COLD_SELECTION_SCHEMA, "h_sha256": hsha, "n_h": int(len(H)),
                        "condition_id": cond.condition_id, "t_pred": int(t_val), "tasks": {},
                        "definition": "GT = val positives on non-H videos (music GT from non-H video interactions), "
                                      "minus items seen before T_val in the unfiltered log; mask = unfiltered seen "
                                      "items; candidates = full catalog incl. H rows; population = non-empty GT "
                                      "after mask and >= 1 non-H history event before T_val"}
    for task in TASKS:
        tgt, act = task_parts(task)
        base = Fr.build_frames(ev_non_h, lc_non_h, P["catalogs"], P["meta"], hidx, "val", tgt, act)["masked"]
        n_non_h = cold_hidx.history_length(base.users, t_val).astype(np.int64)
        keep = n_non_h >= 1
        f = _restrict_frame(base, keep)
        n_non_h = n_non_h[keep]
        f = dataclasses.replace(f, hist_len_full=n_non_h, hist_len_capped=np.minimum(n_non_h, Fr.HISTORY_CAP))
        h = hashlib.sha256(json.dumps({"schema": COLD_SELECTION_SCHEMA, "h_sha256": hsha,
                                       "condition_id": cond.condition_id, "base": f.population_id}).encode())
        pid = h.hexdigest()
        fid = Fr.compute_frame_id(pid, "masked", f.gt_indptr, f.gt_idx, f.mask_indptr, f.mask_idx)
        n_gt = np.diff(f.gt_indptr)
        if (n_gt < 1).any() or (f.hist_len_full < 1).any():
            raise AssertionError("cold selection frame has an empty-GT or empty-history user")
        # Counts: `base_*` = the build_frames pipeline counts of the base population (non-H GT
        # after mask, >= 1 unfiltered history event), BEFORE the non-H-history filter. All other
        # keys describe the final selection population `f` and are recomputed from it.
        # base.is_train_oov_user is relative to the train users of ev_non_h (= cold train users).
        normal_train_users = np.unique(ev.loc[ev.split == 0, "user_id"].to_numpy())
        col = "video_row" if tgt == "video" else "music_row"
        cold_train_rows = np.unique(cond.events.loc[cond.events.split == 0, col].to_numpy())
        counts = {**{f"base_{k}": v for k, v in base.counts.items()},
                  "base_population_users": int(len(base.users)),
                  "selection_users_excluded_only_h_history": int((~keep).sum()),
                  "evaluated_users": int(len(f.users)), "gt_pairs_effective": int(len(f.gt_idx)),
                  "mask_pairs": int(len(f.mask_idx)),
                  "evaluated_users_cold_train_oov": int(f.is_train_oov_user.sum()),
                  "evaluated_users_normal_train_oov": int((~np.isin(f.users, normal_train_users)).sum()),
                  "gt_pairs_cold_train_unseen_item": int((~np.isin(f.gt_idx, cold_train_rows)).sum())}
        f = dataclasses.replace(f, population_id=pid, frame_id=fid, counts=counts)
        frames[("val", task, "masked")] = f
        meta["tasks"][task] = {"frame_id": fid, "population_id": pid, "n_users": int(len(f.users)),
                               "gt_pairs": int(len(f.gt_idx)), **counts}
    return frames, meta


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

        if len(cond.exclude_video_ids):
            # Cold-video condition (b1): H validation labels never reach the selection score.
            self.frames, self.frames_meta = build_cold_selection_frames(P, cond)
            self.frames_meta["kind"] = "cold_val_selection"
        else:
            # Normal condition: unchanged, the M3 val masked frames (frame_ids asserted).
            self.frames = build_eval_frames(P, ("val",), ("masked",), frames_summary)
            self.frames_meta = {"kind": "normal_val_masked", "schema": Fr.SCHEMA_VERSION,
                                "tasks": {k[1]: {"frame_id": f.frame_id, "population_id": f.population_id,
                                                 "n_users": int(len(f.users)), "gt_pairs": int(len(f.gt_idx))}
                                          for k, f in self.frames.items()}}
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
