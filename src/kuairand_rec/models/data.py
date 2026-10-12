"""Training samples, candidate sets, false-negative sets and batches (PLAN.md §3d D1-2, D1-3, D1-5).

Everything here is derived from ONE training condition (`TrainCondition`, i.e. one
`fit_condition` output). In the cold-video condition H was removed from the events *before*
fitting, so H is absent from the samples, the train vocab (hence the candidates), Pos_train and
every rolling history, by construction; `TrainCondition` additionally asserts it.

* Samples: every train event positive for any action. Each carries 4 task masks; the music
  tasks additionally require music_id != 0 (P1-5).
* Candidates (D1-3): video = catalog rows whose video is in the train vocab; music = music
  catalog rows whose music is in the train vocab (non-zero). Columns are unique items, so
  duplicate positives share a column.
* Pos_train(u, task) (Q7a): candidate columns of u's train positives for that task only
  (music: the non-zero music of those videos). Used only to remove loss terms.
* History (Q2a): rolling, t_pred = the event's own time, train events only, built per batch by
  the single `build_history_batch`.
* like_oversample k (Q6): each like sample gets k-1 copies whose long_view masks are off.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from kuairand_rec.data import catalog as catm
from kuairand_rec.data.history import HistoryIndex
from kuairand_rec.eval.views import train_view
from kuairand_rec.models import features as F
from kuairand_rec.models.loss import TaskTargets
from kuairand_rec.models.towers import TASKS

ACTION_OF = {"video_long_view": "long_view", "video_is_like": "is_like",
             "music_long_view": "long_view", "music_is_like": "is_like"}


def condition_id(name: str, exclude_video_ids=()) -> str:
    """D1-5: hash of the condition name and the sorted held-out video ids."""
    ids = sorted(int(x) for x in np.asarray(exclude_video_ids).ravel())
    return hashlib.sha256(json.dumps({"name": name, "exclude_video_ids": ids}).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- condition
@dataclass
class TrainCondition:
    """All training inputs of one condition. Built from `fit_condition` output (or the saved
    normal-condition artifacts, which were produced by the same function)."""

    name: str
    condition_id: str
    exclude_video_ids: np.ndarray
    events: pd.DataFrame            # model-side events of the condition (H removed)
    vocabs: catm.Vocabs
    video_catalog: pd.DataFrame     # encoded with this condition's vocabs
    music_catalog: pd.DataFrame
    cand_video_rows: np.ndarray     # [C_v] catalog rows, ascending
    cand_music_rows: np.ndarray     # [C_m]
    video_col: np.ndarray           # [n_video_catalog] catalog row -> candidate column, -1 = none
    music_col: np.ndarray           # [n_music_catalog]
    num_stats: dict | None = None   # train-only numeric scalers of this condition (provenance)

    @classmethod
    def build(cls, name: str, events: pd.DataFrame, vocabs: catm.Vocabs, video_catalog: pd.DataFrame,
              music_catalog: pd.DataFrame, exclude_video_ids=(), num_stats: dict | None = None) -> "TrainCondition":
        ex = np.asarray(sorted(int(x) for x in np.asarray(exclude_video_ids).ravel()), dtype=np.int64)
        vc, mc = video_catalog, music_catalog
        F.check_shared_music_inputs(vc, mc)
        cand_v = vc.loc[vc.video_idx >= vocabs.video.offset, "video_row"].to_numpy(np.int64)
        cand_m = mc.loc[mc.music_idx >= vocabs.music.offset, "music_row"].to_numpy(np.int64)
        # Candidates are exactly the train-vocab items (one column per in-vocab id).
        assert len(cand_v) == len(vocabs.video.ids) and len(cand_m) == len(vocabs.music.ids)
        vcol = np.full(len(vc), -1, np.int64)
        vcol[cand_v] = np.arange(len(cand_v))
        mcol = np.full(len(mc), -1, np.int64)
        mcol[cand_m] = np.arange(len(cand_m))
        cond = cls(name, condition_id(name, ex), ex, events, vocabs, vc, mc, cand_v, cand_m, vcol, mcol, num_stats)
        cond._check_exclusion()
        return cond

    def _check_exclusion(self) -> None:
        """D1-5: H must be absent from the events, the vocab and the candidates."""
        if not len(self.exclude_video_ids):
            return
        ex = self.exclude_video_ids
        if self.events.video_id.isin(ex).any():
            raise ValueError("held-out videos present in the condition's events")
        if np.isin(self.vocabs.video.ids, ex).any():
            raise ValueError("held-out videos present in the train video vocab")
        if np.isin(self.video_catalog.video_id.to_numpy()[self.cand_video_rows], ex).any():
            raise ValueError("held-out videos present in the candidate set")

    def train_events(self) -> pd.DataFrame:
        return train_view(self.events)


def normal_condition(P: dict) -> TrainCondition:
    """Normal condition from `load_processed` artifacts (they came from `fit_condition`)."""
    return TrainCondition.build("normal", P["events"], P["vocabs"], P["video_catalog"], P["music_catalog"],
                                num_stats=P["meta"].get("numeric_video_stats_train"))


def condition_from_fit(name: str, fit: dict, exclude_video_ids=()) -> TrainCondition:
    """Any condition (incl. cold-video) from a `fit_condition(..., exclude_video_ids=H)` output."""
    return TrainCondition.build(name, fit["events"], fit["vocabs"], fit["video_features"],
                                fit["music_features"], exclude_video_ids, fit.get("num_stats"))


def build_condition(spec, P: dict, video_meta: pd.DataFrame | None = None) -> TrainCondition:
    """The only way the trainer / exporter / DirectScorer obtain a condition.

    spec = "normal" -> the saved M2 artifacts. spec = {"name": ..., "exclude_video_ids": [...]}
    (or "exclude_video_ids_file", one id per line) -> refit everything with `fit_condition` on
    the H-removed log, so the cold condition has its own vocabs, scalers, feature tables and
    event encodings (D1-5). H selection itself is M5.
    """
    if spec in (None, "normal") or (isinstance(spec, dict) and spec.get("name") == "normal"
                                     and not spec.get("exclude_video_ids") and not spec.get("exclude_video_ids_file")):
        return normal_condition(P)
    from kuairand_rec.data import io
    from kuairand_rec.data.pipeline import fit_condition
    if "exclude_video_ids_file" in spec:
        from pathlib import Path
        ids = [int(x) for x in Path(spec["exclude_video_ids_file"]).read_text().split()]
    else:
        ids = [int(x) for x in spec["exclude_video_ids"]]
    if not ids:
        raise ValueError("a non-normal condition needs a non-empty exclude_video_ids")
    vm = io.load_video_features() if video_meta is None else video_meta
    fit = fit_condition(P["events"], vm, P["catalogs"], list(ACTIONS), ids)
    return condition_from_fit(spec["name"], fit, ids)


ACTIONS = ("long_view", "is_like")


# ----------------------------------------------------------------------------- samples
@dataclass
class Samples:
    """One row per (possibly oversampled) training sample."""

    event_pos: np.ndarray      # [N] row in the condition's train events (for debugging)
    user_id: np.ndarray        # [N]
    user_idx: np.ndarray       # [N] train-vocab user index (always in vocab for train users)
    time_ms: np.ndarray        # [N] t_pred for the rolling history = the event's own time
    video_col: np.ndarray      # [N] video candidate column
    music_col: np.ndarray      # [N] music candidate column, -1 for music_id 0
    task_mask: dict            # task -> [N] bool
    is_copy: np.ndarray        # [N] bool, an oversampling copy

    def __len__(self) -> int:
        return len(self.user_id)

    def counts(self) -> dict:
        return {"rows": len(self), "copies": int(self.is_copy.sum()),
                **{f"valid_{t}": int(m.sum()) for t, m in self.task_mask.items()}}


def build_samples(cond: TrainCondition, actions=("long_view", "is_like")) -> Samples:
    tr = cond.train_events().reset_index(drop=True)
    pos = np.flatnonzero(tr[list(actions)].to_numpy().any(axis=1))
    s = tr.iloc[pos]
    vcol = cond.video_col[s.video_row.to_numpy()]
    mrow = s.music_row.to_numpy()
    mcol = np.where(mrow >= 0, cond.music_col[np.maximum(mrow, 0)], -1)
    # Every train event's video (and its non-zero music) is in the train vocab -> a candidate.
    if (vcol < 0).any() or ((mrow >= 0) & (mcol < 0)).any():
        raise AssertionError("a train label is not in the candidate set")
    if (s.user_idx.to_numpy() < cond.vocabs.user.offset).any():
        raise AssertionError("a train sample has an OOV user")
    has_music = mrow >= 0
    lv, like = s.long_view.to_numpy() == 1, s.is_like.to_numpy() == 1
    masks = {"video_long_view": lv, "video_is_like": like,
             "music_long_view": lv & has_music, "music_is_like": like & has_music}
    return Samples(pos.astype(np.int64), s.user_id.to_numpy(np.int64), s.user_idx.to_numpy(np.int64),
                   s.time_ms.to_numpy(np.int64), vcol.astype(np.int64), mcol.astype(np.int64),
                   masks, np.zeros(len(s), bool))


def oversample_likes(s: Samples, k: int) -> Samples:
    """Append k-1 copies of every like sample with both long_view masks off (D1-3 Q6)."""
    if k < 1:
        raise ValueError("like_oversample must be >= 1")
    if k == 1:
        return s
    like = np.flatnonzero(s.task_mask["video_is_like"])
    rep = np.tile(like, k - 1)
    cat = lambda a: np.concatenate([a, a[rep]])
    masks = {}
    for t, m in s.task_mask.items():
        extra = np.zeros(len(rep), bool) if t.endswith("long_view") else m[rep]
        masks[t] = np.concatenate([m, extra])
    return Samples(cat(s.event_pos), cat(s.user_id), cat(s.user_idx), cat(s.time_ms), cat(s.video_col),
                   cat(s.music_col), masks, np.concatenate([s.is_copy, np.ones(len(rep), bool)]))


# ----------------------------------------------------------------------------- Pos_train
@dataclass
class PosTrain:
    """Per task CSR over train-vocab user index: distinct candidate columns of the user's
    train positives for that task. Built from the base (non-oversampled) samples."""

    indptr: dict      # task -> [n_user_vocab + 1]
    cols: dict        # task -> [nnz]

    @classmethod
    def build(cls, s: Samples, n_user: int) -> "PosTrain":
        if s.is_copy.any():
            raise ValueError("build Pos_train from the base samples")
        indptr, cols = {}, {}
        for t in TASKS:
            m = s.task_mask[t]
            col = (s.video_col if t.startswith("video") else s.music_col)[m]
            u = s.user_idx[m]
            assert (col >= 0).all()
            n_c = int(col.max()) + 1 if len(col) else 1
            key = np.unique(u * n_c + col)          # dedup: one entry per (user, item)
            uu, cc = key // n_c, key % n_c
            ip = np.zeros(n_user + 1, np.int64)
            np.add.at(ip, uu + 1, 1)
            indptr[t], cols[t] = np.cumsum(ip), cc.astype(np.int64)
        return cls(indptr, cols)

    def of(self, task: str, user_idx: int) -> np.ndarray:
        ip = self.indptr[task]
        return self.cols[task][ip[user_idx]:ip[user_idx + 1]]

    def fn_pairs(self, task: str, user_idx: np.ndarray, label: np.ndarray, valid: np.ndarray):
        """(rows, cols) to mask for a batch: Pos_train(u_r, task) minus the row's own label."""
        ip, cols = self.indptr[task], self.cols[task]
        r = np.flatnonzero(valid)
        lo, hi = ip[user_idx[r]], ip[user_idx[r] + 1]
        n = hi - lo
        rows = np.repeat(r, n)
        off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        c = cols[np.repeat(lo, n) + off]
        keep = c != label[rows]                     # the current label is never masked
        return rows[keep], c[keep]


# ----------------------------------------------------------------------------- batches
@dataclass
class TrainBatch:
    sample_idx: np.ndarray            # [B] rows of the (expanded) sample list
    user_id: np.ndarray
    user_idx: torch.Tensor            # [B]
    history: F.HistoryTensors | None
    user_feats: torch.Tensor | None   # U1 only
    targets: dict                     # task -> TaskTargets

    def only(self, tasks) -> "TrainBatch":
        """Copy with every task outside `tasks` emptied (used by tests / diagnostics)."""
        tg = {}
        for t, tt in self.targets.items():
            if t in tasks:
                tg[t] = tt
            else:
                e = tt.fn_rows[:0]
                tg[t] = TaskTargets(torch.zeros_like(tt.valid), torch.full_like(tt.label, -1), e, e)
        return TrainBatch(self.sample_idx, self.user_id, self.user_idx, self.history, self.user_feats, tg)


class BatchBuilder:
    """Turns sample indices into a model-ready batch (history, targets, FN pairs)."""

    def __init__(self, cond: TrainCondition, samples: Samples, pos: PosTrain, hcfg: F.HistoryConfig,
                 use_history: bool = True, user_table: F.UserFeatureTable | None = None):
        self.samples, self.pos, self.hcfg = samples, pos, hcfg
        self.use_history, self.user_table = use_history, user_table
        # Rolling history over the condition's train events only (D1-2).
        self.hidx: HistoryIndex = F.train_history_index(cond.events)

    def __call__(self, idx: np.ndarray, device=None) -> TrainBatch:
        s = self.samples
        idx = np.asarray(idx, dtype=np.int64)
        uidx = s.user_idx[idx]
        hist = None
        if self.use_history:
            hist = F.build_history_batch(self.hidx, s.user_id[idx], s.time_ms[idx], self.hcfg).to_torch(device)
        uf = None
        if self.user_table is not None:
            uf = torch.as_tensor(self.user_table.lookup(s.user_id[idx]), dtype=torch.long, device=device)
        targets = {}
        for t in TASKS:
            valid = s.task_mask[t][idx]
            col = (s.video_col if t.startswith("video") else s.music_col)[idx]
            label = np.where(valid, col, -1)
            fr, fc = self.pos.fn_pairs(t, uidx, label, valid)
            targets[t] = TaskTargets(torch.as_tensor(valid, device=device),
                                     torch.as_tensor(label, dtype=torch.long, device=device),
                                     torch.as_tensor(fr, dtype=torch.long, device=device),
                                     torch.as_tensor(fc, dtype=torch.long, device=device))
        return TrainBatch(idx, s.user_id[idx], torch.as_tensor(uidx, dtype=torch.long, device=device),
                          hist, uf, targets)


def epoch_batches(n: int, batch_size: int, seed: int, epoch: int) -> list[np.ndarray]:
    """One epoch = one permutation of the (expanded) sample list; the last batch may be short."""
    perm = np.random.default_rng([seed, epoch]).permutation(n)
    return [perm[a:a + batch_size] for a in range(0, n, batch_size)]
