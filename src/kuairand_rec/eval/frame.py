"""Evaluation frames: GT, seen mask, population and exclusion counts (PLAN.md §3b P1-1,
P1-5, P1-6, P1-7; §3c Q4 and frame identity).

One frame per (split, target, action, kind) with kind in:
  masked  -- main protocol: GT minus items seen before t_pred; same items masked at ranking
  nomask  -- secondary view: *same users* as masked (fixed population), GT keeps seen items,
             nothing masked at ranking
  cold    -- users with non-empty GT but no history before t_pred (excluded from main
             metrics, Pop only)

GT pipeline for one task (fixed order so that every positive is counted exactly once):
  test-period positive events
    -> sentinel (music_id 0, music only) / out-of-catalog removed and counted
    -> dedup to (user, catalog row) pairs; shared music collapses to one music row
    -> seen mask: pairs whose item the user interacted with before t_pred (uncapped)
  Label-conflict events were dropped in M2 and never enter GT, mask or history; they are
  only counted as an *upper bound* of possibly-lost GT pairs (`{a}_any` = some raw row was
  positive, so the true label is unknown).
Population = users with |GT| >= 1 after the mask and >= 1 history event before t_pred.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kuairand_rec.data import catalog as catm
from kuairand_rec.data.history import HistoryIndex
from kuairand_rec.eval.views import SPLIT_ID, prediction_time

SCHEMA_VERSION = "m3-frame-v1"
HISTORY_CAP = 100
TARGETS = ("video", "music")


@dataclass(frozen=True)
class EvalFrame:
    split: str
    target: str
    action: str
    kind: str                     # masked | nomask | cold
    t_pred: int
    users: np.ndarray             # [n] user_id, ascending
    gt_indptr: np.ndarray         # CSR over users -> catalog rows (sorted within user)
    gt_idx: np.ndarray
    mask_indptr: np.ndarray       # CSR of masked catalog rows (empty rows for nomask/cold)
    mask_idx: np.ndarray
    n_items: int
    hist_len_full: np.ndarray
    hist_len_capped: np.ndarray
    is_train_oov_user: np.ndarray
    population_id: str
    frame_id: str
    counts: dict = field(default_factory=dict)

    @property
    def masked(self) -> bool:
        return self.kind == "masked"

    @property
    def task(self) -> str:
        return f"{self.target}_{self.action}"

    def gt_rows(self, i: int) -> np.ndarray:
        return self.gt_idx[self.gt_indptr[i]: self.gt_indptr[i + 1]]


# ------------------------------------------------------------------- identity hashes
def _h_update(h, arr) -> None:
    a = np.ascontiguousarray(np.asarray(arr, dtype="<i8"))
    h.update(str(a.shape).encode())
    h.update(a.tobytes())


def compute_population_id(split, target, action, t_pred, catalog_ids, users) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"schema": SCHEMA_VERSION, "split": split, "target": target,
                         "action": action, "t_pred": int(t_pred)}, sort_keys=True).encode())
    _h_update(h, catalog_ids)   # ordered: a permuted catalog changes the id
    _h_update(h, users)         # ordered evaluated users
    return h.hexdigest()


def compute_frame_id(population_id, kind, gt_indptr, gt_idx, mask_indptr, mask_idx) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"population_id": population_id, "kind": kind}, sort_keys=True).encode())
    for a in (gt_indptr, gt_idx, mask_indptr, mask_idx):
        _h_update(h, a)
    return h.hexdigest()


# ------------------------------------------------------------------- helpers
def _item_col(target: str) -> str:
    return {"video": "video_row", "music": "music_row"}[target]


def catalog_ids(cat: catm.Catalogs, target: str) -> np.ndarray:
    return {"video": cat.video_ids, "music": cat.music_ids}[target]


def _pairs_to_csr(users: np.ndarray, pu: np.ndarray, pi: np.ndarray):
    """pairs (pu, pi) restricted to `users` (sorted) -> CSR with sorted, unique rows."""
    pos = np.searchsorted(users, pu)
    ok = (pos < len(users)) & (users[np.minimum(pos, len(users) - 1)] == pu)
    pos, pi = pos[ok], pi[ok]
    order = np.lexsort((pi, pos))
    pos, pi = pos[order], pi[order]
    indptr = np.zeros(len(users) + 1, dtype=np.int64)
    np.add.at(indptr, pos + 1, 1)
    return np.cumsum(indptr), pi.astype(np.int64)


def _csr_take(indptr, idx, sel):
    """Rows `sel` (bool over rows) of a CSR."""
    n = np.diff(indptr)
    keep_el = np.repeat(sel, n)
    new_ptr = np.concatenate([[0], np.cumsum(n[sel])]).astype(np.int64)
    return new_ptr, idx[keep_el]


def _make(split, target, action, kind, t_pred, users, gptr, gidx, mptr, midx, n_items,
          hfull, oov, cat, counts) -> EvalFrame:
    pid = compute_population_id(split, target, action, t_pred, catalog_ids(cat, target), users)
    fid = compute_frame_id(pid, kind, gptr, gidx, mptr, midx)
    return EvalFrame(split, target, action, kind, int(t_pred), users, gptr, gidx, mptr, midx, n_items,
                     hfull, np.minimum(hfull, HISTORY_CAP), oov, pid, fid, counts)


# ------------------------------------------------------------------- main builder
def build_frames(events: pd.DataFrame, label_conflicts: pd.DataFrame, cat: catm.Catalogs, meta: dict,
                 hidx: HistoryIndex, split: str, target: str, action: str) -> dict[str, EvalFrame]:
    """Return {"masked", "nomask", "cold"} frames for one task. `hidx` must be built on the
    same full `events` (it supplies the uncapped pre-t_pred mask and history lengths)."""
    if target not in TARGETS:
        raise ValueError(target)
    s, t_pred, col = SPLIT_ID[split], prediction_time(meta, split), _item_col(target)
    n_items = len(catalog_ids(cat, target))

    pos = events[(events.split == s) & (events[action] == 1)]
    if not (pos.time_ms >= t_pred).all():
        raise AssertionError("GT events before t_pred")
    rows = pos[col].to_numpy(np.int64)
    sentinel = (pos.music_id.to_numpy() == catm.MUSIC_SENTINEL) if target == "music" else np.zeros(len(pos), bool)
    ooc = (rows < 0) & ~sentinel
    ok = rows >= 0
    c = {"pos_events": int(len(pos)), "pos_events_sentinel": int(sentinel.sum()),
         "pos_events_ooc": int(ooc.sum()), "pos_events_in_catalog": int(ok.sum())}

    pairs = pd.DataFrame({"u": pos.user_id.to_numpy()[ok], "i": rows[ok]}).drop_duplicates()
    c["gt_pairs_raw"] = int(len(pairs))
    c["dedup_events_collapsed"] = c["pos_events_in_catalog"] - c["gt_pairs_raw"]
    c["users_with_raw_gt"] = int(pairs.u.nunique())

    # Label-conflict accounting: upper bound only, never added to GT.
    lc = label_conflicts[(label_conflicts.split == s) & (label_conflicts[f"{action}_any"] == 1)]
    lc_rows = cat.video_row(lc.video_id.to_numpy())
    if target == "music":
        lc_rows = np.where(lc_rows >= 0, cat.video_music_row[np.maximum(lc_rows, 0)], -1)
    lcp = pd.DataFrame({"u": lc.user_id.to_numpy(), "i": lc_rows})
    lcp = lcp[lcp.i >= 0].drop_duplicates()
    lcp = lcp.merge(pairs, how="left", indicator=True)
    c["label_conflict_events_any_positive"] = int(len(lc))
    c["label_conflict_possible_gt_pairs"] = int((lcp["_merge"] == "left_only").sum())

    # Raw GT CSR over all users with raw GT, then the uncapped pre-t_pred seen mask.
    u_all = np.unique(pairs.u.to_numpy())
    gptr, gidx = _pairs_to_csr(u_all, pairs.u.to_numpy(), pairs.i.to_numpy())
    sptr, sidx = hidx.seen_rows(u_all, t_pred, target)
    n_gt_raw = np.diff(gptr)
    row_of = np.repeat(np.arange(len(u_all)), n_gt_raw)
    key_gt = row_of * (n_items + 1) + gidx
    key_seen = np.repeat(np.arange(len(u_all)), np.diff(sptr)) * (n_items + 1) + sidx
    seen_gt = np.isin(key_gt, key_seen)
    n_masked = np.bincount(row_of[seen_gt], minlength=len(u_all))
    n_gt_eff = n_gt_raw - n_masked
    hfull = hidx.history_length(u_all, t_pred).astype(np.int64)
    oov_all = ~np.isin(u_all, events.loc[events.split == 0, "user_id"].unique())

    in_pop = (n_gt_eff >= 1) & (hfull >= 1)
    is_cold = (n_gt_raw >= 1) & (hfull == 0)
    c["gt_pairs_masked_seen"] = int(seen_gt.sum())
    c["users_all_gt_masked"] = int(((n_gt_eff == 0) & (hfull >= 1)).sum())
    c["users_cold_no_history"] = int(is_cold.sum())
    c["cold_gt_pairs"] = int(n_gt_raw[is_cold].sum())
    c["evaluated_users"] = int(in_pop.sum())
    c["evaluated_users_train_oov"] = int((in_pop & oov_all).sum())

    # Train-unseen items in the evaluated GT (kept in GT per P1-6b; counted).
    tr = events[events.split == 0]
    tr_rows = np.unique(tr[col].to_numpy())
    train_seen = np.zeros(n_items, bool)
    train_seen[tr_rows[tr_rows >= 0]] = True

    pop_users = u_all[in_pop]
    # masked frame: GT minus seen, mask = seen rows
    keep_el = ~seen_gt & np.repeat(in_pop, n_gt_raw)
    g_row = row_of[keep_el]
    m_gptr = np.concatenate([[0], np.cumsum(np.bincount(g_row, minlength=len(u_all))[in_pop])]).astype(np.int64)
    m_gidx = gidx[keep_el]
    m_mptr, m_midx = _csr_take(sptr, sidx, in_pop)
    c_masked = dict(c, gt_pairs_effective=int(len(m_gidx)),
                    gt_pairs_train_unseen_item=int((~train_seen[m_gidx]).sum()))
    masked = _make(split, target, action, "masked", t_pred, pop_users, m_gptr, m_gidx, m_mptr, m_midx,
                   n_items, hfull[in_pop], oov_all[in_pop], cat, c_masked)

    # nomask frame: same users, raw GT (incl. seen items), no ranking mask
    n_gptr, n_gidx = _csr_take(gptr, gidx, in_pop)
    empty = np.zeros(len(pop_users) + 1, np.int64)
    c_nomask = dict(c, gt_pairs_effective=int(len(n_gidx)),
                    gt_pairs_train_unseen_item=int((~train_seen[n_gidx]).sum()))
    nomask = _make(split, target, action, "nomask", t_pred, pop_users, n_gptr, n_gidx, empty,
                   np.zeros(0, np.int64), n_items, hfull[in_pop], oov_all[in_pop], cat, c_nomask)
    assert nomask.population_id == masked.population_id

    # cold frame: no history -> nothing seen -> raw GT == effective GT
    c_gptr, c_gidx = _csr_take(gptr, gidx, is_cold)
    c_cold = dict(c, gt_pairs_effective=int(len(c_gidx)),
                  gt_pairs_train_unseen_item=int((~train_seen[c_gidx]).sum()))
    cold = _make(split, target, action, "cold", t_pred, u_all[is_cold], c_gptr, c_gidx,
                 np.zeros(int(is_cold.sum()) + 1, np.int64), np.zeros(0, np.int64), n_items,
                 hfull[is_cold], oov_all[is_cold], cat, c_cold)
    return {"masked": masked, "nomask": nomask, "cold": cold}


def frame_table(f: EvalFrame) -> pd.DataFrame:
    """Per-user frame rows for saving (M5 joins per-user metrics on user_id)."""
    n_gt = np.diff(f.gt_indptr)
    return pd.DataFrame({
        "user_id": f.users, "n_gt": n_gt,
        "gt_rows": [f.gt_rows(i).tolist() for i in range(len(f.users))],
        "n_masked_candidates": np.diff(f.mask_indptr),
        "hist_len_full": f.hist_len_full, "hist_len_capped": f.hist_len_capped,
        "bucket": history_bucket(f.hist_len_capped),
        "is_train_oov_user": f.is_train_oov_user,
    })


def history_bucket(capped: np.ndarray) -> np.ndarray:
    """P1-1/P1-2 buckets on the capped length actually fed to the model."""
    return np.where(capped == 0, "0", np.where(capped <= 5, "1-5", np.where(capped <= 20, "6-20", ">20")))
