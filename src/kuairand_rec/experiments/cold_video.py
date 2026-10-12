"""Controlled cold-video experiment (PLAN.md §3b P1-9, §3d D1-5).

Commands:
    # 1. Draw H once (train data only) -> results/m5/cold_video/H_video_ids.txt + H_selection.json
    uv run python -m kuairand_rec.experiments.cold_video select --config configs/cold_video.yaml
    # 2. (GPU) train the cold model with the unchanged M4 runner (H removed via fit_condition):
    #    python -m kuairand_rec.models.train --config configs/threetower_cold_u0.yaml
    # 3. Evaluate every comparison group on the fixed cold frames (CPU):
    uv run python -m kuairand_rec.experiments.cold_video evaluate --config configs/cold_video.yaml \
        --cold_run <cold_u0 run dir> --seen_run <u0 run dir>

Selection (P1-9): ~10% of train-seen videos, stratified by train popularity, fixed seed, no
val/test data. Implementation choices (within P1-9, recorded in H_selection.json):
  * popularity = number of train events on the video (label-free interaction count);
  * strata = n_strata equal-size rank bins of that count (rank ties broken by video_id);
  * per stratum, round(fraction * n_stratum) videos drawn without replacement from the sorted ids
    with numpy default_rng([seed, stratum]).

Removal: H goes through the only cold entry point, `fit_condition(..., exclude_video_ids=H)`,
which drops every H event from *all* splits first and then refits vocabs, scalers, metadata
vocabs, popularity and feature tables (`models.data.build_condition`). So H is absent from
training targets, rolling and frozen histories, candidates, Pos_train, Pop, ALS and every vocab;
an H video is scored at test time through the content path (video-ID OOV).

Fixed evaluation frames (built once from the *unfiltered* cleaned log, shared by every model and
condition; video tasks only, P1-9 scope):
  GT_H(u)    = test positives on H, deduped, minus videos u interacted with before T_test
               (= the M3 masked-frame GT restricted to H rows);
  mask       = all pre-T_test interactions from the unfiltered log (evaluation only);
  population = |GT_H(u)| >= 1 and >= 1 non-H history event before T_test;
  candidates = (a) "full": the full video catalog minus the mask;
               (b) "h_only": H minus the mask (implemented by also masking every non-H row).
Comparison groups (labels never mixed): cold_threetower; cold-condition baselines (cold_pop,
cold_als[_like_aux], random_expected); seen-video reference (seen_threetower, seen_pop, seen_als),
which is a reference point, not an upper bound. Main claims: cold_threetower vs the cold-condition
baselines with a paired user bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kuairand_rec.data import catalog as catm
from kuairand_rec.data.history import HistoryIndex
from kuairand_rec.eval import frame as Fr
from kuairand_rec.eval.evaluator import KS, EvalContext, evaluate, summarize
from kuairand_rec.eval.views import history_view, prediction_time, train_view
from kuairand_rec.experiments import bootstrap as B
from kuairand_rec.experiments import sources as S

COLD_TASKS = ("video_long_view", "video_is_like")
ACTIONS = ("long_view", "is_like")
FRAME_KINDS = ("cold_video_full", "cold_video_h")


# ============================================================================ H selection
@dataclass(frozen=True)
class SelectionConfig:
    fraction: float = 0.10
    n_strata: int = 10
    seed: int = 0
    popularity: str = "train_events"

    @classmethod
    def from_dict(cls, d: dict | None) -> "SelectionConfig":
        return cls(**dict(d or {}))


def train_popularity(train: pd.DataFrame) -> pd.DataFrame:
    """Train-seen videos with their label-free train interaction count, sorted by video_id."""
    catm.assert_train_only(train)
    p = train.groupby("video_id").size().rename("n_train_events").reset_index()
    return p.sort_values("video_id", kind="mergesort").reset_index(drop=True)


def select_holdout(train: pd.DataFrame, cfg: SelectionConfig) -> tuple[np.ndarray, pd.DataFrame]:
    """Stratified draw of H from train-seen videos. Reads only train events (asserted).

    Returns (sorted H video ids, per-video table: video_id, n_train_events, rank, stratum, selected)."""
    if cfg.popularity != "train_events":
        raise ValueError("only popularity = train_events is implemented")
    if not (0 < cfg.fraction < 1) or cfg.n_strata < 1:
        raise ValueError("bad selection config")
    p = train_popularity(train)
    # Rank ascending by count, ties by video_id (mergesort on the id-sorted table is stable).
    order = np.argsort(p.n_train_events.to_numpy(), kind="mergesort")
    rank = np.empty(len(p), np.int64)
    rank[order] = np.arange(len(p))
    p["rank"] = rank
    p["stratum"] = (rank * cfg.n_strata) // len(p)
    p["selected"] = False
    for s in range(cfg.n_strata):
        ids = np.sort(p.loc[p.stratum == s, "video_id"].to_numpy(np.int64))
        k = int(round(cfg.fraction * len(ids)))
        pick = np.random.default_rng([cfg.seed, s]).choice(ids, size=k, replace=False)
        p.loc[p.video_id.isin(pick), "selected"] = True
    H = np.sort(p.loc[p.selected, "video_id"].to_numpy(np.int64))
    return H, p


def h_sha256(H) -> str:
    ids = sorted(int(x) for x in np.asarray(H).ravel())
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def read_h(path) -> np.ndarray:
    ids = np.array([int(x) for x in Path(path).read_text().split()], dtype=np.int64)
    if len(ids) == 0 or not (np.diff(ids) > 0).all():
        raise ValueError(f"{path}: H must be non-empty, sorted and unique")
    return ids


def holdout_report(events: pd.DataFrame, video_meta: pd.DataFrame, cat: catm.Catalogs, H: np.ndarray,
                   table: pd.DataFrame, cold_cond, normal_vocabs) -> dict:
    """Descriptive statistics of H (outcomes, never selection criteria) + removal checks."""
    ev = events
    isH = ev.video_id.isin(H).to_numpy()
    vm = video_meta.set_index("video_id")
    cold_train = train_view(cold_cond.events)
    strata = []
    for s, g in table.groupby("stratum"):
        strata.append({"stratum": int(s), "n_videos": int(len(g)), "n_selected": int(g.selected.sum()),
                       "train_events_min": int(g.n_train_events.min()), "train_events_max": int(g.n_train_events.max()),
                       "selected_train_events_median": float(g.loc[g.selected, "n_train_events"].median())
                       if g.selected.any() else None})
    hm = vm.loc[H]
    other_train_videos = np.setdiff1d(np.unique(cold_train.video_id.to_numpy()), H)
    authors_cold = set(vm.loc[other_train_videos, "author_id"].tolist())
    music_cold = set(vm.loc[other_train_videos, "music_id"].tolist()) - {catm.MUSIC_SENTINEL}
    nz = hm.music_id != catm.MUSIC_SENTINEL
    removed = {}
    for sname, sid in (("train", 0), ("val", 1), ("test", 2)):
        m = isH & (ev.split.to_numpy() == sid)
        removed[sname] = {"events": int(m.sum()), **{f"{a}_pos": int(ev.loc[m, a].sum()) for a in ACTIONS},
                          "users": int(ev.loc[m, "user_id"].nunique())}
    test = ev[(ev.split == 2) & isH]
    val = ev[(ev.split == 1) & isH]
    sizes = lambda v: {n: int(len(getattr(v, n).ids)) for n in ("user", "video", "author", "music")}
    return {
        "n_h": int(len(H)), "n_train_seen_videos": int(len(table)), "fraction_realised": float(len(H) / len(table)),
        "h_sha256": h_sha256(H), "strata": strata,
        "train_events_on_h": {"total": int(table.loc[table.selected, "n_train_events"].sum()),
                              "share_of_train_events": float(table.loc[table.selected, "n_train_events"].sum()
                                                             / table.n_train_events.sum())},
        "events_removed_by_split": removed,
        "h_author_in_cold_train": {"n": int(hm.author_id.isin(authors_cold).sum()),
                                   "share": float(hm.author_id.isin(authors_cold).mean())},
        "h_music_nonzero": int(nz.sum()), "h_music_zero": int((~nz).sum()),
        "h_music_in_cold_train": {"n": int(hm.music_id[nz].isin(music_cold).sum()),
                                  "share_of_nonzero": float(hm.music_id[nz].isin(music_cold).mean()) if nz.any() else None},
        "h_with_test_positive (outcome, not a criterion)": {a: int(test.loc[test[a] == 1, "video_id"].nunique())
                                                            for a in ACTIONS},
        "h_test_positive_events": {a: int(test[a].sum()) for a in ACTIONS},
        "h_val_positive_events": {a: int(val[a].sum()) for a in ACTIONS},
        "vocab_in_vocab_sizes": {"normal": sizes(normal_vocabs), "cold": sizes(cold_cond.vocabs)},
        "cold_candidates": {"video": int(len(cold_cond.cand_video_rows)), "music": int(len(cold_cond.cand_music_rows))},
        "cold_condition_id": cold_cond.condition_id,
    }


def removal_checks(cold_cond, H: np.ndarray, cat: catm.Catalogs, meta: dict) -> dict:
    """D1-5 / P1-9 leakage checks on a built cold condition (all must be True)."""
    from kuairand_rec.models import data as D
    from kuairand_rec.models import features as F

    hrows = cat.video_row(H)
    ev = cold_cond.events
    vc = cold_cond.video_catalog
    out = {
        "h_rows_in_catalog": bool((hrows >= 0).all()),
        "no_h_event_any_split": not bool(ev.video_id.isin(H).any()),
        "no_h_in_video_vocab": not bool(np.isin(cold_cond.vocabs.video.ids, H).any()),
        "no_h_in_candidates": not bool(np.isin(hrows, cold_cond.cand_video_rows).any()),
        "h_video_idx_is_oov": bool((vc.loc[vc.video_id.isin(H), "video_idx"] == catm.OOV).all()),
    }
    try:
        samples = D.build_samples(cold_cond)    # itself refuses a label outside the candidates
        out["no_h_training_sample"] = not bool(np.isin(cold_cond.cand_video_rows[samples.video_col], hrows).any())
        pos = D.PosTrain.build(samples, cold_cond.vocabs.user.size)
        cols = pos.cols["video_long_view"], pos.cols["video_is_like"]
        out["no_h_in_pos_train"] = not any(np.isin(cold_cond.cand_video_rows[c], hrows).any() for c in cols)
    except AssertionError:
        out["no_h_training_sample"] = out["no_h_in_pos_train"] = False
    for split in ("val", "test"):
        hidx, t = F.frozen_history_index(ev, split, meta)
        users = np.unique(ev.user_id.to_numpy())
        pos_m, _ = hidx.model_history(users, t, 100)
        rows = hidx.gather(pos_m, "video_row", pad=-1)
        out[f"no_h_in_frozen_history_{split}"] = not bool(np.isin(rows[rows >= 0], hrows).any())
        sptr, sidx = hidx.seen_rows(users, t, "video")
        out[f"no_h_in_model_side_seen_{split}"] = not bool(np.isin(sidx, hrows).any())
    tr = F.train_history_index(ev)
    out["no_h_in_rolling_train_history_index"] = not bool(np.isin(tr.events.video_row.to_numpy(), hrows).any())
    return out


# ============================================================================ cold frames
@dataclass(frozen=True)
class ColdVideoFrame(Fr.EvalFrame):
    """An EvalFrame whose mask is always applied by the shared evaluator (both candidate sets
    are masked rankings). kind = cold_video_full | cold_video_h."""

    @property
    def masked(self) -> bool:
        return True


def _subset_csr(indptr, idx, keep_el, keep_rows):
    """Keep elements `keep_el` (bool over elements) and rows `keep_rows` (bool over rows)."""
    row_of = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    sel = keep_el & keep_rows[row_of]
    cnt = np.bincount(row_of[sel], minlength=len(indptr) - 1)[keep_rows]
    return np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64), idx[sel].astype(np.int64)


def build_cold_frames(base_masked: Fr.EvalFrame, H: np.ndarray, cat: catm.Catalogs,
                      non_h_hist_len: np.ndarray, cold_train_users: np.ndarray | None = None) -> dict:
    """Cold frames from the M3 test masked frame of a video task (unfiltered log).

    base_masked: `build_frames(...)["masked"]` (test, video, action) on the unfiltered log.
    non_h_hist_len: [len(base.users)] number of non-H events with t < T_test per base user.
    cold_train_users: user ids with >= 1 train event in the cold condition (= cold user vocab).
    Exactness: GT_H(u) is a subset of the M3 masked GT, and >= 1 non-H history event implies
    >= 1 history event, so every cold-population user is in the M3 masked population.

    GT / mask / population (hence population_id and frame_id) come from the unfiltered log only.
    Per-user metadata describes the *cold model's actual input*: hist_len_full = number of non-H
    events before T_test (= the cold frozen history length), hist_len_capped = min(that, 100),
    is_train_oov_user = not in the cold train users (if given). The normal-train OOV flag of the
    same users is returned separately (`normal_train_oov`), never mixed into the frame.
    """
    f = base_masked
    if f.kind != "masked" or f.target != "video" or f.split != "test":
        raise ValueError("cold frames derive from the test video masked frame")
    n_items = f.n_items
    is_h = np.zeros(n_items, bool)
    hrows = cat.video_row(H)
    if (hrows < 0).any():
        raise ValueError("H contains ids outside the video catalog")
    is_h[hrows] = True
    gt_h_el = is_h[f.gt_idx]
    n_gt_h = np.bincount(np.repeat(np.arange(len(f.users)), np.diff(f.gt_indptr))[gt_h_el], minlength=len(f.users))
    in_pop = (n_gt_h >= 1) & (non_h_hist_len >= 1)
    users = f.users[in_pop]
    gptr, gidx = _subset_csr(f.gt_indptr, f.gt_idx, gt_h_el, in_pop)
    mptr, midx = _subset_csr(f.mask_indptr, f.mask_idx, np.ones(len(f.mask_idx), bool), in_pop)
    # (b) H-only candidates: mask = seen rows U every non-H catalog row (sorted, unique per user).
    non_h = np.flatnonzero(~is_h)
    rows_b, ptr_b = [], [0]
    for i in range(len(users)):
        r = np.union1d(midx[mptr[i]:mptr[i + 1]], non_h)
        rows_b.append(r)
        ptr_b.append(ptr_b[-1] + len(r))
    hptr = np.asarray(ptr_b, np.int64)
    hidx_ = np.concatenate(rows_b).astype(np.int64) if rows_b else np.zeros(0, np.int64)
    # Cold model's actual input (non-H history, cap 100) and the two OOV notions, kept apart.
    nh = np.asarray(non_h_hist_len, dtype=np.int64)[in_pop]
    nh_capped = np.minimum(nh, Fr.HISTORY_CAP)
    normal_oov = f.is_train_oov_user[in_pop]
    cold_oov = (~np.isin(users, np.asarray(cold_train_users)) if cold_train_users is not None
                else np.zeros(len(users), bool))
    buckets = Fr.history_bucket(nh_capped)
    counts = {
        "h_size": int(len(H)), "h_sha256": h_sha256(H),
        "base_masked_frame_id": f.frame_id, "base_population_id": f.population_id,
        "base_users_with_gt_h": int((n_gt_h >= 1).sum()),
        "users_excluded_only_h_history": int(((n_gt_h >= 1) & (non_h_hist_len == 0)).sum()),
        "evaluated_users": int(in_pop.sum()), "gt_h_pairs": int(len(gidx)),
        "evaluated_users_normal_train_oov": int(normal_oov.sum()),
        "evaluated_users_cold_train_oov": int(cold_oov.sum()) if cold_train_users is not None else None,
        "gt_h_videos": int(len(np.unique(gidx))),
        "history_length_definition": "non-H events with t < T_test (cold model input), capped at 100",
        "bucket_counts_non_h_history": {k: int((buckets == k).sum()) for k in ("0", "1-5", "6-20", ">20")},
        "users_bucket_differs_from_unfiltered_history": int((buckets != Fr.history_bucket(f.hist_len_capped[in_pop])).sum()),
    }
    if sum(counts["bucket_counts_non_h_history"].values()) != len(users) or counts["bucket_counts_non_h_history"]["0"]:
        raise AssertionError("cold buckets must partition the population with no empty-history user")
    out = {}
    for kind, (mp, mi) in (("cold_video_full", (mptr, midx)), ("cold_video_h", (hptr, hidx_))):
        pid = Fr.compute_population_id("test", "video", f.action, f.t_pred, cat.video_ids, users)
        fid = Fr.compute_frame_id(pid, kind, gptr, gidx, mp, mi)
        c = dict(counts, n_candidates_mean=float(np.mean(n_items - np.diff(mp))) if len(users) else 0.0)
        out[kind] = ColdVideoFrame("test", "video", f.action, kind, f.t_pred, users, gptr, gidx, mp, mi, n_items,
                                   nh, nh_capped, cold_oov, pid, fid, c)
    out["non_h_hist_len"] = nh
    out["unfiltered_hist_len"] = f.hist_len_full[in_pop]
    out["normal_train_oov"] = normal_oov
    out["cold_train_oov"] = cold_oov
    return out


def cold_label_conflict_accounting(label_conflicts: pd.DataFrame, events: pd.DataFrame, H: np.ndarray,
                                   frame: Fr.EvalFrame, base_masked: Fr.EvalFrame, action: str,
                                   t_pred: int, non_h_hist_len_of) -> dict:
    """Upper bound of GT_H pairs possibly lost to M2 label-conflict dropping (P1-1 accounting).

    Read-only: uses the saved `label_conflict_events` (test split, `{action}_any` = some dropped
    raw row was positive). Never adds anything to GT, never changes the mask. A conflict pair
    (u, v in H) is a *possible* GT_H pair if it is not already in raw GT_H; it would only have
    counted if v was not seen by u before T_test and u has >= 1 non-H history event."""
    lc = label_conflicts[(label_conflicts.split == 2) & (label_conflicts[f"{action}_any"] == 1)
                         & label_conflicts.video_id.isin(H)]
    pairs = lc[["user_id", "video_id"]].drop_duplicates()
    if len(pairs) == 0:
        return {"events": 0, "possible_gt_h_pairs": 0, "details": []}
    pos = events[(events.split == 2) & (events[action] == 1) & events.video_id.isin(H)]
    raw_gt = set(zip(pos.user_id.tolist(), pos.video_id.tolist()))
    pre = events[events.time_ms < t_pred]
    details = []
    for u, v in pairs.itertuples(index=False):
        seen = bool(((pre.user_id == u) & (pre.video_id == v)).any())
        nh = int(non_h_hist_len_of(np.array([u]))[0])
        details.append({"user_id": int(u), "video_id": int(v), "already_in_raw_gt": (int(u), int(v)) in raw_gt,
                        "seen_before_t_test": seen, "non_h_history_events": nh,
                        "user_in_cold_population": bool(np.isin(u, frame.users)),
                        "user_in_m3_masked_population": bool(np.isin(u, base_masked.users)),
                        "would_be_counted_if_positive": (not seen) and nh >= 1 and (int(u), int(v)) not in raw_gt})
    return {"events": int(len(lc)),
            "possible_gt_h_pairs": int(sum(d["would_be_counted_if_positive"] for d in details)),
            "conflict_pairs_not_in_gt": int(sum(not d["already_in_raw_gt"] for d in details)),
            "details": details,
            "note": "upper bound only; label unknown (dropped in M2), never added to GT / mask"}


def cold_frames_metadata(P: dict, H: np.ndarray, cold_cond, hidx: HistoryIndex | None = None):
    """Fixed cold frames for both video tasks + metadata (counts, buckets, OOV, label conflicts).
    Returns ({(task, kind): ColdVideoFrame}, {task: {kind: meta}}, {task: build_cold_frames output})."""
    ev, cat, meta = P["events"], P["catalogs"], P["meta"]
    hidx = HistoryIndex(ev) if hidx is None else hidx
    non_h_idx = HistoryIndex(ev[~ev.video_id.isin(H)])
    cold_users = np.asarray(cold_cond.vocabs.user.ids)
    frames, md, extra = {}, {}, {}
    for task in COLD_TASKS:
        act = task.split("_", 1)[1]
        base = Fr.build_frames(ev, P["label_conflict_events"], cat, meta, hidx, "test", "video", act)["masked"]
        nh = non_h_idx.history_length(base.users, base.t_pred)
        cf = build_cold_frames(base, H, cat, nh, cold_users)
        lca = cold_label_conflict_accounting(P["label_conflict_events"], ev, H, cf["cold_video_full"], base, act,
                                             base.t_pred, lambda u: non_h_idx.history_length(u, base.t_pred))
        md[task] = {}
        for k in FRAME_KINDS:
            frames[(task, k)] = cf[k]
            cf[k].counts["label_conflict_h"] = lca
            md[task][k] = {"frame_id": cf[k].frame_id, "population_id": cf[k].population_id,
                           "n_users": int(len(cf[k].users)), **cf[k].counts}
        extra[task] = cf
    return frames, md, extra


def non_h_history_length(events: pd.DataFrame, H: np.ndarray, users: np.ndarray, t_pred: int) -> np.ndarray:
    """Number of non-H events with t < t_pred (unfiltered log minus H; all splits before t_pred)."""
    return HistoryIndex(events[~events.video_id.isin(H)]).history_length(users, t_pred)


# ============================================================================ random reference
def random_reference(frame: Fr.EvalFrame, ks=KS, name: str = "random_expected") -> pd.DataFrame:
    """Per-user expectation of a uniformly random ranking of the user's own candidates (P1-9).

    C_u = n_items - |mask_u| (candidates after masking; GT_u is inside them by construction).
      E[Recall@K | u] = min(K, C_u) / C_u
      E[NDCG@K  | u] = (|GT_u| / C_u) * sum_{r <= min(K, C_u)} 1/log2(r+1) / IDCG_u
    Same columns as `evaluate`, so it can be paired in the bootstrap (hits@K = expected hits)."""
    n_gt = np.diff(frame.gt_indptr).astype(np.float64)
    C = (frame.n_items - np.diff(frame.mask_indptr)).astype(np.float64)
    if (C < n_gt).any() or (C < 1).any():
        raise AssertionError("candidate count smaller than |GT|")
    if frame.masked:
        # GT must lie inside the candidates (never masked).
        for i in range(len(frame.users)):
            if np.isin(frame.gt_rows(i), frame.mask_idx[frame.mask_indptr[i]:frame.mask_indptr[i + 1]]).any():
                raise AssertionError("GT item is masked")
    disc = 1.0 / np.log2(np.arange(2, max(ks) + 2))
    cum = np.concatenate([[0.0], np.cumsum(disc)])
    out = {"user_id": frame.users, "n_gt": n_gt.astype(np.int64)}
    for k in ks:
        m = np.minimum(k, C)
        out[f"hits@{k}"] = n_gt * m / C
        out[f"recall@{k}"] = m / C
        out[f"ndcg@{k}"] = (n_gt / C) * cum[m.astype(np.int64)] / cum[np.minimum(n_gt, k).astype(np.int64)]
    df = pd.DataFrame(out)
    df["n_gt_unscorable"] = 0
    df["has_signal"] = True
    df["frame_id"], df["population_id"], df["model"] = frame.frame_id, frame.population_id, name
    return df


# ============================================================================ baselines
def als_configs(baselines_dir: Path) -> dict:
    """Selected M3 ALS configs (val-selected, no test), reused unchanged for the cold condition."""
    from kuairand_rec.baselines.als import ALSConfig
    sel = json.loads((Path(baselines_dir) / "run_meta.json").read_text())["als_selected"]
    keep = {f.name for f in ALSConfig.__dataclass_fields__.values()}
    return {k: ALSConfig(**{a: v for a, v in c.items() if a in keep}) for k, c in sel.items()}


def baseline_scorers(events: pd.DataFrame, cat: catm.Catalogs, als_cfg: dict, prefix: str) -> dict:
    """Pop + ALS fitted on train_view(events). `events` = cold (H removed) or normal events."""
    from kuairand_rec.baselines import als as A
    from kuairand_rec.baselines.popularity import PopularityScorer

    tr = train_view(events)
    pop = PopularityScorer(tr, cat, list(ACTIONS))
    pop.name = f"{prefix}_pop"
    out = {pop.name: (pop, {"fit": "train_view counts (P1-6b: train-unseen items score 0)"})}
    for variant in ("als", "als_like_aux"):
        fitted, prov = {}, {}
        for task in COLD_TASKS:
            key = f"{task}/{variant}"
            if key not in als_cfg:
                continue
            act = task.split("_", 1)[1]
            c = als_cfg[key]
            m = A.fit_als(tr, len(cat.video_ids), "video", act, c)
            fitted[("video", act)] = m
            prov[task] = {**c.__dict__, "n_scorable_items": int(len(m.item_rows)), "train_fingerprint": m.fingerprint}
        if fitted:
            name = f"{prefix}_{variant}"
            out[name] = (A.ALSScorer(fitted, "foldin", name), prov)
    return out


# ============================================================================ evaluate CLI
def _ctx(events, meta):
    return EvalContext("test", prediction_time(meta, "test"), history_view(events, "test", meta))


def evaluate_cold(cfg: dict, cold_run: str | None = None, seen_run: str | None = None,
                  out_dir: str | None = None, P: dict | None = None, video_meta: pd.DataFrame | None = None) -> dict:
    """`P` / `video_meta` are injectable for fixture tests (default: the M2 artifacts + raw metadata)."""
    from kuairand_rec.data.pipeline import load_processed
    from kuairand_rec.models import artifacts as A
    from kuairand_rec.models import data as D
    from kuairand_rec.models.export import ExportedScorer

    t0 = time.time()
    out = Path(out_dir) if out_dir else S.resolve(cfg["output_dir"])
    (out / "per_user").mkdir(parents=True, exist_ok=True)
    P = load_processed(S.resolve(cfg.get("processed_dir", "data/processed"))) if P is None else P
    ev, cat, meta = P["events"], P["catalogs"], P["meta"]
    sel = json.loads(S.resolve(cfg["selection_json"]).read_text())
    H = read_h(S.resolve(cfg["h_file"]))
    if h_sha256(H) != sel["h_sha256"]:
        raise ValueError("H file does not match H_selection.json")
    cname = cfg["condition_name"]
    cond = D.build_condition({"name": cname, "exclude_video_ids": H.tolist()}, P, video_meta)
    checks = removal_checks(cond, H, cat, meta)
    if not all(checks.values()):
        raise AssertionError(f"cold removal checks failed: {[k for k, v in checks.items() if not v]}")
    data_fp = A.data_fingerprint(ev, meta)

    # Fixed frames: unfiltered log, shared by every model and condition (cold and seen reference).
    hidx = HistoryIndex(ev)
    frames, md, extra = cold_frames_metadata(P, H, cond, hidx)
    fsum = {f"{t}/{k}": v for t, d in md.items() for k, v in d.items()}
    for (task, k), f in frames.items():
        tab = Fr.frame_table(f)          # hist_len_* / bucket = cold model input (non-H history)
        tab["normal_train_oov"] = extra[task]["normal_train_oov"]
        tab["unfiltered_hist_len_full"] = extra[task]["unfiltered_hist_len"]
        tab.to_parquet(out / "per_user" / f"frame_{task}_{k}.parquet", index=False)

    ctx_cold, ctx_seen = _ctx(cond.events, meta), _ctx(ev, meta)
    als_cfg = als_configs(S.resolve(cfg.get("baselines_dir", "results/baselines")))
    models: dict = {}       # name -> (scorer, ctx, group, provenance)
    for name, (sc, prov) in baseline_scorers(cond.events, cat, als_cfg, "cold").items():
        models[name] = (sc, ctx_cold, "cold_condition_baseline", prov)
    for name, (sc, prov) in baseline_scorers(ev, cat, als_cfg, "seen").items():
        models[name] = (sc, ctx_seen, "seen_video_reference", prov)
    provenance = {}
    if cold_run:
        src = S.ModelSource("cold_threetower", "threetower", Path(cold_run), expect_condition_id=cond.condition_id)
        provenance["cold_threetower"] = S.verify_threetower_run(src, data_fp)
        sc = ExportedScorer(Path(cold_run) / "export", expect_condition_id=cond.condition_id, name="cold_threetower")
        models["cold_threetower"] = (sc, ctx_cold, "cold_video_model", provenance["cold_threetower"])
    if seen_run:
        normal_id = D.condition_id("normal")
        src = S.ModelSource("seen_threetower", "threetower", Path(seen_run), expect_condition_id=normal_id)
        provenance["seen_threetower"] = S.verify_threetower_run(src, data_fp)
        sc = ExportedScorer(Path(seen_run) / "export", expect_condition_id=normal_id, name="seen_threetower")
        models["seen_threetower"] = (sc, ctx_seen, "seen_video_reference", provenance["seen_threetower"])

    rows, per_user = [], {}
    for (task, kind), f in frames.items():
        per_user[(task, kind)] = {}
        for name, (sc, ctx, group, _) in models.items():
            if name.endswith("als_like_aux") and not task.endswith("is_like"):
                continue
            pu = evaluate(sc, f, ctx)
            pu["model"] = name
            per_user[(task, kind)][name] = pu
            rows.append({**summarize(pu, f), "model": name, "group": group, "frame": kind, "task": task})
        rr = random_reference(f)
        per_user[(task, kind)]["random_expected"] = rr
        rows.append({**summarize(rr, f), "model": "random_expected", "group": "cold_condition_baseline",
                     "frame": kind, "task": task})
        for name, pu in per_user[(task, kind)].items():
            pu.to_parquet(out / "per_user" / f"{name}_{task}_{kind}.parquet", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "cold_metrics.csv", index=False)

    # Reproduction check: the seen-reference baselines refitted here must equal M3 exactly.
    repro = (seen_baseline_reproduction(models, P, S.resolve(cfg.get("baselines_dir", "results/baselines")), hidx)
             if cfg.get("check_seen_reproduction", True) else {"skipped": True})

    bc = cfg["bootstrap"]
    comparisons = [tuple(c) for c in cfg["comparisons"]]
    mrows, drows = [], []
    for (task, kind), pus in per_user.items():
        f = frames[(task, kind)]
        res = B.paired_bootstrap(pus, list(S.METRICS), int(bc["n_boot"]), int(bc["seed"]), f"{f.population_id}/{kind}")
        common = {"task": task, "frame": kind, "population_id": f.population_id, "frame_id": f.frame_id}
        mrows += [{**common, **r} for r in B.model_rows(res, float(bc["ci"]))]
        pairs = [(a, b) for a, b in comparisons if a in pus and b in pus]
        drows += [{**common, **r} for r in B.diff_rows(res, pairs, list(S.METRICS), float(bc["ci"]))]
    pd.DataFrame(mrows).to_csv(out / "cold_bootstrap_models.csv", index=False)
    pd.DataFrame(drows).to_csv(out / "cold_bootstrap_diffs.csv", index=False)
    meta_out = {"config": cfg, "h_sha256": sel["h_sha256"], "n_h": int(len(H)), "condition_id": cond.condition_id,
                "removal_checks": checks, "frames": fsum, "seen_baseline_reproduction": repro,
                "models": {n: {"group": g, "provenance": p} for n, (_, _, g, p) in models.items()},
                "pending": [n for n, r in (("cold_threetower", cold_run), ("seen_threetower", seen_run)) if not r],
                "bootstrap": bc, "comparisons": [list(c) for c in comparisons], "data_fingerprint": data_fp,
                "runtime_s": round(time.time() - t0, 1)}
    (out / "cold_meta.json").write_text(json.dumps(meta_out, indent=1, default=str))
    (out / "COLD_VIDEO.md").write_text(render_cold_md(summary, pd.DataFrame(drows), meta_out))
    return {"summary": summary, "meta": meta_out, "per_user": per_user, "frames": frames}


def seen_baseline_reproduction(models: dict, P: dict, baselines_dir: Path, hidx) -> dict:
    """Seen Pop / ALS refitted here == the M3 per-user tables on the M3 test masked frames."""
    out = {}
    ctx = _ctx(P["events"], P["meta"])
    for task in COLD_TASKS:
        act = task.split("_", 1)[1]
        f = Fr.build_frames(P["events"], P["label_conflict_events"], P["catalogs"], P["meta"], hidx,
                            "test", "video", act)["masked"]
        for name, m3 in (("seen_pop", "pop"), ("seen_als", "als"), ("seen_als_like_aux", "als_like_aux")):
            if name not in models or (m3 == "als_like_aux" and act != "is_like"):
                continue
            pu = evaluate(models[name][0], f, ctx)
            ref = pd.read_parquet(baselines_dir / "per_user" / m3 / f"test_{task}_masked.parquet")
            cols = ["user_id", "n_gt", *S.METRICS]
            same = pu[cols].reset_index(drop=True).equals(ref[cols].reset_index(drop=True))
            out[f"{task}/{name}"] = bool(same)
            if not same:
                raise AssertionError(f"seen baseline {name} does not reproduce M3 on {task}")
    return out


def render_cold_md(summary: pd.DataFrame, diffs: pd.DataFrame, meta: dict) -> str:
    L = ["# M5 controlled cold-video experiment (P1-9)", "",
         f"H: {meta['n_h']} videos, sha256 `{meta['h_sha256'][:16]}`, cold condition_id `{meta['condition_id']}`. "
         "Frames built once from the unfiltered log (GT_H = test positives on H minus seen; population = "
         "non-empty GT_H and >= 1 non-H history event).", ""]
    if meta["pending"]:
        L += [f"Pending (not produced yet): {', '.join(meta['pending'])}.", ""]
    L += ["## Frames", "", "| frame | users | GT_H pairs | GT_H videos | excluded (only-H history) | mean candidates "
          "| normal-train OOV | cold-train OOV | label-conflict possible GT_H pairs |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in meta["frames"].items():
        L.append(f"| {k} | {v['n_users']:,} | {v['gt_h_pairs']:,} | {v['gt_h_videos']:,} | "
                 f"{v['users_excluded_only_h_history']:,} | {v['n_candidates_mean']:.0f} | "
                 f"{v['evaluated_users_normal_train_oov']:,} | {v['evaluated_users_cold_train_oov']:,} | "
                 f"{v['label_conflict_h']['possible_gt_h_pairs']} (of {v['label_conflict_h']['events']} conflict events) |")
    L += ["", "History buckets (non-H history = the cold model's input, cap 100): "
          + "; ".join(f"{k}: {v['bucket_counts_non_h_history']}" for k, v in meta["frames"].items() if k.endswith("full")), ""]
    L += ["## Metrics (mean over evaluated users)", "",
          "| task | frame | group | model | R@10 | R@20 | R@50 | N@20 | GT unscorable |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        name = r.model
        if r.model == "cold_pop" and r.frame == "cold_video_h":
            name = "cold_pop (fixed tie-break ranking, all H scores 0) [a]"
        elif r.model == "random_expected":
            name = "random_expected (analytical) [b]"
        L.append(f"| {r.task} | {r.frame} | {r.group} | {name} | {r['recall@10']:.4f} | {r['recall@20']:.4f} | "
                 f"{r['recall@50']:.4f} | {r['ndcg@20']:.4f} | {r.gt_unscorable:,} |")
    L += ["", "[a] On the H-only candidate set every cold-condition Pop score is 0 (H has no train interactions), so "
          "its ranking is the deterministic tie-break (catalog row = video_id ascending): a fixed arbitrary order, "
          "NOT a random ranking.",
          "[b] Closed-form expectation of a uniformly random ranking of each user's own candidates (P1-9); not a "
          "sampled model.", ""]
    if len(diffs):
        L += ["", "## Paired bootstrap differences (a - b), recall@20 / ndcg@20", "",
              "| task | frame | a - b | metric | diff [95% CI] |", "|---|---|---|---|---|"]
        for _, r in diffs[diffs.metric.isin(["recall@20", "ndcg@20"])].iterrows():
            L.append(f"| {r.task} | {r.frame} | {r.model_a} - {r.model_b} | {r.metric} | "
                     f"{r['diff']:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}] |")
    L += ["", "Seen-video reference rows are context only (not an upper bound, not a cold baseline).", ""]
    return "\n".join(L) + "\n"


# ============================================================================ select CLI
def run_select(cfg: dict, overwrite: bool = False, P: dict | None = None,
               video_meta: pd.DataFrame | None = None) -> dict:
    from kuairand_rec.data import io
    from kuairand_rec.data.pipeline import load_processed
    from kuairand_rec.models import data as D

    P = load_processed(S.resolve(cfg.get("processed_dir", "data/processed"))) if P is None else P
    scfg = SelectionConfig.from_dict(cfg["selection"])
    H, table = select_holdout(train_view(P["events"]), scfg)
    h_path, j_path = S.resolve(cfg["h_file"]), S.resolve(cfg["selection_json"])
    if h_path.exists() and not overwrite:
        old = read_h(h_path)
        if not np.array_equal(old, H):
            raise RuntimeError(f"{h_path} exists with a different H; refusing to overwrite (H is drawn once)")
    vm = io.load_video_features() if video_meta is None else video_meta
    cond = D.build_condition({"name": cfg["condition_name"], "exclude_video_ids": H.tolist()}, P, vm)
    rep = holdout_report(P["events"], vm, P["catalogs"], H, table, cond, P["vocabs"])
    rep["removal_checks"] = removal_checks(cond, H, P["catalogs"], P["meta"])
    # Share of the validation (selection) frames' GT that falls on H: the M4 Validator scores the
    # cold model on the standard val masked frames (see PROGRESS.md M5 open question).
    hidx = HistoryIndex(P["events"])
    hrows = P["catalogs"].video_row(H)
    rep["val_masked_gt_on_h"] = {}
    for act in ACTIONS:
        f = Fr.build_frames(P["events"], P["label_conflict_events"], P["catalogs"], P["meta"], hidx,
                            "val", "video", act)["masked"]
        rep["val_masked_gt_on_h"][f"video_{act}"] = {"pairs": int(np.isin(f.gt_idx, hrows).sum()),
                                                    "share": float(np.isin(f.gt_idx, hrows).mean())}
    # Fixed cold frames (sizes only; no model is scored here).
    _, md, _ = cold_frames_metadata(P, H, cond, hidx)
    rep["cold_frames"] = {t: {k: {kk: vv for kk, vv in v.items() if kk not in ("population_id", "n_users")}
                              for k, v in d.items()} for t, d in md.items()}
    # Cold validation selection frames (decision b1): what the cold model's Validator scores.
    from kuairand_rec.models.evaluation import build_cold_selection_frames
    _, rep["cold_val_selection_frames"] = build_cold_selection_frames(P, cond, hidx)
    h_path.parent.mkdir(parents=True, exist_ok=True)
    h_path.write_text("\n".join(str(int(x)) for x in H) + "\n")
    table.to_csv(h_path.parent / "H_strata.csv", index=False)
    doc = {"selection_config": scfg.__dict__, "condition_name": cfg["condition_name"],
           "inputs": "train_view(events) only (split == 0); no val/test data or labels",
           "popularity_definition": "number of train events per train-seen video (label-free)",
           "strata_definition": f"{scfg.n_strata} equal-size rank bins of the train event count (ties by video_id)",
           "draw": "per stratum round(fraction * n) ids without replacement, numpy default_rng([seed, stratum])",
           "h_file": str(cfg["h_file"]), "h_ids": H.tolist(), **rep}
    from kuairand_rec.utils import git_sha
    doc["git_sha"] = git_sha()
    j_path.write_text(json.dumps(doc, indent=1, default=str))
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--config", required=True)
    s.add_argument("--overwrite", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--config", required=True)
    e.add_argument("--cold_run", default=None)
    e.add_argument("--seen_run", default=None)
    e.add_argument("--out_dir", default=None)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.cmd == "select":
        d = run_select(cfg, a.overwrite)
        print(json.dumps({k: d[k] for k in ("n_h", "h_sha256", "removal_checks")}))
        return 0 if all(d["removal_checks"].values()) else 1
    r = evaluate_cold(cfg, a.cold_run, a.seen_run, a.out_dir)
    print(json.dumps({"rows": len(r["summary"]), "pending": r["meta"]["pending"], "runtime_s": r["meta"]["runtime_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
