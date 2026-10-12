"""M5 fix round: cold-condition validation selection (decision b1) and selection-vs-export
consistency. The normal condition must be bit-for-bit unchanged.

Fixture (m5_fixtures + val extras below), H = {13, 14}. Catalog: v10/v11 share m100, v12 m0,
v13 m101, v14 m102, v15 m103. T_val = 100 min.
Val positives (t in [100, 200) min):
  u1: v15 lv+like @120 (non-H), v13 lv+like @195 (H).
  u4: v11 lv+like @150 (non-H); only pre-T_val event = train v13 (H) -> only-H history.
  u7 (added): train v14 @40 (H only) -> only-H history; val lv v10 @160 (non-H).
  u8 (added): train v12 @20; val lv v13 @170 (H only) -> no non-H val GT.
  u9 (added): train v12 @25; val lv v10 @165 and v11 @166 (shared music m100, non-H) and lv v13 @167 (H).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import m4_fixtures as fx
import m5_fixtures as m5
from kuairand_rec.data.history import HistoryIndex
from kuairand_rec.eval import frame as Fr
from kuairand_rec.models import data as D
from kuairand_rec.models import evaluation as E
from kuairand_rec.models import features as F
from kuairand_rec.models import train as TR
from kuairand_rec.models.towers import TASKS, TowerConfig

MIN = fx.MIN
H = [13, 14]
R = fx._row
ROWS = m5.ROWS + [R(7, 14, 40 * MIN, lv=1), R(7, 10, 160 * MIN, lv=1),
                  R(8, 12, 20 * MIN, lv=1), R(8, 13, 170 * MIN, lv=1),
                  R(9, 12, 25 * MIN, lv=1), R(9, 10, 165 * MIN, lv=1), R(9, 11, 166 * MIN, lv=1),
                  R(9, 13, 167 * MIN, lv=1)]


@pytest.fixture(scope="module")
def P():
    return m5.make_P(ROWS)


@pytest.fixture(scope="module")
def cold(P):
    return D.build_condition({"name": "cold_fx", "exclude_video_ids": H}, P, fx.VIDEO_META)


@pytest.fixture(scope="module")
def sel(P, cold):
    return E.build_cold_selection_frames(P, cold)


def _gt(f, ids):
    return {int(u): sorted(ids[f.gt_rows(i)].tolist()) for i, u in enumerate(f.users)}


# ============================================================================ frame definition
def test_cold_selection_frames_hand_computed(P, sel):
    frames, meta = sel
    vid, mid = P["catalogs"].video_ids, P["catalogs"].music_ids
    v = frames[("val", "video_long_view", "masked")]
    # u1: v15 (non-H); v13 is H -> dropped. u4: non-H GT v11, but its only pre-T_val event is
    # train v13 (H) -> only-H history -> excluded. u7: same (train v14). u8: only H GT -> no GT.
    # u9: v10, v11.
    assert _gt(v, vid) == {1: [15], 9: [10, 11]}
    assert v.counts["selection_users_excluded_only_h_history"] == 2           # u4, u7
    # Music: u9's v10 / v11 share m100 -> one non-H music GT (m100); u1: m103 (v15).
    m = frames[("val", "music_long_view", "masked")]
    assert _gt(m, mid) == {1: [103], 9: [100]}
    assert meta["h_sha256"] == E.h_sha256(H) and meta["n_h"] == 2 and meta["schema"] == E.COLD_SELECTION_SCHEMA
    assert set(meta["tasks"]) == set(TASKS)
    for t in TASKS:
        assert meta["tasks"][t]["frame_id"] == frames[("val", t, "masked")].frame_id


def test_no_h_gt_and_shared_music_gt_from_non_h_videos(P, cold, sel):
    frames, _ = sel
    cat = P["catalogs"]
    hrows = cat.video_row(H)
    h_music = cat.video_music_row[hrows]
    ev = P["events"]
    val = ev[(ev.split == 1)]
    for t in TASKS:
        f = frames[("val", t, "masked")]
        tgt, act = t.split("_", 1)
        if tgt == "video":
            assert not np.isin(f.gt_idx, hrows).any()
        # Every GT item has >= 1 non-H val positive for that user (music: via a non-H video).
        nonh = val[(val[act] == 1) & ~val.video_id.isin(H)]
        col = "video_row" if tgt == "video" else "music_row"
        allowed = set(zip(nonh.user_id, nonh[col]))
        for i, u in enumerate(f.users):
            assert all((int(u), int(r)) in allowed for r in f.gt_rows(i))
    assert (h_music >= 0).all()                                               # m101 / m102: H-only music here


def _shared_music_P():
    """v16 (non-H) shares music m101 with v13 (H). u9 has a val lv on v16 and on v13."""
    shared = pd.concat([fx.VIDEO_META, pd.DataFrame([{**fx.VIDEO_META.iloc[3].to_dict(), "video_id": 16,
                                                      "author_id": 5}])], ignore_index=True)
    rows = ROWS + [R(9, 16, 168 * MIN, lv=1, dur=10_000.0), R(2, 16, 30 * MIN, lv=1, dur=10_000.0),
                   R(10, 12, 30 * MIN, lv=1), R(10, 13, 171 * MIN, lv=1)]
    fit = fx.make(rows, video_meta=shared)
    lc = m5.make_P()["label_conflict_events"]
    P2 = {"events": fit["events"], "label_conflict_events": lc, "catalogs": fit["catalogs"], "meta": dict(fx.META),
          "video_catalog": fit["video_features"], "music_catalog": fit["music_features"], "vocabs": fit["vocabs"]}
    return P2, shared


def test_shared_music_gt_kept_only_via_non_h_video():
    P2, shared = _shared_music_P()
    c2 = D.build_condition({"name": "cold_fx", "exclude_video_ids": H}, P2, shared)
    f2, _ = E.build_cold_selection_frames(P2, c2)
    mids = P2["catalogs"].music_ids
    g = _gt(f2[("val", "music_long_view", "masked")], mids)
    # u9: m100 (v10 / v11) and m101 via the NON-H video v16 -> kept although v13 (H) also carries m101.
    assert g[9] == [100, 101]
    # u10: its only m101 positive is on v13 (H) -> no music GT -> not in the music selection frame.
    assert 10 not in g
    # Normal M3 val frame (for contrast) does contain u10's m101 GT via v13.
    normal = E.build_eval_frames(P2, ("val",), ("masked",), frames_summary=None)
    assert 101 in _gt(normal[("val", "music_long_view", "masked")], mids)[10]


def test_mask_is_unfiltered_strict_past_and_candidates_full_catalog(P, sel):
    frames, _ = sel
    hidx = HistoryIndex(P["events"])
    t_val = P["meta"]["t_val_ms"]
    for t in TASKS:
        f = frames[("val", t, "masked")]
        tgt = t.split("_", 1)[0]
        ip, rows = hidx.seen_rows(f.users, t_val, tgt)       # unfiltered (incl. H), strict t < T_val
        assert np.array_equal(f.mask_indptr, ip) and np.array_equal(f.mask_idx, rows)
        assert f.n_items == (len(P["catalogs"].video_ids) if tgt == "video" else len(P["catalogs"].music_ids))
    # H rows are seen-masked only where actually seen (u1 saw v13 / v14 in train), never wholesale.
    v = frames[("val", "video_long_view", "masked")]
    r13, r14 = P["catalogs"].video_row(H)
    i1 = int(np.flatnonzero(v.users == 1)[0])
    i9 = int(np.flatnonzero(v.users == 9)[0])
    assert r13 in v.mask_idx[v.mask_indptr[i1]:v.mask_indptr[i1 + 1]]
    assert r13 not in v.mask_idx[v.mask_indptr[i9]:v.mask_indptr[i9 + 1]] and r14 not in v.mask_idx[v.mask_indptr[i9]:v.mask_indptr[i9 + 1]]


def test_no_empty_gt_or_empty_non_h_history_users(P, cold, sel):
    frames, _ = sel
    hc, t = F.frozen_history_index(cold.events, "val", P["meta"])
    for f in frames.values():
        assert (np.diff(f.gt_indptr) >= 1).all()
        assert (hc.history_length(f.users, t) >= 1).all()
        assert np.array_equal(f.hist_len_full, hc.history_length(f.users, t))
        assert np.array_equal(f.hist_len_capped, np.minimum(f.hist_len_full, 100))
        assert np.array_equal(f.is_train_oov_user, ~np.isin(f.users, cold.events.loc[cold.events.split == 0, "user_id"]))


def test_selection_counts_match_final_frame(P, cold, sel):
    """Non-`base_` counts describe the final selection frame; `base_` counts the pre-filter base."""
    frames, meta = sel
    normal_train = set(P["events"].loc[P["events"].split == 0, "user_id"])
    cold_train = cold.events[cold.events.split == 0]
    for (s, task, k), f in frames.items():
        c = f.counts
        assert c == {kk: vv for kk, vv in meta["tasks"][task].items()
                     if kk not in ("frame_id", "population_id", "n_users", "gt_pairs")}
        assert c["evaluated_users"] == len(f.users) == meta["tasks"][task]["n_users"]
        assert c["gt_pairs_effective"] == len(f.gt_idx) == meta["tasks"][task]["gt_pairs"]
        assert c["mask_pairs"] == len(f.mask_idx)
        assert c["evaluated_users_cold_train_oov"] == int(f.is_train_oov_user.sum()) \
            == sum(int(u) not in set(cold_train.user_id) for u in f.users)
        assert c["evaluated_users_normal_train_oov"] == sum(int(u) not in normal_train for u in f.users)
        col = "video_row" if task.startswith("video") else "music_row"
        assert c["gt_pairs_cold_train_unseen_item"] == int((~np.isin(f.gt_idx, cold_train[col].unique())).sum())
        # base vs final: base population = final + excluded only-H-history users.
        assert c["base_population_users"] == c["base_evaluated_users"] == len(f.users) + c["selection_users_excluded_only_h_history"]
        assert c["base_gt_pairs_effective"] >= c["gt_pairs_effective"]
        # No unprefixed stale base counts left in the final counts.
        for stale in ("evaluated_users_train_oov", "users_with_raw_gt", "gt_pairs_raw", "gt_pairs_train_unseen_item"):
            assert stale not in c
    # Fixture: video_lv base = {u1, u4, u7, u9}; u4 / u7 have only-H history -> excluded (2).
    v = frames[("val", "video_long_view", "masked")]
    assert v.counts["selection_users_excluded_only_h_history"] == 2
    assert v.counts["base_population_users"] == 4 and v.counts["evaluated_users"] == 2
    # u4, u7 are cold-train OOV (their only train event is on H), so the base cold-OOV count is 2
    # and the final one is 0 -- the previously reported (unfiltered) value would have been wrong.
    assert v.counts["base_evaluated_users_train_oov"] == 2 and v.counts["evaluated_users_cold_train_oov"] == 0


def test_frame_ids_are_namespaced_and_differ_from_normal(P, cold, sel):
    frames, _ = sel
    normal = E.build_eval_frames(P, ("val",), ("masked",), frames_summary=None)
    for k, f in frames.items():
        assert f.frame_id != normal[k].frame_id and f.population_id != normal[k].population_id
    # Another H / condition -> another id even for identical GT (namespaced by H sha + condition id).
    c3 = D.build_condition({"name": "cold_other", "exclude_video_ids": H}, P, fx.VIDEO_META)
    f3, _ = E.build_cold_selection_frames(P, c3)
    k = ("val", "video_long_view", "masked")
    assert np.array_equal(f3[k].gt_idx, frames[k].gt_idx) and f3[k].frame_id != frames[k].frame_id
    with pytest.raises(ValueError):
        E.build_cold_selection_frames(P, D.normal_condition(P))


# ============================================================================ validator behaviour
def _model(cold, seed=0):
    import torch
    torch.manual_seed(seed)
    from kuairand_rec.models.towers import ThreeTowerModel
    return ThreeTowerModel(F.FeatureSizes.from_vocabs(cold.vocabs), TowerConfig(), F.HistoryConfig())


def _flip(P, mask):
    ev = P["events"].copy()
    ev.loc[mask(ev), ["long_view", "is_like"]] = 1 - ev.loc[mask(ev), ["long_view", "is_like"]]
    return dict(P, events=ev)


def test_cold_validator_uses_selection_frames_and_ignores_h_val_labels(P, cold):
    m = _model(cold)
    v = E.Validator(P, cold, frames_summary=None)
    assert v.frames_meta["kind"] == "cold_val_selection"
    base = v(m, 0)
    sel_ids = {k[1]: f.frame_id for k, f in v.frames.items()}
    assert base["frame_ids"] == sel_ids
    # H val labels flipped: score, tie and frame ids bitwise unchanged.
    Ph = _flip(P, lambda e: (e.split == 1) & e.video_id.isin(H))
    vh = E.Validator(Ph, cold, frames_summary=None)
    rh = vh(m, 0)
    assert rh["score"] == base["score"] and rh["tie"] == base["tie"] and rh["frame_ids"] == base["frame_ids"]
    # Negative control: non-H val labels flipped -> the selection frame (and generally the score) changes.
    Pn = _flip(P, lambda e: (e.split == 1) & ~e.video_id.isin(H))
    vn = E.Validator(Pn, cold, frames_summary=None)
    assert vn.frames_meta["tasks"]["video_long_view"]["frame_id"] != v.frames_meta["tasks"]["video_long_view"]["frame_id"]


def test_normal_validator_unchanged(P):
    """Normal condition: exactly the M3 val masked frames (same objects / ids as build_eval_frames)."""
    cond = D.normal_condition(P)
    v = E.Validator(P, cond, frames_summary=None)
    ref = E.build_eval_frames(P, ("val",), ("masked",), frames_summary=None)
    assert v.frames_meta["kind"] == "normal_val_masked"
    assert {k: f.frame_id for k, f in v.frames.items()} == {k: f.frame_id for k, f in ref.items()}
    # The normal frames still contain val GT on H videos (u1's v13), i.e. no filtering happened.
    hrow = P["catalogs"].video_row([13])[0]
    assert np.isin(hrow, ref[("val", "video_long_view", "masked")].gt_idx)


# ============================================================================ selection vs export
def _hist(frame_ids, score=0.5, tie=0.2):
    h = {"epoch": 0, "step": 1, "improved": True, "score": score, "tie": tie, "frame_ids": frame_ids}
    for t in TASKS:
        h[f"{t}/recall@20"], h[f"{t}/ndcg@20"] = score, tie
    return h


def _summ(frame_ids, score=0.5, tie=0.2):
    return pd.DataFrame([{"split": "val", "kind": "masked", "target": t.split("_", 1)[0], "action": t.split("_", 1)[1],
                          "recall@20": score, "ndcg@20": tie, "frame_id": frame_ids[t]} for t in TASKS])


def test_selection_vs_export_refuses_frame_mismatch():
    ids = {t: f"id_{t}" for t in TASKS}
    sel = {"best_epoch": 0, "val_history": [_hist(ids)]}
    r = TR.compare_selection_with_export(sel, _summ(ids))
    assert r["frames_checked_equal"] and r["frame_ids"] == ids and r["max_abs_diff"] == 0
    other = dict(ids, video_is_like="normal_val_frame")
    with pytest.raises(ValueError, match="different frames"):
        TR.compare_selection_with_export(sel, _summ(other))
    # Legacy summaries without frame ids (M4 readiness tests) keep working, flagged as unchecked.
    r = TR.compare_selection_with_export(sel, _summ(ids).drop(columns="frame_id"))
    assert not r["frames_checked_equal"]


def test_cold_run_end_to_end_selection_records_and_consistency(P, tmp_path):
    """Real M4 runner on the cold fixture: selection uses the cold selection frames, export metrics
    are compared on the same frames, and the reported val/test metrics stay on the M3 frames."""
    mp = pytest.MonkeyPatch()
    hf = tmp_path / "H.txt"
    hf.write_text("13\n14\n")
    try:
        out = m5.run_m4(m5.base_cfg("cold_fx", tmp_path, condition={"name": "cold_fx", "exclude_video_ids_file": str(hf)}),
                        P, mp, tmp_path / "run")
    finally:
        mp.undo()
    sel = json.loads((out / "selection.json").read_text())
    sve = json.loads((out / "selection_vs_export.json").read_text())
    sf = sel["selection_frames"]
    assert sf["kind"] == "cold_val_selection" and sf["h_sha256"] == E.h_sha256(H)
    assert sf["condition_id"] == D.condition_id("cold_fx", H)
    cold = D.build_condition({"name": "cold_fx", "exclude_video_ids": H}, P, fx.VIDEO_META)
    want, _ = E.build_cold_selection_frames(P, cold)
    want_ids = {k[1]: f.frame_id for k, f in want.items()}
    assert {t: v["frame_id"] for t, v in sf["tasks"].items()} == want_ids
    assert all(h["frame_ids"] == want_ids for h in sel["val_history"])
    assert sve["compare_frames"] == "cold_val_selection" and sve["frames_checked_equal"] and sve["frame_ids"] == want_ids
    se = pd.read_csv(out / "selection_frame_export_metrics.csv")
    assert set(se.frame_id) == set(want_ids.values())
    # Reported metrics: the standard M3 val/test frames, not the selection frames.
    met = pd.read_csv(out / "metrics.csv")
    normal = E.build_eval_frames(P, frames_summary=None)
    for _, r in met.iterrows():
        assert r.frame_id == normal[(r.split, f"{r.target}_{r.action}", r.kind)].frame_id
    assert not set(met.frame_id) & set(want_ids.values())


def test_normal_run_end_to_end_unchanged_compare_path(P, tmp_path):
    mp = pytest.MonkeyPatch()
    try:
        out = m5.run_m4(m5.base_cfg("u0", tmp_path), P, mp, tmp_path / "run")
    finally:
        mp.undo()
    sel = json.loads((out / "selection.json").read_text())
    sve = json.loads((out / "selection_vs_export.json").read_text())
    normal = E.build_eval_frames(P, ("val",), ("masked",), frames_summary=None)
    ids = {k[1]: f.frame_id for k, f in normal.items()}
    assert sel["selection_frames"]["kind"] == "normal_val_masked"
    assert {t: v["frame_id"] for t, v in sel["selection_frames"]["tasks"].items()} == ids
    assert sve["compare_frames"] == "normal_val_masked" and sve["frames_checked_equal"] and sve["frame_ids"] == ids
    assert not (out / "selection_frame_export_metrics.csv").exists()
    assert json.loads((out / "parity.json").read_text())["ok"]
