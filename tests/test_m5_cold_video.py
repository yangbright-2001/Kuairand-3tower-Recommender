"""M5 cold-video experiment on hand-built fixtures (PLAN.md §3b P1-9, §3d D1-5).

Covers: H selection (train-only, stratified, deterministic, label/val/test-invariant), removal
(leakage) through the only cold entry point, fixed cold frames (hand-computed GT_H, population,
both candidate sets, identical to the unfiltered M3 frame), the per-user random reference
(closed form vs simulated random rankings), and the end-to-end evaluate path with a cold model
trained by the unchanged M4 runner.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import m4_fixtures as fx
import m5_fixtures as m5
from kuairand_rec.data import catalog as catm
from kuairand_rec.eval import frame as Fr
from kuairand_rec.eval import metrics as M
from kuairand_rec.eval.views import train_view
from kuairand_rec.experiments import cold_video as C
from kuairand_rec.models import data as D
from kuairand_rec.models import features as F
from kuairand_rec.models.evaluation import build_eval_frames

MIN = fx.MIN
H = np.array(m5.H)


@pytest.fixture(scope="module")
def P():
    return m5.make_P()


def _synthetic_train(n_videos=400, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.zipf(1.6, n_videos).clip(1, 500)
    rows = [(int(u), 1000 + v, int(t)) for v, c in enumerate(counts) for t, u in enumerate(rng.integers(0, 50, c))]
    df = pd.DataFrame(rows, columns=["user_id", "video_id", "time_ms"])
    df["split"] = np.int8(0)
    df["long_view"] = rng.integers(0, 2, len(df))
    df["is_like"] = rng.integers(0, 2, len(df))
    return df


# ============================================================================ selection
def test_selection_stratified_deterministic_and_train_only():
    tr = _synthetic_train()
    cfg = C.SelectionConfig(fraction=0.1, n_strata=10, seed=0)
    H1, tab = C.select_holdout(tr, cfg)
    H2, _ = C.select_holdout(tr.sample(frac=1, random_state=3), cfg)        # row order irrelevant
    assert np.array_equal(H1, H2) and (np.diff(H1) > 0).all()
    # 10 equal-size strata over popularity ranks; round(10%) per stratum; H subset of train-seen.
    sizes = tab.groupby("stratum").size()
    assert sizes.max() - sizes.min() <= 1 and len(sizes) == 10
    sel = tab.groupby("stratum").selected.sum()
    assert (sel == (0.1 * sizes).round().astype(int)).all()
    assert set(H1) <= set(tr.video_id)
    # Strata are ordered by train popularity (monotone boundaries).
    g = tab.groupby("stratum").n_train_events
    assert (g.max().to_numpy()[:-1] <= g.min().to_numpy()[1:]).all()
    # Seed changes the draw, labels do not (popularity = label-free event count).
    assert not np.array_equal(H1, C.select_holdout(tr, C.SelectionConfig(seed=1))[0])
    flipped = tr.assign(long_view=1 - tr.long_view, is_like=1 - tr.is_like)
    assert np.array_equal(H1, C.select_holdout(flipped, cfg)[0])


def test_selection_refuses_non_train_rows_and_ignores_val_test(P):
    ev = P["events"]
    with pytest.raises(ValueError, match="non-train"):
        C.select_holdout(ev, C.SelectionConfig(fraction=0.5, n_strata=2))
    tr = train_view(ev)
    a = C.select_holdout(tr, C.SelectionConfig(fraction=0.5, n_strata=2))[0]
    # Changing / adding val-test events cannot change H (selection only ever sees train_view).
    ev2 = ev.copy()
    later = ev2.split > 0
    ev2.loc[later, ["long_view", "is_like"]] = 1 - ev2.loc[later, ["long_view", "is_like"]]
    b = C.select_holdout(train_view(ev2), C.SelectionConfig(fraction=0.5, n_strata=2))[0]
    assert np.array_equal(a, b) and set(a) <= set(tr.video_id)


def test_run_select_writes_h_once(tmp_path, P):
    cfg = {"condition_name": "cold_fx", "h_file": str(tmp_path / "H.txt"), "selection_json": str(tmp_path / "s.json"),
           "selection": {"fraction": 0.4, "n_strata": 2, "seed": 0}}
    d = C.run_select(cfg, P=P, video_meta=fx.VIDEO_META)
    Hs = C.read_h(tmp_path / "H.txt")
    assert d["h_ids"] == Hs.tolist() and d["h_sha256"] == C.h_sha256(Hs) and all(d["removal_checks"].values())
    assert d["cold_condition_id"] == D.condition_id("cold_fx", Hs)
    js = json.loads((tmp_path / "s.json").read_text())
    assert js["selection_config"]["seed"] == 0 and "train_view" in js["inputs"]
    assert set(js["cold_frames"]) == {"video_long_view", "video_is_like"}
    # Re-running is idempotent; a different H is refused (H is drawn once).
    C.run_select(cfg, P=P, video_meta=fx.VIDEO_META)
    (tmp_path / "H.txt").write_text("10\n")
    with pytest.raises(RuntimeError, match="refusing"):
        C.run_select(cfg, P=P, video_meta=fx.VIDEO_META)


# ============================================================================ removal / leakage
@pytest.fixture(scope="module")
def cold(P):
    return D.build_condition({"name": "cold_fx", "exclude_video_ids": H.tolist()}, P, fx.VIDEO_META)


def test_removal_checks_pass_and_refit(P, cold):
    checks = C.removal_checks(cold, H, P["catalogs"], P["meta"])
    assert all(checks.values()), checks
    # Vocabs / scalers / feature tables are refitted (D1-5), not reused from the normal condition.
    assert set(cold.vocabs.video.ids) == set(P["vocabs"].video.ids) - set(H)
    assert cold.num_stats != P.get("num_stats", None)
    vc = cold.video_catalog.set_index("video_id")
    # v14's author a4 and music m102 only occur on v14 -> OOV in the cold condition (content path).
    assert vc.loc[14, "author_idx"] == catm.OOV and vc.loc[14, "music_idx"] == catm.OOV
    assert vc.loc[13, "video_idx"] == catm.OOV
    # u5's only train event is on H -> u5 is not even in the cold user vocab.
    assert 5 not in cold.vocabs.user.ids and 5 in P["vocabs"].user.ids


def test_cold_inputs_invariant_to_any_change_on_h(P, cold):
    """Leakage: perturbing every H event (labels, feedback, play time) leaves every cold model
    input bitwise unchanged -- samples, Pos_train, candidates, feature tables, frozen histories."""
    ev = P["events"].copy()
    h = ev.video_id.isin(H)
    for c in ("long_view", "is_like", "is_click", "is_follow"):
        ev.loc[h, c] = 1 - ev.loc[h, c]
    ev.loc[h, "play_time_ms"] = 999_999
    P2 = dict(P, events=ev)
    c2 = D.build_condition({"name": "cold_fx", "exclude_video_ids": H.tolist()}, P2, fx.VIDEO_META)
    from kuairand_rec.models import artifacts as A
    assert A.condition_fingerprint(c2) == A.condition_fingerprint(cold)
    s1, s2 = D.build_samples(cold), D.build_samples(c2)
    assert all(np.array_equal(getattr(s1, f), getattr(s2, f)) for f in ("user_id", "time_ms", "video_col"))
    for split in ("val", "test"):
        h1, t = F.frozen_history_index(cold.events, split, P["meta"])
        h2, _ = F.frozen_history_index(c2.events, split, P["meta"])
        users = np.unique(P["events"].user_id)
        b1, b2 = F.build_history_batch(h1, users, t, F.HistoryConfig()), F.build_history_batch(h2, users, t, F.HistoryConfig())
        assert np.array_equal(b1.video_row, b2.video_row) and np.array_equal(b1.feedback, b2.feedback)


def test_negative_control_removal_check_detects_leak(P, cold):
    """If H events were left in the model-side events, the checks must fail."""
    leaky = D.TrainCondition.__new__(D.TrainCondition)
    leaky.__dict__.update(cold.__dict__)
    leaky.events = P["events"]                    # unfiltered: H present in histories
    with pytest.raises(Exception):
        leaky._check_exclusion()
    checks = C.removal_checks(leaky, H, P["catalogs"], P["meta"])
    assert not checks["no_h_event_any_split"] and not checks["no_h_in_frozen_history_test"]


# ============================================================================ fixed cold frames
@pytest.fixture(scope="module")
def frames(P):
    return build_eval_frames(P, frames_summary=None)


def _cold_frames(P, frames, task="video_long_view", Hh=H):
    base = frames[("test", task, "masked")]
    nh = C.non_h_history_length(P["events"], Hh, base.users, base.t_pred)
    return C.build_cold_frames(base, np.asarray(Hh), P["catalogs"], nh)


def _gt_ids(f, cat):
    return {int(u): sorted(cat.video_ids[f.gt_rows(i)].tolist()) for i, u in enumerate(f.users)}


def test_cold_frames_hand_computed(P, frames):
    cat = P["catalogs"]
    base = frames[("test", "video_long_view", "masked")]
    assert _gt_ids(base, cat) == {2: [15], 4: [10, 14], 5: [14]}
    cf = _cold_frames(P, frames)
    full, honly = cf["cold_video_full"], cf["cold_video_h"]
    # GT_H: u4 -> {v14}; u5 has GT_H {v14} but only-H history -> excluded; u2's GT is not on H.
    assert _gt_ids(full, cat) == {4: [14]} and _gt_ids(honly, cat) == {4: [14]}
    assert full.counts["users_excluded_only_h_history"] == 1 and full.counts["base_users_with_gt_h"] == 2
    # Full-catalog mask = u4's unfiltered pre-T_test interactions {v11, v13} (incl. H video v13).
    assert sorted(cat.video_ids[full.mask_idx].tolist()) == [11, 13]
    # H-only candidates: everything except unseen H rows is masked -> only v14 remains.
    cand = np.setdiff1d(np.arange(full.n_items), honly.mask_idx)
    assert cat.video_ids[cand].tolist() == [14]
    assert full.masked and honly.masked and full.population_id == honly.population_id
    assert full.frame_id != honly.frame_id


def test_cold_frames_derive_from_unfiltered_m3_frame(P, frames):
    """GT / mask / population come from the unfiltered log, independent of the training condition
    (P1-9): they equal an independent pandas recomputation and are subsets of the M3 masked frame."""
    ev, cat, t = P["events"], P["catalogs"], P["meta"]["t_test_ms"]
    for task, act in (("video_long_view", "long_view"), ("video_is_like", "is_like")):
        base = frames[("test", task, "masked")]
        cf = _cold_frames(P, frames, task)["cold_video_full"]
        pre = ev[ev.time_ms < t]
        seen = pre.groupby("user_id").video_id.apply(set).to_dict()
        pos = ev[(ev.split == 2) & (ev[act] == 1) & ev.video_id.isin(H)]
        want = {}
        for u, g in pos.groupby("user_id"):
            gt = sorted(set(g.video_id) - seen.get(u, set()))
            non_h_hist = int((~pre[pre.user_id == u].video_id.isin(H)).sum())
            if gt and non_h_hist >= 1:
                want[int(u)] = gt
        assert _gt_ids(cf, cat) == want
        bpos = {int(u): i for i, u in enumerate(base.users)}
        for i, u in enumerate(cf.users):
            j = bpos[int(u)]
            assert set(cf.gt_rows(i)) <= set(base.gt_rows(j))
            assert np.array_equal(cf.mask_idx[cf.mask_indptr[i]:cf.mask_indptr[i + 1]],
                                  base.mask_idx[base.mask_indptr[j]:base.mask_indptr[j + 1]])


def test_cold_history_metadata_is_the_cold_model_input(P, frames, cold):
    """hist_len_* / bucket of a cold frame = the cold model's frozen T_test history (non-H, cap 100),
    not the unfiltered history; OOV flag = cold-train OOV; normal-train OOV kept separately."""
    from kuairand_rec.data.history import HistoryIndex
    from kuairand_rec.models import features as F
    hc, t = F.frozen_history_index(cold.events, "test", P["meta"])
    for task in ("video_long_view", "video_is_like"):
        base = frames[("test", task, "masked")]
        nh = C.non_h_history_length(P["events"], H, base.users, base.t_pred)
        cf = C.build_cold_frames(base, H, P["catalogs"], nh, cold.vocabs.user.ids)
        for k in C.FRAME_KINDS:
            f = cf[k]
            _, ln = hc.model_history(f.users, t, 100)                       # actual cold model input
            assert np.array_equal(f.hist_len_capped, ln)
            assert np.array_equal(f.hist_len_full, hc.history_length(f.users, t))
            assert (f.hist_len_capped <= 100).all() and (f.hist_len_full >= 1).all()
            b = Fr.history_bucket(f.hist_len_capped)
            bc = f.counts["bucket_counts_non_h_history"]
            assert sum(bc.values()) == len(f.users) and bc["0"] == 0
            assert all(bc[x] == int((b == x).sum()) for x in ("1-5", "6-20", ">20"))
            assert np.array_equal(f.is_train_oov_user, ~np.isin(f.users, cold.vocabs.user.ids))
            assert f.counts["evaluated_users_cold_train_oov"] == int(f.is_train_oov_user.sum())
            assert np.array_equal(cf["normal_train_oov"], base.is_train_oov_user[np.isin(base.users, f.users)])
            assert f.counts["evaluated_users_normal_train_oov"] == int(cf["normal_train_oov"].sum())
    # Hand case (video_long_view): u4 has 2 pre-T_test events (train v13 on H, val v11) -> non-H 1.
    base = frames[("test", "video_long_view", "masked")]
    cf = C.build_cold_frames(base, H, P["catalogs"], C.non_h_history_length(P["events"], H, base.users, base.t_pred),
                             cold.vocabs.user.ids)
    f = cf["cold_video_full"]
    assert f.users.tolist() == [4] and f.hist_len_full.tolist() == [1] and cf["unfiltered_hist_len"].tolist() == [2]
    # u4's only train event is on H -> u4 is cold-train OOV but normal-train in-vocab.
    assert f.is_train_oov_user.tolist() == [True] and cf["normal_train_oov"].tolist() == [False]


def test_cold_frame_identity_unchanged_by_metadata_fix(P, frames, cold):
    """The history / OOV metadata fix changes no GT / mask / population byte: frame ids are the
    same with or without the cold-train users, and equal the pre-fix definition (GT_H + unfiltered
    mask + population of the M3 masked frame)."""
    base = frames[("test", "video_long_view", "masked")]
    nh = C.non_h_history_length(P["events"], H, base.users, base.t_pred)
    a = C.build_cold_frames(base, H, P["catalogs"], nh)
    b = C.build_cold_frames(base, H, P["catalogs"], nh, cold.vocabs.user.ids)
    for k in C.FRAME_KINDS:
        assert a[k].frame_id == b[k].frame_id and a[k].population_id == b[k].population_id
        pid = Fr.compute_population_id("test", "video", "long_view", base.t_pred, P["catalogs"].video_ids, a[k].users)
        assert Fr.compute_frame_id(pid, k, a[k].gt_indptr, a[k].gt_idx, a[k].mask_indptr, a[k].mask_idx) == a[k].frame_id


def test_cold_label_conflict_accounting():
    """Upper bound of GT_H pairs lost to M2 label-conflict drops, from the saved conflict table only."""
    P = m5.make_P()
    lc = pd.DataFrame({"user_id": [4, 2, 4, 2], "video_id": [13, 14, 10, 13], "time_ms": [210 * MIN, 205 * MIN, 230 * MIN, 50 * MIN],
                       "long_view_any": [1, 1, 1, 1], "is_like_any": [0, 0, 0, 0], "split": [2, 2, 2, 0]})
    P["label_conflict_events"] = lc
    fr = build_eval_frames(P, splits=("test",), kinds=("masked",), frames_summary=None)
    cold = D.build_condition({"name": "cold_fx", "exclude_video_ids": H.tolist()}, P, fx.VIDEO_META)
    frames_, md, _ = C.cold_frames_metadata(P, H, cold)
    a = md["video_long_view"]["cold_video_full"]["label_conflict_h"]
    # Test split, long_view_any, video in H: (4, 13) and (2, 14). (4, 10) is not on H; (2, 13) is train.
    assert a["events"] == 2
    d = {(x["user_id"], x["video_id"]): x for x in a["details"]}
    assert set(d) == {(4, 13), (2, 14)}
    assert d[(4, 13)]["seen_before_t_test"] and not d[(4, 13)]["would_be_counted_if_positive"]   # u4 saw v13 in train
    assert d[(2, 14)]["seen_before_t_test"] and not d[(2, 14)]["would_be_counted_if_positive"]   # u2 cycles v10..v14
    assert a["possible_gt_h_pairs"] == 0 and a["conflict_pairs_not_in_gt"] == 2
    # An unseen H video with a non-H-history user is a possible GT_H pair (u4 never saw v14 before T_test).
    lc2 = pd.DataFrame({"user_id": [4], "video_id": [14], "time_ms": [215 * MIN], "long_view_any": [1],
                        "is_like_any": [1], "split": [2]})
    base = fr[("test", "video_long_view", "masked")]
    r = C.cold_label_conflict_accounting(lc2, P["events"], H, frames_[("video_long_view", "cold_video_full")], base,
                                         "long_view", base.t_pred, lambda u: C.non_h_history_length(P["events"], H, u, base.t_pred))
    assert r["details"][0]["already_in_raw_gt"] and r["possible_gt_h_pairs"] == 0        # (4,14) is already GT
    lc3 = lc2.assign(user_id=[6], video_id=[13])                                           # u6: test-only, no history
    r = C.cold_label_conflict_accounting(lc3, P["events"], H, frames_[("video_long_view", "cold_video_full")], base,
                                         "long_view", base.t_pred, lambda u: C.non_h_history_length(P["events"], H, u, base.t_pred))
    assert r["details"][0]["non_h_history_events"] == 0 and r["possible_gt_h_pairs"] == 0
    lc4 = lc2.assign(user_id=[5], video_id=[14])                                           # u5: already GT, only-H history
    lc5 = lc2.assign(user_id=[4], video_id=[15])                                           # not on H -> ignored
    assert C.cold_label_conflict_accounting(lc5, P["events"], H, frames_[("video_long_view", "cold_video_full")], base,
                                            "long_view", base.t_pred, lambda u: u * 0)["events"] == 0
    lc6 = pd.DataFrame({"user_id": [2], "video_id": [13], "time_ms": [215 * MIN], "long_view_any": [1],
                        "is_like_any": [0], "split": [2]})
    P6 = dict(P, events=P["events"][~((P["events"].user_id == 2) & (P["events"].video_id == 13))])
    r = C.cold_label_conflict_accounting(lc6, P6["events"], H, frames_[("video_long_view", "cold_video_full")], base,
                                         "long_view", base.t_pred, lambda u: C.non_h_history_length(P6["events"], H, u, base.t_pred))
    assert r["possible_gt_h_pairs"] == 1 and r["details"][0]["would_be_counted_if_positive"]
    del lc4
    f = frames_[("video_long_view", "cold_video_full")]
    assert f.frame_id == md["video_long_view"]["cold_video_full"]["frame_id"]
    assert len(f.gt_idx) == md["video_long_view"]["cold_video_full"]["gt_h_pairs"]              # GT untouched
    assert md["video_is_like"]["cold_video_full"]["label_conflict_h"]["events"] == 0             # is_like_any = 0


def test_cold_frames_identical_across_conditions(P, frames, cold):
    """The frame is built from the unfiltered log only: training-condition artifacts cannot enter."""
    a = _cold_frames(P, frames)
    P2 = dict(P, events=P["events"])                     # same log
    b = _cold_frames(P2, build_eval_frames(P2, frames_summary=None))
    for k in C.FRAME_KINDS:
        assert a[k].frame_id == b[k].frame_id
    # Negative control: a frame built from the cold (H-removed) events has no GT on H at all and a
    # mask without H -- exactly what the fixed frame must avoid (P1-9).
    from kuairand_rec.data.history import HistoryIndex
    wrong = Fr.build_frames(cold.events, P["label_conflict_events"], P["catalogs"], P["meta"],
                            HistoryIndex(cold.events), "test", "video", "long_view")["masked"]
    hrows = P["catalogs"].video_row(H)
    assert not np.isin(wrong.gt_idx, hrows).any() and np.isin(a["cold_video_full"].gt_idx, hrows).all()
    assert np.isin(a["cold_video_full"].mask_idx, hrows).any() and not np.isin(wrong.mask_idx, hrows).any()


def test_h_only_mask_is_sorted_unique_and_keeps_gt(P, frames):
    cf = _cold_frames(P, frames)["cold_video_h"]
    for i in range(len(cf.users)):
        m = cf.mask_idx[cf.mask_indptr[i]:cf.mask_indptr[i + 1]]
        assert (np.diff(m) > 0).all() and not np.isin(cf.gt_rows(i), m).any()


# ============================================================================ random reference
def test_random_reference_closed_form_matches_simulation():
    """E[metric] under a uniformly random ranking of the user's own candidates (P1-9)."""
    rng = np.random.default_rng(0)
    n_items, users = 60, np.array([1, 2, 3])
    gt = [np.array([3, 7]), np.array([10]), np.array([0, 1, 2, 4, 5])]
    masks = [np.arange(20, 60), np.array([0, 1, 2]), np.arange(30, 60)]       # C_u = 20, 57, 30
    gptr = np.concatenate([[0], np.cumsum([len(g) for g in gt])]).astype(np.int64)
    mptr = np.concatenate([[0], np.cumsum([len(m) for m in masks])]).astype(np.int64)
    f = C.ColdVideoFrame("test", "video", "long_view", "cold_video_h", 0, users, gptr, np.concatenate(gt),
                         mptr, np.concatenate(masks), n_items, np.ones(3, np.int64), np.ones(3, np.int64),
                         np.zeros(3, bool), "p", "f")
    ref = C.random_reference(f)
    sims = []
    for _ in range(6000):
        s = rng.random((3, n_items)).astype(np.float32)
        M.apply_mask(s, mptr, f.mask_idx)
        top, valid = M.rank_topk(s, 50)
        sims.append(pd.DataFrame(M.per_user_metrics(top, valid, gptr, f.gt_idx, n_items)))
    sim = pd.concat(sims).groupby(level=0).mean()
    for k in (10, 20, 50):
        assert np.allclose(ref[f"recall@{k}"], sim[f"recall@{k}"], atol=0.02)
        assert np.allclose(ref[f"ndcg@{k}"], sim[f"ndcg@{k}"], atol=0.02)
    # Exact values for user 1 (C = 20, |GT| = 2): Recall@10 = 10/20, Recall@20 = Recall@50 = 1.
    assert ref["recall@10"][0] == 0.5 and ref["recall@20"][0] == 1.0 and ref["recall@50"][0] == 1.0


def test_random_reference_rejects_masked_gt():
    f = C.ColdVideoFrame("test", "video", "long_view", "cold_video_full", 0, np.array([1]),
                         np.array([0, 1]), np.array([2]), np.array([0, 1]), np.array([2]), 5,
                         np.ones(1, np.int64), np.ones(1, np.int64), np.zeros(1, bool), "p", "f")
    with pytest.raises(AssertionError):
        C.random_reference(f)


# ============================================================================ end to end
@pytest.fixture(scope="module")
def e2e(P, frames, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("m5cold")
    try:
        bdir = m5.write_baselines(P, frames, tmp / "baselines")
        hf = tmp / "H.txt"
        hf.write_text("\n".join(map(str, H)) + "\n")
        (tmp / "sel.json").write_text(json.dumps({"h_sha256": C.h_sha256(H)}))
        cond_spec = {"name": "cold_fx", "exclude_video_ids_file": str(hf)}
        cold_run = m5.run_m4(m5.base_cfg("cold_fx", tmp, condition=cond_spec), P, mp, tmp / "runs" / "cold")
        seen_run = m5.run_m4(m5.base_cfg("u0", tmp), P, mp, tmp / "runs" / "u0")
    finally:
        mp.undo()
    cfg = {"output_dir": str(tmp / "out"), "baselines_dir": str(bdir), "condition_name": "cold_fx",
           "h_file": str(hf), "selection_json": str(tmp / "sel.json"),
           "bootstrap": {"n_boot": 100, "seed": 0, "ci": 0.95},
           "comparisons": [["cold_threetower", "cold_pop"], ["cold_threetower", "cold_als"],
                           ["cold_threetower", "random_expected"], ["seen_threetower", "cold_threetower"]]}
    r = C.evaluate_cold(cfg, str(cold_run), str(seen_run), P=P, video_meta=fx.VIDEO_META)
    return {"r": r, "cfg": cfg, "cold_run": cold_run, "seen_run": seen_run, "tmp": tmp}


def test_cold_run_uses_cold_condition_and_m4_runner(e2e):
    ck = __import__("torch").load(e2e["cold_run"] / "best.pt", weights_only=False)
    assert ck["condition_id"] == D.condition_id("cold_fx", H) and sorted(ck["exclude_video_ids"]) == H.tolist()
    man = json.loads((e2e["cold_run"] / "export" / "manifest.json").read_text())
    assert man["condition_id"] == ck["condition_id"] and json.loads((e2e["cold_run"] / "parity.json").read_text())["ok"]


def test_evaluate_cold_groups_frames_and_checks(e2e):
    r = e2e["r"]
    s, meta = r["summary"], r["meta"]
    assert all(meta["removal_checks"].values()) and all(meta["seen_baseline_reproduction"].values())
    groups = dict(zip(s.model, s.group))
    assert groups["cold_threetower"] == "cold_video_model" and groups["random_expected"] == "cold_condition_baseline"
    assert groups["cold_pop"] == groups["cold_als"] == "cold_condition_baseline"
    assert groups["seen_threetower"] == groups["seen_pop"] == "seen_video_reference"
    # Every model is scored on the same frame per (task, kind).
    assert (s.groupby(["task", "frame"]).frame_id.nunique() == 1).all()
    # P1-6b: cold ALS cannot score H (-inf) -> all GT_H unscorable, zero hits; cold Pop scores H 0.
    als = s[s.model == "cold_als"]
    assert (als.gt_unscorable == als.gt_pairs).all() and (als["recall@50"] == 0).all()
    ca = r["per_user"][("video_long_view", "cold_video_full")]
    assert (ca["cold_als"]["n_gt_unscorable"] == ca["cold_als"]["n_gt"]).all()


def test_cold_model_scores_h_from_content_only(P, e2e):
    """H rows of the cold export are finite and come from the trained content path: zeroing the
    (fixed-zero) OOV video-ID row changes nothing, and every H video_idx is OOV."""
    from kuairand_rec.models import artifacts as A
    from kuairand_rec.models.export import ExportedScorer
    cond = D.build_condition({"name": "cold_fx", "exclude_video_ids_file": str(e2e["tmp"] / "H.txt")}, P, fx.VIDEO_META)
    sc = ExportedScorer(e2e["cold_run"] / "export", expect_condition_id=cond.condition_id)
    rows = P["catalogs"].video_row(H)
    assert np.isfinite(sc.I["video_long_view"][rows]).all()
    model, _ = A.load_model(e2e["cold_run"] / "best.pt", cond)
    assert float(model.video_tower.video_emb.effective_weight()[catm.OOV].detach().abs().sum()) == 0.0
    with pytest.raises(ValueError):
        ExportedScorer(e2e["seen_run"] / "export", expect_condition_id=cond.condition_id)


def test_evaluate_cold_outputs_and_bootstrap(e2e):
    out = __import__("pathlib").Path(e2e["cfg"]["output_dir"])
    for f in ("cold_metrics.csv", "cold_bootstrap_models.csv", "cold_bootstrap_diffs.csv", "cold_meta.json", "COLD_VIDEO.md"):
        assert (out / f).exists(), f
    d = pd.read_csv(out / "cold_bootstrap_diffs.csv")
    assert {"cold_threetower"} == set(d.model_a) - {"seen_threetower"}
    assert set(d.frame) == set(C.FRAME_KINDS)
    meta = json.loads((out / "cold_meta.json").read_text())
    assert meta["pending"] == [] and meta["models"]["cold_threetower"]["provenance"]["parity_ok"]


def test_evaluate_cold_refuses_mismatched_h_or_condition(P, e2e, tmp_path):
    cfg = dict(e2e["cfg"], output_dir=str(tmp_path / "o"), selection_json=str(tmp_path / "s.json"))
    (tmp_path / "s.json").write_text(json.dumps({"h_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="H file"):
        C.evaluate_cold(cfg, P=P, video_meta=fx.VIDEO_META)
    # A normal-condition run passed as the cold model is refused (condition_id mismatch).
    with pytest.raises(ValueError, match="condition_id"):
        C.evaluate_cold(dict(e2e["cfg"], output_dir=str(tmp_path / "o2")), str(e2e["seen_run"]), None,
                        P=P, video_meta=fx.VIDEO_META)
