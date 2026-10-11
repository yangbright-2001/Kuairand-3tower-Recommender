"""EvalFrame tests on the hand-built fixture (PLAN.md §3b P1-1/5/6/7, §3c Q4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from m3_fixtures import META, make_events

from kuairand_rec.data import catalog as catm
from kuairand_rec.data.history import HistoryIndex
from kuairand_rec.eval import frame as F
from kuairand_rec.eval.views import history_view, train_view


@pytest.fixture(scope="module")
def fx():
    return make_events()


def _frames(fx, target, action="long_view", split="test"):
    ev, lc, cat, h = fx
    return F.build_frames(ev, lc, cat, META, h, split, target, action)


def _gt(f):
    return {int(u): f.gt_rows(i).tolist() for i, u in enumerate(f.users)}


def test_video_frame_hand_computed(fx):
    fr = _frames(fx, "video")
    m = fr["masked"]
    # u1: raw {0,1,2,3}, v13 (row 3) seen at t=100 -> masked; u2 cold; u3 all masked; u4 {1}
    assert _gt(m) == {1: [0, 1, 2], 4: [1]}
    c = m.counts
    assert c["pos_events"] == 7 and c["pos_events_sentinel"] == 0 and c["pos_events_ooc"] == 0
    assert c["gt_pairs_raw"] == 7 and c["gt_pairs_masked_seen"] == 2
    assert c["users_cold_no_history"] == 1 and c["users_all_gt_masked"] == 1
    assert c["evaluated_users"] == 2 and c["evaluated_users_train_oov"] == 1
    assert m.is_train_oov_user.tolist() == [False, True]
    # u4's test history includes its val event (frozen at T_test).
    assert m.hist_len_full.tolist() == [1, 1]
    # mask rows: u1 {3}, u4 {0}
    assert m.mask_idx.tolist() == [3, 0]


def test_music_dedup_sentinel_and_mask_from_video_music(fx):
    m = _frames(fx, "music")["masked"]
    c = m.counts
    # u1 positives: v10,v11 (shared m100 -> one pair), v12 (sentinel), v13 (m101, seen)
    assert c["pos_events_sentinel"] == 1
    assert c["dedup_events_collapsed"] == 1
    # u4: GT m100 masked because its val video v10 carries m100 -> excluded
    assert _gt(m) == {1: [0]}
    assert c["users_all_gt_masked"] == 2  # u3 (m102 via v14) and u4 (m100 via v10)


def test_label_conflict_counted_as_upper_bound_only(fx):
    m = _frames(fx, "video")["masked"]
    # (u4, v13) is not in raw GT -> 1 possible lost pair; (u1, v10) is in GT -> not counted
    assert m.counts["label_conflict_events_any_positive"] == 2
    assert m.counts["label_conflict_possible_gt_pairs"] == 1
    assert 3 not in _gt(m)[4]  # never added to GT


def test_cold_frame(fx):
    cold = _frames(fx, "video")["cold"]
    assert cold.users.tolist() == [2] and _gt(cold) == {2: [0]}
    assert cold.hist_len_full.tolist() == [0]


def test_nomask_same_population_different_frame_id(fx):
    fr = _frames(fx, "video")
    m, n = fr["masked"], fr["nomask"]
    assert (m.users == n.users).all() and m.population_id == n.population_id
    assert m.frame_id != n.frame_id
    # nomask keeps seen GT (u1 row 3) and masks nothing; u3 is still excluded (fixed population)
    assert _gt(n) == {1: [0, 1, 2, 3], 4: [1]}
    assert len(n.mask_idx) == 0


def test_time_boundaries(fx):
    ev, lc, cat, h = fx
    m = _frames(fx, "video")["masked"]
    # u1's event at exactly T_test (v10@2000) is GT, not mask.
    assert 0 in _gt(m)[1] and 0 not in m.mask_idx[m.mask_indptr[0]:m.mask_indptr[1]]
    # u3's event at T_val - 1 = 999 is in the val-time mask/history; val frame mask has no val events.
    val = F.build_frames(ev, lc, cat, META, h, "val", "video", "long_view")["masked"]
    assert len(val.users) == 0  # no val positives in the fixture
    sptr, sidx = h.seen_rows(np.array([3, 4]), META["t_val_ms"], "video")
    assert sidx.tolist() == [4]  # u3's v14@999 seen; u4's val v10@1500 is not


def test_frame_id_sensitivity(fx):
    ev, lc, cat, h = fx
    base = _frames(fx, "video")["masked"]
    again = _frames(fx, "video")["masked"]
    assert base.frame_id == again.frame_id
    # permuted catalog ids -> different population_id / frame_id
    cat2 = catm.Catalogs(cat.video_ids[[1, 0, 2, 3, 4]], cat.music_ids, cat.video_music_row)
    pid2 = F.compute_population_id("test", "video", "long_view", 2000, cat2.video_ids, base.users)
    assert pid2 != base.population_id
    # user order
    assert F.compute_population_id("test", "video", "long_view", 2000, cat.video_ids,
                                   base.users[::-1]) != base.population_id
    # one GT element changed / one mask element changed
    gidx = base.gt_idx.copy(); gidx[-1] = 2
    assert F.compute_frame_id(base.population_id, "masked", base.gt_indptr, gidx,
                              base.mask_indptr, base.mask_idx) != base.frame_id
    midx = base.mask_idx.copy(); midx[0] = 2
    assert F.compute_frame_id(base.population_id, "masked", base.gt_indptr, base.gt_idx,
                              base.mask_indptr, midx) != base.frame_id


def test_ooc_counted():
    ev, lc, cat, _ = make_events()
    # an extra test positive on a video outside the catalog
    extra = ev.iloc[[0]].copy()
    extra[["video_id", "time_ms", "split", "long_view", "video_row", "music_row", "music_id"]] = [99, 2900, 2, 1, -1, -1, 555]
    ev2 = pd.concat([ev, extra], ignore_index=True)
    f = F.build_frames(ev2, lc, cat, META, HistoryIndex(ev2), "test", "video", "long_view")["masked"]
    assert f.counts["pos_events_ooc"] == 1
    fm = F.build_frames(ev2, lc, cat, META, HistoryIndex(ev2), "test", "music", "long_view")["masked"]
    assert fm.counts["pos_events_ooc"] == 1 and fm.counts["pos_events_sentinel"] == 1


def test_views_boundaries(fx):
    ev = fx[0]
    assert (train_view(ev).split == 0).all()
    assert history_view(ev, "val", META).split.max() == 0
    assert history_view(ev, "test", META).split.max() == 1
    assert history_view(ev, "test", META).time_ms.max() < 2000
    bad = ev.copy()
    bad.loc[bad.time_ms == 1500, "split"] = 0  # val event mislabelled as train
    with pytest.raises(ValueError):
        history_view(bad, "val", META)
