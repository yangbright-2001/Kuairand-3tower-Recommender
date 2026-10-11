"""M2 correctness tests on small hand-built data (PLAN.md §3b P1-2..P1-6b, P1-9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuairand_rec.data import catalog as catm
from kuairand_rec.data import clean, split, train_stats
from kuairand_rec.data.history import HistoryIndex, csr_row
from kuairand_rec.data.pipeline import attach_ids, encode_events

ACTIONS = ["long_view", "is_like"]
LABEL_COLS = ["long_view", "is_like"]


def _log(rows):
    cols = ["user_id", "video_id", "time_ms", "long_view", "is_like", "play_time_ms", "hourmin", "src_file"]
    return pd.DataFrame(rows, columns=cols)


# ============================================================================ A. cleaning
def test_cleaning_hand_computed():
    log = _log([
        # (u0,v1,100): exact duplicate x3 -> 1 event, 2 redundant rows
        (0, 1, 100, 1, 0, 500, 1900, 0),
        (0, 1, 100, 1, 0, 500, 1900, 0),
        (0, 1, 100, 1, 0, 500, 1900, 1),   # src_file is provenance, not compared
        # (u0,v2,100): hourmin conflict, labels agree -> keep row with max play_time
        (0, 2, 100, 0, 0, 300, 1900, 0),
        (0, 2, 100, 0, 0, 900, 2000, 1),
        # (u0,v3,200): long_view conflict -> whole event dropped
        (0, 3, 200, 0, 0, 100, 1900, 0),
        (0, 3, 200, 1, 0, 200, 1900, 0),
        # (u0,v1,300): genuine repeat of (u0,v1) at a later time -> kept
        (0, 1, 300, 0, 1, 50, 2100, 0),
        # (u1,v1,100): different user same video/time -> independent event
        (1, 1, 100, 0, 0, 10, 1900, 0),
    ])
    ev, rep = clean.clean_logs(log, LABEL_COLS, ACTIONS)
    ex = rep["exclusions"]
    assert ex["exact_duplicate_redundant_rows_removed"] == 2
    assert ex["conflict_groups_total"] == 2
    assert ex["label_conflict_events_dropped"] == 1
    assert ex["label_conflict_events_by_label"] == {"long_view": 1, "is_like": 0}
    assert ex["label_conflict_rows_dropped"] == 2
    assert ex["conflict_rows_removed_by_representative"] == 1
    # keys: (0,1,100) (0,2,100) (0,3,200) (0,1,300) (1,1,100) -> 5; one dropped -> 4
    assert rep["before"]["rows"] == 9 and rep["before"]["distinct_events"] == 5
    assert rep["after"]["rows"] == rep["after"]["distinct_events"] == 4
    assert rep["row_conservation_ok"] and rep["event_conservation_ok"]
    assert not ((ev.user_id == 0) & (ev.video_id == 3)).any()
    # Representative is the whole max-play row (hourmin 2000 comes with play 900).
    r = ev[(ev.user_id == 0) & (ev.video_id == 2)].iloc[0]
    assert (r.play_time_ms, r.hourmin) == (900, 2000)
    # Genuine repeat kept: (u0, v1) at both 100 and 300, labels untouched.
    rep_rows = ev[(ev.user_id == 0) & (ev.video_id == 1)].sort_values("time_ms")
    assert rep_rows.time_ms.tolist() == [100, 300]
    assert rep_rows.long_view.tolist() == [1, 0] and rep_rows.is_like.tolist() == [0, 1]
    assert rep["after"]["long_view_positive_rows"] == 1 and rep["after"]["is_like_positive_rows"] == 1
    # before: lv rows 3 (dup copies) + 1 (conflict row) = 4; events with any lv: (0,1,100), (0,3,200)
    assert rep["before"]["long_view_positive_rows"] == 4
    assert rep["before"]["long_view_positive_events"] == 2


def test_cleaning_deterministic_under_row_order():
    log = _log([
        (0, 2, 100, 0, 0, 900, 1900, 0),
        (0, 2, 100, 0, 0, 900, 2000, 1),   # play tie -> broken by remaining columns
        (0, 4, 50, 1, 0, 5, 1800, 0),
    ])
    a, _ = clean.clean_logs(log, LABEL_COLS, ACTIONS)
    b, _ = clean.clean_logs(log.iloc[::-1].reset_index(drop=True), LABEL_COLS, ACTIONS)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[a.video_id == 2, "hourmin"].item() == 1900


def test_like_conflict_dropped_and_nan_aware_duplicates():
    log = _log([
        (0, 1, 10, 0, 0, 5, 1900, 0), (0, 1, 10, 0, 1, 5, 1900, 0),  # is_like conflict
        (0, 2, 10, 0, 0, 5, np.nan, 0), (0, 2, 10, 0, 0, 5, np.nan, 1),  # NaN == NaN -> exact dup
    ])
    ev, rep = clean.clean_logs(log, LABEL_COLS, ACTIONS)
    assert rep["exclusions"]["label_conflict_events_by_label"] == {"long_view": 0, "is_like": 1}
    assert rep["exclusions"]["exact_duplicate_redundant_rows_removed"] == 1
    assert ev.video_id.tolist() == [2]


def test_duration_zero_marked_missing_labels_untouched():
    ev = pd.DataFrame({"duration_ms": [0, 5000], "long_view": [1, 0]})
    out = clean.mark_missing_duration(ev)
    assert np.isnan(out.duration_ms[0]) and out.duration_ms[1] == 5000
    assert out.long_view.tolist() == [1, 0]


# ============================================================================ B. split
def test_cut_points_cst():
    t_val = split.local_to_epoch_ms("2022-04-17 00:00:00", "Asia/Shanghai")
    # 2022-04-17 00:00 CST = 2022-04-16 16:00 UTC
    assert t_val == int(pd.Timestamp("2022-04-16 16:00:00", tz="UTC").value // 10**6)


def test_split_boundaries_and_same_timestamp():
    tv, tt = 1000, 2000
    t = np.array([0, 999, 1000, 1000, 1001, 1999, 2000, 2000, 5000])
    s = split.assign_split(t, tv, tt)
    assert s.tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2]
    # every timestamp maps to exactly one split
    assert pd.DataFrame({"t": t, "s": s}).groupby("t").s.nunique().max() == 1
    with pytest.raises(ValueError):
        split.assign_split(t, tt, tv)


def test_split_conservation():
    ev = pd.DataFrame({"user_id": [0, 0, 1, 1, 2], "video_id": [1, 2, 1, 3, 3],
                       "time_ms": [5, 15, 25, 15, 1], "long_view": [1, 0, 1, 1, 0], "is_like": [0, 0, 1, 0, 0]})
    ev["split"] = split.assign_split(ev.time_ms.to_numpy(), 10, 20)
    rows = split.split_summary(ev, ACTIONS, "UTC")
    assert [r["events"] for r in rows] == [2, 2, 1]
    assert sum(r["long_view_positive_events"] for r in rows) == ev.long_view.sum()
    assert sum(r["is_like_positive_events"] for r in rows) == ev.is_like.sum()


# ============================================================================ C. catalogs / vocab
def _video_meta():
    # music 0 = sentinel (videos 3, 5); music 77 shared by videos 1 and 2; video 6 never in logs.
    return pd.DataFrame({
        "video_id": [1, 2, 3, 4, 5, 6],
        "author_id": [10, 10, 30, 40, 50, 60],
        "music_id": [77, 77, 0, 44, 0, 99],
        "video_type": ["NORMAL"] * 5 + ["AD"],
        "upload_type": ["Web"] * 6,
        "upload_dt": ["2022-04-10"] * 6,
        "music_type": [9.0, 9.0, np.nan, 4.0, np.nan, 8.0],
        "tag": ["1,2", "2", None, "3", "1", "5"],
        "video_duration": [1000.0, 2000.0, np.nan, 4000.0, 5000.0, 6000.0],
        "server_width": [720.0] * 6, "server_height": [1280.0] * 6,
    })


def test_catalogs_music_sentinel_and_shared_music():
    cat = catm.build_catalogs(_video_meta())
    assert cat.video_ids.tolist() == [1, 2, 3, 4, 5, 6]          # includes never-logged video 6
    assert cat.music_ids.tolist() == [44, 77, 99]                 # non-zero only, 99 never logged
    assert cat.video_music_row.tolist() == [1, 1, -1, 0, -1, 2]   # shared music -> same row
    assert cat.music_row([0, 44, 12345]).tolist() == [-1, 0, -1]
    assert cat.video_row([6, 7]).tolist() == [5, -1]


def _toy_events():
    # Train: users 0,1 on videos 1,2,3. Val/test introduce user 2, video 4 (author 40, music 44).
    ev = pd.DataFrame({
        "user_id":   [0, 0, 1, 1, 2, 0],
        "video_id":  [1, 3, 2, 1, 4, 4],
        "time_ms":   [1, 2, 3, 4, 15, 25],
        "long_view": [1, 1, 0, 1, 1, 1],
        "is_like":   [0, 1, 1, 0, 0, 0],
    })
    ev["split"] = split.assign_split(ev.time_ms.to_numpy(), 10, 20)
    return ev


def test_train_only_vocab_and_oov():
    vm = _video_meta()
    cat = catm.build_catalogs(vm)
    ev = attach_ids(_toy_events(), vm, cat)
    voc = catm.build_vocabs(ev[ev.split == 0], vm)
    assert voc.user.ids.tolist() == [0, 1]
    assert voc.video.ids.tolist() == [1, 2, 3]
    assert voc.author.ids.tolist() == [10, 30]           # author 40 only on val/test video
    assert voc.music.ids.tolist() == [77]                 # 0 is not an id; 44 not train-seen
    # Index layout: PAD=0, OOV=1, ids from 2 (music from 3, NO_MUSIC=2).
    assert voc.video.encode([1, 2, 3, 4, 6]).tolist() == [2, 3, 4, 1, 1]
    assert voc.user.encode([0, 1, 2]).tolist() == [2, 3, 1]
    assert voc.encode_music([77, 0, 44, 99]).tolist() == [3, catm.NO_MUSIC, catm.OOV, catm.OOV]
    enc = encode_events(ev, voc)
    test_row = enc[enc.split == 2].iloc[0]
    assert (test_row.video_idx, test_row.author_idx, test_row.music_idx) == (1, 1, 1)
    val_row = enc[enc.split == 1].iloc[0]
    assert val_row.user_idx == catm.OOV                   # user 2 first appears in val
    assert (enc.loc[enc.split == 0, ["user_idx", "video_idx", "author_idx"]] > catm.OOV).all().all()
    # Metadata categories: AD only on never-logged video 6 -> OOV.
    assert voc.meta["video_type"].encode(np.array(["NORMAL", "AD"], dtype=object)).tolist() == [2, 1]
    assert voc.meta["tag"].ids.tolist() == ["1", "2"]     # tags of train videos 1,2,3


def test_cold_video_exclusion_hook():
    vm = _video_meta()
    cat = catm.build_catalogs(vm)
    ev = attach_ids(_toy_events(), vm, cat)
    cold = train_stats.remove_videos(ev, [1])
    assert 1 not in cold.video_id.values
    voc = catm.build_vocabs(cold[cold.split == 0], vm)
    assert voc.video.ids.tolist() == [2, 3]
    assert voc.user.ids.tolist() == [0, 1]                # both still have non-H train events
    assert voc.music.ids.tolist() == [77]                 # still via video 2
    # Same result via the exclude argument.
    voc2 = catm.build_vocabs(ev[ev.split == 0], vm, exclude_video_ids=[1])
    assert voc2.video.ids.tolist() == [2, 3]


def test_train_only_popularity():
    vm = _video_meta()
    cat = catm.build_catalogs(vm)
    ev = attach_ids(_toy_events(), vm, cat)
    with pytest.raises(ValueError):
        train_stats.popularity(ev, cat, ACTIONS)          # val/test rows must be refused
    pop = train_stats.popularity(ev[ev.split == 0], cat, ACTIONS)
    v = pop["video"].set_index("video_id")
    assert v.train_events.tolist() == [2, 1, 1, 0, 0, 0]
    assert v.train_long_view.tolist() == [2, 0, 1, 0, 0, 0]   # val/test lv on video 4 ignored
    assert v.train_is_like.tolist() == [0, 1, 1, 0, 0, 0]
    m = pop["music"].set_index("music_id")
    # music 77 = videos 1+2; video 3 (music 0) contributes to no music row.
    assert m.loc[77, "train_events"] == 3 and m.loc[77, "train_long_view"] == 2
    assert m.loc[77, "train_is_like"] == 1
    assert m.loc[44, "train_events"] == 0 and m.loc[99, "train_events"] == 0
    assert m.train_events.sum() == 3                       # 4 train events minus 1 sentinel


def test_numeric_stats_train_only_and_missing_flag():
    vm = _video_meta()
    cat = catm.build_catalogs(vm)
    ev = attach_ids(_toy_events(), vm, cat)
    st = train_stats.numeric_video_stats(ev[ev.split == 0], vm)
    # train videos 1,2,3; video 3 duration is NaN
    exp = np.log1p([1000.0, 2000.0])
    assert st["video_duration"]["mean"] == pytest.approx(exp.mean())
    assert st["video_duration"]["n_missing_train_videos"] == 1
    voc = catm.build_vocabs(ev[ev.split == 0], vm)
    f = catm.encode_video_features(vm, cat, voc, st)
    assert f.video_duration_missing.tolist() == [0, 0, 1, 0, 0, 0]
    assert f.loc[2, "video_duration_z"] == 0.0
    assert f.music_row.tolist() == cat.video_music_row.tolist()
    assert f[["tag0_idx", "tag1_idx"]].iloc[0].tolist() == [2, 3]   # "1","2"
    assert f.loc[3, "tag0_idx"] == catm.OOV                          # tag "3" only on video 4


# ============================================================================ D. history
def _hist_events():
    """User 7: events at t=10(v1), 20(v2), 20(v3), 30(v1 repeat), 40(v4). User 8: one event at 20."""
    ev = pd.DataFrame({
        "user_id":  [7, 7, 7, 7, 7, 8],
        "video_id": [1, 3, 2, 1, 4, 9],
        "time_ms":  [10, 20, 20, 30, 40, 20],
        "long_view": [1, 0, 1, 0, 1, 1],
    })
    ev["video_row"] = ev.video_id
    ev["music_row"] = ev.video_id.map({1: 0, 2: 0, 3: -1, 4: 1, 9: 2})   # v1,v2 share music 0-row
    return ev


def test_history_strictly_before_and_same_timestamp():
    h = HistoryIndex(_hist_events())
    pos, n = h.model_history(np.array([7, 7, 7, 7, 8]), np.array([10, 20, 21, 40, 20]), max_len=4)
    vid = h.gather(pos, "video_id", pad=-1)
    assert n.tolist() == [0, 1, 3, 4, 0]
    assert vid[0].tolist() == [-1, -1, -1, -1]           # nothing before the first event
    assert vid[1].tolist() == [-1, -1, -1, 1]            # t=20 events never see each other
    assert vid[2].tolist() == [-1, 1, 2, 3]              # same-ts order by video_id: v2 before v3
    assert vid[3].tolist() == [1, 2, 3, 1]               # target at t=40 excluded; repeat v1 kept
    assert vid[4].tolist() == [-1] * 4                   # other users never leak in
    # Feedback tokens come from the same past events.
    assert h.gather(pos, "long_view", pad=-1)[3].tolist() == [1, 1, 0, 0]


def test_history_no_future_leakage_property():
    rng = np.random.default_rng(0)
    ev = pd.DataFrame({"user_id": rng.integers(0, 5, 400), "video_id": rng.integers(0, 30, 400),
                       "time_ms": rng.integers(0, 50, 400)}).drop_duplicates(["user_id", "video_id", "time_ms"])
    h = HistoryIndex(ev)
    # Train usage: every event predicts itself from its own history.
    pos, n = h.model_history(h.events.user_id.to_numpy(), h.events.time_ms.to_numpy(), max_len=100)
    t_hist = h.gather(pos, "time_ms", pad=-1)
    u_hist = h.gather(pos, "user_id", pad=-1)
    tq = h.events.time_ms.to_numpy()[:, None]
    uq = h.events.user_id.to_numpy()[:, None]
    valid = pos >= 0
    assert (t_hist[valid] < np.broadcast_to(tq, pos.shape)[valid]).all()
    assert (u_hist[valid] == np.broadcast_to(uq, pos.shape)[valid]).all()
    # Uncapped count equals the brute-force strict count.
    brute = [int(((ev.user_id == u) & (ev.time_ms < t)).sum()) for u, t in zip(uq[:, 0], tq[:, 0])]
    assert n.tolist() == brute


def test_frozen_history_information_boundary():
    ev = pd.DataFrame({"user_id": [0, 0, 0, 0], "video_id": [1, 2, 3, 4], "time_ms": [5, 12, 18, 25]})
    ev["split"] = split.assign_split(ev.time_ms.to_numpy(), 10, 20)
    h = HistoryIndex(ev)
    pos_v, _ = h.model_history(np.array([0]), 10, max_len=10)    # val frozen at T_val
    pos_t, _ = h.model_history(np.array([0]), 20, max_len=10)    # test frozen at T_test
    sv = h.gather(pos_v, "split", pad=-1)[0]
    st = h.gather(pos_t, "split", pad=-1)[0]
    assert set(sv[sv >= 0]) == {0}                               # val sees train only
    assert set(st[st >= 0]) == {0, 1}                            # test sees train + val
    assert 2 not in st                                           # never test events


def test_history_cap_vs_uncapped_seen_mask():
    # 105 distinct videos, one per timestamp; the oldest 5 fall outside the 100 cap.
    ev = pd.DataFrame({"user_id": 0, "video_id": np.arange(105), "time_ms": np.arange(105) + 1})
    ev["video_row"] = ev.video_id
    ev["music_row"] = ev.video_id // 2                           # music shared by video pairs
    h = HistoryIndex(ev)
    pos, n = h.model_history(np.array([0]), 1000, max_len=100)
    vids = h.gather(pos, "video_id", pad=-1)[0]
    assert n[0] == 100 and vids[0] == 5 and vids[-1] == 104      # most recent 100
    indptr, rows = h.seen_rows(np.array([0]), 1000, "video")
    seen = csr_row(indptr, rows, 0)
    assert len(seen) == 105 and {0, 1, 2, 3, 4} <= set(seen)     # mask is uncapped
    assert not ({0, 1, 2, 3, 4} & set(vids))
    _, mrows = h.seen_rows(np.array([0]), 1000, "music")
    assert len(mrows) == 53                                      # 105 videos -> music 0..52


def test_seen_mask_dedup_music_sentinel_and_strictness():
    h = HistoryIndex(_hist_events())
    indptr, rows = h.seen_rows(np.array([7, 7, 8]), np.array([30, 41, 21]), "video")
    assert csr_row(indptr, rows, 0).tolist() == [1, 2, 3]        # t<30: v1, v2, v3
    assert csr_row(indptr, rows, 1).tolist() == [1, 2, 3, 4]     # repeat v1 deduped
    assert csr_row(indptr, rows, 2).tolist() == [9]
    mi, mr = h.seen_rows(np.array([7, 7]), np.array([30, 41]), "music")
    assert csr_row(mi, mr, 0).tolist() == [0]                    # v1,v2 share music row 0; v3 sentinel skipped
    assert csr_row(mi, mr, 1).tolist() == [0, 1]


def test_history_insensitive_to_input_order():
    ev = _hist_events()
    a = HistoryIndex(ev)
    b = HistoryIndex(ev.sample(frac=1.0, random_state=1))
    pa, _ = a.model_history(np.array([7]), 41, max_len=5)
    pb, _ = b.model_history(np.array([7]), 41, max_len=5)
    assert (a.gather(pa, "video_id", -1) == b.gather(pb, "video_id", -1)).all()


# ============================================================================ review fixes (2026-10-11)
def test_label_conflict_events_returned_for_gt_accounting():
    log = _log([
        (0, 3, 200, 0, 0, 100, 1900, 0), (0, 3, 200, 1, 0, 200, 1900, 0),   # lv conflict, lv-any=1
        (1, 4, 300, 0, 1, 5, 1900, 0), (1, 4, 300, 0, 0, 5, 1900, 0),       # like conflict, like-any=1
        (2, 5, 400, 1, 0, 5, 1900, 0),
    ])
    ev, rep, dropped = clean.clean_logs(log, LABEL_COLS, ACTIONS, return_dropped=True)
    assert ev.video_id.tolist() == [5]
    assert dropped[["user_id", "video_id", "time_ms"]].values.tolist() == [[0, 3, 200], [1, 4, 300]]
    assert dropped.long_view_any.tolist() == [1, 0] and dropped.is_like_any.tolist() == [0, 1]
    assert len(dropped) == rep["exclusions"]["label_conflict_events_dropped"]


def test_train_only_fitters_require_split_column():
    vm = _video_meta()
    cat = catm.build_catalogs(vm)
    ev = attach_ids(_toy_events(), vm, cat).drop(columns="split")
    with pytest.raises(ValueError):
        train_stats.popularity(ev, cat, ACTIONS)
    with pytest.raises(ValueError):
        train_stats.numeric_video_stats(ev, vm)


def test_build_vocabs_train_only_guard():
    vm = _video_meta()
    cat = catm.build_catalogs(vm)
    ev = attach_ids(_toy_events(), vm, cat)
    # (1) no split column -> refused
    with pytest.raises(ValueError, match="split"):
        catm.build_vocabs(ev[ev.split == 0].drop(columns="split"), vm)
    # (2) any non-train row -> refused (would admit val user 2 / test video 4)
    with pytest.raises(ValueError, match="non-train"):
        catm.build_vocabs(ev, vm)
    with pytest.raises(ValueError, match="non-train"):
        catm.build_vocabs(ev[ev.split <= 1], vm)
    # (3) legal train-only input: unchanged behaviour, incl. the cold exclusion path
    voc = catm.build_vocabs(ev[ev.split == 0], vm)
    assert voc.user.ids.tolist() == [0, 1] and voc.video.ids.tolist() == [1, 2, 3]
    assert voc.author.ids.tolist() == [10, 30] and voc.music.ids.tolist() == [77]
    cold = catm.build_vocabs(ev[ev.split == 0], vm, exclude_video_ids=[1])
    assert cold.video.ids.tolist() == [2, 3]


def test_empty_vocab_maps_everything_to_oov():
    v = catm._vocab("x", np.array([], dtype=np.int64))
    assert v.encode([1, 2]).tolist() == [catm.OOV, catm.OOV]


def test_music_features_shared_music_and_sentinel():
    vm = _video_meta()
    cat = catm.build_catalogs(vm)
    ev = attach_ids(_toy_events(), vm, cat)
    voc = catm.build_vocabs(ev[ev.split == 0], vm)
    mf = catm.encode_music_features(vm, cat, voc)
    assert mf.music_id.tolist() == [44, 77, 99]
    assert mf.n_videos_metadata.tolist() == [1, 2, 1]          # 77 shared by videos 1, 2
    assert mf.music_idx.tolist() == [catm.OOV, 3, catm.OOV]    # only 77 is train-seen
    bad = vm.copy()
    bad.loc[bad.video_id == 2, "music_type"] = 4.0               # same music, different type
    with pytest.raises(ValueError):
        catm.encode_music_features(bad, cat, voc)


def test_fit_condition_cold_refits_everything_consistently():
    from kuairand_rec.data.pipeline import fit_condition
    vm = _video_meta()
    cat = catm.build_catalogs(vm)
    ev = attach_ids(_toy_events(), vm, cat)
    normal = fit_condition(ev, vm, cat, ACTIONS)
    # Stale normal-condition indices on the input must be overwritten, not reused.
    stale = normal["events"]
    cold = fit_condition(stale, vm, cat, ACTIONS, exclude_video_ids=[1, 3])
    ce, cv = cold["events"], cold["vocabs"]
    assert not ce.video_id.isin([1, 3]).any()
    assert cv.video.ids.tolist() == [2] and cv.author.ids.tolist() == [10]
    assert cv.user.ids.tolist() == [1]                          # user 0 only had H train events
    assert (ce.video_idx == cv.video.encode(ce.video_id.to_numpy())).all()
    assert ce.loc[ce.user_id == 0, "user_idx"].eq(catm.OOV).all()
    # Video features: H rows stay in the catalog but are content-only (video OOV);
    # video 1's author 10 is still in vocab via video 2; video 3's author 30 is OOV.
    vf = cold["video_features"].set_index("video_id")
    assert vf.loc[[1, 3], "video_idx"].tolist() == [catm.OOV, catm.OOV]
    assert vf.loc[1, "author_idx"] == 2 and vf.loc[3, "author_idx"] == catm.OOV
    assert len(vf) == len(cat.video_ids)
    # Popularity: H gets 0; music 77 counts only video 2.
    pv = cold["popularity"]["video"].set_index("video_id")
    assert pv.loc[[1, 3], "train_events"].tolist() == [0, 0]
    assert cold["popularity"]["music"].set_index("music_id").loc[77, "train_events"] == 1
    # Numeric stats now fitted on video 2 only.
    assert cold["num_stats"]["video_duration"]["mean"] == pytest.approx(np.log1p(2000.0))
