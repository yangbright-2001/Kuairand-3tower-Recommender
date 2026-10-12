"""M4-1 feature / history leakage tests (PLAN.md §3d D1-2, D1-6, D1-9 L1-L5)."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

import m4_fixtures as fx
from kuairand_rec.data import catalog as catm
from kuairand_rec.data.history import HistoryIndex
from kuairand_rec.models import features as F

MIN = fx.MIN
CFG = F.HistoryConfig()
T_VAL, T_TEST = fx.META["t_val_ms"], fx.META["t_test_ms"]


@pytest.fixture(scope="module")
def fit():
    return fx.make()


def _hist(fit, split_name, users):
    hidx, t = F.frozen_history_index(fit["events"], split_name, fx.META)
    return F.build_history_batch(hidx, users, t, CFG)


def _vid(fit, video_ids):
    return fit["catalogs"].video_row(np.asarray(video_ids))


def _tokens(hb, i):
    """(video_row list, valid count) of user i, oldest -> newest."""
    return hb.video_row[i][hb.valid[i]].tolist()


def _assert_batches_equal(a: F.HistoryBatch, b: F.HistoryBatch):
    for f in fields(F.HistoryBatch):
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert x.dtype == y.dtype and x.shape == y.shape, f.name
        assert x.tobytes() == y.tobytes(), f"{f.name} differs"   # bitwise


# ============================================================================ L1
def test_l1_strict_past_same_timestamp_and_cap_at_three_t_pred(fit):
    ev = fit["events"]
    # (a) train: t_pred = target time, index on train events only.
    hidx = F.train_history_index(ev)
    assert (hidx.events.split == 0).all()
    hb = F.build_history_batch(hidx, [1, 1, 1], [50 * MIN, 60 * MIN, 10 * MIN], CFG)
    # target v10@50: v11 at the same timestamp is excluded, only v13@10 remains.
    assert _tokens(hb, 0) == _vid(fit, [13]).tolist()
    # target v12@60: both same-timestamp events at 50 are now strictly past; ordered by video_id.
    assert _tokens(hb, 1) == _vid(fit, [13, 10, 11]).tolist()
    assert hb.length.tolist() == [1, 3, 0]          # target v13@10 has no past event
    # (b) T_val: train events only (val event v15@120 must not appear).
    hb = _hist(fit, "val", [1])
    assert _tokens(hb, 0) == _vid(fit, [13, 10, 11, 12, 14]).tolist()
    # (c) T_test: train + val events, no test events.
    hb = _hist(fit, "test", [1])
    assert _tokens(hb, 0) == _vid(fit, [13, 10, 11, 12, 14, 15, 13]).tolist()
    # Every token is strictly before t_pred at all three settings (also asserted in the builder).
    for split_name, t in (("val", T_VAL), ("test", T_TEST)):
        hidx, _ = F.frozen_history_index(ev, split_name, fx.META)
        users = np.unique(ev.user_id)
        pos, _ = hidx.model_history(users, t, CFG.max_len)
        assert (hidx.gather(pos, "time_ms", pad=0)[pos >= 0] < t).all()


def test_l1_cap_keeps_most_recent_100_right_aligned(fit):
    ev = fit["events"]
    hb = _hist(fit, "val", [2])
    assert hb.length[0] == 100 and hb.valid.shape == (1, 100) and hb.valid.all()
    u2 = ev[(ev.user_id == 2) & (ev.time_ms < T_VAL)].sort_values(["time_ms", "video_id"])
    assert len(u2) == 130
    assert hb.video_row[0].tolist() == u2.video_row.to_numpy()[-100:].tolist()
    # Right alignment: newest token is in the last column; short histories are left-padded.
    hb1 = _hist(fit, "val", [1])
    assert hb1.valid[0, -5:].all() and not hb1.valid[0, :-5].any()
    assert (hb1.video_row[0, :-5] == -1).all()


def test_empty_history_and_unknown_user(fit):
    hb = _hist(fit, "test", [3, 999])               # u3: test-only; 999: never seen
    assert hb.length.tolist() == [0, 0]
    assert not hb.valid.any() and (hb.video_row == -1).all()
    assert (hb.feedback == 0).all() and (hb.hidden == 0).all()
    assert (hb.play_bucket == F.PLAY_NONE).all() and (hb.recency_bucket == F.RECENCY_PAD).all()


# ============================================================================ L2 / L5
def test_l2_hidden_iff_age_below_d(fit):
    hidx = F.train_history_index(fit["events"])
    t_ev = 99 * MIN                                     # u4's only event (like = 1)
    D = CFG.feedback_delay_ms
    hb = F.build_history_batch(hidx, [4, 4, 4], [t_ev + 1, t_ev + D - 1, t_ev + D], CFG)
    assert hb.hidden[:, -1].tolist() == [1.0, 1.0, 0.0]
    # Hidden: all feedback zero, play bucket = PLAY_NONE. Visible at exactly age == D.
    assert (hb.feedback[:2, -1] == 0).all() and (hb.play_bucket[:2, -1] == F.PLAY_NONE).all()
    lv, like = F.FEEDBACK_BINARY_COLS.index("long_view"), F.FEEDBACK_BINARY_COLS.index("is_like")
    assert hb.feedback[2, -1, lv] == 1 and hb.feedback[2, -1, like] == 1
    assert hb.play_bucket[2, -1] >= F.PLAY_ZERO
    # Video identity and recency are kept even when feedback is hidden.
    assert (hb.video_row[:, -1] == _vid(fit, [13])[0]).all()
    assert (hb.recency_bucket[:, -1] > F.RECENCY_PAD).all()


def test_l2_hidden_at_t_val(fit):
    hb = _hist(fit, "val", [1, 4])
    # u1: v14@85 is 15 min old -> hidden; older tokens visible. u4: v13@99 -> hidden.
    assert hb.hidden[0, -5:].tolist() == [0, 0, 0, 0, 1]
    assert hb.hidden[1, -1] == 1


def test_l5_val_events_younger_than_d_hidden_at_t_test(fit):
    hb = _hist(fit, "test", [1])
    # last token = val event v13@195 (age 5 min); the val event v15@120 (80 min) is visible.
    assert hb.video_row[0, -1] == _vid(fit, [13])[0] and hb.hidden[0, -1] == 1
    assert (hb.feedback[0, -1] == 0).all()
    assert hb.video_row[0, -2] == _vid(fit, [15])[0] and hb.hidden[0, -2] == 0
    assert hb.feedback[0, -2].sum() > 0


# ============================================================================ L3 / L4
def _rows_modified(pred):
    """Raw fixture rows where `pred(row)` rows get every feedback / play / label field changed."""
    rows = fx.raw_rows()
    for r in rows:
        if pred(r):
            for c in fx.FB:
                r[c] = 1 - r[c]
            r["play_time_ms"] = r["play_time_ms"] * 7 + 123_456
            r["duration_ms"] = 1_000.0
    return rows


def test_l3_hidden_token_payload_cannot_reach_inputs(fit):
    # Tokens hidden at each t_pred (age < D). Hiding is per t_pred: u1 v14@85 is hidden at
    # T_val but 115 min old (visible) at T_test, so each split modifies only its own set.
    hidden_at = {"val": {(1, 14, 85 * MIN), (4, 13, 99 * MIN)}, "test": {(1, 13, 195 * MIN)}}
    for s, keys in hidden_at.items():
        hb = _hist(fit, s, [1, 2, 3, 4])
        assert int(hb.hidden.sum()) == len(keys)
        fit2 = fx.make(_rows_modified(lambda r, k=keys: (r["user_id"], r["video_id"], r["time_ms"]) in k))
        _assert_batches_equal(hb, _hist(fit2, s, [1, 2, 3, 4]))
    fit2 = fx.make(_rows_modified(lambda r: (r["user_id"], r["video_id"], r["time_ms"]) == (4, 13, 99 * MIN)))
    # Sanity: the modification is real and visible when the token is old enough.
    late = F.build_history_batch(HistoryIndex(fit2["events"]), [4], [99 * MIN + CFG.feedback_delay_ms], CFG)
    base = F.build_history_batch(HistoryIndex(fit["events"]), [4], [99 * MIN + CFG.feedback_delay_ms], CFG)
    assert late.feedback.tobytes() != base.feedback.tobytes()


def test_l4_events_at_or_after_t_pred_cannot_reach_inputs(fit):
    users = [1, 2, 3, 4]
    for s, t in (("val", T_VAL), ("test", T_TEST)):
        fit2 = fx.make(_rows_modified(lambda r, t=t: r["time_ms"] >= t))
        _assert_batches_equal(_hist(fit, s, users), _hist(fit2, s, users))
        # Also add new events at exactly t_pred and later, including a new video id for a user.
        rows = fx.raw_rows() + [fx._row(1, 12, t, lv=1, like=1), fx._row(4, 10, t + 1, lv=1),
                                fx._row(2, 15, t + 5 * MIN)]
        _assert_batches_equal(_hist(fit, s, users), _hist(fx.make(rows), s, users))
    # Rolling train history: changing the target and every later train event of u1 is invisible.
    hidx = F.train_history_index(fit["events"])
    fit3 = fx.make(_rows_modified(lambda r: r["user_id"] == 1 and r["time_ms"] >= 50 * MIN))
    hidx3 = F.train_history_index(fit3["events"])
    _assert_batches_equal(F.build_history_batch(hidx, [1], [50 * MIN], CFG),
                          F.build_history_batch(hidx3, [1], [50 * MIN], CFG))


def test_builder_rejects_future_tokens(fit):
    # If an index ever returned a token at t_pred (a broken window), the builder must refuse.
    class Broken(HistoryIndex):
        def model_history(self, users, t_pred, max_len=100):
            return super().model_history(users, np.asarray(t_pred) + 1, max_len)

    with pytest.raises(AssertionError):
        F.build_history_batch(Broken(fit["events"]), [1], [50 * MIN], CFG)


# ============================================================================ buckets
def test_play_and_recency_buckets():
    pb = F.play_bucket([0, 5, 20, 100, 250, 1000, 10_000, 5], [100, 100, 100, 100, 100, 100, 100, np.nan])
    # 0 -> PLAY_ZERO; .05 -> (0,.1); .2 -> [.1,.25); 1.0 -> [1,2); 2.5 -> [2,inf); 100 -> [2,inf); NaN dur
    assert pb.tolist() == [2, 3, 4, 8, 9, 9, 9, F.PLAY_DURATION_MISSING]
    assert pb.max() < F.N_PLAY_BUCKETS
    # recency: 1 + floor(log2(1 + age_min)), clipped so frozen T_test ages reuse a trained row.
    rb = F.recency_bucket(np.array([1, MIN, 3 * MIN, 7 * 24 * 60 * MIN, 17 * 24 * 60 * MIN]), 13)
    assert rb.tolist() == [1, 2, 3, 14, 14]
    assert rb.max() < CFG.n_recency_rows


def test_recency_clip_covers_frozen_ages(fit):
    hb = _hist(fit, "test", [1, 2])
    assert hb.recency_bucket.max() < CFG.n_recency_rows


def test_history_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        F.HistoryConfig.from_dict({"max_length": 50})
    assert F.HistoryConfig.from_dict({"feedback_delay_ms": 1}).feedback_delay_ms == 1


# ============================================================================ item features
def test_video_music_features_follow_catalog_rows(fit):
    vc, mc = fit["video_features"], fit["music_features"]
    vf, mf = F.VideoFeatures.from_catalog(vc), F.MusicFeatures.from_catalog(mc)
    assert len(vf) == 6 and len(mf) == 4
    F.check_shared_music_inputs(vc, mc)
    v15 = int(fit["catalogs"].video_row([15])[0])
    # Val-only video: video, author, music are train-OOV (content path); metadata stays encoded.
    assert vf.video_idx[v15] == catm.OOV and vf.author_idx[v15] == catm.OOV and vf.music_idx[v15] == catm.OOV
    v12 = int(fit["catalogs"].video_row([12])[0])
    assert vf.music_idx[v12] == catm.NO_MUSIC and vf.num_missing[v12, 0] == 1 and vf.num[v12, 0] == 0
    with pytest.raises(ValueError):
        F.VideoFeatures.from_catalog(vc.iloc[::-1])
    bad = vc.copy()
    bad.loc[bad.video_id == 11, "music_idx"] = catm.OOV
    with pytest.raises(ValueError):
        F.check_shared_music_inputs(bad, mc)


# ============================================================================ U1
def _uf():
    uf = fx.USER_FEATURES.copy()
    uf["onehot_feat4"] = [np.nan, 1.0, 2.0, np.nan, 3.0, 1.0, 1.0]
    return uf


def test_u1_vocab_train_users_only_with_threshold(fit):
    train = fit["events"][fit["events"].split == 0]
    assert set(train.user_id) == {1, 2, 4}
    voc = F.fit_user_feature_vocab(_uf(), train, min_count=2)
    # onehot_feat0: train users all "0"; value "1" exists only on non-train users -> OOV.
    assert list(voc.vocabs["onehot_feat0"].ids) == ["0"]
    # NaN is its own category (u1, u4) and kept; "1.0" (u2 only) is rare -> OOV.
    assert list(voc.vocabs["onehot_feat4"].ids) == ["<NA>"]
    assert list(voc.vocabs["is_live_streamer"].ids) == ["-124"]
    assert list(voc.vocabs["is_video_author"].ids) == ["0"]            # "1": u2 only -> rare
    assert voc.summary["is_video_author"] == {"n_values_train_users": 2, "n_kept": 1, "n_train_users_on_rare": 1}
    tab = F.build_user_feature_table(_uf(), voc)
    codes = tab.lookup([1, 2, 3, 999])
    j = {f: i for i, f in enumerate(voc.fields)}
    assert codes.shape == (4, len(F.U1_FIELDS))
    assert codes[0, j["onehot_feat4"]] == 2 and codes[1, j["onehot_feat4"]] == catm.OOV
    assert codes[1, j["is_video_author"]] == catm.OOV and codes[0, j["is_video_author"]] == 2
    assert codes[2, j["register_days_range"]] == catm.OOV            # "8-14" never on a train user
    assert codes[2, j["is_live_streamer"]] == catm.OOV               # streamer value "1" not in train users
    assert (codes[3] == catm.OOV).all()                              # user missing from snapshot
    assert (codes >= catm.OOV).all()                                 # PAD never produced


def test_u1_vocab_refuses_non_train_events(fit):
    with pytest.raises(ValueError):
        F.fit_user_feature_vocab(_uf(), fit["events"], min_count=1)
    with pytest.raises(ValueError):
        F.fit_user_feature_vocab(pd.concat([_uf(), _uf().iloc[:1]]), fit["events"][fit["events"].split == 0])
