"""Pop / ALS / evaluator-integration tests (PLAN.md §3b P1-1, P1-6b; §3c Q1-Q3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from m3_fixtures import META, make_events, split_of

from kuairand_rec.baselines import als as A
from kuairand_rec.baselines.popularity import PopularityScorer, popularity_counts
from kuairand_rec.data import catalog as catm
from kuairand_rec.data.history import HistoryIndex
from kuairand_rec.data.pipeline import attach_ids
from kuairand_rec.eval import frame as F
from kuairand_rec.eval.evaluator import EvalContext, evaluate
from kuairand_rec.eval.views import history_view, train_view

ACTIONS = ["long_view", "is_like"]


# ----------------------------------------------------------------- random fixture for ALS
def _random_events(seed=0, n_users=60, n_videos=40):
    """Random log on a catalog with gaps: video ids 0..n_videos-1 but every 3rd video is
    never in train (non-contiguous scorable rows); users ids are sparse (multiples of 7)."""
    rng = np.random.default_rng(seed)
    vm = pd.DataFrame({"video_id": np.arange(n_videos), "author_id": np.arange(n_videos) % 5,
                       "music_id": np.where(np.arange(n_videos) % 10 == 0, 0, 1000 + np.arange(n_videos) // 2)})
    rows = []
    for u in range(n_users):
        for _ in range(rng.integers(3, 15)):
            v = int(rng.integers(n_videos))
            t = int(rng.integers(0, 3000))
            if t < 1000 and v % 3 == 2:      # gap: these videos never appear in train
                continue
            lv = int(rng.random() < 0.5)
            rows.append((7 * u + 3, v, t, lv, int(lv and rng.random() < 0.3)))
    ev = pd.DataFrame(rows, columns=["user_id", "video_id", "time_ms", "long_view", "is_like"])
    ev = ev.drop_duplicates(["user_id", "video_id", "time_ms"])
    ev["split"] = split_of(ev.time_ms)
    cat = catm.build_catalogs(vm)
    ev = attach_ids(ev, vm, cat).sort_values(["user_id", "time_ms", "video_id"], kind="mergesort")
    return ev.reset_index(drop=True), cat


CFG = A.ALSConfig(factors=4, regularization=0.1, alpha=5.0, iterations=5, seed=0)


@pytest.fixture(scope="module")
def rnd():
    return _random_events()


# ----------------------------------------------------------------- confidence semantics
def test_implicit_version_pinned():
    # The confidence semantics below were verified for this version; re-verify on upgrade.
    import implicit
    assert implicit.__version__ == "0.7.3"


def test_implicit_confidence_semantics_match_closed_form():
    """implicit 0.7.3 uses a stored entry d as the confidence C (p = 1); our fold-in with
    C = d must reproduce recalculate_user exactly."""
    from implicit.cpu.als import AlternatingLeastSquares
    rng = np.random.default_rng(1)
    M = (rng.random((25, 15)) < 0.25) * rng.choice([2.0, 6.0], size=(25, 15))
    Cm = sp.csr_matrix(M.astype(np.float32))
    mod = AlternatingLeastSquares(factors=3, regularization=0.2, alpha=1.0, iterations=3, random_state=0,
                                  num_threads=1)
    mod.fit(Cm, show_progress=False)
    Y = mod.item_factors.astype(np.float64)
    for u in range(5):
        row = Cm[u]
        lib = mod.recalculate_user(u, row)
        c = row.data.astype(np.float64)
        Yi = Y[row.indices]
        ours = np.linalg.solve(Y.T @ Y + (Yi.T * (c - 1)) @ Yi + 0.2 * np.eye(3), Yi.T @ c)
        assert np.allclose(ours, lib, atol=1e-4)


def test_fold_in_matches_recalculate_user_on_fitted_model(rnd):
    from implicit.cpu.als import AlternatingLeastSquares
    ev, cat = rnd
    tr = train_view(ev)
    m = A.fit_als(tr, len(cat.video_ids), "video", "long_view", CFG)
    hist = history_view(ev, "test", META)
    users = np.unique(hist.user_id)
    X, has, _ = A.fold_in(m, hist, users)
    # rebuild the same confidence rows in the compact index and ask implicit
    lib = AlternatingLeastSquares(factors=CFG.factors, regularization=CFG.regularization, alpha=1.0)
    lib.item_factors = np.array(m.Y)   # writable copy; m.Y stays frozen
    conf = A.build_confidence(hist, "video", "long_view", 0.0, CFG.alpha)
    conf = conf[np.isin(conf.item, m.item_rows)]
    for u in users[has][:10]:
        cu = conf[conf.user_id == u]
        cols = np.searchsorted(m.item_rows, cu.item.to_numpy())
        row = sp.csr_matrix((cu.c.to_numpy(np.float32), (np.zeros(len(cols), int), cols)),
                            shape=(1, len(m.item_rows)))
        x_lib = lib.recalculate_user(0, row)
        assert np.allclose(X[np.searchsorted(users, u)], x_lib, atol=1e-4)


# ----------------------------------------------------------------- compact index mapping
def test_noncontiguous_rows_and_users_mapping(rnd):
    ev, cat = rnd
    tr = train_view(ev)
    m = A.fit_als(tr, len(cat.video_ids), "video", "long_view", CFG)
    conf = A.build_confidence(tr, "video", "long_view", 0.0, CFG.alpha)
    # scorable rows are exactly the catalog rows with train signal; gaps (v % 3 == 2) excluded
    assert m.item_rows.tolist() == sorted(conf.item.unique().tolist())
    assert not any(cat.video_ids[r] % 3 == 2 for r in m.item_rows)
    assert m.train_users.tolist() == sorted(conf.user_id.unique().tolist())
    # Refit implicit directly on a dense full matrix in a *different* order would not be
    # comparable; instead check that the train factors reproduce the training matrix rows:
    # user u's top train-factor score over scorable items is consistent with the compact
    # mapping, i.e. X_train[k] belongs to train_users[k]. Re-fold-in of that user from the
    # same confidence rows with frozen Y must be close to X_train[k] after many iterations.
    m2 = A.fit_als(tr, len(cat.video_ids), "video", "long_view",
                   A.ALSConfig(factors=4, regularization=0.1, alpha=5.0, iterations=60, seed=0))
    Xr, has, _ = A.fold_in(m2, tr, m2.train_users)
    assert has.all()
    assert np.allclose(Xr, m2.X_train, atol=2e-2)
    # scores land on the right catalog columns
    sc = A.ALSScorer({("video", "long_view"): m2}, "nofoldin", "x")
    ctx = EvalContext("test", 2000, history_view(ev, "test", META))
    s = sc.score(m2.train_users[:3], "video", "long_view", ctx)
    assert np.isneginf(s[:, ~m2.scorable]).all()
    assert np.allclose(s[:, m2.item_rows], m2.X_train[:3] @ m2.Y.T)


# ----------------------------------------------------------------- train-only + frozen
def test_fit_refuses_non_train(rnd):
    ev, cat = rnd
    with pytest.raises(ValueError):
        A.fit_als(ev, len(cat.video_ids), "video", "long_view", CFG)


def test_val_test_events_do_not_change_fit(rnd):
    ev, cat = rnd
    m1 = A.fit_als(train_view(ev), len(cat.video_ids), "video", "long_view", CFG)
    ev2 = ev.copy()
    later = ev2.split > 0
    ev2.loc[later, "long_view"] = 1 - ev2.loc[later, "long_view"]   # flip every val/test label
    ev2 = pd.concat([ev2, ev2[later].assign(time_ms=lambda d: d.time_ms + 1)], ignore_index=True)
    m2 = A.fit_als(train_view(ev2), len(cat.video_ids), "video", "long_view", CFG)
    assert np.array_equal(m1.Y, m2.Y) and np.array_equal(m1.X_train, m2.X_train)
    assert m1.fingerprint == m2.fingerprint
    with pytest.raises(ValueError):
        m1.Y[0, 0] = 1.0   # frozen
    # scoring with test history leaves Y untouched
    y_before = m1.Y.tobytes()
    sc = A.ALSScorer({("video", "long_view"): m1}, "foldin", "als")
    sc.score(np.unique(ev.user_id)[:5], "video", "long_view",
             EvalContext("test", 2000, history_view(ev, "test", META)))
    assert m1.Y.tobytes() == y_before


def test_fingerprint_includes_labels(rnd):
    ev, cat = rnd
    tr = train_view(ev)
    f1 = A.train_fingerprint(tr, "video", "long_view")
    assert f1 == A.train_fingerprint(tr.sample(frac=1.0, random_state=3), "video", "long_view")  # order-free
    tr2 = tr.copy()
    tr2.iloc[0, tr2.columns.get_loc("long_view")] ^= 1
    assert A.train_fingerprint(tr2, "video", "long_view") != f1


def test_fit_deterministic(rnd):
    ev, cat = rnd
    a = A.fit_als(train_view(ev), len(cat.video_ids), "video", "long_view", CFG)
    b = A.fit_als(train_view(ev), len(cat.video_ids), "video", "long_view", CFG)
    assert np.array_equal(a.Y, b.Y) and np.array_equal(a.X_train, b.X_train)


# ----------------------------------------------------------------- nofoldin vs refold
def test_nofoldin_uses_raw_train_factors_not_refold(rnd):
    ev, cat = rnd
    m = A.fit_als(train_view(ev), len(cat.video_ids), "video", "long_view",
                  A.ALSConfig(factors=4, regularization=0.1, alpha=5.0, iterations=1, seed=0))
    ctx = EvalContext("test", 2000, history_view(ev, "test", META))
    users = m.train_users[:8]
    nof = A.ALSScorer({("video", "long_view"): m}, "nofoldin", "nof").score(users, "video", "long_view", ctx)
    assert np.allclose(nof[:, m.item_rows], m.X_train[:8] @ m.Y.T)
    Xr, _, _ = A.fold_in(m, train_view(ev), users)   # als_refold_train diagnostic
    assert not np.allclose(Xr, m.X_train[:8], atol=1e-3)   # with 1 iteration they differ


# ----------------------------------------------------------------- aux signal + scorable
def test_aux_scorable_and_shared_confidence():
    # train: u1 like on v0; u2 long_view (no like) on v1; v2 only seen without signal
    vm = pd.DataFrame({"video_id": [0, 1, 2], "author_id": [0, 0, 0], "music_id": [5, 6, 7]})
    cat = catm.build_catalogs(vm)
    ev = pd.DataFrame([(1, 0, 10, 1, 1), (2, 1, 20, 1, 0), (3, 2, 30, 0, 0), (2, 0, 40, 0, 0)],
                      columns=["user_id", "video_id", "time_ms", "long_view", "is_like"])
    ev["split"] = 0
    ev = attach_ids(ev, vm, cat)
    pure = A.fit_als(ev, 3, "video", "is_like", A.ALSConfig(factors=2, alpha=4, aux_w=0.0, iterations=3))
    aux = A.fit_als(ev, 3, "video", "is_like", A.ALSConfig(factors=2, alpha=4, aux_w=0.3, iterations=3))
    assert pure.item_rows.tolist() == [0]
    assert aux.item_rows.tolist() == [0, 1]   # long_view-only item becomes scorable with aux
    c = A.build_confidence(ev, "video", "is_like", 0.3, 4.0)
    # (u1,v0): r = 1 + 0.3 = 1.3, C = 1 + 4*1.3 = 6.2 ; (u2,v1): r = 0.3, C = 2.2
    assert c.set_index(["user_id", "item"]).c.to_dict() == pytest.approx({(1, 0): 6.2, (2, 1): 2.2})
    with pytest.raises(ValueError):
        A.build_confidence(ev, "video", "long_view", 0.3, 4.0)


def test_music_confidence_any_over_shared_videos():
    vm = pd.DataFrame({"video_id": [0, 1, 2], "author_id": [0, 0, 0], "music_id": [5, 5, 0]})
    cat = catm.build_catalogs(vm)
    ev = pd.DataFrame([(1, 0, 10, 1, 0), (1, 1, 20, 1, 0), (1, 2, 30, 1, 0)],
                      columns=["user_id", "video_id", "time_ms", "long_view", "is_like"])
    ev["split"] = 0
    ev = attach_ids(ev, vm, cat)
    c = A.build_confidence(ev, "music", "long_view", 0.0, 10.0)
    assert len(c) == 1 and c.item.tolist() == [0] and c.c.tolist() == [11.0]   # binary, sentinel dropped


# ----------------------------------------------------------------- evaluator integration
@pytest.fixture(scope="module")
def fx():
    return make_events()


def test_pop_hand_computed(fx):
    ev, lc, cat, h = fx
    tr = train_view(ev)
    # train long_view: only u5 on v10 -> video [1,0,0,0,0]; music m100 = v10 + v11 = 1
    assert popularity_counts(tr, cat, "video", "long_view").tolist() == [1, 0, 0, 0, 0]
    assert popularity_counts(tr, cat, "music", "long_view").tolist() == [1, 0, 0]
    with pytest.raises(ValueError):
        popularity_counts(ev, cat, "video", "long_view")


def test_pop_end_to_end_metrics(fx):
    ev, lc, cat, h = fx
    f = F.build_frames(ev, lc, cat, META, h, "test", "video", "long_view")["masked"]
    ctx = EvalContext("test", 2000, history_view(ev, "test", META))
    pu = evaluate(PopularityScorer(train_view(ev), cat, ACTIONS), f, ctx, ks=(1, 2, 3))
    # scores [1,0,0,0,0], ties by row. u1 GT {0,1,2}, mask {3}: ranking 0,1,2 -> all hit at K=3
    # u4 GT {1}, mask {0}: ranking 1,2,3 -> hit at rank 1
    r = pu.set_index("user_id")
    assert r.loc[1, "recall@1"] == pytest.approx(1 / 3) and r.loc[1, "recall@3"] == 1.0
    assert r.loc[4, "ndcg@1"] == 1.0
    assert (pu.frame_id == f.frame_id).all()


def test_unscorable_gt_is_item_level_and_kept_in_denominator(fx):
    ev, lc, cat, h = fx
    f = F.build_frames(ev, lc, cat, META, h, "test", "video", "long_view")["masked"]
    m = A.fit_als(train_view(ev), 5, "video", "long_view", A.ALSConfig(factors=2, iterations=2))
    assert m.item_rows.tolist() == [0]   # only v10 has train long_view signal
    pu = evaluate(A.ALSScorer({("video", "long_view"): m}, "foldin", "als"), f,
                  EvalContext("test", 2000, history_view(ev, "test", META)), ks=(1, 2, 3)).set_index("user_id")
    # u1 GT {0,1,2}: rows 1,2 unscorable; u4 GT {1}: unscorable. Denominator unchanged.
    assert pu.loc[1, "n_gt_unscorable"] == 2 and pu.loc[4, "n_gt_unscorable"] == 1
    assert pu.loc[1, "n_gt"] == 3
    assert not pu.has_signal.any()   # neither user has a scorable long_view before T_test


class _BadScorer:
    name = "bad"

    def score(self, users, target, action, ctx):
        return np.zeros((len(users), 2), np.float32)

    def signal(self, users, target, action, ctx):
        return np.ones(len(users), bool)

    def item_scorable(self, target, action):
        return np.ones(5, bool)


def test_evaluator_guards(fx):
    ev, lc, cat, h = fx
    f = F.build_frames(ev, lc, cat, META, h, "test", "video", "long_view")["masked"]
    with pytest.raises(ValueError):
        evaluate(_BadScorer(), f, EvalContext("test", 2000, history_view(ev, "test", META)))
    with pytest.raises(ValueError):   # val history for a test frame
        evaluate(PopularityScorer(train_view(ev), cat, ACTIONS), f,
                 EvalContext("val", 1000, history_view(ev, "val", META)))
    with pytest.raises(ValueError):   # history containing GT-period events
        evaluate(PopularityScorer(train_view(ev), cat, ACTIONS), f, EvalContext("test", 2000, ev))


def test_no_signal_users_zero_and_kept(rnd):
    ev, cat = rnd
    h = HistoryIndex(ev)
    lc = pd.DataFrame(columns=["user_id", "video_id", "time_ms", "long_view_any", "is_like_any", "split"])
    f = F.build_frames(ev, lc, cat, META, h, "test", "video", "is_like")["masked"]
    m = A.fit_als(train_view(ev), len(cat.video_ids), "video", "is_like", CFG)
    ctx = EvalContext("test", 2000, history_view(ev, "test", META))
    for mode in ("foldin", "nofoldin"):
        sc = A.ALSScorer({("video", "is_like"): m}, mode, mode)
        pu = evaluate(sc, f, ctx)
        assert len(pu) == len(f.users)            # nobody leaves the population
        ns = ~pu.has_signal
        assert ns.any()                           # the fixture has like-less users
        assert (pu.loc[ns, ["recall@50", "ndcg@50"]].to_numpy() == 0).all()
        s = sc.score(f.users[ns.to_numpy()], "video", "is_like", ctx)
        assert np.isneginf(s).all()
    # train-OOV evaluated users have no train factors -> no signal in nofoldin
    oov = f.users[f.is_train_oov_user]
    if len(oov):
        assert not A.ALSScorer({("video", "is_like"): m}, "nofoldin", "n").signal(oov, "video", "is_like", ctx).any()


def test_scores_invariant_to_future_events(rnd):
    """Deleting / flipping every event at t >= T_test leaves Pop/ALS test scores unchanged."""
    ev, cat = rnd
    users = np.unique(ev.user_id)
    def scores(e):
        m = A.fit_als(train_view(e), len(cat.video_ids), "video", "long_view", CFG)
        ctx = EvalContext("test", 2000, history_view(e, "test", META))
        a = A.ALSScorer({("video", "long_view"): m}, "foldin", "a").score(users, "video", "long_view", ctx)
        p = PopularityScorer(train_view(e), cat, ACTIONS).score(users, "video", "long_view", ctx)
        return a, p
    a1, p1 = scores(ev)
    ev2 = ev[ev.split < 2].copy()
    a2, p2 = scores(ev2)
    ev3 = ev.copy()
    ev3.loc[ev3.split == 2, "long_view"] ^= 1
    a3, p3 = scores(ev3)
    assert np.array_equal(a1, a2) and np.array_equal(a1, a3)
    assert np.array_equal(p1, p2) and np.array_equal(p1, p3)


def test_val_foldin_uses_train_only(rnd):
    ev, cat = rnd
    m = A.fit_als(train_view(ev), len(cat.video_ids), "video", "long_view", CFG)
    hv = history_view(ev, "val", META)
    assert hv.split.max() == 0
    # a user whose only long_view positives are in val: no val signal, has test signal
    tr_pos = set(ev[(ev.split == 0) & (ev.long_view == 1)].user_id)
    val_pos = set(ev[(ev.split == 1) & (ev.long_view == 1)].user_id)
    only_val = sorted(val_pos - tr_pos)
    new = pd.DataFrame([(999999, int(cat.video_ids[m.item_rows[0]]), 1500, 1, 0)],
                       columns=["user_id", "video_id", "time_ms", "long_view", "is_like"])
    vm = pd.DataFrame({"video_id": cat.video_ids, "author_id": 0,
                       "music_id": np.where(cat.video_music_row >= 0, cat.music_ids[np.maximum(cat.video_music_row, 0)], 0)})
    new["split"] = split_of(new.time_ms)
    e2 = pd.concat([ev, attach_ids(new, vm, cat)], ignore_index=True)
    sc = A.ALSScorer({("video", "long_view"): m}, "foldin", "als")
    u = np.array([999999] + only_val[:1], np.int64)
    u.sort()
    assert not sc.signal(u, "video", "long_view", EvalContext("val", 1000, history_view(e2, "val", META))).any()
    assert sc.signal(u, "video", "long_view", EvalContext("test", 2000, history_view(e2, "test", META))).all()


def test_user_order_permutation_invariance(rnd):
    ev, cat = rnd
    m = A.fit_als(train_view(ev), len(cat.video_ids), "video", "long_view", CFG)
    ctx = EvalContext("test", 2000, history_view(ev, "test", META))
    sc = A.ALSScorer({("video", "long_view"): m}, "foldin", "als")
    users = np.unique(ev.user_id)
    perm = np.random.default_rng(0).permutation(len(users))
    s1 = sc.score(users, "video", "long_view", ctx)
    s2 = sc.score(users[perm], "video", "long_view", ctx)
    assert np.array_equal(s1[perm], s2)
