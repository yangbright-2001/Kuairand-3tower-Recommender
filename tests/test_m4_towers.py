"""M4-1 tower tests (PLAN.md §3d D1-1, D1-4, D1-6, D1-9: OOV, gradients, stability, U0/U1).

The losses here are throwaway test objectives (plain CE over the catalog), not the D1-3
training loss, which is M4-2.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as Fn

import m4_fixtures as fx
from kuairand_rec.data import catalog as catm
from kuairand_rec.models import features as F
from kuairand_rec.models import towers as T

HCFG = F.HistoryConfig()


@pytest.fixture(scope="module")
def fit():
    return fx.make()


@pytest.fixture(scope="module")
def feats(fit):
    return F.VideoFeatures.from_catalog(fit["video_features"]), F.MusicFeatures.from_catalog(fit["music_features"])


def _model(fit, seed=0, user_vocab=None, **cfg):
    torch.manual_seed(seed)
    sizes = F.FeatureSizes.from_vocabs(fit["vocabs"], user_vocab)
    return T.ThreeTowerModel(sizes, T.TowerConfig(**cfg), HCFG)


def _users(fit, split_name, users):
    hidx, t = F.frozen_history_index(fit["events"], split_name, fx.META)
    hb = F.build_history_batch(hidx, users, t, HCFG)
    uidx = torch.as_tensor(fit["vocabs"].user.encode(np.asarray(users)))
    return uidx, hb.to_torch()


def _forward(model, feats, uidx, hist, user_feats=None):
    vf, mf = feats
    enc = model.encode_items(vf, mf)
    items = model.item_embeddings(enc)
    users = model.user_embeddings(model.encode_users(uidx, hist, enc, user_feats))
    return users, items


def _all_scores(model, feats, uidx, hist, user_feats=None):
    with torch.no_grad():
        u, i = _forward(model, feats, uidx, hist, user_feats)
        return {t: T.ThreeTowerModel.scores(u, i, t) for t in T.TASKS}


def _task_loss(u, i, task, label=0):
    logits = T.ThreeTowerModel.scores(u, i, task) / 0.05
    return Fn.cross_entropy(logits, torch.full((logits.shape[0],), label, dtype=torch.long))


USERS = [1, 2, 3, 4, 999]          # 3: train-OOV, empty history; 999: unknown, empty history


# ============================================================================ shapes / heads
def test_shapes_heads_and_normalisation(fit, feats):
    m = _model(fit).eval()
    uidx, hist = _users(fit, "test", USERS)
    u, i = _forward(m, feats, uidx, hist)
    assert set(u) == set(T.TASKS) and set(i) == set(T.TASKS)
    assert len(m.user_heads.proj) == 4 and len(m.video_heads.proj) == 2 and len(m.music_heads.proj) == 2
    for t in T.TASKS:
        assert u[t].shape == (5, 64)
        n_items = 6 if t.startswith("video") else 4
        assert i[t].shape == (n_items, 64)
        assert torch.allclose(u[t].norm(dim=-1), torch.ones(5), atol=1e-5)
        assert torch.allclose(i[t].norm(dim=-1), torch.ones(n_items), atol=1e-5)
    enc = m.encode_items(*feats)
    assert enc.v_base.shape == (6, 128) and enc.m_base.shape == (4, 128)


def test_no_batchnorm_and_layernorm_bases(fit):
    m = _model(fit)
    assert not any(isinstance(x, torch.nn.modules.batchnorm._BatchNorm) for x in m.modules())
    for tower in (m.music_tower, m.video_tower, m.user_tower):
        assert isinstance(tower.mlp[-1], torch.nn.LayerNorm)


def test_eval_is_deterministic_and_batch_independent(fit, feats):
    m = _model(fit).eval()
    uidx, hist = _users(fit, "test", USERS)
    s1, s2 = _all_scores(m, feats, uidx, hist), _all_scores(m, feats, uidx, hist)
    for t in T.TASKS:
        assert torch.equal(s1[t], s2[t])
    # A user's embedding does not depend on the other users in the batch.
    u_one, h_one = _users(fit, "test", [1])
    s_one = _all_scores(m, feats, u_one, h_one)
    for t in T.TASKS:
        assert torch.allclose(s_one[t][0], s1[t][0], atol=1e-6)


# ============================================================================ shared music tower
def test_single_music_tower_no_duplicated_parameters(fit):
    m = _model(fit)
    towers = [x for x in m.modules() if isinstance(x, T.MusicTower)]
    assert towers == [m.music_tower]                       # one module; video tower holds no copy
    ids = {id(p) for p in m.parameters()}
    assert len(ids) == len(list(m.parameters()))           # no duplicated parameter copies


def test_video_loss_reaches_music_tower_unless_use_music_false(fit, feats):
    uidx, hist = _users(fit, "val", [1, 2, 4])
    for use_music, expect in ((True, True), (False, False)):
        m = _model(fit, use_music=use_music, dropout=0.0, id_dropout=0.0).train()
        u, i = _forward(m, feats, uidx, hist)
        (_task_loss(u, i, "video_long_view") + _task_loss(u, i, "video_is_like")).backward()
        g = [p.grad for p in m.music_tower.parameters()]
        nonzero = any(x is not None and x.abs().sum() > 0 for x in g)
        assert nonzero == expect
        if expect:
            assert all(torch.isfinite(x).all() for x in g if x is not None)


def test_each_task_trains_only_its_own_heads(fit, feats):
    uidx, hist = _users(fit, "val", [1, 2, 4])
    for task in T.TASKS:
        m = _model(fit, dropout=0.0, id_dropout=0.0).train()
        u, i = _forward(m, feats, uidx, hist)
        _task_loss(u, i, task, label=1).backward()
        item_heads = m.video_heads if task.startswith("video") else m.music_heads
        other_item = m.music_heads if task.startswith("video") else m.video_heads
        for heads, own in ((m.user_heads, True), (item_heads, True), (other_item, False)):
            for t, lin in heads.proj.items():
                grads = [lin.weight.grad, lin.bias.grad]
                if own and t == task:
                    assert all(gr is not None and torch.isfinite(gr).all() and gr.abs().sum() > 0 for gr in grads)
                else:
                    assert all(gr is None or gr.abs().sum() == 0 for gr in grads), (task, t)


# ============================================================================ OOV / fixed-zero rows
def test_pad_oov_rows_stay_zero_after_adamw_steps(fit, feats):
    m = _model(fit, dropout=0.0).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.1)
    uidx, hist = _users(fit, "test", USERS)
    for _ in range(5):
        opt.zero_grad()
        u, i = _forward(m, feats, uidx, hist)
        sum(_task_loss(u, i, t) for t in T.TASKS).backward()
        for _, emb in m.fixed_zero_embeddings():
            assert (emb.weight.grad[list(emb.zero_rows)] == 0).all()
        opt.step()
    embs = m.fixed_zero_embeddings()
    assert len(embs) >= 8
    for name, emb in embs:
        assert (emb.weight[list(emb.zero_rows)] == 0).all(), name
    # NO_MUSIC is learnable and was trained (v12 has music_id 0 and is in train).
    assert m.music_tower.music_emb.zero_rows == (catm.PAD, catm.OOV)
    assert m.music_tower.music_emb.weight[catm.NO_MUSIC].abs().sum() > 0


def test_perturbing_pad_oov_rows_leaves_scores_unchanged(fit, feats):
    m = _model(fit).eval()
    uidx, hist = _users(fit, "test", USERS)
    before = _all_scores(m, feats, uidx, hist)
    with torch.no_grad():
        for _, emb in m.fixed_zero_embeddings():
            emb.weight[list(emb.zero_rows)] = torch.randn(len(emb.zero_rows), emb.weight.shape[1]) * 100
    after = _all_scores(m, feats, uidx, hist)
    for t in T.TASKS:
        assert torch.equal(before[t], after[t])


def test_oov_items_and_users_are_finite_content_path(fit, feats):
    m = _model(fit).eval()
    vf, _ = feats
    v15 = int(fit["catalogs"].video_row([15])[0])
    assert vf.video_idx[v15] == catm.OOV and vf.author_idx[v15] == catm.OOV
    uidx, hist = _users(fit, "test", USERS)
    assert uidx[2] == catm.OOV and uidx[4] == catm.OOV and hist.length[2] == 0 and hist.length[4] == 0
    s = _all_scores(m, feats, uidx, hist)
    for t in T.TASKS:
        assert torch.isfinite(s[t]).all()
    # Train-OOV user with empty history and an unknown user take the identical path (equal up
    # to BLAS row-position rounding, which is not bitwise across batch rows).
    for t in T.TASKS:
        assert torch.allclose(s[t][2], s[t][4], atol=1e-6, rtol=0)
    # Users 1's T_test history contains the train-OOV video v15: represented by content.
    assert v15 in hist.video_row[0].tolist()


def test_id_dropout_shared_across_candidates_history_and_music(fit, feats):
    # With p = 1 every ID slot is dropped: train-mode item encodings must equal the eval-mode
    # encodings of features whose video / author / music ids are all OOV (the zero-row path).
    m = _model(fit, dropout=0.0, id_dropout=1.0)
    vf, mf = feats
    m.train()
    enc_tr = m.encode_items(vf, mf)
    oov = lambda x: torch.full_like(x, catm.OOV)
    # Non-zero music ids -> OOV; music_id-0 videos keep NO_MUSIC (dropout drops that vocab
    # entry too, which the eval reference reproduces by zeroing the NO_MUSIC row).
    has_m = vf.music_row >= 0
    vf_oov = dataclasses.replace(vf, video_idx=oov(vf.video_idx), author_idx=oov(vf.author_idx),
                                 music_idx=torch.where(has_m, catm.OOV, vf.music_idx))
    mf_oov = dataclasses.replace(mf, music_idx=oov(mf.music_idx))
    m.eval()
    with torch.no_grad():
        m.music_tower.music_emb.weight[catm.NO_MUSIC] = 0.0
    enc_ev = m.encode_items(vf_oov, mf_oov)
    assert torch.allclose(enc_tr.v_base, enc_ev.v_base, atol=1e-6)
    assert torch.allclose(enc_tr.m_base, enc_ev.m_base, atol=1e-6)
    # Eval mode applies no ID dropout at all.
    assert m._item_keep(torch.device("cpu")) == {"video": None, "author": None, "music": None}


# ============================================================================ shared music output
MIN_ = fx.MIN
# Extended catalog: two more music_id-0 videos, one with the same music_type as v12 (NaN) and
# one with a different music_type (9.0), so NO_MUSIC is exercised with >1 type.
EXT_META = pd.concat([fx.VIDEO_META, pd.DataFrame({
    "video_id": [16, 17], "author_id": [1, 2], "music_id": [0, 0], "video_type": ["NORMAL"] * 2,
    "upload_dt": ["2022-04-10"] * 2, "upload_type": ["Kmovie"] * 2, "visible_status": [0.0] * 2,
    "video_duration": [20_000.0, 25_000.0], "server_width": [720.0] * 2, "server_height": [1280.0] * 2,
    "music_type": [9.0, np.nan], "tag": ["1", "2"]})], ignore_index=True)


@pytest.fixture(scope="module")
def ext():
    rows = fx.raw_rows() + [fx._row(2, 16, 95 * MIN_, lv=1, dur=20_000.0),
                            fx._row(4, 17, 30 * MIN_, lv=1, dur=25_000.0)]
    f = fx.make(rows, EXT_META)
    vc, mc = f["video_features"], f["music_features"]
    F.check_shared_music_inputs(vc, mc)
    return f, (F.VideoFeatures.from_catalog(vc), F.MusicFeatures.from_catalog(mc))


def _vrow(f, vid):
    return int(f["catalogs"].video_row([vid])[0])


def _mrow(f, mid):
    return int(f["catalogs"].music_row([mid])[0])


def _capture(model):
    """Record every MusicTower output and the music input actually consumed by the video tower."""
    rec = {"music_out": [], "video_music_in": []}
    h1 = model.music_tower.register_forward_hook(lambda mod, a, out: rec["music_out"].append(out))
    h2 = model.video_tower.register_forward_pre_hook(lambda mod, a: rec["video_music_in"].append(a[2]))
    return rec, (h1, h2)


def test_train_mode_video_music_input_is_the_music_candidate_base(ext):
    f, (vf, mf) = ext
    m = _model(f, dropout=0.5, id_dropout=0.3).train()
    rec, hooks = _capture(m)
    for seed in range(3):
        torch.manual_seed(seed)
        enc = m.encode_items(vf, mf)
        assert len(rec["music_out"]) == seed + 1                       # exactly one call per forward
        consumed = rec["video_music_in"][-1]
        assert consumed is enc.video_music_base
        has = vf.music_row >= 0
        # Bitwise: same tensor values as the music candidates' m_base (same dropout sample).
        assert torch.equal(consumed[has], enc.m_base[vf.music_row[has]])
        assert torch.equal(enc.m_base, rec["music_out"][-1][: len(mf)])
    for h in hooks:
        h.remove()
    # Dropout is genuinely active: two train forwards differ.
    torch.manual_seed(10)
    a = m.encode_items(vf, mf).m_base
    torch.manual_seed(11)
    assert not torch.allclose(a, m.encode_items(vf, mf).m_base)


def test_videos_sharing_music_use_one_sample(ext):
    f, (vf, mf) = ext
    m = _model(f, dropout=0.5).train()
    enc = m.encode_items(vf, mf)
    r10, r11, r100 = _vrow(f, 10), _vrow(f, 11), _mrow(f, 100)
    assert int(vf.music_row[r10]) == int(vf.music_row[r11]) == r100
    assert torch.equal(enc.video_music_base[r10], enc.video_music_base[r11])
    assert torch.equal(enc.video_music_base[r10], enc.m_base[r100])


def test_video_loss_gradient_reaches_music_ids_through_the_gather(ext):
    f, (vf, mf) = ext
    m = _model(f, dropout=0.3, id_dropout=0.0).train()
    enc = m.encode_items(vf, mf)
    # Video-only objective: the music heads / music candidates are not used.
    m.video_heads(enc.v_base)["video_long_view"].pow(2).sum().backward()
    g = m.music_tower.music_emb.weight.grad
    m100 = int(mf.music_idx[_mrow(f, 100)])
    assert torch.isfinite(g).all()
    assert g[m100].abs().sum() > 0                                # shared non-zero music
    assert g[catm.NO_MUSIC].abs().sum() > 0                       # music_id-0 videos
    assert (g[[catm.PAD, catm.OOV]] == 0).all()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m.music_heads.parameters())


def test_use_music_false_ablation(ext):
    f, (vf, mf) = ext
    m = _model(f, use_music=False, dropout=0.3).train()
    enc = m.encode_items(vf, mf)
    assert enc.video_music_base is None and enc.m_base.shape == (len(mf), 128)
    assert m.video_tower.mlp[0].in_features == _model(f).video_tower.mlp[0].in_features - 128
    with pytest.raises(ValueError):
        m.video_tower(vf, {}, enc.m_base)
    m.video_heads(enc.v_base)["video_long_view"].pow(2).sum().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m.music_tower.parameters())
    # Music retrieval still trains the music tower.
    m.zero_grad()
    m.music_heads(m.encode_items(vf, mf).m_base)["music_long_view"].pow(2).sum().backward()
    assert m.music_tower.music_emb.weight.grad.abs().sum() > 0
    # And with use_music=True the video tower refuses a missing music input.
    with pytest.raises(ValueError):
        _model(f).video_tower(vf, {}, None)


def test_oov_music_and_no_music_paths(ext):
    f, (vf, mf) = ext
    m = _model(f).eval()
    with torch.no_grad():
        enc = m.encode_items(vf, mf)
        mt = m.music_tower
        # Independent recomputations below are compared with atol 1e-5: a separate (smaller)
        # matmul can differ by ~1 ulp (~1e-6 at |x| ~ 2); same-forward comparisons stay bitwise.
        # Eval: identical to calling the music tower on each video's own inputs (old semantics).
        direct = mt(vf.music_idx, vf.music_type_idx)
        assert torch.allclose(enc.video_music_base, direct, atol=1e-5, rtol=0)

        # OOV music: v15 -> m103 is val-only, so its id is OOV but it keeps its catalog row.
        r15, r103 = _vrow(f, 15), _mrow(f, 103)
        assert int(vf.music_idx[r15]) == catm.OOV and int(vf.music_row[r15]) == r103
        assert torch.equal(enc.video_music_base[r15], enc.m_base[r103])
        zero_id = mt.mlp(torch.cat([torch.zeros(1, m.cfg.id_dim), mt.type_emb(mf.music_type_idx[r103:r103 + 1])], -1))
        assert torch.allclose(enc.m_base[r103], zero_id[0], atol=1e-5, rtol=0)

        # NO_MUSIC: v12 and v17 (music_type NaN) share one sample; v16 (music_type 9.0) differs.
        r12, r16, r17 = _vrow(f, 12), _vrow(f, 16), _vrow(f, 17)
        assert (vf.music_row[[r12, r16, r17]] == -1).all() and (vf.music_idx[[r12, r16, r17]] == catm.NO_MUSIC).all()
        assert int(vf.music_type_idx[r12]) == int(vf.music_type_idx[r17]) != int(vf.music_type_idx[r16])
        assert torch.equal(enc.video_music_base[r12], enc.video_music_base[r17])
        assert not torch.allclose(enc.video_music_base[r12], enc.video_music_base[r16])
        for r in (r12, r16):
            want = mt(torch.tensor([catm.NO_MUSIC]), vf.music_type_idx[r:r + 1])[0]
            assert torch.allclose(enc.video_music_base[r], want, atol=1e-5, rtol=0)
            # Not the OOV-music path with the same music_type.
            oov = mt(torch.tensor([catm.OOV]), vf.music_type_idx[r:r + 1])[0]
            assert not torch.allclose(enc.video_music_base[r], oov)
        # NO_MUSIC is no music candidate: m_base has exactly the catalog rows, and changing the
        # NO_MUSIC embedding moves music_id-0 videos only.
        assert enc.m_base.shape[0] == len(mf) == len(f["catalogs"].music_ids)
        mt.music_emb.weight[catm.NO_MUSIC] += 1.0
        enc2 = m.encode_items(vf, mf)
    assert torch.equal(enc2.m_base, enc.m_base)
    moved = (enc2.v_base != enc.v_base).any(1)
    assert moved.tolist() == (vf.music_row < 0).tolist()


def test_music_gather_guard_rejects_inconsistent_inputs(ext):
    f, (vf, mf) = ext
    m = _model(f).eval()
    r10 = _vrow(f, 10)
    bad = vf.music_type_idx.clone()
    bad[r10] = bad[r10] + 1
    with pytest.raises(ValueError):
        m.encode_items(dataclasses.replace(vf, music_type_idx=bad), mf)
    bad_idx = vf.music_idx.clone()
    bad_idx[_vrow(f, 12)] = catm.OOV                                # music_id 0 must stay NO_MUSIC
    with pytest.raises(ValueError):
        m.encode_items(dataclasses.replace(vf, music_idx=bad_idx), mf)


# ============================================================================ history encoder
def test_empty_history_is_nan_free_and_zero(fit, feats):
    m = _model(fit).train()
    uidx, hist = _users(fit, "test", [3, 999])
    enc = m.encode_items(*feats)
    pooled = m.user_tower.history(hist, enc.v_base)
    assert torch.isfinite(pooled).all() and (pooled == 0).all()
    u = m.user_embeddings(m.encode_users(uidx, hist, enc))
    loss = sum(_task_loss(u, m.item_embeddings(enc), t) for t in T.TASKS)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_history_changes_user_embedding_and_pad_content_does_not(fit, feats):
    m = _model(fit).eval()
    uidx, hist = _users(fit, "val", [1])
    enc = m.encode_items(*feats)
    with torch.no_grad():
        full = m.encode_users(uidx, hist, enc)
        empty = dataclasses.replace(
            hist, video_row=torch.full_like(hist.video_row, -1), feedback=torch.zeros_like(hist.feedback),
            hidden=torch.zeros_like(hist.hidden), play_bucket=torch.zeros_like(hist.play_bucket),
            recency_bucket=torch.zeros_like(hist.recency_bucket), valid=torch.zeros_like(hist.valid),
            length=torch.zeros_like(hist.length))
        assert not torch.allclose(full, m.encode_users(uidx, empty, enc))
        # PAD positions gather v_base row 0 and must zero it: u4's only token is v13 (row 3),
        # so garbage in row 0 must not change u4's embedding.
        u4, h4 = _users(fit, "val", [4])
        assert h4.video_row[h4.valid].tolist() == [3] and (h4.video_row[~h4.valid] == -1).all()
        v2 = enc.v_base.clone()
        v2[0] += 1000.0
        assert torch.equal(m.encode_users(u4, h4, enc), m.encode_users(u4, h4, T.ItemEncodings(v2, enc.m_base)))


def test_history_tokens_use_shared_video_tower_gradient(fit, feats):
    # User-side loss alone must update the video tower through the history tokens (Q3a).
    m = _model(fit, dropout=0.0, id_dropout=0.0).train()
    uidx, hist = _users(fit, "val", [1, 2])
    enc = m.encode_items(*feats)
    u = m.user_embeddings(m.encode_users(uidx, hist, enc))
    u["video_long_view"].sum().backward()
    assert m.video_tower.video_emb.weight.grad.abs().sum() > 0
    m2 = _model(fit, dropout=0.0, id_dropout=0.0, use_history=False).train()
    enc2 = m2.encode_items(*feats)
    m2.user_embeddings(m2.encode_users(uidx, None, enc2))["video_long_view"].sum().backward()
    assert m2.video_tower.video_emb.weight.grad is None or m2.video_tower.video_emb.weight.grad.abs().sum() == 0


def test_history_batch_validation(fit, feats):
    m = _model(fit).eval()
    uidx, hist = _users(fit, "val", [1])
    enc = m.encode_items(*feats)
    bad = dataclasses.replace(hist, length=hist.length + 1)
    with pytest.raises(ValueError):
        m.encode_users(uidx, bad, enc)
    with pytest.raises(ValueError):
        m.encode_users(uidx, None, enc)                     # use_history=True needs a history
    short = F.build_history_batch(F.frozen_history_index(fit["events"], "val", fx.META)[0], [1],
                                  fx.META["t_val_ms"], F.HistoryConfig(max_len=10)).to_torch()
    with pytest.raises(ValueError):
        m.encode_users(uidx, short, enc)


def test_nan_input_is_not_silently_masked(fit, feats):
    # NaN reaching the towers must propagate (the trainer's fail-fast check relies on it).
    m = _model(fit).eval()
    vf, mf = feats
    num = vf.num.clone()
    num[0, 0] = float("nan")
    uidx, hist = _users(fit, "val", [1])
    s = _all_scores(m, (dataclasses.replace(vf, num=num), mf), uidx, hist)
    assert not torch.isfinite(s["video_long_view"]).all()


# ============================================================================ U0 / U1
def _u1(fit):
    train = fit["events"][fit["events"].split == 0]
    voc = F.fit_user_feature_vocab(fx.USER_FEATURES, train, min_count=1)
    return voc, F.build_user_feature_table(fx.USER_FEATURES, voc)


def test_u0_u1_inputs(fit, feats):
    uidx, hist = _users(fit, "test", USERS)
    m0 = _model(fit).eval()
    voc, tab = _u1(fit)
    m1 = _model(fit, user_vocab=voc).eval()
    uf = torch.as_tensor(tab.lookup(USERS))
    assert not m0.user_tower.is_u1 and m1.user_tower.is_u1
    assert len(m1.user_tower.field_embs) == len(F.U1_FIELDS) == 21
    with pytest.raises(ValueError):
        _forward(m0, feats, uidx, hist, uf)                  # U0 refuses user features
    with pytest.raises(ValueError):
        _forward(m1, feats, uidx, hist, None)                # U1 requires them
    with pytest.raises(ValueError):
        _forward(m1, feats, uidx, hist, uf[:, :5])
    s = _all_scores(m1, feats, uidx, hist, uf)
    assert all(torch.isfinite(s[t]).all() for t in T.TASKS)
    # U1 features change the user embedding: u1 (in vocab) vs the same user with all-OOV fields.
    s_oov = _all_scores(m1, feats, uidx, hist, torch.full_like(uf, catm.OOV))
    assert not torch.allclose(s[T.TASKS[0]][0], s_oov[T.TASKS[0]][0])
    # Unknown user 999 is absent from the snapshot: all-OOV codes -> identical to s_oov row.
    assert torch.allclose(s[T.TASKS[0]][4], s_oov[T.TASKS[0]][4], atol=1e-6, rtol=0)
    # The item side is identical between U0 and U1 given the same seed (only the user tower differs).
    e0, e1 = m0.encode_items(*feats), m1.encode_items(*feats)
    assert torch.equal(e0.v_base, e1.v_base) and torch.equal(e0.m_base, e1.m_base)


def test_tower_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        T.TowerConfig.from_dict({"base_dimension": 128})
    assert T.TowerConfig.from_dict({"use_music": False}).use_music is False
