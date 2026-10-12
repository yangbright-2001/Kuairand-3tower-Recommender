"""M4-2 loss / training-data / trainer tests (PLAN.md §3d D1-3, D1-5, D1-9: F1-F6, gradients,
stability, oversampling, leakage, shared item forward).

Hand-built fixture (catalog of `m4_fixtures.VIDEO_META`: v10, v11 share music m100; v12 has
music 0; v13 -> m101, v14 -> m102; v15 -> m103 is val-only, so not a candidate).
Train (t < 100 min):
  uA=1: v10 lv (t10), v13 lv+like (t20), v14 lv (t30), v12 lv+like (t40, music 0),
        v10 lv again (t60: a repeated positive of the same pair)
  uB=2: v10 lv (t15), v11 like only (t25), v14 click only (t35, not a sample)
  uC=3: v11 lv (t12), v13 like only (t22), v10 lv (t70)
  uD=4: v10..v14 lv (t5..t9)  -> every video / music column is a uD lv positive
Val / test (never samples, never in Pos_train, never in rolling train histories):
  uA v11 lv+like (t120), uC v15 lv (t130), uB v13 lv (t210)

Expected Pos_train (written by hand, in item ids):
  video_lv:   uA {10,12,13,14}  uB {10}       uC {10,11}  uD {10,11,12,13,14}
  video_like: uA {12,13}        uB {11}       uC {13}     uD {}
  music_lv:   uA {100,101,102}  uB {100}      uC {100}    uD {100,101,102}   (v12: no music)
  music_like: uA {101}          uB {100}      uC {101}    uD {}
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

import m4_fixtures as fx
from kuairand_rec.data import catalog as catm
from kuairand_rec.models import data as D
from kuairand_rec.models import features as F
from kuairand_rec.models import loss as L
from kuairand_rec.models import train as TR
from kuairand_rec.models.towers import TASKS, ThreeTowerModel, TowerConfig

MIN = fx.MIN
R = fx._row
HCFG = F.HistoryConfig()
TAU = 0.05


def rows():
    return [
        R(1, 10, 10 * MIN, lv=1), R(1, 13, 20 * MIN, lv=1, like=1), R(1, 14, 30 * MIN, lv=1),
        R(1, 12, 40 * MIN, lv=1, like=1), R(1, 10, 60 * MIN, lv=1),
        R(2, 10, 15 * MIN, lv=1), R(2, 11, 25 * MIN, like=1), R(2, 14, 35 * MIN),
        R(3, 11, 12 * MIN, lv=1), R(3, 13, 22 * MIN, like=1), R(3, 10, 70 * MIN, lv=1),
        *[R(4, v, (5 + i) * MIN, lv=1) for i, v in enumerate([10, 11, 12, 13, 14])],
        R(1, 11, 120 * MIN, lv=1, like=1), R(3, 15, 130 * MIN, lv=1), R(2, 13, 210 * MIN, lv=1),
    ]


POS = {
    "video_long_view": {1: {10, 12, 13, 14}, 2: {10}, 3: {10, 11}, 4: {10, 11, 12, 13, 14}},
    "video_is_like": {1: {12, 13}, 2: {11}, 3: {13}, 4: set()},
    "music_long_view": {1: {100, 101, 102}, 2: {100}, 3: {100}, 4: {100, 101, 102}},
    "music_is_like": {1: {101}, 2: {100}, 3: {101}, 4: set()},
}


class World:
    """Condition + samples + Pos_train + a full batch, with id <-> column helpers."""

    def __init__(self, exclude=(), k=1):
        self.fit = fx.make(rows(), exclude_video_ids=exclude)
        self.cond = D.condition_from_fit("cold" if len(exclude) else "normal", self.fit, exclude)
        self.base = D.build_samples(self.cond)
        self.pos = D.PosTrain.build(self.base, self.cond.vocabs.user.size)
        self.samples = D.oversample_likes(self.base, k)
        self.builder = D.BatchBuilder(self.cond, self.samples, self.pos, HCFG)
        self.cat = self.fit["catalogs"]
        self.tr = self.cond.train_events().reset_index(drop=True)

    def vcol(self, vid):
        return int(self.cond.video_col[self.cat.video_row([vid])[0]])

    def mcol(self, mid):
        return int(self.cond.music_col[self.cat.music_row([mid])[0]])

    def col(self, task, item):
        return self.vcol(item) if task.startswith("video") else self.mcol(item)

    def uidx(self, u):
        return int(self.cond.vocabs.user.encode([u])[0])

    def sample(self, u, vid, t_min):
        ev = self.tr.iloc[self.samples.event_pos]
        hit = np.flatnonzero((ev.user_id.to_numpy() == u) & (ev.video_id.to_numpy() == vid)
                             & (ev.time_ms.to_numpy() == t_min * MIN) & ~self.samples.is_copy)
        assert len(hit) == 1
        return int(hit[0])

    def batch(self, idx=None):
        return self.builder(np.arange(len(self.samples)) if idx is None else np.asarray(idx))


@pytest.fixture(scope="module")
def W():
    return World()


def masked_cols(batch, task, r):
    tt = batch.targets[task]
    return set(tt.fn_cols[tt.fn_rows == r].tolist())


# ============================================================================ samples / candidates
def test_samples_candidates_and_task_masks(W):
    # 5 (uA) + 2 (uB; the click-only v14 row is no sample) + 3 (uC) + 5 (uD); val/test excluded.
    assert len(W.samples) == 15 and not W.samples.is_copy.any()
    ev = W.tr.iloc[W.samples.event_pos]
    assert (ev.split == 0).all()
    assert W.cond.cand_video_rows.tolist() == W.cat.video_row([10, 11, 12, 13, 14]).tolist()
    assert W.cond.cand_music_rows.tolist() == W.cat.music_row([100, 101, 102]).tolist()
    assert W.cond.video_col[W.cat.video_row([15])[0]] == -1        # val-only video: no column
    assert W.cond.music_col[W.cat.music_row([103])[0]] == -1
    i = W.sample(1, 12, 40)                                         # music_id 0 row
    m = {t: bool(W.samples.task_mask[t][i]) for t in TASKS}
    assert m == {"video_long_view": True, "video_is_like": True, "music_long_view": False, "music_is_like": False}
    i = W.sample(2, 11, 25)                                         # like only
    m = {t: bool(W.samples.task_mask[t][i]) for t in TASKS}
    assert m == {"video_long_view": False, "video_is_like": True, "music_long_view": False, "music_is_like": True}


def test_pos_train_matches_hand_written_sets(W):
    for task, per_user in POS.items():
        for u, items in per_user.items():
            got = set(W.pos.of(task, W.uidx(u)).tolist())
            assert got == {W.col(task, x) for x in items}, (task, u)
    # Val positive uA v11 (t120) and test positive uB v13 are not in Pos_train.
    assert W.vcol(11) not in W.pos.of("video_long_view", W.uidx(1))
    assert W.vcol(13) not in W.pos.of("video_long_view", W.uidx(2))


# ============================================================================ F1 duplicate positives
def test_f1_duplicate_positives_share_one_column(W):
    b = W.batch()
    lab = b.targets["video_long_view"].label
    rows_v10 = [W.sample(1, 10, 10), W.sample(1, 10, 60), W.sample(2, 10, 15), W.sample(3, 10, 70)]
    assert {int(lab[r]) for r in rows_v10} == {W.vcol(10)}           # same user and different users
    assert len(set(W.cond.cand_video_rows.tolist())) == len(W.cond.cand_video_rows)  # unique columns
    # The repeated same-pair positive: neither row masks the shared label column.
    for r in rows_v10[:2]:
        assert W.vcol(10) not in masked_cols(b, "video_long_view", r)
    # Pos_train stores the pair once.
    assert W.pos.of("video_long_view", W.uidx(1)).tolist().count(W.vcol(10)) == 1


# ============================================================================ F2 same-user FN mask
def test_f2_same_user_same_task_positives_masked_label_kept(W):
    b = W.batch()
    t = "video_long_view"
    r = W.sample(1, 10, 10)
    assert masked_cols(b, t, r) == {W.vcol(v) for v in (12, 13, 14)}
    r = W.sample(1, 13, 20)
    assert masked_cols(b, t, r) == {W.vcol(v) for v in (10, 12, 14)}
    # Other users' rows are not affected by uA's positives.
    assert masked_cols(b, t, W.sample(2, 10, 15)) == set()
    assert masked_cols(b, t, W.sample(3, 11, 12)) == {W.vcol(10)}   # uC's own later positive (Q7a)
    # The label logit stays finite for every valid row of every task.
    m = ThreeTowerModel(F.FeatureSizes.from_vocabs(W.cond.vocabs), TowerConfig(), HCFG).eval()
    items = TR.ItemInputs.from_condition(W.cond)
    with torch.no_grad():
        enc = m.encode_items(items.vf, items.mf)
        ie = m.item_embeddings(enc)
        ue = m.user_embeddings(m.encode_users(b.user_idx, b.history, enc))
    for task in TASKS:
        tgt = task.split("_")[0]
        vi, lab, lg = L.masked_logits(ue[task], ie[task][items.cand_rows[tgt]], b.targets[task], TAU)
        assert torch.isfinite(lg[torch.arange(len(vi)), lab]).all()
        assert int(torch.isneginf(lg).sum()) == int(b.targets[task].fn_rows.numel())


def test_fn_mask_equals_brute_force_definition(W):
    # Independent recomputation from the raw train events (pandas), per row and task.
    b = W.batch()
    ev = W.tr.iloc[W.samples.event_pos].reset_index(drop=True)
    tr = W.tr
    for task in TASKS:
        tgt, act = task.split("_", 1)
        for r in range(len(ev)):
            if not b.targets[task].valid[r]:
                assert masked_cols(b, task, r) == set()
                continue
            p = tr[(tr.user_id == ev.user_id[r]) & (tr[act] == 1)]
            if tgt == "video":
                pos = {W.vcol(v) for v in p.video_id}
                lab = W.vcol(ev.video_id[r])
            else:
                pos = {W.mcol(mm) for mm in p.music_id if mm != 0}
                lab = W.mcol(ev.music_id[r])
            assert int(b.targets[task].label[r]) == lab
            assert masked_cols(b, task, r) == pos - {lab}, (task, r)


# ============================================================================ F3 same task only
def test_f3_mask_is_task_specific(W):
    b = W.batch()
    r = W.sample(1, 13, 20)                                         # uA like row on v13
    # v10 and v14 are uA long_view positives but not likes -> remain negatives in like.
    assert masked_cols(b, "video_is_like", r) == {W.vcol(12)}
    assert W.vcol(10) not in masked_cols(b, "video_is_like", r)
    assert masked_cols(b, "music_is_like", r) == set()             # uA's only like music is m101 = label
    assert masked_cols(b, "music_long_view", r) == {W.mcol(100), W.mcol(102)}
    # uC: v13 is a like positive, not a long_view positive -> not masked in uC's lv rows.
    assert W.vcol(13) not in masked_cols(b, "video_long_view", W.sample(3, 11, 12))


# ============================================================================ F4 shared music
def test_f4_shared_music(W):
    b = W.batch()
    m100 = W.mcol(100)
    # v10 and v11 share m100: their music labels are the same column.
    assert int(b.targets["music_long_view"].label[W.sample(1, 10, 10)]) == m100
    assert int(b.targets["music_long_view"].label[W.sample(3, 11, 12)]) == m100
    assert int(b.targets["music_is_like"].label[W.sample(2, 11, 25)]) == m100
    # Video task: v11 is not a uA positive, so it is NOT masked in uA's v10 row although it
    # shares v10's music.
    assert W.vcol(11) not in masked_cols(b, "video_long_view", W.sample(1, 10, 10))
    # uC has lv positives on both v10 and v11 (one music): Pos_train holds m100 once, and both
    # music rows keep it as their label (no self-masking through the sibling video).
    assert W.pos.of("music_long_view", W.uidx(3)).tolist() == [m100]
    for r in (W.sample(3, 11, 12), W.sample(3, 10, 70)):
        assert masked_cols(b, "music_long_view", r) == set()
    # ... while in the video task each masks the sibling video.
    assert masked_cols(b, "video_long_view", W.sample(3, 10, 70)) == {W.vcol(11)}


# ============================================================================ F5 music 0 / label-only
def test_f5_music_zero_and_label_only_rows(W):
    b = W.batch()
    r = W.sample(1, 12, 40)
    for t in ("music_long_view", "music_is_like"):
        assert not b.targets[t].valid[r] and int(b.targets[t].label[r]) == -1
        assert (b.targets[t].fn_rows != r).all()
    # uD is a lv positive of every candidate: every uD lv row is label-only -> CE == 0 exactly.
    uD = [W.sample(4, v, 5 + i) for i, v in enumerate([10, 11, 12, 13, 14])]
    for r in uD:
        assert len(masked_cols(b, "video_long_view", r)) == 4
    torch.manual_seed(0)
    u = torch.nn.functional.normalize(torch.randn(len(W.samples), 8), dim=-1).requires_grad_()
    c = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1).requires_grad_()
    bb = W.batch(uD)
    loss, info = L.task_loss(u[:5], c, bb.targets["video_long_view"], TAU)
    assert info["n_valid"] == 5 and loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(u.grad).all() and (u.grad == 0).all() and (c.grad == 0).all()
    # Music rows of uD: v12 has no music -> 4 valid rows, each label-only over 3 music columns.
    ml, mi = L.task_loss(u[:5], c[:3], bb.targets["music_long_view"], TAU)
    assert mi["n_valid"] == 4 and ml.item() == 0.0


# ============================================================================ F6 hand-computed CE
def _hand_ce(u, c, rows_labels_masks, tau):
    vals = []
    for r, lab, masked in rows_labels_masks:
        z = (u[r] @ c.T) / tau
        keep = [j for j in range(len(c)) if j == lab or j not in masked]
        zz = z[keep]
        vals.append(np.log(np.exp(zz - zz.max()).sum()) + zz.max() - z[lab])
    return float(np.mean(vals))


def test_f6_full_softmax_matches_hand_computation(W):
    b = W.batch()
    rng = np.random.default_rng(0)
    n = len(W.samples)
    ue = {t: rng.normal(size=(n, 6)) for t in TASKS}
    ie = {"video_long_view": rng.normal(size=(6, 6)), "video_is_like": rng.normal(size=(6, 6)),
          "music_long_view": rng.normal(size=(4, 6)), "music_is_like": rng.normal(size=(4, 6))}
    ue = {t: x / np.linalg.norm(x, axis=1, keepdims=True) for t, x in ue.items()}
    ie = {t: x / np.linalg.norm(x, axis=1, keepdims=True) for t, x in ie.items()}
    cand = TR.ItemInputs.from_condition(W.cond).cand_rows
    T64 = lambda d: {t: torch.tensor(x, dtype=torch.float64) for t, x in d.items()}
    total, info = L.multitask_loss(T64(ue), T64(ie), cand, b.targets, TAU)
    ev = W.tr.iloc[W.samples.event_pos].reset_index(drop=True)
    want_total = 0.0
    for task in TASKS:
        tgt, act = task.split("_", 1)
        c = ie[task][(W.cond.cand_video_rows if tgt == "video" else W.cond.cand_music_rows)]
        spec = []
        for r in range(n):
            if ev[act][r] != 1 or (tgt == "music" and ev.music_id[r] == 0):
                continue
            item = ev.video_id[r] if tgt == "video" else ev.music_id[r]
            masked = {W.col(task, x) for x in POS[task][ev.user_id[r]]} - {W.col(task, item)}
            spec.append((r, W.col(task, item), masked))
        want = _hand_ce(ue[task], c, spec, TAU)
        assert info[task]["n_valid"] == len(spec)
        assert info[task]["loss"] == pytest.approx(want, rel=1e-10, abs=1e-10), task
        want_total += want
    assert float(total) == pytest.approx(want_total, rel=1e-10)
    # Equal weights: the total is the plain sum of the 4 task means.
    assert float(total) == pytest.approx(sum(info[t]["loss"] for t in TASKS), rel=1e-12)


def test_f6_cold_condition_excludes_h_everywhere():
    Wc = World(exclude=[13])
    c = Wc.cond
    assert c.condition_id != World().cond.condition_id
    assert 13 not in c.vocabs.video.ids
    assert Wc.cat.video_row([13])[0] not in c.cand_video_rows              # not a candidate
    assert c.video_col[Wc.cat.video_row([13])[0]] == -1
    # m101 only occurs on v13 -> not in the cold music vocab -> not a music candidate.
    assert c.music_col[Wc.cat.music_row([101])[0]] == -1
    ev = Wc.tr.iloc[Wc.samples.event_pos]
    assert not (ev.video_id == 13).any()                                    # not a training sample
    for t in TASKS:                                                         # not in Pos_train
        cols = Wc.pos.cols[t]
        assert (cols >= 0).all() and (cols < (len(c.cand_video_rows) if t.startswith("video")
                                              else len(c.cand_music_rows))).all()
    assert set(Wc.pos.of("video_long_view", Wc.uidx(1)).tolist()) == {Wc.vcol(v) for v in (10, 12, 14)}
    assert Wc.pos.of("music_is_like", Wc.uidx(1)).size == 0                 # uA's like music was m101
    b = Wc.batch()
    r13 = int(Wc.cat.video_row([13])[0])
    assert not (b.history.video_row == r13).any()                           # not in rolling history
    for t in TASKS:
        assert (b.targets[t].fn_cols >= 0).all()
    # The condition refuses inconsistent inputs: H still present in the events.
    with pytest.raises(ValueError):
        D.TrainCondition.build("bad", World().fit["events"], c.vocabs, Wc.fit["video_features"],
                               Wc.fit["music_features"], exclude_video_ids=[13])


# ============================================================================ loss guards
def test_task_loss_guards_and_empty_task(W):
    b = W.batch()
    u = torch.nn.functional.normalize(torch.randn(len(W.samples), 4), dim=-1)
    c = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    tt = b.targets["video_long_view"]
    # Mask that would remove the label column is refused.
    r0 = int(torch.nonzero(tt.valid)[0])
    bad = dataclasses.replace(tt, fn_rows=torch.cat([tt.fn_rows, torch.tensor([r0])]),
                              fn_cols=torch.cat([tt.fn_cols, tt.label[r0:r0 + 1]]))
    with pytest.raises(ValueError):
        L.task_loss(u, c, bad, TAU)
    # Label outside the candidate set is refused.
    with pytest.raises(ValueError):
        L.task_loss(u, c[:2], tt, TAU)
    # Empty task -> None (skipped, never 0/0); all-empty batch -> total None.
    empty = W.batch().only(())
    for t in TASKS:
        assert L.task_loss(u, c, empty.targets[t], TAU)[0] is None
    cand = {"video": torch.arange(5), "music": torch.arange(3)}
    ue = {t: u for t in TASKS}
    ie = {t: c for t in TASKS}
    assert L.multitask_loss(ue, ie, cand, empty.targets, TAU)[0] is None


# ============================================================================ model-level gradients
def _model(cond, seed=0, **cfg):
    torch.manual_seed(seed)
    return ThreeTowerModel(F.FeatureSizes.from_vocabs(cond.vocabs), TowerConfig(**cfg), HCFG)


def test_each_task_produces_gradients_only_on_its_own_heads(W):
    items = TR.ItemInputs.from_condition(W.cond)
    b = W.batch()
    for task in TASKS:
        m = _model(W.cond, dropout=0.0, id_dropout=0.0).train()
        loss, info = TR.forward_loss(m, items, b.only([task]), TAU)
        assert [t for t in TASKS if info[t]["n_valid"]] == [task]
        loss.backward()
        own_item = m.video_heads if task.startswith("video") else m.music_heads
        other_item = m.music_heads if task.startswith("video") else m.video_heads
        for heads, own in ((m.user_heads, True), (own_item, True), (other_item, False)):
            for t, lin in heads.proj.items():
                for g in (lin.weight.grad, lin.bias.grad):
                    if own and t == task:
                        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, (task, t)
                    else:
                        assert g is None or g.abs().sum() == 0, (task, t)
        # Every task reaches the shared item towers (music via the video tower or directly).
        assert m.music_tower.music_emb.weight.grad.abs().sum() > 0


def test_video_loss_reaches_music_tower_unless_use_music_false(W):
    items = TR.ItemInputs.from_condition(W.cond)
    b = W.batch().only(["video_long_view", "video_is_like"])
    for use_music in (True, False):
        m = _model(W.cond, dropout=0.0, id_dropout=0.0, use_music=use_music).train()
        TR.forward_loss(m, items, b, TAU)[0].backward()
        g = [p.grad for p in m.music_tower.parameters() if p.grad is not None]
        assert any(x.abs().sum() > 0 for x in g) == use_music


def test_train_step_encodes_catalog_once_and_shares_it(W):
    """One step = one encode_items, one MusicTower call, one VideoTower call; the history tokens
    and the video candidates consume the same v_base tensor, and the video tower consumes the
    music candidates' m_base sample (train mode, dropout on)."""
    m = _model(W.cond, dropout=0.5, id_dropout=0.3)
    items = TR.ItemInputs.from_condition(W.cond)
    opt, sched = TR.make_optimizer(m, TR.TrainConfig(warmup_steps=0))
    rec = {"enc": [], "music": [], "video": [], "hist_vbase": [], "video_music_in": [], "heads_in": []}
    orig = m.encode_items

    def wrapped(*a, **k):
        out = orig(*a, **k)
        rec["enc"].append(out)
        return out

    m.encode_items = wrapped
    hooks = [m.music_tower.register_forward_hook(lambda mod, a, o: rec["music"].append(o)),
             m.video_tower.register_forward_hook(lambda mod, a, o: rec["video"].append(o)),
             m.video_tower.register_forward_pre_hook(lambda mod, a: rec["video_music_in"].append(a[2])),
             m.user_tower.history.register_forward_pre_hook(lambda mod, a: rec["hist_vbase"].append(a[1])),
             m.video_heads.register_forward_pre_hook(lambda mod, a: rec["heads_in"].append(a[0]))]
    tc = TR.TrainConfig(warmup_steps=0)
    for step in range(3):
        TR.train_step(m, opt, sched, items, W.batch(), tc, step)
        assert len(rec["enc"]) == len(rec["music"]) == len(rec["video"]) == step + 1
        enc = rec["enc"][-1]
        assert rec["hist_vbase"][-1] is enc.v_base and rec["heads_in"][-1] is enc.v_base
        assert rec["video"][-1] is enc.v_base
        has = items.vf.music_row >= 0
        assert torch.equal(rec["video_music_in"][-1][has], enc.m_base[items.vf.music_row[has]])
        assert torch.equal(enc.m_base, rec["music"][-1][: len(items.mf)])
    for h in hooks:
        h.remove()


# ============================================================================ stability / fail fast
def test_finite_on_empty_history_no_likes_and_oov_user(W):
    items = TR.ItemInputs.from_condition(W.cond)
    m = _model(W.cond).train()
    # First events of each user have an empty rolling history.
    first = [W.sample(4, 10, 5), W.sample(3, 11, 12)]
    b = W.batch(first)
    assert (b.history.length == 0).all()
    assert not b.targets["video_is_like"].valid.any()                  # no like rows in this batch
    loss, info = TR.forward_loss(m, items, b, TAU)
    assert info["video_is_like"]["n_valid"] == 0 and "loss" not in info["video_is_like"]
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    # OOV user id + empty history (the cold-user path), e.g. after user-ID dropout.
    m.zero_grad()
    b_oov = dataclasses.replace(b, user_idx=torch.full_like(b.user_idx, catm.OOV))
    loss, _ = TR.forward_loss(m, items, b_oov, TAU)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_all_empty_batch_fails_loudly(W):
    m = _model(W.cond)
    opt, sched = TR.make_optimizer(m, TR.TrainConfig())
    with pytest.raises(ValueError):
        TR.train_step(m, opt, sched, TR.ItemInputs.from_condition(W.cond), W.batch().only(()), TR.TrainConfig(), 0)


def _params(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def test_nan_loss_fails_fast_with_dump_and_no_update(W, tmp_path):
    m = _model(W.cond)
    opt, sched = TR.make_optimizer(m, TR.TrainConfig(warmup_steps=0))
    with torch.no_grad():
        m.user_tower.mlp[0].weight[0, 0] = float("nan")
    before = _params(m)
    with pytest.raises(TR.NonFiniteError):
        TR.train_step(m, opt, sched, TR.ItemInputs.from_condition(W.cond), W.batch(), TR.TrainConfig(), 7, tmp_path)
    dumps = list(tmp_path.glob("nonfinite_step7.json"))
    assert len(dumps) == 1 and "batch_rows" in json.loads(dumps[0].read_text())
    after = _params(m)
    assert all(torch.allclose(before[k], after[k], rtol=0, atol=0, equal_nan=True) for k in before)
    assert opt.state == {} or all(len(s) == 0 for s in opt.state.values())   # no Adam step taken


def test_nan_gradient_fails_fast_before_optimizer_step(W, tmp_path):
    m = _model(W.cond)
    opt, sched = TR.make_optimizer(m, TR.TrainConfig(warmup_steps=0))
    m.video_heads.proj["video_long_view"].weight.register_hook(lambda g: g * float("nan"))
    before = _params(m)
    with pytest.raises(TR.NonFiniteError, match="grad norm"):
        TR.train_step(m, opt, sched, TR.ItemInputs.from_condition(W.cond), W.batch(), TR.TrainConfig(), 3, tmp_path)
    assert (tmp_path / "nonfinite_step3.json").exists()
    after = _params(m)
    assert all(torch.equal(before[k], after[k]) for k in before)


# ============================================================================ oversampling
def test_like_oversample_k4_rows_and_masks():
    W1, W4 = World(), World(k=4)
    b, s = W1.base, W4.samples
    n_like = int(b.task_mask["video_is_like"].sum())
    assert n_like == 4
    assert len(s) == len(b) + 3 * n_like and int(s.is_copy.sum()) == 3 * n_like
    c = W4.samples.counts()
    assert c["valid_video_long_view"] == int(b.task_mask["video_long_view"].sum())
    assert c["valid_music_long_view"] == int(b.task_mask["music_long_view"].sum())
    assert c["valid_video_is_like"] == 4 * n_like
    assert c["valid_music_is_like"] == 4 * int(b.task_mask["music_is_like"].sum())
    for t in ("video_long_view", "music_long_view"):
        assert not s.task_mask[t][s.is_copy].any()                       # copies: long_view off
    # Pos_train is from the base samples only and identical for k = 1 / 4.
    for t in TASKS:
        assert np.array_equal(W1.pos.cols[t], W4.pos.cols[t]) and np.array_equal(W1.pos.indptr[t], W4.pos.indptr[t])
    with pytest.raises(ValueError):
        D.PosTrain.build(s, W4.cond.vocabs.user.size)
    assert D.oversample_likes(b, 1) is b
    with pytest.raises(ValueError):
        D.oversample_likes(b, 0)


def test_epoch_is_one_permutation():
    bs = D.epoch_batches(23, 5, seed=0, epoch=1)
    assert sorted(np.concatenate(bs).tolist()) == list(range(23)) and [len(x) for x in bs] == [5, 5, 5, 5, 3]
    assert all(np.array_equal(a, b) for a, b in zip(bs, D.epoch_batches(23, 5, 0, 1)))
    assert not np.array_equal(np.concatenate(bs), np.concatenate(D.epoch_batches(23, 5, 0, 2)))


# ============================================================================ leakage
def test_rolling_history_is_strict_past_train_only(W):
    b = W.batch()
    s = W.samples
    tr = W.tr
    for r in range(len(s)):
        u, t = s.user_id[r], s.time_ms[r]
        want = tr[(tr.user_id == u) & (tr.time_ms < t)].sort_values(["time_ms", "video_id"]).video_row.tolist()[-HCFG.max_len:]
        got = b.history.video_row[r][b.history.valid[r]].tolist()
        assert got == want, r
    # uA's last train row (t60) does not see the val event v11 (t120) nor itself.
    r = W.sample(1, 10, 60)
    assert W.cat.video_row([11])[0] not in b.history.video_row[r].tolist()


def test_val_test_events_do_not_change_any_training_input():
    """Changing / removing every val and test event leaves samples, Pos_train, candidates and
    every batch tensor bitwise unchanged."""
    base = World()
    alt_rows = [r for r in rows() if r["time_ms"] < fx.META["t_val_ms"]]
    alt_rows += [R(2, 12, 150 * MIN, lv=1, like=1), R(4, 11, 250 * MIN, lv=1, like=1)]
    alt = World.__new__(World)
    alt.fit = fx.make(alt_rows)
    # v15 no longer occurs at all; vocabs are train-only so they are identical.
    alt.cond = D.condition_from_fit("normal", alt.fit)
    alt.base = D.build_samples(alt.cond)
    alt.pos = D.PosTrain.build(alt.base, alt.cond.vocabs.user.size)
    alt.samples = alt.base
    alt.builder = D.BatchBuilder(alt.cond, alt.samples, alt.pos, HCFG)
    assert np.array_equal(base.cond.cand_video_rows, alt.cond.cand_video_rows)
    assert np.array_equal(base.cond.cand_music_rows, alt.cond.cand_music_rows)
    for t in TASKS:
        assert np.array_equal(base.pos.cols[t], alt.pos.cols[t])
    b1, b2 = base.batch(), alt.builder(np.arange(len(alt.samples)))
    for f in dataclasses.fields(F.HistoryTensors):
        assert torch.equal(getattr(b1.history, f.name), getattr(b2.history, f.name)), f.name
    for t in TASKS:
        for f in ("valid", "label", "fn_rows", "fn_cols"):
            assert torch.equal(getattr(b1.targets[t], f), getattr(b2.targets[t], f)), (t, f)


def test_later_train_events_do_not_change_earlier_histories(W):
    """Rolling history of a row is unaffected by any train event at t >= its time (only the
    FN mask may use later train positives, by design, Q7a)."""
    r_early = W.sample(1, 13, 20)
    b1 = W.batch([r_early])
    alt = [x for x in rows() if not (x["user_id"] == 1 and x["time_ms"] >= 20 * MIN)]
    alt += [R(1, 11, 20 * MIN, lv=1, like=1, play=99_000), R(1, 14, 50 * MIN, lv=1, like=1)]
    W2 = World.__new__(World)
    W2.fit = fx.make(alt)
    W2.cond = D.condition_from_fit("normal", W2.fit)
    W2.samples = D.build_samples(W2.cond)
    W2.pos = D.PosTrain.build(W2.samples, W2.cond.vocabs.user.size)
    W2.builder = D.BatchBuilder(W2.cond, W2.samples, W2.pos, HCFG)
    W2.cat, W2.tr = W2.fit["catalogs"], W2.cond.train_events().reset_index(drop=True)
    h2 = F.build_history_batch(W2.builder.hidx, [1], [20 * MIN], HCFG)
    for f in ("video_row", "feedback", "hidden", "play_bucket", "recency_bucket", "valid", "length"):
        assert np.array_equal(getattr(b1.history, f).numpy(), getattr(h2, f)), f


# ============================================================================ optimizer / trainer
def test_param_groups_weight_decay_assignment(W):
    m = _model(W.cond)
    g = TR.param_groups(m, 0.01)
    decay, nodecay = {id(p) for p in g[0]["params"]}, {id(p) for p in g[1]["params"]}
    assert g[0]["weight_decay"] == 0.01 and g[1]["weight_decay"] == 0.0
    assert len(decay) + len(nodecay) == len(list(m.parameters())) and not decay & nodecay
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            assert id(mod.weight) in decay and (mod.bias is None or id(mod.bias) in nodecay)
        if isinstance(mod, (torch.nn.LayerNorm,)):
            assert id(mod.weight) in nodecay and id(mod.bias) in nodecay
    for _, emb in m.fixed_zero_embeddings():
        assert id(emb.weight) in nodecay
    assert id(m.user_tower.history.pos_emb) in nodecay and id(m.user_tower.history.pool_query) in nodecay


def test_warmup_schedule():
    m = torch.nn.Linear(2, 2)
    opt, sched = TR.make_optimizer(m, TR.TrainConfig(lr=1e-3, warmup_steps=4))
    lrs = []
    for _ in range(6):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs == pytest.approx([2.5e-4, 5e-4, 7.5e-4, 1e-3, 1e-3, 1e-3])


def test_trainer_fixture_end_to_end_two_epochs(W, tmp_path):
    tc = TR.TrainConfig(batch_size=4, max_epochs=2, warmup_steps=0, lr=1e-2, seed=0)
    tr = TR.Trainer(W.cond, TowerConfig(), HCFG, tc, out_dir=tmp_path)
    res = tr.fit()
    assert res["steps"] == 2 * int(np.ceil(15 / 4)) and len(tr.log) == res["steps"]
    assert all(np.isfinite(r["loss"]) and np.isfinite(r["grad_norm"]) for r in tr.log)
    # PAD / OOV rows stay exactly zero after real optimizer steps (Q5b).
    for name, emb in tr.model.fixed_zero_embeddings():
        assert (emb.weight[list(emb.zero_rows)] == 0).all(), name
    p = tr.save("last.pt")
    ck = torch.load(p, weights_only=False)
    assert ck["condition_id"] == W.cond.condition_id and ck["step"] == res["steps"]


def test_trainer_reduces_loss_on_fixture(W):
    """Overfit check: without dropout, training steps reduce the full-batch loss."""
    tcfg = TowerConfig(dropout=0.0, id_dropout=0.0)
    tc = TR.TrainConfig(batch_size=15, max_epochs=40, warmup_steps=0, lr=3e-3, seed=0)
    tr = TR.Trainer(W.cond, tcfg, HCFG, tc)
    b = W.batch()
    with torch.no_grad():
        l0 = float(TR.forward_loss(tr.model.eval(), tr.items, b, TAU)[0])
    tr.fit()
    with torch.no_grad():
        l1 = float(TR.forward_loss(tr.model.eval(), tr.items, b, TAU)[0])
    assert l1 < 0.5 * l0


def test_trainer_is_deterministic(W):
    def run():
        tr = TR.Trainer(W.cond, TowerConfig(), HCFG, TR.TrainConfig(batch_size=4, max_epochs=1, warmup_steps=0, seed=3))
        tr.fit()
        return [r["loss"] for r in tr.log], _params(tr.model)
    a, pa = run()
    b, pb = run()
    assert a == b and all(torch.equal(pa[k], pb[k]) for k in pa)


def test_output_dir_inside_worktree_is_refused(tmp_path):
    from kuairand_rec import paths
    with pytest.raises(ValueError):
        TR.assert_outside_worktree(paths.PROJECT_ROOT / "runs" / "x")
    TR.assert_outside_worktree(tmp_path)


def test_train_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        TR.TrainConfig.from_dict({"learning_rate": 1e-3})
    assert TR.TrainConfig.from_dict({"like_oversample": 4}).like_oversample == 4
    assert TR.TrainConfig().like_oversample == 1 and TR.TrainConfig().tau == 0.05
