"""M4-1 invariants on the real M2 artifacts. Skipped unless the pipeline has been run."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from kuairand_rec import paths
from kuairand_rec.data.pipeline import load_processed
from kuairand_rec.models import features as F


def test_feedback_cols_match_data_config():
    cfg = yaml.safe_load((paths.PROJECT_ROOT / "configs" / "data.yaml").read_text())
    assert tuple(c for c in cfg["feedback_cols"] if c != "play_time_ms") == F.FEEDBACK_BINARY_COLS
    assert cfg["history_max_len"] == F.HistoryConfig().max_len


needs_data = pytest.mark.skipif(not (paths.PROCESSED_DIR / "meta.json").exists(),
                                reason="run `python -m kuairand_rec.data.pipeline` first")


@needs_data
def test_real_item_features_and_frozen_histories():
    P = load_processed()
    F.check_shared_music_inputs(P["video_catalog"], P["music_catalog"])
    vf, mf = F.VideoFeatures.from_catalog(P["video_catalog"]), F.MusicFeatures.from_catalog(P["music_catalog"])
    assert len(vf) == len(P["catalogs"].video_ids) and len(mf) == len(P["catalogs"].music_ids)
    ev, cfg = P["events"], F.HistoryConfig()
    users = np.unique(ev.loc[ev.split == 2, "user_id"])
    hidx, t = F.frozen_history_index(ev, "test", P["meta"])
    hb = F.build_history_batch(hidx, users, t, cfg)
    assert hb.length.max() <= cfg.max_len
    assert hb.recency_bucket.max() < cfg.n_recency_rows
    # D1-2 impact at T_test: only a few hundred hidden tokens.
    assert 0 < hb.hidden.sum() < 1000


@needs_data
def test_frozen_recency_buckets_all_occur_in_train_target_histories():
    """P1-6b: every non-PAD recency row used at T_val / T_test must have been used by the
    rolling train-target histories (all train events positive for any action, D1-3)."""
    P = load_processed()
    ev, cfg = P["events"], F.HistoryConfig()
    tr = ev[ev.split == 0]
    tgt = tr[(tr.long_view == 1) | (tr.is_like == 1)]
    hb = F.build_history_batch(F.train_history_index(ev), tgt.user_id.to_numpy(), tgt.time_ms.to_numpy(), cfg)
    train_used = set(np.unique(hb.recency_bucket[hb.valid]).tolist())
    assert F.RECENCY_PAD not in train_used
    for split_name in ("val", "test"):
        hidx, t = F.frozen_history_index(ev, split_name, P["meta"])
        fb = F.build_history_batch(hidx, np.unique(ev.user_id), t, cfg)
        used = set(np.unique(fb.recency_bucket[fb.valid]).tolist())
        assert F.RECENCY_PAD not in used
        assert used <= train_used, (split_name, sorted(used - train_used))


# ----------------------------------------------------------------------------- M4-2 (real data)
@pytest.fixture(scope="module")
def real_cond():
    from kuairand_rec.models import data as D
    P = load_processed()
    return P, D.normal_condition(P)


@needs_data
def test_real_training_samples_candidates_and_pos_train(real_cond):
    from kuairand_rec.models import data as D
    P, cond = real_cond
    s = D.build_samples(cond)
    # D1-3 numbers: 345,365 rows, 18,771 likes; candidates = train vocab sizes.
    assert len(s) == 345_365 and int(s.task_mask["video_is_like"].sum()) == 18_771
    assert len(cond.cand_video_rows) == 7_529 and len(cond.cand_music_rows) == 7_151
    ev = cond.train_events().reset_index(drop=True).iloc[s.event_pos]
    assert (ev.split == 0).all()
    assert (s.task_mask["music_long_view"] <= (ev.music_id.to_numpy() != 0)).all()
    assert ((ev.music_id.to_numpy() == 0) & s.task_mask["music_is_like"]).sum() == 0
    pos = D.PosTrain.build(s, cond.vocabs.user.size)
    # Brute-force Pos_train for a sample of users, all 4 tasks.
    tr = cond.train_events()
    rng = np.random.default_rng(0)
    for u in rng.choice(np.unique(s.user_id), 200, replace=False):
        ui = int(cond.vocabs.user.encode([u])[0])
        tu = tr[tr.user_id == u]
        for task in D.ACTION_OF:
            tgt, act = task.split("_", 1)
            p = tu[tu[act] == 1]
            if tgt == "video":
                want = set(cond.video_col[p.video_row].tolist())
            else:
                want = set(cond.music_col[p.music_row[p.music_row >= 0]].tolist())
            assert set(pos.of(task, ui).tolist()) == want, (u, task)


@needs_data
def test_real_one_train_step_is_finite_and_updates(real_cond):
    import torch
    from kuairand_rec.models import train as TR
    from kuairand_rec.models.towers import TowerConfig
    P, cond = real_cond
    tr = TR.Trainer(cond, TowerConfig(), F.HistoryConfig(), TR.TrainConfig(batch_size=64, warmup_steps=0, max_steps=1))
    before = {n: p.detach().clone() for n, p in tr.model.named_parameters()}
    tr.fit()
    rec = tr.log[0]
    assert np.isfinite(rec["loss"]) and np.isfinite(rec["grad_norm"]) and rec["grad_norm"] > 0
    assert all(rec[f"n_{t}"] >= 0 for t in ("video_long_view", "video_is_like"))
    changed = [n for n, p in tr.model.named_parameters() if not torch.equal(before[n], p)]
    assert any(n.startswith("music_tower") for n in changed) and any(n.startswith("user_tower") for n in changed)
    for name, emb in tr.model.fixed_zero_embeddings():
        assert (emb.weight[list(emb.zero_rows)] == 0).all(), name


# ----------------------------------------------------------------------------- M4-3 (real data)
@needs_data
def test_real_eval_frames_match_m3_frame_ids(real_cond):
    from kuairand_rec.models.evaluation import build_eval_frames
    P, _ = real_cond
    frames = build_eval_frames(P)            # raises on any frame_id mismatch vs frames_summary.json
    assert len(frames) == 2 * 4 * 3


@needs_data
def test_real_small_export_parity_and_retrieve(real_cond, tmp_path):
    """Random-init checkpoint of the real normal condition: export a stratified user subset,
    then strict parity (checks 1-3 on those users) with a reloaded model."""
    import torch
    from kuairand_rec.models import artifacts as A
    from kuairand_rec.models import parity as PR
    from kuairand_rec.models.evaluation import build_eval_frames
    from kuairand_rec.models.export import ExportedScorer, export_embeddings, retrieve
    from kuairand_rec.models.inference import DirectScorer
    from kuairand_rec.models import train as TR
    from kuairand_rec.models.towers import TowerConfig
    P, cond = real_cond
    tr = TR.Trainer(cond, TowerConfig(), F.HistoryConfig(), TR.TrainConfig(), out_dir=tmp_path)
    tr.save("best.pt")
    model, _ = A.load_model(tmp_path / "best.pt", cond)
    frames = build_eval_frames(P, splits=("test",), kinds=("masked", "cold"))
    users, cover = PR.select_parity_users(frames, 256, seed=1)
    assert min(cover["strata"].values()) > 0
    export_embeddings(model, cond, P, tmp_path / "export", tmp_path / "best.pt", users=users)
    ex = ExportedScorer(tmp_path / "export", expect_condition_id=cond.condition_id)
    direct = DirectScorer(model, cond, P["meta"])
    r = PR.check_scores_and_topk(direct, ex, users, frames)
    assert r["ok"] and r["max_abs"] == 0.0, {k: v["max_abs"] for k, v in r["tasks"].items()}
    r3 = PR.check_retrieve(ex, users[:64], frames)
    assert r3["ok"], r3["issues"]
    # Exported seen mask == M3 frame mask for every sampled frame user.
    f = frames[("test", "video_long_view", "masked")]
    pos = {u: i for i, u in enumerate(f.users.tolist())}
    for u in users[:64]:
        if u in pos:
            i = pos[u]
            assert np.array_equal(ex.seen_rows(u, "video"), np.sort(f.mask_idx[f.mask_indptr[i]:f.mask_indptr[i + 1]]))
    top = retrieve(ex, int(users[0]), "music", "is_like", 10)
    assert len(top) == 10 and all(np.isfinite(s) for _, s in top)
