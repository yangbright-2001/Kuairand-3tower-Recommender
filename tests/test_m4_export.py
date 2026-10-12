"""M4-3 tests: validation / checkpoint selection, export, DirectScorer / ExportedScorer,
retrieve() and strict parity (PLAN.md §3d D1-5, D1-7, D1-8, D1-9 "Export", "Cold condition").

Fixture = `m4_fixtures` (6 videos, 4 music; u1 has val + test events, u3 is a test-only cold
user, v15 is a val-only train-OOV video), encoded through the real M2 path. Frames come from the
real M3 `build_frames` (no frames_summary reference for the fixture).
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

import m4_fixtures as fx
from kuairand_rec.eval import metrics as M
from kuairand_rec.eval.evaluator import EvalContext
from kuairand_rec.models import artifacts as A
from kuairand_rec.models import data as D
from kuairand_rec.models import evaluation as E
from kuairand_rec.models import export as X
from kuairand_rec.models import features as F
from kuairand_rec.models import inference as I
from kuairand_rec.models import parity as PR
from kuairand_rec.models import train as TR
from kuairand_rec.models.towers import TASKS, TowerConfig

HCFG = F.HistoryConfig()
MIN = fx.MIN


# Extra test-period events with unseen GT so the test masked frames are non-empty:
# u4 (saw only v13) and u2 (never saw v15) get test positives.
ROWS = fx.raw_rows() + [fx._row(4, 14, 220 * MIN, lv=1, like=1), fx._row(4, 10, 225 * MIN, lv=1),
                        fx._row(2, 15, 240 * MIN, lv=1, like=1), fx._row(4, 11, 150 * MIN, lv=1, like=1)]


def fit_of(exclude=()):
    return fx.make(ROWS, exclude_video_ids=exclude)


def make_P() -> dict:
    """`load_processed`-like dict for the fixture (normal condition encodings)."""
    fit = fit_of()
    lc = pd.DataFrame({"user_id": pd.Series([], dtype=np.int64), "video_id": pd.Series([], dtype=np.int64),
                       "time_ms": pd.Series([], dtype=np.int64), "split": pd.Series([], dtype=np.int8),
                       "long_view_any": pd.Series([], dtype=np.int64), "is_like_any": pd.Series([], dtype=np.int64)})
    return {"events": fit["events"], "label_conflict_events": lc, "catalogs": fit["catalogs"], "meta": dict(fx.META),
            "video_catalog": fit["video_features"], "music_catalog": fit["music_features"], "vocabs": fit["vocabs"],
            "num_stats": fit["num_stats"]}


def cold_condition(P, H):
    return D.condition_from_fit("cold_test", fit_of(H), H)


@pytest.fixture(scope="module")
def P():
    return make_P()


@pytest.fixture(scope="module")
def cond(P):
    return D.condition_from_fit("normal", fit_of())


@pytest.fixture(scope="module")
def run(P, cond, tmp_path_factory):
    """Train 2 epochs with validation, export the best checkpoint (fixture end-to-end)."""
    out = tmp_path_factory.mktemp("run")
    tc = TR.TrainConfig(batch_size=16, max_epochs=2, warmup_steps=0, lr=1e-2, seed=0)
    tr = TR.Trainer(cond, TowerConfig(), HCFG, tc, out_dir=out)
    val = E.Validator(P, cond, frames_summary=None, user_batch_size=2)
    res = tr.fit(validate=val)
    model, ck = A.load_model(out / "best.pt", cond)
    frames = E.build_eval_frames(P, frames_summary=None)
    man = X.export_embeddings(model, cond, P, out / "export", out / "best.pt", {"exp": "fixture"},
                              frame_ids={"/".join(k): f.frame_id for k, f in frames.items()})
    return {"out": out, "trainer": tr, "res": res, "model": model, "ck": ck, "frames": frames, "manifest": man,
            "validator": val}


# ============================================================================ validation / selection
def test_validator_uses_val_masked_frames_only(P, cond, run):
    val = run["validator"]
    assert set(val.frames) == {("val", t, "masked") for t in TASKS}
    assert set(val.ctx) == {"val"} and val.t_val == fx.META["t_val_ms"]
    h = run["res"]["val_history"]
    assert [r["epoch"] for r in h] == [0, 1] and all(np.isfinite(r["score"]) for r in h)
    r = h[0]
    assert r["score"] == pytest.approx(np.mean([r[f"{t}/recall@20"] for t in TASKS]))
    assert r["tie"] == pytest.approx(np.mean([r[f"{t}/ndcg@20"] for t in TASKS]))


def test_validation_score_ignores_test_events(P, cond, run):
    """Changing every test-period label leaves the validation score bitwise unchanged."""
    P2 = dict(P)
    ev = P["events"].copy()
    t = ev.split == 2
    ev.loc[t, ["long_view", "is_like"]] = 1 - ev.loc[t, ["long_view", "is_like"]]
    P2["events"] = ev
    m = run["model"]
    v1 = E.Validator(P, cond, frames_summary=None)(m, 0)
    v2 = E.Validator(P2, cond, frames_summary=None)(m, 0)
    assert v1["score"] == v2["score"] and v1["tie"] == v2["tie"]


def test_validation_restores_train_mode(P, cond, run):
    m = run["model"]
    m.train()
    E.Validator(P, cond, frames_summary=None)(m, 0)
    assert m.training
    m.eval()


class _Seq:
    """Fake validator returning a fixed (score, tie) sequence and snapshotting the model."""

    def __init__(self, seq):
        self.seq, self.snap = list(seq), []

    def __call__(self, model, epoch):
        self.snap.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        s, t = self.seq[epoch]
        return {"score": s, "tie": t}


def test_selection_rule_tie_break_and_patience(cond, tmp_path):
    seq = [(0.1, 0.5), (0.2, 0.1), (0.2, 0.3), (0.2, 0.3), (0.15, 0.9), (0.19, 0.9), (0.5, 0.5)]
    tc = TR.TrainConfig(batch_size=8, max_epochs=7, warmup_steps=0, patience=3, seed=0)
    tr = TR.Trainer(cond, TowerConfig(), HCFG, tc, out_dir=tmp_path)
    v = _Seq(seq)
    res = tr.fit(validate=v)
    # Epoch 2 beats epoch 1 on the tie-break; epoch 3 is an exact tie (earliest wins); then
    # epochs 3, 4, 5 are 3 non-improvements -> stop before epoch 6 (which would be best).
    assert res["best_epoch"] == 2 and res["best_score"] == 0.2 and res["best_tie"] == 0.3
    assert [r["epoch"] for r in res["val_history"]] == [0, 1, 2, 3, 4, 5]
    assert [r["improved"] for r in res["val_history"]] == [True, True, True, False, False, False]
    ck = torch.load(tmp_path / "best.pt", weights_only=False)
    assert ck["epoch"] == 2
    assert all(torch.equal(ck["model"][k], v.snap[2][k]) for k in ck["model"])
    assert not all(torch.equal(ck["model"][k], v.snap[5][k]) for k in ck["model"])


def test_non_finite_validation_fails(cond):
    tr = TR.Trainer(cond, TowerConfig(), HCFG, TR.TrainConfig(batch_size=8, max_epochs=1, warmup_steps=0))
    with pytest.raises(TR.NonFiniteError):
        tr.fit(validate=lambda m, e: {"score": float("nan"), "tie": 0.0})


def test_max_steps_still_validates_partial_epoch(cond, tmp_path):
    tr = TR.Trainer(cond, TowerConfig(), HCFG, TR.TrainConfig(batch_size=8, max_epochs=3, max_steps=2,
                                                               warmup_steps=0), out_dir=tmp_path)
    res = tr.fit(validate=_Seq([(0.1, 0.1)] * 3))
    assert res["steps"] == 2 and len(res["val_history"]) == 1 and res["val_history"][0]["partial_epoch"]
    assert (tmp_path / "best.pt").exists()


# ============================================================================ checkpoint provenance
def test_checkpoint_records_condition_and_fingerprint(run, cond):
    ck = run["ck"]
    assert ck["condition_id"] == cond.condition_id
    assert ck["fingerprint"] == A.condition_fingerprint(cond)
    assert ck["num_stats"] == cond.num_stats and ck["tower_config"] == TowerConfig().to_dict()
    for k in ("vocabs", "video_features", "music_features", "model_events", "candidates", "num_stats"):
        assert ck["fingerprint"][k]


def test_load_model_rejects_other_condition_or_artifacts(P, run, cond):
    ckpt = run["out"] / "best.pt"
    with pytest.raises(ValueError, match="condition_id"):
        A.load_model(ckpt, cold_condition(P, [13]))
    # Same condition id but different feature table (e.g. scalers from another condition).
    vc = cond.video_catalog.copy()
    vc["video_duration_z"] = vc["video_duration_z"] + 0.5
    bad = D.TrainCondition.build(cond.name, cond.events, cond.vocabs, vc, cond.music_catalog, num_stats=cond.num_stats)
    with pytest.raises(ValueError, match="video_features"):
        A.load_model(ckpt, bad)
    # Same condition id but different scalers.
    ns = json.loads(json.dumps(cond.num_stats))
    ns["video_duration"]["mean"] += 1.0
    bad2 = D.TrainCondition.build(cond.name, cond.events, cond.vocabs, cond.video_catalog, cond.music_catalog,
                                  num_stats=ns)
    with pytest.raises(ValueError, match="num_stats"):
        A.load_model(ckpt, bad2)


def test_reloaded_model_equals_saved_state(run):
    sd = run["model"].state_dict()
    assert all(torch.equal(sd[k], run["ck"]["model"][k]) for k in sd)
    assert not run["model"].training


# ============================================================================ export
def test_export_covers_every_catalog_row_and_user(P, run):
    sc = X.ExportedScorer(run["out"] / "export")
    man = sc.manifest
    assert man["n_items"] == {"video": 6, "music": 4} and man["t_pred"] == {"val": fx.META["t_val_ms"],
                                                                         "test": fx.META["t_test_ms"]}
    assert np.array_equal(sc.catalog_ids["video"], P["catalogs"].video_ids)
    assert np.array_equal(sc.user_ids["test"], np.unique(P["events"].user_id))
    for t in TASKS:
        assert sc.I[t].shape[0] == (6 if t.startswith("video") else 4)
        assert np.allclose(np.linalg.norm(sc.I[t], axis=1), 1, atol=1e-5)
        for s in X.SPLITS:
            assert np.isfinite(sc.U[s][t]).all() and np.allclose(np.linalg.norm(sc.U[s][t], axis=1), 1, atol=1e-5)
    assert man["condition_id"] == run["ck"]["condition_id"]
    assert man["checkpoint_sha256"] == A.sha256_file(run["out"] / "best.pt")
    assert set(man["frame_ids"]) == {"/".join(k) for k in run["frames"]}


def test_export_matches_independent_model_forward(P, cond, run):
    """Exported vectors = a fresh eval-mode forward of the checkpoint (batch 1)."""
    sc = X.ExportedScorer(run["out"] / "export")
    m = run["model"]
    items = TR.ItemInputs.from_condition(cond)
    with I.inference_settings(1):
        enc, ie = I.encode_items(m, items.vf, items.mf)
        hidx, t = F.frozen_history_index(cond.events, "test", P["meta"])
        for u in sc.user_ids["test"]:
            hb = F.build_history_batch(hidx, [u], t, HCFG).to_torch()
            ue = m.user_embeddings(m.encode_users(torch.as_tensor(cond.vocabs.user.encode([u])), hb, enc))
            r = sc.rows([u], "test")[0]
            for task in TASKS:
                assert np.array_equal(ue[task][0].numpy(), sc.U["test"][task][r])
    for task in TASKS:
        assert np.array_equal(ie[task].numpy(), sc.I[task])


def test_exported_seen_mask_equals_frame_mask_csr(run):
    sc = X.ExportedScorer(run["out"] / "export")
    n = 0
    for task in TASKS:
        tgt = task.split("_")[0]
        f = run["frames"][("test", task, "masked")]
        for i, u in enumerate(f.users):
            assert np.array_equal(np.sort(sc.seen_rows(u, tgt)), np.sort(f.mask_idx[f.mask_indptr[i]:f.mask_indptr[i + 1]]))
            n += 1
    assert n > 0


def test_exported_scorer_rejects_tampering_and_wrong_condition(P, run, tmp_path):
    d = tmp_path / "exp"
    shutil.copytree(run["out"] / "export", d)
    with pytest.raises(ValueError, match="condition_id"):
        X.ExportedScorer(d, expect_condition_id=cold_condition(P, [13]).condition_id)
    U = np.load(d / "users_test_video_long_view.npy")
    U[0, 0] += 1e-3
    np.save(d / "users_test_video_long_view.npy", U)
    with pytest.raises(ValueError, match="sha256"):
        X.ExportedScorer(d)


def test_scorer_checks_prediction_time(run):
    sc = X.ExportedScorer(run["out"] / "export")
    with pytest.raises(ValueError):
        sc.score(sc.user_ids["test"][:1], "video", "long_view", EvalContext("test", fx.META["t_val_ms"], pd.DataFrame()))


# ============================================================================ batch independence
def test_canonical_user_encoding_is_batch_independent(P, cond, run):
    m = run["model"]
    items = TR.ItemInputs.from_condition(cond)
    users = np.unique(P["events"].user_id)
    hidx, t = F.frozen_history_index(cond.events, "test", P["meta"])
    with I.inference_settings(1):
        enc, _ = I.encode_items(m, items.vf, items.mf)
        a = I.encode_users(m, enc, hidx, t, users, cond.vocabs, HCFG)
        perm = users[::-1]
        b = I.encode_users(m, enc, hidx, t, perm, cond.vocabs, HCFG)
        c = I.encode_users(m, enc, hidx, t, users[1:2], cond.vocabs, HCFG)
    for task in TASKS:
        assert np.array_equal(a[task][::-1], b[task]) and np.array_equal(a[task][1:2], c[task])


def test_score_rows_is_row_independent():
    rng = np.random.default_rng(0)
    U = rng.normal(size=(37, 64)).astype(np.float32)
    It = rng.normal(size=(501, 64)).astype(np.float32)
    s = I.score_rows(U, It)
    assert np.array_equal(s[5:6], I.score_rows(U[5:6], It)) and np.array_equal(s[::-1], I.score_rows(U[::-1], It))
    assert s.dtype == np.float32


# ============================================================================ retrieve
def test_retrieve_matches_exported_scorer_and_mask(run):
    sc = X.ExportedScorer(run["out"] / "export")
    for task in TASKS:
        tgt, act = task.split("_", 1)
        n_items = sc.I[task].shape[0]
        for u in sc.user_ids["test"]:
            s = sc.score_users([u], tgt, act, "test")
            seen = sc.seen_rows(u, tgt)
            res = X.retrieve(sc, u, tgt, act, k=50)
            rows = np.searchsorted(sc.catalog_ids[tgt], [i for i, _ in res])
            assert len(res) == n_items - len(seen)                      # catalog < k: all unmasked items
            assert not np.isin(rows, seen).any()
            assert [v for _, v in res] == [float(x) for x in s[0, rows]]
            masked = s.copy()
            M.apply_mask(masked, np.array([0, len(seen)]), seen)
            top, valid = M.rank_topk(masked, 50)
            assert np.array_equal(rows, top[0][valid[0]])
            nm = X.retrieve(sc, u, tgt, act, k=50, mask_seen=False)
            assert len(nm) == n_items and set(np.searchsorted(sc.catalog_ids[tgt], [i for i, _ in nm])) >= set(seen)
            assert len(X.retrieve(sc, u, tgt, act, k=2)) == min(2, n_items - len(seen))


def test_retrieve_errors(run):
    sc = X.ExportedScorer(run["out"] / "export")
    with pytest.raises(KeyError):
        X.retrieve(sc, 999_999, "video", "long_view", 5)
    with pytest.raises(ValueError):
        X.retrieve(sc, int(sc.user_ids["test"][0]), "video", "click", 5)
    with pytest.raises(ValueError):
        X.retrieve(sc, int(sc.user_ids["test"][0]), "video", "long_view", 0)


def test_retrieve_tie_break_is_catalog_row_ascending(run, tmp_path):
    d = tmp_path / "exp"
    shutil.copytree(run["out"] / "export", d)
    It = np.load(d / "items_video_long_view.npy")
    It[:] = It[2]                                                      # every video ties exactly
    np.save(d / "items_video_long_view.npy", It)
    sc = X.ExportedScorer(d, verify_files=False)
    u = 3                                                              # cold user: nothing masked
    res = X.retrieve(sc, u, "video", "long_view", 6)
    assert [i for i, _ in res] == sc.catalog_ids["video"].tolist() and len({v for _, v in res}) == 1


# ============================================================================ parity (fixture)
def test_parity_checks_pass_bitwise_on_fixture(P, cond, run):
    model, _ = A.load_model(run["out"] / "best.pt", cond)
    r = PR.parity_checks(model, cond, P, run["out"] / "export", run["out"] / "best.pt", n_users=10,
                         frames=run["frames"], require_all_strata=False)
    assert r["ok"], r
    s = r["check1_2_scores_topk"]
    assert s["max_abs"] == 0.0 and not s["topk_mismatches"]
    assert all(v["bitwise_equal"] for v in s["tasks"].values())
    assert r["check3_retrieve"]["independent_metrics_max_abs_dev"] <= 1e-12
    assert all(v["identical"] for v in r["check4_metrics"]["frames"].values())
    assert r["users"]["strata"]["cold"] >= 1                             # u3


def test_parity_detects_a_perturbed_export(P, cond, run, tmp_path):
    d = tmp_path / "exp"
    shutil.copytree(run["out"] / "export", d)
    man = json.loads((d / "manifest.json").read_text())
    U = np.load(d / "users_test_video_long_view.npy")
    sc = X.ExportedScorer(d)
    users, _ = PR.select_parity_users(run["frames"], 10, 0, require_all_strata=False)
    victim = int(users[0])
    r = sc.rows([victim], "test")[0]
    U[r] = -U[r]                                                       # flips the victim's ranking
    np.save(d / "users_test_video_long_view.npy", U)
    man["files"]["users_test_video_long_view.npy"] = A.sha256_file(d / "users_test_video_long_view.npy")
    (d / "manifest.json").write_text(json.dumps(man))
    model, _ = A.load_model(run["out"] / "best.pt", cond)
    res = PR.parity_checks(model, cond, P, d, run["out"] / "best.pt", n_users=10, frames=run["frames"],
                           require_all_strata=False)
    assert not res["ok"]
    t = res["check1_2_scores_topk"]["tasks"]["test/video_long_view"]
    assert t["max_abs"] > 1e-5 and t["top50_nomask_mismatch_users"] >= 1
    assert res["check1_2_scores_topk"]["topk_mismatches"][0]["user_id"] == victim
    # Other tasks are untouched and still bitwise equal.
    assert res["check1_2_scores_topk"]["tasks"]["test/video_is_like"]["max_abs"] == 0.0
    assert res["check1_2_scores_topk"]["tasks"]["val/video_long_view"]["max_abs"] == 0.0


def test_direct_scorer_builds_its_own_embeddings(P, cond, run, tmp_path):
    """DirectScorer runs the towers itself (bitwise = an independent forward) and never reads
    export files: it works with the export directory gone, and follows the model's weights."""
    model, _ = A.load_model(run["out"] / "best.pt", cond)
    sc = X.ExportedScorer(run["out"] / "export")
    d = tmp_path / "moved"
    shutil.move(str(run["out"] / "export"), d)                       # export unreachable for Direct
    try:
        direct = I.DirectScorer(model, cond, P["meta"])
        users = sc.user_ids["test"]
        items = TR.ItemInputs.from_condition(cond)
        with I.inference_settings(1):
            enc, ie = I.encode_items(model, items.vf, items.mf)
            hidx, t = F.frozen_history_index(cond.events, "test", P["meta"])
            ue = I.encode_users(model, enc, hidx, t, users, cond.vocabs, HCFG)
        de = direct.user_embeddings(users, "test")
        for task in TASKS:
            Id = direct.I[task]
            assert isinstance(Id, np.ndarray) and Id.dtype == np.float32 and Id.flags["C_CONTIGUOUS"]
            assert np.array_equal(Id, ie[task].numpy())                 # catalog row order kept
            assert np.array_equal(de[task], ue[task])
        # A different model gives different Direct embeddings (Direct is not a cache of the export).
        m2, _ = A.load_model(run["out"] / "best.pt", cond)
        with torch.no_grad():
            for p in m2.parameters():
                p.add_(0.01)
        d2 = I.DirectScorer(m2, cond, P["meta"])
        assert not np.array_equal(d2.I["video_long_view"], direct.I["video_long_view"])
        assert not np.array_equal(d2.user_embeddings(users, "test")["video_long_view"], de["video_long_view"])
    finally:
        shutil.move(str(d), run["out"] / "export")


def test_direct_scorer_uses_canonical_score_rows(P, cond, run, monkeypatch):
    """Direct and Exported share the single canonical scoring function (no torch matmul)."""
    model, _ = A.load_model(run["out"] / "best.pt", cond)
    direct = I.DirectScorer(model, cond, P["meta"])
    ex = X.ExportedScorer(run["out"] / "export")
    users = ex.user_ids["test"]
    ctx = EvalContext("test", fx.META["t_test_ms"], pd.DataFrame({"time_ms": []}))
    U_direct = direct.user_embeddings(users, "test")    # towers run (and cache) before the spies
    calls = []
    real = I.score_rows

    def spy(U, It):
        calls.append((U.shape, It.shape))
        return real(U, It)

    monkeypatch.setattr(I, "score_rows", spy)
    monkeypatch.setattr(torch.Tensor, "__matmul__", lambda *a: pytest.fail("torch matmul used for scoring"))
    for task in TASKS:
        tgt, act = task.split("_", 1)
        s = direct.score(users, tgt, act, ctx)
        want = real(U_direct[task], direct.I[task])
        assert s.dtype == np.float32 and np.array_equal(s, want)
        assert np.array_equal(s, ex.score(users, tgt, act, ctx))
    assert len(calls) == 2 * len(TASKS)                                 # Direct + Exported, every task
    assert direct.score(users[:0], "video", "long_view", ctx).shape == (0, 6)


def test_parity_fails_on_tampered_item_matrix(P, cond, run, tmp_path):
    """Swapping two exported item rows (manifest sha re-signed) must fail checks 1, 2 and 4;
    Direct is unaffected because it computes its own item embeddings."""
    d = tmp_path / "exp"
    shutil.copytree(run["out"] / "export", d)
    man = json.loads((d / "manifest.json").read_text())
    f = "items_music_is_like.npy"
    It = np.load(d / f)
    It[[0, 1]] = It[[1, 0]]
    np.save(d / f, It)
    man["files"][f] = A.sha256_file(d / f)
    (d / "manifest.json").write_text(json.dumps(man))
    model, _ = A.load_model(run["out"] / "best.pt", cond)
    res = PR.parity_checks(model, cond, P, d, run["out"] / "best.pt", n_users=10, frames=run["frames"],
                           require_all_strata=False)
    assert not res["ok"]
    s = res["check1_2_scores_topk"]
    for split in ("val", "test"):
        assert s["tasks"][f"{split}/music_is_like"]["max_abs"] > 1e-5
        assert s["tasks"][f"{split}/music_is_like"]["top50_nomask_mismatch_users"] >= 1
        assert s["tasks"][f"{split}/video_long_view"]["max_abs"] == 0.0       # untouched task
    assert not res["check1_2_scores_topk"]["ok"]


def test_parity_refuses_export_from_another_checkpoint(P, cond, run, tmp_path):
    other = tmp_path / "other.pt"
    ck = torch.load(run["out"] / "best.pt", weights_only=False)
    ck["step"] += 1
    torch.save(ck, other)
    model, _ = A.load_model(other, cond)
    with pytest.raises(ValueError, match="different checkpoint"):
        PR.parity_checks(model, cond, P, run["out"] / "export", other, n_users=4, frames=run["frames"],
                         require_all_strata=False)


# ============================================================================ cold condition artifacts
def test_cold_condition_has_own_artifacts_and_is_not_mixed(P, run, tmp_path):
    H = [13]
    cc = cold_condition(P, H)
    assert cc.condition_id == D.condition_id("cold_test", H) != run["ck"]["condition_id"]
    assert 13 not in cc.vocabs.video.ids and not cc.events.video_id.isin(H).any()
    assert A.condition_fingerprint(cc)["vocabs"] != run["ck"]["fingerprint"]["vocabs"]
    tr = TR.Trainer(cc, TowerConfig(), HCFG, TR.TrainConfig(batch_size=8, max_epochs=1, warmup_steps=0),
                    out_dir=tmp_path)
    tr.fit()
    tr.save("best.pt")
    m, _ = A.load_model(tmp_path / "best.pt", cc)
    # The cold checkpoint cannot be loaded with the normal condition (and vice versa, above).
    normal = D.condition_from_fit("normal", fit_of())
    with pytest.raises(ValueError):
        A.load_model(tmp_path / "best.pt", normal)
    man = X.export_embeddings(m, cc, P, tmp_path / "export", tmp_path / "best.pt")
    sc = X.ExportedScorer(tmp_path / "export", expect_condition_id=cc.condition_id)
    with pytest.raises(ValueError):
        X.ExportedScorer(tmp_path / "export", expect_condition_id=normal.condition_id)
    # Evaluation frame / catalog are the unfiltered ones: H stays a catalog row and is scored.
    assert man["n_items"]["video"] == len(P["catalogs"].video_ids)
    r13 = int(P["catalogs"].video_row([13])[0])
    assert np.isfinite(sc.I["video_long_view"][r13]).all()
    # Cold history at T_test never contains H (model-side events are H-free).
    hidx, t = F.frozen_history_index(cc.events, "test", P["meta"])
    hb = F.build_history_batch(hidx, sc.user_ids["test"], t, HCFG)
    assert not (hb.video_row == r13).any()
    # Seen mask in the export is the evaluation-side (unfiltered) mask: u1 saw v13 before T_test.
    assert r13 in sc.seen_rows(1, "video")


def test_build_condition_spec(P):
    assert D.build_condition("normal", P).condition_id == D.condition_id("normal")
    with pytest.raises(ValueError):
        D.build_condition({"name": "cold", "exclude_video_ids": []}, P)
