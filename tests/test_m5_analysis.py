"""M5 main-comparison analysis on the fixture: frame parity of every per-user input, history
buckets, paired bootstrap wiring, three-tower provenance, and ablation configs through the
unchanged M4 runner."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import m4_fixtures as fx
import m5_fixtures as m5
from kuairand_rec import paths
from kuairand_rec.eval.frame import history_bucket
from kuairand_rec.experiments import analysis as AN
from kuairand_rec.experiments import buckets as BK
from kuairand_rec.experiments import sources as S
from kuairand_rec.models import data as D
from kuairand_rec.models.evaluation import build_eval_frames
from kuairand_rec.models.towers import TASKS

CFG = paths.PROJECT_ROOT / "configs"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    P = m5.make_P()
    frames = build_eval_frames(P, frames_summary=None)
    tmp = tmp_path_factory.mktemp("m5main")
    mp = pytest.MonkeyPatch()
    try:
        bdir = m5.write_baselines(P, frames, tmp / "baselines")
        u0 = m5.run_m4(m5.base_cfg("u0", tmp), P, mp, tmp / "runs" / "u0")
        nh = m5.run_m4(m5.base_cfg("abl_nohist", tmp, model={"use_history": False}), P, mp, tmp / "runs" / "nohist")
        nm = m5.run_m4(m5.base_cfg("abl_nomusic", tmp, model={"use_music": False}), P, mp, tmp / "runs" / "nomusic")
    finally:
        mp.undo()
    normal = D.condition_id("normal")
    cfg = {"output_dir": str(tmp / "out"), "split": "test", "frame_kinds": ["masked"], "tasks": list(TASKS),
           "metrics": list(S.METRICS), "bootstrap": {"n_boot": 200, "seed": 0, "ci": 0.95},
           "models": {"pop": {"kind": "baseline", "dir": str(bdir), "m3_name": "pop"},
                      "als": {"kind": "baseline", "dir": str(bdir), "m3_name": "als"},
                      "als_like_aux": {"kind": "baseline", "dir": str(bdir), "m3_name": "als_like_aux",
                                       "tasks": ["video_is_like", "music_is_like"]},
                      "u0": {"kind": "threetower", "run_dir": str(u0), "expect_condition_id": normal},
                      "u1": {"kind": "threetower", "run_dir": None},
                      "abl_nohist": {"kind": "threetower", "run_dir": str(nh), "expect_condition_id": normal},
                      "abl_nomusic": {"kind": "threetower", "run_dir": str(nm), "expect_condition_id": normal}},
           "comparisons": [["u0", "pop"], ["u0", "als"], ["u0", "als_like_aux"], ["u1", "u0"],
                           ["abl_nohist", "u0"], ["abl_nomusic", "u0"]]}
    r = AN.run(cfg, P=P, frames_summary=None)
    return {"P": P, "frames": frames, "cfg": cfg, "r": r, "bdir": bdir, "u0": u0, "nh": nh, "nm": nm, "tmp": tmp}


# ============================================================================ frame parity
def test_every_input_matches_its_frame(world):
    meta = world["r"]["meta"]
    assert meta["pending_models"] == ["u1"]
    for key, v in meta["inputs"].items():
        assert v["metrics_csv_max_abs_dev"] <= 1e-12 and len(v["sha256"]) == 64, key
    m = world["r"]["models"]
    # One frame per (task, kind) across all models; masked and cold have distinct populations.
    assert (m.groupby(["task", "frame_kind"]).frame_id.nunique() == 1).all()
    for (task, kind), g in m.groupby(["task", "frame_kind"]):
        f = world["frames"][("test", task, kind)]
        assert g.frame_id.iloc[0] == f.frame_id
        assert (g[g.stratum == "all"].n_users == len(f.users)).all()


def test_point_estimates_equal_reported_metrics(world):
    m = world["r"]["models"]
    m = m[(m.stratum == "all") & (m.frame_kind == "masked")]
    for name, src in (("u0", world["u0"]), ("abl_nohist", world["nh"])):
        rep = pd.read_csv(src / "metrics.csv")
        for _, r in m[m.model == name].iterrows():
            t, a = r.task.split("_", 1)
            want = rep[(rep.split == "test") & (rep.kind == "masked") & (rep.target == t) & (rep.action == a)][r.metric]
            assert r["mean"] == pytest.approx(float(want.iloc[0]), abs=1e-15)
    pop = pd.read_csv(world["bdir"] / "metrics.csv")
    for _, r in m[m.model == "pop"].iterrows():
        t, a = r.task.split("_", 1)
        want = pop[(pop.model == "pop") & (pop.split == "test") & (pop.kind == "masked") & (pop.target == t) & (pop.action == a)]
        assert r["mean"] == pytest.approx(float(want[r.metric].iloc[0]), abs=1e-15)


def test_rejects_per_user_tables_from_another_frame(world, tmp_path):
    f = world["frames"][("test", "video_long_view", "masked")]
    src = S.ModelSource("u0", "threetower", world["u0"])
    df, _ = S.load_per_user(src, f)
    bad_dir = tmp_path / "run"
    shutil.copytree(world["u0"], bad_dir)
    p = bad_dir / "per_user" / "test_video_long_view_masked.parquet"
    cases = {
        "frame_id": df.assign(frame_id="x" * 64),
        "users": df.iloc[::-1].reset_index(drop=True),
        "n_gt": df.assign(n_gt=df.n_gt + 1),
        "outside": df.assign(**{"recall@20": 1.5}),
    }
    for want, bad in cases.items():
        bad.to_parquet(p, index=False)
        with pytest.raises(ValueError, match=want):
            S.load_per_user(S.ModelSource("u0", "threetower", bad_dir), f)
    # A per-user table whose mean differs from the run's own metrics.csv is refused.
    df.assign(**{"recall@20": df["recall@20"] * 0.5}).to_parquet(p, index=False)
    d2, _ = S.load_per_user(S.ModelSource("u0", "threetower", bad_dir), f)
    with pytest.raises(ValueError, match="metrics.csv"):
        S.check_summary_consistency(S.ModelSource("u0", "threetower", bad_dir), d2, f)


def test_threetower_provenance_checks(world, tmp_path):
    from kuairand_rec.models import artifacts as A
    P = world["P"]
    fp = A.data_fingerprint(P["events"], P["meta"])
    prov = S.verify_threetower_run(S.ModelSource("u0", "threetower", world["u0"],
                                                 expect_condition_id=D.condition_id("normal")), fp)
    assert prov["parity_ok"] and prov["condition_id"] == D.condition_id("normal")
    with pytest.raises(ValueError, match="expected"):
        S.verify_threetower_run(S.ModelSource("u0", "threetower", world["u0"], expect_condition_id="deadbeef"), fp)
    d = tmp_path / "run"
    shutil.copytree(world["u0"], d)
    sel = json.loads((d / "selection.json").read_text())
    (d / "selection.json").write_text(json.dumps({**sel, "test_used_for_selection": True}))
    with pytest.raises(ValueError, match="test_used_for_selection"):
        S.verify_threetower_run(S.ModelSource("u0", "threetower", d), fp)
    (d / "selection.json").write_text(json.dumps(sel))
    par = json.loads((d / "parity.json").read_text())
    (d / "parity.json").write_text(json.dumps({**par, "ok": False}))
    with pytest.raises(ValueError, match="parity"):
        S.verify_threetower_run(S.ModelSource("u0", "threetower", d), fp)
    (d / "parity.json").write_text(json.dumps(par))
    ev = P["events"].copy()
    ev.loc[ev.index[0], "long_view"] = 1 - ev.loc[ev.index[0], "long_view"]
    with pytest.raises(ValueError, match="fingerprint"):
        S.verify_threetower_run(S.ModelSource("u0", "threetower", d), A.data_fingerprint(ev, P["meta"]))


# ============================================================================ buckets
def test_history_buckets_partition_and_match_frame_definition(world):
    for task in TASKS:
        f = world["frames"][("test", task, "masked")]
        b = BK.frame_buckets(f)
        assert np.array_equal(b, history_bucket(np.minimum(f.hist_len_full, 100)))
        assert set(b) <= set(BK.BUCKETS)
        c = BK.bucket_counts(f)
        assert c.n_users.sum() == len(f.users) and c.gt_pairs.sum() == len(f.gt_idx)
    # Hand-computed (video_long_view): u2 has 130 train events -> capped 100 -> ">20";
    # u4 has 2 pre-T_test events (train v13, val v11) -> "1-5"; u5 has 1 -> "1-5".
    f = world["frames"][("test", "video_long_view", "masked")]
    assert dict(zip(f.users.tolist(), BK.frame_buckets(f).tolist())) == {2: ">20", 4: "1-5", 5: "1-5"}
    assert f.hist_len_full.tolist() == [130, 2, 1] and f.hist_len_capped.tolist() == [100, 2, 1]


def test_bucket_bootstrap_rows_reconcile_with_overall(world):
    m = world["r"]["models"]
    m = m[m.frame_kind == "masked"]
    for (task, model, metric), g in m.groupby(["task", "model", "metric"]):
        allr = g[g.stratum == "all"].iloc[0]
        bk = g[g.stratum != "all"]
        assert bk.n_users.sum() == allr.n_users
        assert (bk["mean"] * bk.n_users).sum() / allr.n_users == pytest.approx(allr["mean"], abs=1e-12)
    for k, v in world["r"]["meta"]["checks"].items():
        if k.startswith("masked/"):
            assert v["bucket_partition_max_dev"] <= 1e-12


def test_buckets_reject_history_free_users(world):
    with pytest.raises(ValueError, match="without history"):
        BK.frame_buckets(world["frames"][("test", "video_long_view", "cold")])


# ============================================================================ bootstrap wiring
def test_diffs_cover_configured_pairs_and_skip_missing(world):
    d = world["r"]["diffs"]
    masked = d[(d.frame_kind == "masked") & (d.stratum == "all")]
    pairs = set(zip(masked.model_a, masked.model_b))
    assert ("u0", "pop") in pairs and ("abl_nohist", "u0") in pairs and ("abl_nomusic", "u0") in pairs
    assert not any("u1" in p for p in pairs)                                      # pending model
    aux = masked[masked.model_b == "als_like_aux"]
    assert set(aux.task) == {"video_is_like", "music_is_like"}                    # task-restricted model
    # Cold-user frame: only models that were evaluated there (pop + three-tower; never ALS).
    cold = world["r"]["models"][world["r"]["models"].frame_kind == "cold"]
    assert "als" not in set(cold.model) and {"pop", "u0"} <= set(cold.model)


def test_analysis_is_deterministic(world, tmp_path):
    cfg = dict(world["cfg"], output_dir=str(tmp_path / "again"))
    r2 = AN.run(cfg, P=world["P"], frames_summary=None)
    pd.testing.assert_frame_equal(world["r"]["models"], r2["models"])
    pd.testing.assert_frame_equal(world["r"]["diffs"], r2["diffs"])
    # Adding / removing a model does not change any other model's bootstrap rows (per-population stream).
    cfg3 = json.loads(json.dumps(cfg))
    del cfg3["models"]["abl_nomusic"]
    cfg3["output_dir"] = str(tmp_path / "fewer")
    r3 = AN.run(cfg3, P=world["P"], frames_summary=None)
    a = world["r"]["models"]
    a = a[a.model != "abl_nomusic"].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, r3["models"].reset_index(drop=True))


def test_outputs_written(world):
    out = Path(world["cfg"]["output_dir"])
    for f in ("bootstrap_models.csv", "bootstrap_diffs.csv", "bucket_counts.csv", "analysis_meta.json", "MAIN_ANALYSIS.md"):
        assert (out / f).exists()
    meta = json.loads((out / "analysis_meta.json").read_text())
    us = meta["user_sets"]["test/video_long_view/masked"]
    ids = pd.read_csv(out / us["file"]).user_id.to_numpy()
    assert S.user_set_id(ids) == us["user_ids_sha256"] and len(ids) == us["n_users"]
    assert meta["n_boot"] == 200 and meta["seed"] == 0 and "recall@K" in meta["metric_definitions"]


# ============================================================================ ablation runs (M4 runner)
def test_ablation_checkpoints_carry_flags(world):
    import torch
    nh = torch.load(world["nh"] / "best.pt", weights_only=False)
    nm = torch.load(world["nm"] / "best.pt", weights_only=False)
    u0 = torch.load(world["u0"] / "best.pt", weights_only=False)
    assert nh["tower_config"]["use_history"] is False and nh["tower_config"]["use_music"] is True
    assert nm["tower_config"]["use_music"] is False and nm["tower_config"]["use_history"] is True
    assert u0["tower_config"]["use_history"] and u0["tower_config"]["use_music"]
    assert nh["condition_id"] == nm["condition_id"] == u0["condition_id"] == D.condition_id("normal")
    assert not any(k.startswith("user_tower.history") for k in nh["model"])


def test_no_history_ablation_ignores_history():
    """use_history=False: user embeddings do not change when the user's history changes."""
    import torch
    from kuairand_rec.models import features as F
    from kuairand_rec.models.inference import encode_items, encode_users, inference_settings
    from kuairand_rec.models.towers import ThreeTowerModel, TowerConfig
    from kuairand_rec.models.train import ItemInputs

    P = m5.make_P()
    cond = D.normal_condition(P)
    torch.manual_seed(0)
    m = ThreeTowerModel(F.FeatureSizes.from_vocabs(cond.vocabs), TowerConfig(use_history=False), F.HistoryConfig())
    items = ItemInputs.from_condition(cond)
    users = np.array([1, 2, 4])
    ev2 = cond.events[~((cond.events.user_id == 2) & (cond.events.time_ms > 30 * fx.MIN))]
    with inference_settings(1):
        enc, _ = encode_items(m, items.vf, items.mf)
        a = encode_users(m, enc, F.frozen_history_index(cond.events, "test", P["meta"])[0], P["meta"]["t_test_ms"],
                         users, cond.vocabs, F.HistoryConfig())
        b = encode_users(m, enc, F.frozen_history_index(ev2, "test", P["meta"])[0], P["meta"]["t_test_ms"],
                         users, cond.vocabs, F.HistoryConfig())
    for t in TASKS:
        assert np.array_equal(a[t], b[t])


def _load(name):
    return yaml.safe_load((CFG / name).read_text())


def test_m5_configs_differ_from_u0_only_where_intended():
    u0 = _load("threetower_u0.yaml")
    diff = lambda c: {k for k in set(u0) | set(c) if u0.get(k) != c.get(k)}
    nh, nm, cold = _load("threetower_abl_nohist.yaml"), _load("threetower_abl_nomusic.yaml"), _load("threetower_cold_u0.yaml")
    assert diff(nh) == {"exp", "model"} and nh["model"] == {"use_history": False}
    assert diff(nm) == {"exp", "model"} and nm["model"] == {"use_music": False}
    assert diff(cold) == {"exp", "condition"}
    cv = _load("cold_video.yaml")
    assert cold["condition"]["name"] == cv["condition_name"]
    assert cold["condition"]["exclude_video_ids_file"] == cv["h_file"]
    assert cv["selection"] == {"fraction": 0.1, "n_strata": 10, "seed": 0, "popularity": "train_events"}
    for c in (nh, nm, cold):
        assert c["require_clean_git"] is True and c["device"] == "cuda" and c["train"] == u0["train"]
    a = _load("m5_analysis.yaml")
    assert a["bootstrap"] == {"n_boot": 1000, "seed": 0, "ci": 0.95} and a["split"] == "test"
    assert set(a["tasks"]) == set(TASKS) and a["frame_kinds"] == ["masked"]
