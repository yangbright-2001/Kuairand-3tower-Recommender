"""M4-4 readiness: formal U0/U1 configs, training-only benchmark, selection-vs-export report."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

import m4_fixtures as fx
from kuairand_rec import paths
from kuairand_rec.models import benchmark as B
from kuairand_rec.models import data as D
from kuairand_rec.models import features as F
from kuairand_rec.models import train as TR
from kuairand_rec.models.towers import TASKS, TowerConfig

CFG = paths.PROJECT_ROOT / "configs"


def _load(name):
    return yaml.safe_load((CFG / name).read_text())


def test_formal_configs_parse_and_follow_d1():
    for name in ("threetower_u0.yaml", "threetower_u1.yaml"):
        c = _load(name)
        tc = TR.TrainConfig.from_dict(c["train"])
        TowerConfig.from_dict(c["model"])
        F.HistoryConfig.from_dict(c["history"])
        assert c["require_clean_git"] is True and c["condition"] == "normal" and c["device"] == "cuda"
        assert (tc.tau, tc.lr, tc.warmup_steps, tc.weight_decay, tc.clip_grad_norm) == (0.05, 1e-3, 300, 0.01, 1.0)
        assert (tc.batch_size, tc.max_epochs, tc.patience, tc.like_oversample, tc.seed, tc.max_steps) == \
            (1024, 30, 3, 1, 0, None)
        assert TowerConfig.from_dict(c["model"]) == TowerConfig()          # D1-1 defaults, ablation flags on
        assert F.HistoryConfig.from_dict(c["history"]) == F.HistoryConfig()
        assert c["eval"]["threads"] == 1 and c["eval"]["parity_users"] >= 2048


def test_u1_differs_from_u0_only_in_user_metadata():
    u0, u1 = _load("threetower_u0.yaml"), _load("threetower_u1.yaml")
    diff = {k for k in set(u0) | set(u1) if u0.get(k) != u1.get(k)}
    assert diff == {"exp", "user_features", "u1_min_count"}
    assert u0["user_features"] == "u0" and u1["user_features"] == "u1" and u1["u1_min_count"] == 5


def _trainer(tmp_path=None, **tc):
    cond = D.condition_from_fit("normal", fx.make())
    return TR.Trainer(cond, TowerConfig(), F.HistoryConfig(),
                      TR.TrainConfig(batch_size=16, warmup_steps=0, **tc), out_dir=tmp_path)


def test_benchmark_reuses_trainer_and_reports(tmp_path):
    tr = _trainer(tmp_path)
    before = {k: v.clone() for k, v in tr.model.state_dict().items()}
    r = B.benchmark_steps(tr, steps=12, skip=2)                   # 80 samples / 16 -> crosses an epoch
    assert r["steps"] == 12 and r["timed_steps"] == 10 and tr.step == 12
    assert r["steps_per_epoch"] == 5 and r["epoch_s_estimate"] == pytest.approx(r["step_s"]["mean"] * 5)
    assert r["loss"]["all_finite"] and r["grad_norm_pre_clip"]["all_finite"]
    assert 0 <= r["data_share"] <= 1 and r["steps_per_s"] > 0 and len(r["per_step"]) == 12
    assert any(not np.array_equal(before[k], v) for k, v in tr.model.state_dict().items())   # real updates
    assert not (tmp_path / "best.pt").exists() and not (tmp_path / "export").exists()


def test_benchmark_steps_match_normal_training(tmp_path):
    """Same seed -> the benchmark's losses are exactly the Trainer.fit losses (same loop body)."""
    a = _trainer(seed=1, max_epochs=1, max_steps=5)
    a.fit()
    b = _trainer(seed=1)
    r = B.benchmark_steps(b, steps=5, skip=0)
    assert [x["loss"] for x in a.log] == [x["loss"] for x in r["per_step"]]


def _summary(vals):
    return pd.DataFrame([{"split": "val", "kind": "masked", "target": t.split("_", 1)[0],
                          "action": t.split("_", 1)[1], "recall@20": vals[t][0], "ndcg@20": vals[t][1]} for t in TASKS])


def _hist(epoch, score_by_task, improved):
    h = {"epoch": epoch, "step": 10 * (epoch + 1), "improved": improved}
    for t, (r, n) in score_by_task.items():
        h[f"{t}/recall@20"], h[f"{t}/ndcg@20"] = r, n
    h["score"] = float(np.mean([v[0] for v in score_by_task.values()]))
    h["tie"] = float(np.mean([v[1] for v in score_by_task.values()]))
    return h


def test_selection_vs_export_report():
    best = {t: (0.10, 0.05) for t in TASKS}
    far = {t: (0.08, 0.04) for t in TASKS}
    sel = {"best_epoch": 1, "val_history": [_hist(0, far, True), _hist(1, best, True)]}
    exp = {t: (0.10, 0.05 + 1e-7) for t in TASKS}
    r = TR.compare_selection_with_export(sel, _summary(exp))
    assert r["score"]["abs_diff"] == 0 and r["tie"]["abs_diff"] == pytest.approx(1e-7)
    assert not r["selection_may_be_sensitive"] and r["max_abs_diff"] == pytest.approx(1e-7)
    # A later epoch tied on the score and within the observed tie-break difference -> flagged.
    near = {t: (0.10, 0.05 - 5e-8) for t in TASKS}
    sel["val_history"].append(_hist(2, near, False))
    r = TR.compare_selection_with_export(sel, _summary(exp))
    assert r["selection_may_be_sensitive"] and [e["epoch"] for e in r["near_tie_epochs"]] == [2]
    # A score difference also defines the sensitivity window.
    exp2 = {t: (0.10 + 1e-4, 0.05) for t in TASKS}
    sel2 = {"best_epoch": 1, "val_history": [_hist(0, {t: (0.10 - 1e-4, 0.05) for t in TASKS}, True),
                                             _hist(1, best, True)]}
    assert TR.compare_selection_with_export(sel2, _summary(exp2))["selection_may_be_sensitive"]


def test_benchmark_cli_refuses_dirty_worktree_for_formal_config(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(B, "assert_clean_worktree", lambda: calls.append(1) or (_ for _ in ()).throw(RuntimeError("dirty")))
    with pytest.raises(RuntimeError, match="dirty"):
        B.run(_load("threetower_u0.yaml"), 1, str(tmp_path / "b"))
    assert calls == [1]
