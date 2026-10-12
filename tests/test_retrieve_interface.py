"""`retrieve(user_id, target, action, k)` wrapper (models/retrieve.py): equal to export.retrieve on a
fixture export, and on the formal U0 export it reproduces the reported per-user test metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import test_m4_export as M4
from kuairand_rec import paths
from kuairand_rec.models import artifacts as A
from kuairand_rec.models import evaluation as E
from kuairand_rec.models import export as X
from kuairand_rec.models import train as TR
from kuairand_rec.models.retrieve import Retriever, load_retriever, main
from kuairand_rec.models.towers import TASKS, TowerConfig

FORMAL_U0 = paths.PROJECT_ROOT / "artifacts" / "u0" / "9218a76_seed0_job27"


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    P = M4.make_P()
    cond = M4.D.condition_from_fit("normal", M4.fit_of())
    out = tmp_path_factory.mktemp("rt")
    tr = TR.Trainer(cond, TowerConfig(), M4.HCFG, TR.TrainConfig(batch_size=16, max_epochs=1, warmup_steps=0, seed=0),
                    out_dir=out)
    tr.fit(validate=E.Validator(P, cond, frames_summary=None, user_batch_size=2))
    model, _ = A.load_model(out / "best.pt", cond)
    X.export_embeddings(model, cond, P, out / "export", out / "best.pt", {"exp": "rt"})
    return out / "export"


def test_wrapper_has_assignment_signature_and_equals_export_retrieve(export_dir):
    sc = X.ExportedScorer(export_dir)
    retrieve = load_retriever(export_dir)
    for u in sc.user_ids["test"].tolist():
        for task in TASKS:
            tgt, act = task.split("_", 1)
            for k in (1, 3, 50):
                assert retrieve(u, tgt, act, k) == X.retrieve(sc, u, tgt, act, k)
                assert retrieve(user_id=u, target=tgt, action=act, k=k) == X.retrieve(sc, u, tgt, act, k)
            assert Retriever(export_dir).retrieve(u, tgt, act, 50, mask_seen=False) == \
                X.retrieve(sc, u, tgt, act, 50, mask_seen=False)


def test_wrapper_errors_and_condition_check(export_dir):
    r = Retriever(export_dir)
    with pytest.raises(KeyError):
        r.retrieve(999_999, "video", "long_view", 5)
    with pytest.raises(ValueError):
        r.retrieve(int(r.scorer.user_ids["test"][0]), "video", "click", 5)
    with pytest.raises(ValueError):
        Retriever(export_dir, expect_condition_id="not-this-condition")


def test_cli_prints_json(export_dir, capsys):
    u = int(X.ExportedScorer(export_dir).user_ids["test"][0])
    assert main(["--export_dir", str(export_dir), "--user_id", str(u), "--target", "music", "--action", "is_like",
                 "--k", "2"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["user_id"] == u and len(out["items"]) <= 2 and out["mask_seen"] is True


@pytest.mark.skipif(not (FORMAL_U0 / "export" / "manifest.json").exists(), reason="formal U0 artifacts not present")
def test_formal_u0_retrieve_reproduces_reported_test_metrics():
    """Top-20 from retrieve() vs frame GT == the formal per-user R@20 / N@20 (video long view, 300 users)."""
    r = Retriever(FORMAL_U0 / "export", expect_condition_id="133765eb12b75700")
    fr = pd.read_parquet(paths.RESULTS_DIR / "baselines" / "frames" / "test_video_long_view_masked.parquet")
    pu = pd.read_parquet(FORMAL_U0 / "per_user" / "test_video_long_view_masked.parquet")
    assert np.array_equal(fr.user_id.to_numpy(), pu.user_id.to_numpy())
    vids = r.scorer.catalog_ids["video"]
    disc = 1.0 / np.log2(np.arange(2, 22))
    for i in np.random.default_rng(0).choice(len(fr), 300, replace=False):
        u, gt = int(fr.user_id.iloc[i]), set(np.asarray(fr.gt_rows.iloc[i]).tolist())
        rows = np.searchsorted(vids, [v for v, _ in r.retrieve(u, "video", "long_view", 20)])
        rel = np.array([x in gt for x in rows])
        assert rel.sum() / len(gt) == pytest.approx(pu["recall@20"].iloc[i], abs=1e-12)
        assert (rel * disc).sum() / disc[:min(len(gt), 20)].sum() == pytest.approx(pu["ndcg@20"].iloc[i], abs=1e-12)
