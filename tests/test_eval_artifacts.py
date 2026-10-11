"""Evaluator checks on the real M2 artifacts (skipped if data/processed is not built)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuairand_rec import paths

pytestmark = pytest.mark.skipif(not (paths.PROCESSED_DIR / "events.parquet").exists(),
                                reason="processed artifacts not built")


@pytest.fixture(scope="module")
def real():
    from kuairand_rec.data.history import HistoryIndex
    from kuairand_rec.data.pipeline import load_processed
    d = load_processed()
    return d, HistoryIndex(d["events"])


class _Oracle:
    """+1 on the user's GT rows (looked up from the frame for the test only)."""
    name = "oracle"

    def __init__(self, f):
        self.f = f
        self.pos = {int(u): i for i, u in enumerate(f.users)}

    def score(self, users, target, action, ctx):
        s = np.zeros((len(users), self.f.n_items), np.float32)
        for j, u in enumerate(users):
            s[j, self.f.gt_rows(self.pos[int(u)])] = 1.0
        return s

    def signal(self, users, target, action, ctx):
        return np.ones(len(users), bool)

    def item_scorable(self, target, action):
        return np.ones(self.f.n_items, bool)


@pytest.mark.parametrize("target,action", [("video", "long_view"), ("music", "is_like")])
def test_oracle_and_independent_gt(real, target, action):
    from kuairand_rec.eval import frame as F
    from kuairand_rec.eval.evaluator import EvalContext, evaluate
    from kuairand_rec.eval.views import history_view, prediction_time
    d, h = real
    ev, meta = d["events"], d["meta"]
    fr = F.build_frames(ev, d["label_conflict_events"], d["catalogs"], meta, h, "test", target, action)
    f = fr["masked"]
    T = prediction_time(meta, "test")
    col = f"{target}_row"
    # Independent recomputation of the masked GT with plain pandas.
    pos = ev[(ev.time_ms >= T) & (ev[action] == 1) & (ev[col] >= 0)][["user_id", col]].drop_duplicates()
    seen = ev[(ev.time_ms < T) & (ev[col] >= 0)][["user_id", col]].drop_duplicates()
    g = pos.merge(seen, how="left", indicator=True)
    g = g[g["_merge"] == "left_only"]
    has_hist = set(ev.loc[ev.time_ms < T, "user_id"])
    g = g[g.user_id.isin(has_hist)]
    exp = g.groupby("user_id")[col].apply(lambda s: sorted(s.tolist())).to_dict()
    got = {int(u): f.gt_rows(i).tolist() for i, u in enumerate(f.users)}
    assert got == exp
    # Oracle: perfect scores -> Recall@50 = NDCG@50 = 1 for |GT| <= 50.
    pu = evaluate(_Oracle(f), f, EvalContext("test", T, history_view(ev, "test", meta)))
    small = pu.n_gt <= 50
    assert np.allclose(pu.loc[small, "recall@50"], 1) and np.allclose(pu.loc[small, "ndcg@50"], 1)
    assert (pu["ndcg@10"] <= 1 + 1e-9).all()
    # nomask: same users
    assert (fr["nomask"].users == f.users).all()


def test_results_files_consistent():
    out = paths.RESULTS_DIR / "baselines"
    if not (out / "metrics.csv").exists():
        pytest.skip("baselines not run")
    m = pd.read_csv(out / "metrics.csv")
    for _, g in m.groupby(["split", "target", "action", "kind"]):
        assert g.frame_id.nunique() == 1
    for k in (10, 20, 50):
        assert m[f"recall@{k}"].between(0, 1).all() and m[f"ndcg@{k}"].between(0, 1).all()
    assert (m["recall@20"] >= m["recall@10"] - 1e-12).all() and (m["recall@50"] >= m["recall@20"] - 1e-12).all()
