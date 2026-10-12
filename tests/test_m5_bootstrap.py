"""M5 paired user bootstrap: correctness against a naive implementation and determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuairand_rec.experiments import bootstrap as B


def _pu(n=200, seed=0, models=("a", "b", "c")):
    rng = np.random.default_rng(seed)
    users = np.arange(1000, 1000 + n)
    out = {}
    for j, m in enumerate(models):
        r = np.clip(rng.beta(1 + j, 5, size=n), 0, 1)
        out[m] = pd.DataFrame({"user_id": users, "recall@20": r, "ndcg@20": r * rng.uniform(0.3, 1, n)})
    return out


def test_matches_naive_index_resampling():
    pu = _pu()
    res = B.paired_bootstrap(pu, ["recall@20"], 300, 7, "pop/all")
    for m in pu:
        naive = B.naive_bootstrap_means(pu[m]["recall@20"].to_numpy(), 300, 7, "pop/all")
        assert np.allclose(res.boot[:, res.col(m, "recall@20")], naive, rtol=0, atol=1e-12)
    # Point estimate = the plain equal-weight user mean (= evaluator.summarize).
    assert res.point[res.col("a", "recall@20")] == pytest.approx(pu["a"]["recall@20"].mean(), abs=1e-15)


def test_deterministic_and_independent_of_model_set_and_order():
    pu = _pu()
    r1 = B.paired_bootstrap(pu, ["recall@20", "ndcg@20"], 250, 0, "k")
    r2 = B.paired_bootstrap(pu, ["recall@20", "ndcg@20"], 250, 0, "k")
    assert np.array_equal(r1.boot, r2.boot)
    # Re-ordered / reduced model set and metric order: identical replicates for each column.
    r3 = B.paired_bootstrap({"c": pu["c"], "a": pu["a"]}, ["ndcg@20", "recall@20"], 250, 0, "k")
    for m in ("a", "c"):
        for k in ("recall@20", "ndcg@20"):
            assert np.array_equal(r1.boot[:, r1.col(m, k)], r3.boot[:, r3.col(m, k)])
    # Different seed or population key -> different stream.
    assert not np.array_equal(B.paired_bootstrap(pu, ["recall@20"], 250, 1, "k").boot,
                              B.paired_bootstrap(pu, ["recall@20"], 250, 0, "k").boot[:, ::2])
    assert not np.array_equal(B.paired_bootstrap(pu, ["recall@20"], 50, 0, "k2").boot,
                              B.paired_bootstrap(pu, ["recall@20"], 50, 0, "k").boot)


def test_chunking_does_not_change_replicates(monkeypatch):
    pu = _pu(n=50)
    a = B.paired_bootstrap(pu, ["recall@20"], 300, 3, "k").boot
    monkeypatch.setattr(B, "_CHUNK", 7)
    b = B.paired_bootstrap(pu, ["recall@20"], 300, 3, "k").boot
    assert np.array_equal(a, b)
    # A longer run extends the shorter one (prefix property of the stream).
    c = B.paired_bootstrap(pu, ["recall@20"], 400, 3, "k").boot
    assert np.array_equal(c[:300], a)


def test_paired_difference_equals_bootstrap_of_per_user_difference():
    pu = _pu()
    res = B.paired_bootstrap(pu, ["recall@20"], 400, 5, "k")
    d = B.diff_rows(res, [("b", "a")], ["recall@20"])[0]
    diff_user = pu["b"]["recall@20"].to_numpy() - pu["a"]["recall@20"].to_numpy()
    naive = B.naive_bootstrap_means(diff_user, 400, 5, "k")
    lo, hi = np.quantile(naive, [0.025, 0.975])
    assert d["ci_lo"] == pytest.approx(lo, abs=1e-12) and d["ci_hi"] == pytest.approx(hi, abs=1e-12)
    assert d["diff"] == pytest.approx(diff_user.mean(), abs=1e-15)
    # Pairing matters: the paired SE is far smaller than the unpaired one when models correlate.
    pu2 = {"a": pu["a"], "a2": pu["a"].assign(**{"recall@20": pu["a"]["recall@20"] + 0.01})}
    r2 = B.paired_bootstrap(pu2, ["recall@20"], 200, 0, "k")
    d2 = B.diff_rows(r2, [("a2", "a")], ["recall@20"])[0]
    assert d2["ci_lo"] == pytest.approx(0.01) and d2["ci_hi"] == pytest.approx(0.01) and d2["ci_excludes_0"]


def test_ci_brackets_mean_and_degenerate_cases():
    pu = _pu()
    res = B.paired_bootstrap(pu, ["recall@20"], 500, 0, "k")
    for r in B.model_rows(res):
        assert r["ci_lo"] <= r["mean"] <= r["ci_hi"] and r["boot_se"] > 0
    const = {"m": pd.DataFrame({"user_id": [1, 2, 3], "recall@20": [0.5, 0.5, 0.5]})}
    r = B.model_rows(B.paired_bootstrap(const, ["recall@20"], 100, 0, "k"))[0]
    assert r["ci_lo"] == r["ci_hi"] == r["mean"] == 0.5


def test_rejects_misaligned_or_bad_inputs():
    pu = _pu(n=20)
    bad = {"a": pu["a"], "b": pu["b"].iloc[::-1].reset_index(drop=True)}
    with pytest.raises(ValueError, match="aligned"):
        B.paired_bootstrap(bad, ["recall@20"], 10, 0, "k")
    nan = {"a": pu["a"].assign(**{"recall@20": np.nan})}
    with pytest.raises(ValueError, match="non-finite"):
        B.paired_bootstrap(nan, ["recall@20"], 10, 0, "k")
    with pytest.raises(ValueError):
        B.paired_bootstrap({"a": pu["a"].iloc[:0]}, ["recall@20"], 10, 0, "k")


def test_coverage_of_known_mean_is_close_to_nominal():
    """Sanity of the percentile CI: ~95% coverage of a known Bernoulli mean (n = 400 users)."""
    rng = np.random.default_rng(123)
    hit = 0
    trials = 200
    for t in range(trials):
        x = (rng.random(400) < 0.3).astype(float)
        res = B.paired_bootstrap({"m": pd.DataFrame({"user_id": np.arange(400), "r": x})}, ["r"], 400, t, "k")
        r = B.model_rows(res)[0]
        hit += r["ci_lo"] <= 0.3 <= r["ci_hi"]
    assert 0.88 <= hit / trials <= 0.99
