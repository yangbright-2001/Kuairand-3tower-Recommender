"""Catalog-level definitions and cross-check failure handling for the audit."""

import numpy as np
import pytest

from kuairand_rec.audit import run, stats


def test_metadata_catalog_includes_ids_never_in_logs():
    # Metadata music ids: 0 (sentinel), 10, 11, 12, 13. Id 13 never occurs in any log row.
    meta = np.array([0, 10, 10, 11, 12, 13])  # one entry per metadata video (10 is shared)
    train = np.array([10, 10, 11, 0])
    train_pos = np.array([10])
    c = stats.build_catalogs(meta, train, train_pos)
    np.testing.assert_array_equal(c["metadata"], [0, 10, 11, 12, 13])
    np.testing.assert_array_equal(c["train_seen"], [0, 10, 11])
    np.testing.assert_array_equal(c["train_positive"], [10])
    # A log-derived "metadata" catalog would have been {0, 10, 11}: strictly smaller.
    assert 13 in c["metadata"] and 12 in c["metadata"]


def test_catalog_exclude_sentinel_applies_to_all_levels():
    meta = np.array([0, 10, 11])
    c = stats.build_catalogs(meta, np.array([0, 10]), np.array([0]), exclude=(0,))
    np.testing.assert_array_equal(c["metadata"], [10, 11])
    np.testing.assert_array_equal(c["train_seen"], [10])
    assert len(c["train_positive"]) == 0


def test_catalogs_are_nested():
    rng = np.random.default_rng(0)
    meta = np.arange(50)
    train = rng.choice(40, 200)
    train_pos = train[:30]
    c = stats.build_catalogs(meta, train, train_pos)
    assert np.isin(c["train_positive"], c["train_seen"]).all()
    assert np.isin(c["train_seen"], c["metadata"]).all()


def test_ooc_share():
    cat = np.array([1, 2, 3])
    assert stats.ooc_share(np.array([1, 4, 4, 2]), cat) == 0.5  # row-level, repeats count
    assert stats.ooc_share(np.array([], dtype=int), cat) is None


@pytest.mark.parametrize("items, code", [({"a": True, "b": True}, 0), ({"a": True, "b": False}, 1)])
def test_main_exit_code_reflects_checks(monkeypatch, tmp_path, items, code, capsys):
    monkeypatch.setattr(run, "run_audit", lambda raw, out: {"consistency_checks": items, "meta": {"runtime_s": 0}})
    assert run.main(["--out", str(tmp_path)]) == code
    if code:
        assert "FAILED" in capsys.readouterr().err


def test_checks_failed_list():
    ck = run.Checks()
    ck.add("ok", True)
    ck.add("bad", np.bool_(False))
    assert not ck.all_ok and ck.failed == ["bad"]
