"""Invariants on the real M2 artifacts. Skipped unless the pipeline has been run."""

from __future__ import annotations

import numpy as np
import pytest

from kuairand_rec import paths
from kuairand_rec.data import catalog as catm
from kuairand_rec.data.history import HistoryIndex, csr_row
from kuairand_rec.data.pipeline import load_processed, prediction_time

pytestmark = pytest.mark.skipif(not (paths.PROCESSED_DIR / "meta.json").exists(),
                                reason="run `python -m kuairand_rec.data.pipeline` first")


@pytest.fixture(scope="module")
def P():
    return load_processed()


def test_split_and_keys(P):
    ev, m = P["events"], P["meta"]
    assert not ev.duplicated(["user_id", "video_id", "time_ms"]).any()
    t = ev.time_ms.to_numpy()
    s = ev.split.to_numpy()
    assert (t[s == 0] < m["t_val_ms"]).all()
    assert ((t[s == 1] >= m["t_val_ms"]) & (t[s == 1] < m["t_test_ms"])).all()
    assert (t[s == 2] >= m["t_test_ms"]).all()


def test_vocab_roundtrip_and_train_only(P):
    ev, voc = P["events"], P["vocabs"]
    assert (voc.user.encode(ev.user_id.to_numpy()) == ev.user_idx.to_numpy()).all()
    assert (voc.video.encode(ev.video_id.to_numpy()) == ev.video_idx.to_numpy()).all()
    tr = ev[ev.split == 0]
    assert set(voc.video.ids) == set(tr.video_id.unique())
    assert set(voc.user.ids) == set(tr.user_id.unique())
    # Popularity counts equal train-only recounts.
    pop = P["popularity"]["video"]
    assert pop.train_long_view.sum() == tr.long_view.sum()
    assert pop.train_events.sum() == len(tr)


def test_catalog_sizes(P):
    cat = P["catalogs"]
    assert len(cat.video_ids) == 7583 and len(cat.music_ids) == 7201
    assert catm.MUSIC_SENTINEL not in cat.music_ids


def test_frozen_test_history_and_mask_on_real_data(P):
    ev, m = P["events"], P["meta"]
    h = HistoryIndex(ev)
    t_test = prediction_time(m, "test")
    users = np.unique(ev.loc[ev.split == 2, "user_id"].to_numpy())[:2000]
    pos, n = h.model_history(users, t_test, max_len=100)
    th = h.gather(pos, "time_ms", pad=-1)
    sp = h.gather(pos, "split", pad=-1)
    assert (th[pos >= 0] < t_test).all() and not (sp == 2).any()
    full = h.history_length(users, t_test)
    assert (n == np.minimum(full, 100)).all()
    # For users with >100 events the mask covers items outside the model input.
    indptr, rows = h.seen_rows(users, t_test, "video")
    long_u = np.flatnonzero(full > 100)
    assert len(long_u) > 0
    i = long_u[0]
    input_rows = set(h.gather(pos[i : i + 1], "video_row", pad=-1)[0]) - {-1}
    seen = set(csr_row(indptr, rows, i))
    assert input_rows < seen


def test_review_artifacts(P):
    lc, ev = P["label_conflict_events"], P["events"]
    # Dropped label-conflict keys never appear among cleaned events.
    k = ["user_id", "video_id", "time_ms"]
    assert len(lc.merge(ev[k], on=k)) == 0 and len(lc) == 2
    mc, cat = P["music_catalog"], P["catalogs"]
    assert (mc.music_id.to_numpy() == cat.music_ids).all()
    assert mc.n_videos_metadata.sum() == (cat.video_music_row >= 0).sum()
    # Music vocab index on the music catalog agrees with the per-video index.
    vc = P["video_catalog"]
    has = vc.music_row >= 0
    assert (mc.music_idx.to_numpy()[vc.music_row[has]] == vc.music_idx[has].to_numpy()).all()
    assert (vc.loc[~has, "music_idx"] == 2).all()
