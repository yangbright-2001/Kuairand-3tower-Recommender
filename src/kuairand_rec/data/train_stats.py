"""Train-only fitted statistics and the cold-video filtering hook (PLAN.md §3b P1-6b, P1-9).

Everything here takes *train* events as input. Callers pass `events[events.split == 0]`;
the functions assert this so val/test data cannot leak into fitted quantities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kuairand_rec.data.catalog import Catalogs, assert_train_only

NUMERIC_VIDEO_COLS = ("video_duration", "server_width", "server_height")

_assert_train = assert_train_only  # same guard as build_vocabs


def popularity(train: pd.DataFrame, cat: Catalogs, actions: list[str]) -> dict[str, pd.DataFrame]:
    """Per catalog row: train interaction count and train positive count per action.

    Music count = sum over all videos carrying that music (P1-6b); music_id 0 contributes
    to no music row. Items without train signal get 0 (unscorable is decided per model).
    """
    _assert_train(train)
    vr = train.video_row.to_numpy()
    assert (vr >= 0).all(), "every log video must be in the metadata catalog"
    video = pd.DataFrame({"video_row": np.arange(len(cat.video_ids)), "video_id": cat.video_ids})
    video["train_events"] = np.bincount(vr, minlength=len(cat.video_ids))
    for a in actions:
        video[f"train_{a}"] = np.bincount(vr, weights=train[a].to_numpy(), minlength=len(cat.video_ids)).astype(np.int64)

    music = pd.DataFrame({"music_row": np.arange(len(cat.music_ids)), "music_id": cat.music_ids})
    has_m = cat.video_music_row >= 0
    for c in ["train_events"] + [f"train_{a}" for a in actions]:
        music[c] = np.bincount(
            cat.video_music_row[has_m], weights=video[c].to_numpy()[has_m], minlength=len(cat.music_ids)
        ).astype(np.int64)
    return {"video": video, "music": music}


def numeric_video_stats(train: pd.DataFrame, video_meta: pd.DataFrame) -> dict:
    """Mean/std of log1p(numeric metadata) over train-seen videos (NaN ignored)."""
    _assert_train(train)
    vm = video_meta[video_meta.video_id.isin(train.video_id.unique())]
    out = {}
    for c in NUMERIC_VIDEO_COLS:
        x = np.log1p(vm[c].to_numpy(dtype=float))
        out[c] = {"mean": float(np.nanmean(x)), "std": float(np.nanstd(x)) or 1.0,
                  "n_missing_train_videos": int(np.isnan(x).sum())}
    return out


def remove_videos(events: pd.DataFrame, video_ids) -> pd.DataFrame:
    """Cold-video hook (P1-9): drop every event on the held-out videos.

    Apply to the *full* cleaned log before building train targets, histories (including
    pre-T_test history), vocabs and Pop/ALS fits, so H is absent everywhere a model can
    see it. Evaluation GT / mask / population for the cold frame are built separately from
    the unfiltered log (M5). Selection of H is not implemented in M2.
    """
    return events[~events.video_id.isin(np.asarray(video_ids))].reset_index(drop=True)
