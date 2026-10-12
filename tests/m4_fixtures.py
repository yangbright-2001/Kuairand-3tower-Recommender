"""Hand-built M4 fixture, encoded through the real M2 path (attach_ids -> fit_condition).

Catalog (video -> author, music): v10 a1 m100, v11 a1 m100 (shared music), v12 a2 m0 (no music),
v13 a3 m101, v14 a4 m102, v15 a9 m103. v15 occurs only in val, so video / author / music
of v15 are train-OOV (content path). Cuts: T_val = 100 min, T_test = 200 min. D = 20 min.

Users:
  u1: train events incl. a same-timestamp pair at t = 50 min (v10, v11); val events, one of
      them 5 min before T_test (feedback hidden at T_test, L5); test events.
  u2: 130 train events (cap = 100).
  u3: test events only -> empty history at T_val and T_test; train-OOV user.
  u4: one train event at 99 min (age 1 min at T_val -> hidden).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kuairand_rec.data import catalog as catm
from kuairand_rec.data.pipeline import attach_ids, fit_condition

MIN = 60_000
META = {"t_val_ms": 100 * MIN, "t_test_ms": 200 * MIN}
ACTIONS = ["long_view", "is_like"]
FB = ["is_click", "long_view", "is_like", "is_follow", "is_comment", "is_forward", "is_hate", "is_profile_enter"]

VIDEO_META = pd.DataFrame({
    "video_id": [10, 11, 12, 13, 14, 15],
    "author_id": [1, 1, 2, 3, 4, 9],
    "music_id": [100, 100, 0, 101, 102, 103],
    "video_type": ["NORMAL", "NORMAL", "AD", "NORMAL", "NORMAL", "NORMAL"],
    "upload_dt": ["2022-04-09", "2022-04-10", "2022-04-10", "2022-04-09", "2022-04-11", "2022-04-11"],
    "upload_type": ["Kmovie", "ShortImport", "Kmovie", "Web", "Kmovie", "Web"],
    "visible_status": [0.0] * 6,
    "video_duration": [60_000.0, 30_000.0, np.nan, 10_000.0, 120_000.0, 45_000.0],
    "server_width": [720.0] * 6,
    "server_height": [1280.0, 1280.0, 960.0, 1280.0, 1280.0, 1280.0],
    "music_type": [9.0, 9.0, np.nan, 4.0, 9.0, 11.0],
    "tag": ["1,2", "2", "3", "1", "2,3,4", "5"],
})


def split_of(t):
    t = np.asarray(t)
    return np.where(t < META["t_val_ms"], 0, np.where(t < META["t_test_ms"], 1, 2)).astype(np.int8)


def _row(u, v, t, lv=0, like=0, click=1, play=5_000, dur=None, **fb):
    dur = VIDEO_META.set_index("video_id").video_duration[v] if dur is None else dur
    r = {"user_id": u, "video_id": v, "time_ms": int(t), "is_click": click, "long_view": lv, "is_like": like,
         "is_follow": 0, "is_comment": 0, "is_forward": 0, "is_hate": 0, "is_profile_enter": 0,
         "play_time_ms": play, "duration_ms": dur, "tab": 1}
    r.update(fb)
    return r


def raw_rows() -> list[dict]:
    rows = [
        # u1 train
        _row(1, 13, 10 * MIN, lv=1, play=10_000),
        _row(1, 10, 50 * MIN, lv=1, like=1, play=60_000, is_follow=1),
        _row(1, 11, 50 * MIN, lv=0, play=1_000),           # same timestamp as v10
        _row(1, 12, 60 * MIN, lv=1, play=40_000),          # duration missing
        _row(1, 14, 85 * MIN, lv=1, play=200_000),         # age 15 min at T_val -> hidden
        # u1 val
        _row(1, 15, 120 * MIN, lv=1, like=1, play=40_000), # train-OOV video in history
        _row(1, 13, 195 * MIN, lv=1, like=1, play=9_000, is_comment=1),  # age 5 min at T_test -> hidden
        # u1 test
        _row(1, 11, 200 * MIN, lv=1, play=30_000),
        _row(1, 14, 230 * MIN, lv=1, like=1, play=100_000),
        # u3 test only
        _row(3, 10, 210 * MIN, lv=1, play=60_000),
        # u4 single train event 1 min before T_val
        _row(4, 13, 99 * MIN, lv=1, like=1, play=10_000),
    ]
    # u2: 130 train events, 20 s apart, cycling over train videos
    vids = [10, 11, 12, 13, 14]
    for i in range(130):
        rows.append(_row(2, vids[i % 5], 1 * MIN + i * 20_000, lv=i % 2, like=int(i % 7 == 0),
                         play=1_000 * (i % 13)))
    return rows


def make(rows: list[dict] | None = None, video_meta: pd.DataFrame = VIDEO_META, exclude_video_ids=()) -> dict:
    """Run the real M2 encoding on the fixture. Returns fit_condition output + catalogs.
    `exclude_video_ids` = cold-video H, passed to `fit_condition` (the only cold entry point)."""
    ev = pd.DataFrame(raw_rows() if rows is None else rows)
    ev["split"] = split_of(ev.time_ms)
    cat = catm.build_catalogs(video_meta)
    ev = attach_ids(ev, video_meta, cat)
    ev = ev.sort_values(["user_id", "time_ms", "video_id"], kind="mergesort").reset_index(drop=True)
    fit = fit_condition(ev, video_meta, cat, ACTIONS, exclude_video_ids)
    fit["catalogs"] = cat
    fit["meta"] = META
    return fit


USER_FEATURES = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5, 6, 7],
    **{f"onehot_feat{i}": [0, 0, 1, 0, 0, 1, 0] for i in range(18)},
    "is_live_streamer": [-124, -124, 1, -124, -124, 1, -124],
    "is_video_author": [0, 1, 0, 0, 1, 0, 0],
    "register_days_range": ["730+", "730+", "8-14", "730+", "61-90", "730+", "730+"],
})
