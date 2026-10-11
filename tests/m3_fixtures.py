"""Hand-built M3 fixture (shared by test_eval_frame.py / test_baselines.py).

Catalog (video_id -> music_id): v10->m100, v11->m100 (shared), v12->m0 (sentinel),
v13->m101, v14->m102 (102 never in logs). Video rows 0..4, music rows m100:0, m101:1, m102:2.
Cuts: T_val = 1000, T_test = 2000 (an event at exactly 2000 is test).

Users:
  u1: train v13@100 (no lv); test lv on v10@2000, v11@2100, v12@2200, v13@2300
  u2: test only, lv on v10@2500                        -> cold (no history)
  u3: train v14@999; test lv on v14@2400               -> all GT masked
  u4: val v10@1500 (no lv); test lv on v11@3000        -> train-OOV; music GT masked via v10
  u5: train lv v10@500; no test events                 -> never evaluated
Label conflicts (dropped in M2): test (u4, v13) lv_any=1 -> possible lost pair;
                                 test (u1, v10) lv_any=1 -> pair already in GT, not counted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kuairand_rec.data import catalog as catm
from kuairand_rec.data.history import HistoryIndex
from kuairand_rec.data.pipeline import attach_ids

META = {"t_val_ms": 1000, "t_test_ms": 2000}

VIDEO_META = pd.DataFrame({"video_id": [10, 11, 12, 13, 14], "author_id": [1, 1, 2, 3, 4],
                           "music_id": [100, 100, 0, 101, 102]})

ROWS = [  # user, video, time, long_view, is_like
    (1, 13, 100, 0, 0), (1, 10, 2000, 1, 0), (1, 11, 2100, 1, 1), (1, 12, 2200, 1, 0), (1, 13, 2300, 1, 0),
    (2, 10, 2500, 1, 0),
    (3, 14, 999, 0, 0), (3, 14, 2400, 1, 0),
    (4, 10, 1500, 0, 0), (4, 11, 3000, 1, 0),
    (5, 10, 500, 1, 0),
]


def split_of(t):
    t = np.asarray(t)
    return np.where(t < META["t_val_ms"], 0, np.where(t < META["t_test_ms"], 1, 2)).astype(np.int8)


def make_events(rows=ROWS):
    ev = pd.DataFrame(rows, columns=["user_id", "video_id", "time_ms", "long_view", "is_like"])
    ev["split"] = split_of(ev.time_ms)
    cat = catm.build_catalogs(VIDEO_META)
    ev = attach_ids(ev, VIDEO_META, cat)
    ev = ev.sort_values(["user_id", "time_ms", "video_id"], kind="mergesort").reset_index(drop=True)
    lc = pd.DataFrame({"user_id": [4, 1], "video_id": [13, 10], "time_ms": [2600, 2050],
                       "long_view_any": [1, 1], "is_like_any": [0, 0]})
    lc["split"] = split_of(lc.time_ms)
    return ev, lc, cat, HistoryIndex(ev)
