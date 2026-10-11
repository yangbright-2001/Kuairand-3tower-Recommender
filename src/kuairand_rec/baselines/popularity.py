"""Train-only popularity scorer (P1-6b).

score(item) = number of train positive events for the action on that catalog row; music =
sum over all videos carrying the music. Train-unseen / zero-count items score 0 and fall
last under the shared tie-break (row ascending). Identical scores for every user: the only
personalisation is the evaluator's seen mask.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kuairand_rec.data.catalog import Catalogs, assert_train_only
from kuairand_rec.eval.evaluator import EvalContext


def popularity_counts(train: pd.DataFrame, cat: Catalogs, target: str, action: str) -> np.ndarray:
    assert_train_only(train)
    vr = train.video_row.to_numpy(np.int64)
    if (vr < 0).any():
        raise ValueError("train video outside the catalog")
    v = np.bincount(vr, weights=train[action].to_numpy(np.float64), minlength=len(cat.video_ids))
    if target == "video":
        return v.astype(np.int64)
    has = cat.video_music_row >= 0   # music_id 0 contributes to no music row
    return np.bincount(cat.video_music_row[has], weights=v[has], minlength=len(cat.music_ids)).astype(np.int64)


class PopularityScorer:
    name = "pop"

    def __init__(self, train: pd.DataFrame, cat: Catalogs, actions):
        self._s = {}
        for t in ("video", "music"):
            for a in actions:
                s = popularity_counts(train, cat, t, a).astype(np.float32)
                s.setflags(write=False)
                self._s[(t, a)] = s

    def counts(self, target: str, action: str) -> np.ndarray:
        return self._s[(target, action)]

    def score(self, users, target, action, ctx: EvalContext) -> np.ndarray:
        return np.broadcast_to(self._s[(target, action)], (len(users), len(self._s[(target, action)]))).copy()

    def signal(self, users, target, action, ctx: EvalContext) -> np.ndarray:
        return np.ones(len(users), dtype=bool)

    def item_scorable(self, target, action) -> np.ndarray:
        # P1-6b: unseen items get a defined score (0), not -inf.
        return np.ones(len(self._s[(target, action)]), dtype=bool)
