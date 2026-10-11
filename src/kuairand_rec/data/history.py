"""Past-only user histories and seen-item masks (PLAN.md §3b P1-2, P1-6).

One index serves every consumer: train (t_pred = target event time), val (T_val),
test / export (T_test). Two deliberately separate products:

* `model_history`  -> the model input: the most recent `max_len` events with t < t_pred.
* `seen_rows`      -> the evaluation mask: *all* distinct items with t < t_pred, uncapped.
  It is computed from the full event range, never from the truncated model history.

Strictness: the window end is searchsorted(side="left") on (user, t_pred), so every
event at exactly t_pred -- including the target itself and other same-timestamp
videos -- is excluded. Within a timestamp, events are ordered by video_id (P1-2), which
only matters for the order of earlier, fully included timestamps.

Information boundary comes from t_pred alone: at T_val the window holds train events
only; at T_test it holds train + val events. Callers must never pass t_pred > T_test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# user_id * OFFSET + time_ms is a single sortable int64 key (time_ms < 1e13, user_id < 9e5).
_OFFSET = np.int64(10**13)


def _keys(users, times) -> np.ndarray:
    t = np.asarray(times, dtype=np.int64)
    if len(t) and (t.min() < 0 or t.max() >= _OFFSET):
        raise ValueError("time_ms out of range for the composite key")
    return np.asarray(users, dtype=np.int64) * _OFFSET + t


class HistoryIndex:
    """Events sorted by (user_id, time_ms, video_id); position = event index."""

    def __init__(self, events: pd.DataFrame):
        ev = events.sort_values(["user_id", "time_ms", "video_id"], kind="mergesort").reset_index(drop=True)
        self.events = ev
        self._keys = _keys(ev.user_id.to_numpy(), ev.time_ms.to_numpy())
        assert np.all(np.diff(self._keys) >= 0)
        self.video_row = ev["video_row"].to_numpy(np.int64) if "video_row" in ev else None
        self.music_row = ev["music_row"].to_numpy(np.int64) if "music_row" in ev else None

    # --------------------------------------------------------------- windows
    def window(self, users, t_pred) -> tuple[np.ndarray, np.ndarray]:
        """[lo, hi) event positions of user u with time < t_pred (uncapped)."""
        users = np.asarray(users, dtype=np.int64)
        t_pred = np.broadcast_to(np.asarray(t_pred, dtype=np.int64), users.shape)
        lo = np.searchsorted(self._keys, users * _OFFSET, side="left")
        hi = np.searchsorted(self._keys, _keys(users, t_pred), side="left")  # strict t < t_pred
        return lo, hi

    def history_length(self, users, t_pred, max_len: int | None = None) -> np.ndarray:
        lo, hi = self.window(users, t_pred)
        n = hi - lo
        return np.minimum(n, max_len) if max_len is not None else n

    def model_history(self, users, t_pred, max_len: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Padded [n, max_len] matrix of event positions (-1 = pad), oldest -> newest,
        right-aligned so column max_len-1 is the most recent event. Also returns lengths."""
        lo, hi = self.window(users, t_pred)
        start = np.maximum(lo, hi - max_len)  # cap: keep only the most recent max_len
        length = hi - start
        cols = np.arange(max_len)[None, :]
        pos = hi[:, None] - max_len + cols      # right-aligned positions
        valid = pos >= start[:, None]
        return np.where(valid, pos, -1).astype(np.int64), length.astype(np.int64)

    def gather(self, pos: np.ndarray, column: str, pad=0) -> np.ndarray:
        """Look up an event column for a padded position matrix."""
        vals = self.events[column].to_numpy()
        out = vals[np.where(pos >= 0, pos, 0)]
        return np.where(pos >= 0, out, pad)

    # --------------------------------------------------------------- seen-item mask
    def seen_rows(self, users, t_pred, target: str) -> tuple[np.ndarray, np.ndarray]:
        """CSR (indptr, rows): distinct catalog rows each user interacted with before t_pred.

        Uses the full uncapped window. For music, rows come from the non-zero music of the
        interacted videos (music_id 0 has no catalog row and is skipped).
        """
        item = {"video": self.video_row, "music": self.music_row}[target]
        lo, hi = self.window(users, t_pred)
        n = hi - lo
        q = np.repeat(np.arange(len(lo)), n)
        p = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + np.repeat(lo, n)
        rows = item[p]
        keep = rows >= 0
        q, rows = q[keep], rows[keep]
        pair = np.unique(q.astype(np.int64) * (int(item.max()) + 2) + rows)
        q_u, r_u = pair // (int(item.max()) + 2), pair % (int(item.max()) + 2)
        indptr = np.zeros(len(lo) + 1, dtype=np.int64)
        np.add.at(indptr, q_u + 1, 1)
        return np.cumsum(indptr), r_u.astype(np.int64)


def csr_row(indptr: np.ndarray, data: np.ndarray, i: int) -> np.ndarray:
    return data[indptr[i] : indptr[i + 1]]
