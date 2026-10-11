"""Global temporal split (PLAN.md §3b P1-3).

Rule: t < T_val -> train (0); T_val <= t < T_test -> val (1); t >= T_test -> test (2).
Because the rule depends only on time_ms, all events sharing a timestamp land in the
same split, so a timestamp can never straddle a boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


def local_to_epoch_ms(local: str, tz: str) -> int:
    """'2022-04-17 00:00:00' interpreted in `tz` -> epoch milliseconds."""
    return int(pd.Timestamp(local).tz_localize(tz).tz_convert("UTC").value // 10**6)


def epoch_ms_to_local(ms: int, tz: str) -> str:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def assign_split(times: np.ndarray, t_val: int, t_test: int) -> np.ndarray:
    if not t_val < t_test:
        raise ValueError("t_val must precede t_test")
    # side="right": a time equal to a cut goes to the later split (t >= T).
    return np.searchsorted(np.array([t_val, t_test]), np.asarray(times), side="right").astype(np.int8)


def split_summary(events: pd.DataFrame, actions: list[str], tz: str) -> list[dict]:
    """Per-split counts on cleaned events (one row = one event)."""
    rows = []
    n = len(events)
    for s, name in enumerate(SPLIT_NAMES):
        d = events[events.split == s]
        r = {
            "split": name,
            "events": int(len(d)),
            "share": float(len(d) / n) if n else 0.0,
            "users": int(d.user_id.nunique()),
            "videos": int(d.video_id.nunique()),
            "first_event": epoch_ms_to_local(d.time_ms.min(), tz) if len(d) else None,
            "last_event": epoch_ms_to_local(d.time_ms.max(), tz) if len(d) else None,
        }
        for a in actions:
            r[f"{a}_positive_events"] = int(d[a].sum())
            r[f"{a}_rate"] = float(d[a].mean()) if len(d) else 0.0
            r[f"{a}_positive_users"] = int(d.loc[d[a] == 1, "user_id"].nunique())
        rows.append(r)
    return rows
