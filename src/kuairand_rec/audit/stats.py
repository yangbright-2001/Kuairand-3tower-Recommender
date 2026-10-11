"""Pure, unit-tested statistics used by the dataset audit.

Counting conventions (used consistently in the audit report):
- "row"        = one raw log line.
- "key group"  = rows sharing (user_id, video_id, time_ms).
- "event"      = one distinct (user_id, video_id, time_ms) key group.
Nothing here filters or modifies data; functions only count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# user_id * OFFSET + time_ms gives a single sortable int64 key. time_ms (~1.65e12) < 1e13
# and user_id < 9.2e5 keeps it below the int64 limit.
_USER_TIME_OFFSET = np.int64(10**13)


def classify_key_duplicates(df: pd.DataFrame, key: list[str], value_cols: list[str]) -> dict:
    """Split multi-row key groups into exact duplicates and field conflicts.

    Comparison is over `value_cols` only (non-key fields) and is NaN-aware:
    NaN == NaN counts as identical, while NaN vs a value counts as a difference.
    Returns group- and row-level counts plus, for conflict groups, how many
    groups differ in each field.
    """
    sub = df[key + value_cols]
    size = sub.groupby(key, sort=False, dropna=False).size()
    # drop_duplicates treats NaN as equal, so this counts distinct value tuples per key.
    distinct = (
        sub.drop_duplicates().groupby(key, sort=False, dropna=False).size().reindex(size.index)
    )
    multi = size > 1
    exact = multi & (distinct == 1)
    conflict = distinct > 1

    field_conflicts: dict[str, int] = {}
    if conflict.any():
        conf_keys = size.index[conflict.to_numpy()]
        rows = sub.set_index(key).loc[conf_keys]
        # nunique(dropna=False) counts NaN as its own value, so NaN-vs-value is a conflict.
        per_field = rows.groupby(level=list(range(len(key))))[value_cols].nunique(dropna=False)
        field_conflicts = {c: int(v) for c, v in (per_field > 1).sum().items() if v > 0}

    return {
        "n_rows": int(len(sub)),
        "n_key_groups": int(len(size)),
        "n_multi_row_groups": int(multi.sum()),
        "n_exact_dup_groups": int(exact.sum()),
        "n_rows_in_exact_dup_groups": int(size[exact].sum()),
        "n_conflict_groups": int(conflict.sum()),
        "n_rows_in_conflict_groups": int(size[conflict].sum()),
        # Rows that are exact copies of another row in the same group. Equals sub.duplicated().sum().
        "n_redundant_exact_rows": int((size - distinct).sum()),
        "conflict_fields": field_conflicts,
    }


def same_timestamp_multi_item(
    df: pd.DataFrame, user: str = "user_id", time: str = "time_ms", item: str = "video_id"
) -> dict:
    """(user, time) groups that contain more than one distinct item.

    Reports the number of such groups, the raw rows they contain, and the
    distribution of distinct items per group.
    """
    g = df.groupby([user, time], sort=False)
    n_rows = g.size()
    n_items = g[item].nunique()
    multi = n_items > 1
    return {
        "n_user_time_groups": int(len(n_rows)),
        "n_groups_multi_item": int(multi.sum()),
        "n_rows_in_multi_item_groups": int(n_rows[multi].sum()),
        "distinct_items_per_multi_group": {
            int(k): int(v) for k, v in n_items[multi].value_counts().sort_index().items()
        },
        # Any (user, time) group with >1 row, including same-item exact duplicates.
        "n_groups_multi_row": int((n_rows > 1).sum()),
        "n_rows_in_multi_row_groups": int(n_rows[n_rows > 1].sum()),
    }


def repeat_interactions(
    df: pd.DataFrame, user: str = "user_id", item: str = "video_id", time: str = "time_ms"
) -> dict:
    """Real repeated interactions: a (user, item) pair seen at more than one distinct time."""
    ut = df[[user, item, time]].drop_duplicates().sort_values([user, item, time])
    n_ts = ut.groupby([user, item], sort=False).size()
    rep = n_ts > 1
    # Gap between consecutive distinct timestamps of the same pair.
    same_pair = (ut[user].to_numpy()[1:] == ut[user].to_numpy()[:-1]) & (
        ut[item].to_numpy()[1:] == ut[item].to_numpy()[:-1]
    )
    gaps_h = np.diff(ut[time].to_numpy())[same_pair] / 3.6e6
    return {
        "n_pairs": int(len(n_ts)),
        "n_pairs_repeated": int(rep.sum()),
        "n_events_in_repeated_pairs": int(n_ts[rep].sum()),
        "distinct_times_per_repeated_pair": {
            int(k): int(v) for k, v in n_ts[rep].value_counts().sort_index().items()
        },
        "gap_hours_quantiles": (
            {str(q): float(np.quantile(gaps_h, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
            if len(gaps_h)
            else {}
        ),
    }


def quantile_timestamp_cut(times: np.ndarray, q: float) -> int:
    """Timestamp T for a `time < T` / `time >= T` boundary near row quantile q.

    T is the time of the row at sorted position floor(q * n). Every row with
    time == T falls on the later side, so a single timestamp never straddles
    the boundary.
    """
    t = np.sort(np.asarray(times))
    return int(t[int(np.floor(q * len(t)))])


def nearest_local_midnight_ms(t_ms: int, tz: str) -> int:
    """Local (tz) midnight closest to t_ms, returned as epoch ms."""
    ts = pd.Timestamp(t_ms, unit="ms", tz="UTC").tz_convert(tz)
    lo = ts.normalize()
    hi = lo + pd.Timedelta(days=1)
    best = lo if (ts - lo) <= (hi - ts) else hi
    return int(best.tz_convert("UTC").value // 10**6)


def assign_split(times: np.ndarray, cuts: tuple[int, int]) -> np.ndarray:
    """0 = train (t < T1), 1 = val (T1 <= t < T2), 2 = test (t >= T2)."""
    t1, t2 = cuts
    assert t1 < t2
    return np.searchsorted(np.array([t1, t2]), np.asarray(times), side="right").astype(np.int8)


def _user_time_keys(users: np.ndarray, times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.int64)
    assert times.min(initial=0) >= 0 and times.max(initial=0) < _USER_TIME_OFFSET
    return np.asarray(users, dtype=np.int64) * _USER_TIME_OFFSET + times


def count_strictly_before(
    hist_users: np.ndarray,
    hist_times: np.ndarray,
    query_users: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """For each query (u, t), count history events of user u with time < t (strict).

    Events at exactly t, including the query itself, are excluded. Ties are never
    part of the history.
    """
    keys = np.sort(_user_time_keys(hist_users, hist_times))
    q_users = np.asarray(query_users, dtype=np.int64)
    start = np.searchsorted(keys, q_users * _USER_TIME_OFFSET, side="left")
    end = np.searchsorted(keys, _user_time_keys(q_users, query_times), side="left")
    return (end - start).astype(np.int64)


def count_same_time(
    hist_users: np.ndarray,
    hist_times: np.ndarray,
    query_users: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """For each query (u, t), count history events of user u with time == t."""
    keys = np.sort(_user_time_keys(hist_users, hist_times))
    q = _user_time_keys(query_users, query_times)
    return (np.searchsorted(keys, q, "right") - np.searchsorted(keys, q, "left")).astype(np.int64)


CATALOG_LEVELS = ("metadata", "train_seen", "train_positive")


def build_catalogs(
    metadata_items: np.ndarray,
    train_items: np.ndarray,
    train_pos_items: np.ndarray,
    exclude: tuple = (),
) -> dict[str, np.ndarray]:
    """Three nested candidate catalogs (sorted unique ids), from widest to narrowest.

    - metadata:       every id in the metadata table, whether or not it occurs in any log.
    - train_seen:     ids with >=1 train interaction (any label).
    - train_positive: ids with >=1 train positive for the action.
    The metadata catalog must come from the metadata table itself. Deriving it from log
    rows would silently drop never-interacted items and understate the catalog size.
    `exclude` removes sentinel ids (e.g. music_id=0) from all three levels.
    """
    out = {}
    for name, ids in zip(CATALOG_LEVELS, (metadata_items, train_items, train_pos_items)):
        u = np.unique(np.asarray(ids))
        out[name] = u[~np.isin(u, np.asarray(exclude))] if len(exclude) else u
    return out


def ooc_share(items: np.ndarray, catalog: np.ndarray) -> float | None:
    """Share of (row-level) target items that are not in the catalog. None if no items."""
    items = np.asarray(items)
    return float((~np.isin(items, catalog)).mean()) if len(items) else None


HISTORY_BUCKETS = ("0", "1-5", "6-20", ">20")


def history_bucket(lengths: np.ndarray) -> np.ndarray:
    """Map history lengths to the assignment buckets, plus an explicit zero bucket."""
    lengths = np.asarray(lengths)
    out = np.empty(len(lengths), dtype=object)
    out[lengths == 0] = "0"
    out[(lengths >= 1) & (lengths <= 5)] = "1-5"
    out[(lengths >= 6) & (lengths <= 20)] = "6-20"
    out[lengths > 20] = ">20"
    return out


def official_long_view(play_ms: np.ndarray, duration_ms: np.ndarray, thr_ms: int = 18000) -> np.ndarray:
    """Official KuaiRand long_view rule (kuairand.com data dictionary).

    1 if play >= duration when duration <= 18 s, else 1 if play >= 18 s.
    """
    play_ms, duration_ms = np.asarray(play_ms), np.asarray(duration_ms)
    return np.where(duration_ms <= thr_ms, play_ms >= duration_ms, play_ms >= thr_ms).astype(np.int8)
