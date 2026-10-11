"""Raw-log cleaning (PLAN.md §3b P1-4). Pure functions; raw files are never modified.

Event key = (user_id, video_id, time_ms). Rules, applied in this order:
1. Exact duplicates (all non-key fields equal, NaN-aware) collapse to one row.
2. Key groups that still have >1 row are "conflict groups":
   a. if any label in `label_conflict_cols` disagrees, the whole event is dropped;
   b. otherwise one whole representative row is kept: max play_time_ms, ties broken
      deterministically (see `_representative_order`). No per-field max, so labels and
      other fields always come from the same raw row.
Genuine repeats (same user/video at a different time_ms) are distinct events and are kept.
Labels are never re-derived.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["user_id", "video_id", "time_ms"]
NON_VALUE_COLS = ("src_file",)  # provenance only; not compared when detecting duplicates


def _value_cols(log: pd.DataFrame) -> list[str]:
    return [c for c in log.columns if c not in KEY and c not in NON_VALUE_COLS]


def _counts(df: pd.DataFrame, actions: list[str], event_level: bool) -> dict:
    out = {
        "rows": int(len(df)),
        "distinct_events": int(len(df[KEY].drop_duplicates())) if not event_level else int(len(df)),
        "users": int(df.user_id.nunique()),
        "videos": int(df.video_id.nunique()),
    }
    for a in actions:
        out[f"{a}_positive_rows"] = int(df[a].sum())
        # Event-level: an event counts as positive if any of its rows is positive.
        out[f"{a}_positive_events"] = (
            int(df[a].sum()) if event_level else int(df.loc[df[a] == 1, KEY].drop_duplicates().shape[0])
        )
    return out


def _representative_order(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Sort so the first row of each key group is its deterministic representative.

    Primary: play_time_ms descending (P1-4). Ties: remaining value columns ascending,
    then src_file. Every column participates, so the choice never depends on input order.
    """
    rest = [c for c in value_cols if c != "play_time_ms"]
    by = KEY + ["play_time_ms"] + rest + [c for c in NON_VALUE_COLS if c in df.columns]
    asc = [True] * len(KEY) + [False] + [True] * (len(by) - len(KEY) - 1)
    return df.sort_values(by, ascending=asc, kind="mergesort", na_position="last")


def clean_logs(log: pd.DataFrame, label_conflict_cols: list[str], actions: list[str],
               return_dropped: bool = False):
    """Return (events, report[, dropped]). `events` has exactly one row per kept key, sorted by
    (user_id, time_ms, video_id).

    With return_dropped=True, also returns one row per dropped label-conflict event with
    `{a}_any` = 1 if any of its raw rows was positive for action a. The evaluator needs
    these to count "GT events dropped as label conflicts" (P1-1); they are never used as
    history, mask, or training data.
    """
    value_cols = _value_cols(log)
    report: dict = {"key": KEY, "value_cols_compared": value_cols, "before": _counts(log, actions, False)}

    # 1. Exact duplicates. Sorting by src_file first makes the kept copy deterministic.
    srt = log.sort_values(KEY + [c for c in NON_VALUE_COLS if c in log.columns], kind="mergesort")
    dedup = srt.drop_duplicates(subset=KEY + value_cols, keep="first")
    n_exact_redundant = len(log) - len(dedup)
    report["after_exact_dedup"] = _counts(dedup, actions, False)

    # 2. Remaining multi-row keys are conflict groups.
    gsize = dedup.groupby(KEY, sort=False)[KEY[0]].transform("size")
    conf = dedup[gsize.to_numpy() > 1]
    n_conflict_groups = int(len(conf[KEY].drop_duplicates()))

    # 2a. Label conflicts -> drop the whole event.
    lab_nuniq = conf.groupby(KEY, sort=False)[label_conflict_cols].nunique(dropna=False)
    lab_conflict = lab_nuniq.gt(1)
    bad_keys = lab_nuniq.index[lab_conflict.any(axis=1).to_numpy()]
    by_label = {c: int(lab_conflict[c].sum()) for c in label_conflict_cols}

    # Diagnostic from PROGRESS.md: do label-conflict events overlap play_time conflicts?
    pt_conf = conf.groupby(KEY, sort=False)["play_time_ms"].nunique().gt(1)
    overlap = int(pt_conf.loc[bad_keys].sum()) if len(bad_keys) else 0

    key_idx = pd.MultiIndex.from_frame(dedup[KEY])
    is_bad = key_idx.isin(bad_keys)
    kept = dedup[~is_bad]
    dropped_label = dedup[is_bad]

    # 2b. Non-label conflicts -> one representative row per key.
    kept = _representative_order(kept, value_cols).drop_duplicates(subset=KEY, keep="first")
    n_conflict_rows_collapsed = int(len(dedup) - len(dropped_label) - len(kept))

    events = kept.sort_values(["user_id", "time_ms", "video_id"], kind="mergesort").reset_index(drop=True)
    assert not events.duplicated(KEY).any()

    fields = conf.groupby(KEY, sort=False)[value_cols].nunique(dropna=False).gt(1).sum()
    report["exclusions"] = {
        "exact_duplicate_redundant_rows_removed": int(n_exact_redundant),
        "conflict_groups_total": n_conflict_groups,
        "conflict_fields_groups": {c: int(v) for c, v in fields.items() if v > 0},
        "label_conflict_events_dropped": int(len(bad_keys)),
        "label_conflict_events_by_label": by_label,
        "label_conflict_rows_dropped": int(len(dropped_label)),
        "label_conflict_rows_dropped_positive": {a: int(dropped_label[a].sum()) for a in actions},
        "label_conflict_events_also_play_time_conflict": overlap,
        "conflict_groups_collapsed_to_representative": n_conflict_groups - int(len(bad_keys)),
        "conflict_rows_removed_by_representative": n_conflict_rows_collapsed,
    }
    report["after"] = _counts(events, actions, True)
    # Row conservation: raw = kept + exact redundant + label-conflict rows + collapsed conflict rows.
    report["row_conservation_ok"] = bool(
        len(log) == len(events) + n_exact_redundant + len(dropped_label) + n_conflict_rows_collapsed
    )
    report["event_conservation_ok"] = bool(
        report["before"]["distinct_events"] == len(events) + len(bad_keys)
    )
    if not return_dropped:
        return events, report
    dropped = (dropped_label.groupby(KEY, sort=True)[actions].max()
               .rename(columns={a: f"{a}_any" for a in actions}).reset_index())
    return events, report, dropped


def mark_missing_duration(events: pd.DataFrame) -> pd.DataFrame:
    """duration_ms == 0 means unknown duration (P1-4), not a zero-length video. Labels untouched."""
    out = events.copy()
    out["duration_ms"] = out["duration_ms"].astype("float64").where(out["duration_ms"] != 0, np.nan)
    return out
