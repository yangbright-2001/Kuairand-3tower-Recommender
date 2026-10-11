"""Hand-computed checks for audit counting logic (dup classification, ties, splits, history)."""

import numpy as np
import pandas as pd

from kuairand_rec.audit import stats

KEY = ["user_id", "video_id", "time_ms"]


def test_classify_duplicates_nan_aware():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 3],
            "video_id": [9, 9, 8, 8, 7, 7, 6],
            "time_ms": [5, 5, 5, 5, 4, 4, 1],
            # group (1,9,5): exact dup (NaN == NaN); group (1,8,5): NaN vs 3 -> conflict;
            # group (2,7,4): same a, different b -> conflict; (3,6,1): singleton.
            "a": [np.nan, np.nan, np.nan, 3.0, 1.0, 1.0, 0.0],
            "b": [0, 0, 0, 0, 0, 1, 0],
        }
    )
    r = stats.classify_key_duplicates(df, KEY, ["a", "b"])
    assert r["n_key_groups"] == 4
    assert r["n_multi_row_groups"] == 3
    assert r["n_exact_dup_groups"] == 1 and r["n_rows_in_exact_dup_groups"] == 2
    assert r["n_conflict_groups"] == 2 and r["n_rows_in_conflict_groups"] == 4
    assert r["n_redundant_exact_rows"] == 1 == int(df.duplicated().sum())
    assert r["conflict_fields"] == {"a": 1, "b": 1}


def test_same_timestamp_counts_groups_and_rows():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 3],
            "time_ms": [5, 5, 5, 6, 4, 4, 1],
            "video_id": [1, 2, 2, 3, 7, 7, 6],  # (1,5) has 2 distinct videos over 3 rows
        }
    )
    r = stats.same_timestamp_multi_item(df)
    assert r["n_groups_multi_item"] == 1
    assert r["n_rows_in_multi_item_groups"] == 3
    assert r["distinct_items_per_multi_group"] == {2: 1}
    assert r["n_groups_multi_row"] == 2 and r["n_rows_in_multi_row_groups"] == 5


def test_repeat_interactions_ignores_same_time_copies():
    df = pd.DataFrame(
        {"user_id": [1, 1, 1, 2], "video_id": [9, 9, 9, 9], "time_ms": [0, 0, 7_200_000, 3]}
    )
    r = stats.repeat_interactions(df)
    assert r["n_pairs"] == 2 and r["n_pairs_repeated"] == 1
    assert r["distinct_times_per_repeated_pair"] == {2: 1}
    assert r["gap_hours_quantiles"]["0.5"] == 2.0


def test_quantile_cut_keeps_ties_together():
    times = np.array([1, 2, 3, 3, 3, 4, 5, 6, 7, 8])
    t = stats.quantile_timestamp_cut(times, 0.3)  # sorted[3] == 3
    assert t == 3
    split = stats.assign_split(times, (t, 6))
    assert set(split[times == 3]) == {1}  # all tied rows on the same (later) side
    assert list(split) == [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]


def test_nearest_local_midnight():
    tz = "Asia/Shanghai"
    t = int(pd.Timestamp("2022-04-16 22:07", tz=tz).tz_convert("UTC").value // 10**6)
    m = stats.nearest_local_midnight_ms(t, tz)
    assert pd.Timestamp(m, unit="ms", tz="UTC").tz_convert(tz) == pd.Timestamp("2022-04-17", tz=tz)


def test_count_strictly_before_excludes_ties_and_other_users():
    hu = np.array([1, 1, 1, 1, 2])
    ht = np.array([10, 20, 20, 30, 5])
    qu = np.array([1, 1, 1, 2, 3])
    qt = np.array([20, 31, 10, 100, 50])
    np.testing.assert_array_equal(stats.count_strictly_before(hu, ht, qu, qt), [1, 4, 0, 1, 0])
    np.testing.assert_array_equal(stats.count_same_time(hu, ht, qu, qt), [2, 0, 1, 0, 0])


def test_history_bucket_edges():
    b = stats.history_bucket(np.array([0, 1, 5, 6, 20, 21, 500]))
    assert list(b) == ["0", "1-5", "1-5", "6-20", "6-20", ">20", ">20"]


def test_official_long_view_rule():
    play = np.array([10_000, 9_999, 18_000, 17_999, 0])
    dur = np.array([10_000, 10_000, 60_000, 60_000, 0])
    np.testing.assert_array_equal(stats.official_long_view(play, dur), [1, 0, 1, 0, 1])
