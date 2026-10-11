"""Dataset Audit Phase A: preliminary read-only schema inspection.

SUPERSEDED by `python -m kuairand_rec.audit.run` (Phase C). Kept for provenance only.
Known counting issue fixed in Phase C: "same user same time_ms (different videos)" below counts
extra distinct videos, not affected groups/rows; see kuairand_rec.audit.stats.same_timestamp_multi_item.

Scope (agreed): the two standard logs, user_features_pure.csv, video_features_basic_pure.csv.
NOT read: video_features_statistic_pure.csv (future leakage), random log (out of scope).

Prints only schema, a few sample rows and aggregate statistics. Writes nothing.
Usage: uv run python scripts/audit_phase_a.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kuairand_rec import paths

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 30)


def hr(title: str) -> None:
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def schema(df: pd.DataFrame, name: str, n_sample: int = 3) -> None:
    hr(f"{name}: shape={df.shape}")
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s)
        rows.append(
            dict(
                col=c,
                dtype=str(s.dtype),
                n_null=int(s.isna().sum()),
                n_unique=int(s.nunique(dropna=True)),
                min=s.min() if num else None,
                max=s.max() if num else None,
                n_neg=int((s < 0).sum()) if num else None,
                top=str(s.value_counts(dropna=False).index[0])[:20],
            )
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n-- {n_sample} sample rows (fixed seed) --")
    print(df.sample(n_sample, random_state=0).to_string())


def small_vc(df: pd.DataFrame, cols: list[str], max_card: int = 12) -> None:
    for c in cols:
        vc = df[c].value_counts(dropna=False)
        if len(vc) <= max_card:
            print(f"\n[{c}] value counts:\n" + vc.to_string())
        else:
            print(f"\n[{c}] {len(vc)} distinct; top-5:\n" + vc.head(5).to_string())


def main() -> None:
    # ------------------------------------------------------------------ logs
    logs = []
    for i, f in enumerate(paths.STANDARD_LOGS):
        d = pd.read_csv(paths.RAW_DIR / f)
        d["_file"] = i
        logs.append(d)
        schema(d.drop(columns="_file"), f)
        small_vc(d, ["is_rand", "tab", "is_click", "is_like", "long_view", "is_hate"])
    log = pd.concat(logs, ignore_index=True)

    hr("Standard logs: time consistency")
    for i, d in log.groupby("_file"):
        ts = pd.to_datetime(d["time_ms"], unit="ms", utc=True)
        print(
            f"file{i}: time_ms {ts.min()} .. {ts.max()} | date {d['date'].min()}..{d['date'].max()}"
            f" | days={d['date'].nunique()} | rows/day={len(d) / d['date'].nunique():,.0f}"
        )
    # Which timezone makes time_ms agree with date / hourmin?
    for tz in ["UTC", "Asia/Shanghai"]:
        ts = pd.to_datetime(log["time_ms"], unit="ms", utc=True).dt.tz_convert(tz)
        date_ok = (ts.dt.strftime("%Y%m%d").astype(int) == log["date"]).mean()
        hm = ts.dt.hour * 100
        hour_ok = (hm == log["hourmin"]).mean()
        print(f"tz={tz:14s} date match={date_ok:.4f}  hourmin(=HH00) match={hour_ok:.4f}")
    print("hourmin distinct minute parts:", sorted((log["hourmin"] % 100).unique())[:10])
    f0_max, f1_min = log.loc[log._file == 0, "time_ms"].max(), log.loc[log._file == 1, "time_ms"].min()
    print(f"file0.max < file1.min ? {f0_max < f1_min}  gap_hours={(f1_min - f0_max) / 3.6e6:.2f}")
    print(f"time_ms nulls={log['time_ms'].isna().sum()}  non-monotone within file is expected (sorted by user)")

    hr("Standard logs: rough 70/85 cut positions (row-quantile on sorted time_ms; NOT a decision)")
    t = np.sort(log["time_ms"].to_numpy())
    n = len(t)
    print(f"total rows={n:,}  file0 share={(log._file == 0).mean():.4f}")
    for q in (0.70, 0.85):
        cut = t[int(q * n)]
        ties = int((t == cut).sum())
        in_file = log.loc[log.time_ms == cut, "_file"].unique().tolist()
        print(
            f"q={q:.2f} cut_time={pd.to_datetime(cut, unit='ms', utc=True).tz_convert('Asia/Shanghai')}"
            f"  rows_at_exact_cut_ts={ties}  file={in_file}"
        )
    vc_t = pd.Series(t).value_counts()
    print(f"distinct time_ms={len(vc_t):,}  rows sharing a time_ms with another row={int(vc_t[vc_t > 1].sum()):,}"
          f"  max rows per time_ms={int(vc_t.max())}")

    hr("Standard logs: duplicates")
    print("exact duplicate rows:", int(log.drop(columns="_file").duplicated().sum()))
    print("dup (user,video,time_ms):", int(log.duplicated(["user_id", "video_id", "time_ms"]).sum()))
    uv = log.groupby(["user_id", "video_id"]).size()
    print(f"(user,video) pairs={len(uv):,}  pairs with >1 event={int((uv > 1).sum()):,}"
          f"  max repeats={int(uv.max())}")
    print("same user same time_ms (different videos):",
          int(log.duplicated(["user_id", "time_ms"]).sum() - log.duplicated(["user_id", "video_id", "time_ms"]).sum()))

    hr("Standard logs: label sanity (aggregate only)")
    print(log[["is_click", "is_like", "long_view", "is_follow", "is_comment", "is_forward", "is_hate"]]
          .mean().rename("rate").to_string())
    print("\nlike & long_view co-occurrence:\n",
          pd.crosstab(log["is_like"], log["long_view"]).to_string())
    print("\nlong_view rate by tab:\n", log.groupby("tab")["long_view"].agg(["mean", "size"]).to_string())
    print("\nplay_time_ms == 0 share:", round((log["play_time_ms"] == 0).mean(), 4),
          "  duration_ms==0 share:", round((log["duration_ms"] == 0).mean(), 4))
    ratio = log["play_time_ms"] / log["duration_ms"].replace(0, np.nan)
    print("play/duration ratio quantiles by long_view:")
    print(ratio.groupby(log["long_view"]).quantile([0.01, 0.1, 0.5, 0.9]).unstack().round(3).to_string())
    print("min play_time_ms where long_view=1:", int(log.loc[log.long_view == 1, "play_time_ms"].min()))

    # ------------------------------------------------------------------ users
    users = pd.read_csv(paths.RAW_DIR / paths.USER_FEATURES)
    schema(users, paths.USER_FEATURES)
    small_vc(users, ["user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
                     "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range"])
    print("\nonehot_feat* nunique:", {c: int(users[c].nunique()) for c in users.columns if c.startswith("onehot")})
    print("user_id unique & contiguous 0..N-1 ?", users.user_id.is_unique,
          users.user_id.min(), users.user_id.max(), len(users))

    # ------------------------------------------------------------------ videos
    vids = pd.read_csv(paths.RAW_DIR / paths.VIDEO_FEATURES_BASIC)
    schema(vids, paths.VIDEO_FEATURES_BASIC)
    small_vc(vids, ["video_type", "upload_type", "visible_status", "music_type", "tag"])
    print("\nvideo_id unique & contiguous?", vids.video_id.is_unique, vids.video_id.min(), vids.video_id.max(), len(vids))
    print("tag looks multi-valued (contains ',')?:", int(vids["tag"].astype(str).str.contains(",").sum()))
    print("upload_dt range:", vids["upload_dt"].min(), "..", vids["upload_dt"].max(),
          " nulls:", int(vids["upload_dt"].isna().sum()))
    print(f"distinct author_id={vids.author_id.nunique():,}  distinct music_id={vids.music_id.nunique():,}"
          f"  music_id null={int(vids.music_id.isna().sum())}")
    vpm = vids.groupby("music_id").size()
    print("videos per music_id quantiles:", vpm.quantile([0.5, 0.9, 0.99, 1.0]).to_dict(),
          f" music_ids with >1 video={int((vpm > 1).sum()):,}")
    print("top-5 music_id by #videos:\n", vpm.sort_values(ascending=False).head(5).to_string())

    # ------------------------------------------------------------------ joins
    hr("Join coverage (log IDs vs feature files)")
    lu, lv = set(log.user_id.unique()), set(log.video_id.unique())
    print(f"log users={len(lu):,}  in user_features={len(lu & set(users.user_id)):,}"
          f"  | user_features users not in logs={len(set(users.user_id) - lu):,}")
    print(f"log videos={len(lv):,}  in video_basic={len(lv & set(vids.video_id)):,}"
          f"  | video_basic videos not in logs={len(set(vids.video_id) - lv):,}")
    print("log rows with video missing from video_basic:", int((~log.video_id.isin(vids.video_id)).sum()))
    dur = log[["video_id", "duration_ms"]].merge(vids[["video_id", "video_duration"]], on="video_id")
    print("log duration_ms == video_basic.video_duration share:",
          round((dur.duration_ms == dur.video_duration).mean(), 4))
    per_video_dur = log.groupby("video_id")["duration_ms"].nunique()
    print("videos whose log duration_ms varies across events:", int((per_video_dur > 1).sum()))


if __name__ == "__main__":
    main()
