"""Dataset Audit Phase A follow-ups (read-only, aggregates only).

SUPERSEDED by `python -m kuairand_rec.audit.run` (Phase C). Kept for provenance only.
Known counting issues fixed in Phase C: the conflict check here used nunique(dropna=True),
which misses NaN-vs-value differences; see kuairand_rec.audit.stats.classify_key_duplicates.

Clarifies questions raised by scripts/audit_phase_a.py:
file overlap / duplicates, daily volume, hourmin semantics, long_view rule,
music_id sentinel, duration zeros, tag multiplicity, per-user activity.
Usage: uv run python scripts/audit_phase_a_followup.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kuairand_rec import paths


def hr(t: str) -> None:
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


def main() -> None:
    logs = []
    for i, f in enumerate(paths.STANDARD_LOGS):
        d = pd.read_csv(paths.RAW_DIR / f)
        d["_file"] = i
        logs.append(d)
    log = pd.concat(logs, ignore_index=True)
    vids = pd.read_csv(paths.RAW_DIR / paths.VIDEO_FEATURES_BASIC)
    cols = [c for c in log.columns if c != "_file"]
    ts = pd.to_datetime(log["time_ms"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")

    hr("File overlap window & duplicates")
    f0max = log.loc[log._file == 0, "time_ms"].max()
    f1min = log.loc[log._file == 1, "time_ms"].min()
    ov = log[(log.time_ms >= f1min) & (log.time_ms <= f0max)]
    print(f"overlap window rows: file0={int((ov._file == 0).sum())} file1={int((ov._file == 1).sum())}")
    dup_within = {i: int(d[cols].duplicated().sum()) for i, d in log.groupby("_file")}
    dup_all = int(log[cols].duplicated().sum())
    print(f"exact dup rows within file: {dup_within}  combined: {dup_all}  -> cross-file: {dup_all - sum(dup_within.values())}")
    key = ["user_id", "video_id", "time_ms"]
    k = log.duplicated(key, keep=False)
    g = log[k].groupby(key)
    n_conflict = int((g[cols].nunique().max(axis=1) > 1).sum())
    print(f"(user,video,time_ms) groups with >1 row: {g.ngroups:,}  of which with differing other fields: {n_conflict:,}")
    dups = log[log[cols].duplicated(keep=False)]
    print("exact-dup rows label rates (long_view, is_like):",
          round(dups.long_view.mean(), 3), round(dups.is_like.mean(), 4),
          " | dup rows by tab top-3:", dups.tab.value_counts().head(3).to_dict())

    hr("Daily volume (Asia/Shanghai date of time_ms) and positive rates")
    day = ts.dt.strftime("%m-%d")
    daily = log.groupby(day).agg(rows=("user_id", "size"), users=("user_id", "nunique"),
                                 lv=("long_view", "mean"), like=("is_like", "mean"))
    daily["cum_share"] = daily.rows.cumsum() / daily.rows.sum()
    print(daily.round(4).to_string())

    hr("hourmin semantics (Asia/Shanghai)")
    hh = ts.dt.hour * 100
    diff = ((log["hourmin"] - hh) // 100) % 24
    print("hourmin - hour(time_ms) [hours, mod 24] distribution:\n", diff.value_counts().head(5).to_string())
    print("date - date(time_ms) mismatch share:", round((ts.dt.strftime('%Y%m%d').astype(int) != log.date).mean(), 4))

    hr("long_view rule candidates (agreement with label)")
    p, d = log.play_time_ms, log.duration_ms
    cands = {
        "play>=18s": p >= 18000,
        "play>=duration": p >= d,
        "play>=min(18s,dur)": p >= np.minimum(18000, d),
        "play>=min(18s,dur) & dur>0": (p >= np.minimum(18000, d)) & (d > 0),
    }
    for name, c in cands.items():
        print(f"{name:30s} agreement={(c.astype(int) == log.long_view).mean():.4f}")
    print("long_view=1 with play_time_ms==0:", int(((log.long_view == 1) & (p == 0)).sum()))
    print("is_like=1 with play_time_ms==0:", int(((log.is_like == 1) & (p == 0)).sum()),
          " is_like=1 total:", int(log.is_like.sum()))

    hr("music_id sentinel & sharing (video_features_basic)")
    m0 = vids.music_id == 0
    print(f"videos with music_id==0: {int(m0.sum())}  music_type null among them: {int(vids.loc[m0, 'music_type'].isna().sum())}"
          f"  | music_type null overall: {int(vids.music_type.isna().sum())}")
    print("music_type distribution for music_id==0:", vids.loc[m0, "music_type"].value_counts(dropna=False).to_dict())
    nz = vids[~m0]
    vpm = nz.groupby("music_id").size()
    print(f"non-zero music_ids={len(vpm):,}  videos={len(nz):,}  ids with >1 video={int((vpm > 1).sum())}"
          f"  videos on shared ids={int(vpm[vpm > 1].sum())}  max={int(vpm.max())}")
    print("videos-per-music histogram (non-zero):", vpm.value_counts().sort_index().to_dict())
    lv_pos = log[log.long_view == 1].merge(vids[["video_id", "music_id"]], on="video_id")
    lk_pos = log[log.is_like == 1].merge(vids[["video_id", "music_id"]], on="video_id")
    shared = set(vpm[vpm > 1].index)
    for nm, dfp in [("long_view", lv_pos), ("is_like", lk_pos)]:
        print(f"{nm} positives: music_id==0 share={(dfp.music_id == 0).mean():.4f}"
              f"  on shared non-zero music share={dfp.music_id.isin(shared).mean():.4f}")

    hr("duration / metadata gaps")
    z = log.duration_ms == 0
    zv = log.loc[z, "video_id"].unique()
    print(f"log rows duration_ms==0: {int(z.sum()):,}  distinct videos: {len(zv)}"
          f"  of which video_duration NaN in basic: {int(vids.set_index('video_id').loc[zv, 'video_duration'].isna().sum())}")
    print("video_type AD rows in log:", int(log.video_id.isin(vids.loc[vids.video_type == 'AD', 'video_id']).sum()))
    ntag = vids.tag.dropna().astype(str).str.count(",") + 1
    print("tags per video:", ntag.value_counts().sort_index().to_dict(), " tag null:", int(vids.tag.isna().sum()))
    print("upload_dt counts:", vids.upload_dt.value_counts().to_dict())

    hr("Per-user activity (combined standard logs)")
    per_u = log.groupby("user_id").agg(n=("video_id", "size"), lv=("long_view", "sum"), like=("is_like", "sum"))
    q = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    print("events/user quantiles:", per_u.n.quantile(q).to_dict())
    print("long_view/user quantiles:", per_u.lv.quantile(q).to_dict())
    print("like/user quantiles:", per_u.like.quantile(q).to_dict(),
          f" users with >=1 like: {int((per_u.like > 0).sum()):,} / {len(per_u):,}")


if __name__ == "__main__":
    main()
