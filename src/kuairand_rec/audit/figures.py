"""Report figures for the dataset audit. Every plot is computed from the raw logs/metadata."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker  # noqa: F401  (used as matplotlib.ticker.*)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kuairand_rec.audit import stats  # noqa: E402

TZ = "Asia/Shanghai"
CAND_STYLE = {
    "row_quantile_raw": ("tab:red", "-", "row-quantile (raw rows)"),
    "row_quantile_distinct_events": ("tab:purple", ":", "row-quantile (distinct events)"),
    "local_midnight_nearest": ("tab:gray", "--", "nearest CST midnight"),
}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def music_sharing(vids: pd.DataFrame, R: dict, out: Path) -> None:
    vpm = vids.groupby("music_id").size()
    nz = vpm.drop(0, errors="ignore")
    hist = nz.value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    ax = axes[0]
    ax.bar(hist.index.astype(str), hist.values, color="tab:blue", label="non-zero music_id")
    ax.bar(["0 (sentinel)"], [1], color="tab:red", label=f"music_id=0: 1 id, {int(vpm.get(0, 0))} videos")
    ax.set_yscale("log")
    ax.set_xlabel("videos per music_id")
    ax.set_ylabel("number of music_ids (log)")
    ax.set_title(f"Videos per music_id (metadata, {len(vids):,} videos)")
    ax.legend(fontsize=8)

    ax = axes[1]
    cats = ["unique_nonzero", "shared_nonzero", "music_id=0"]
    colors = ["tab:blue", "tab:orange", "tab:red"]
    pc = R["b1_entities_music_author"]["music"]["positive_rows_share_by_music_category"]
    vc = R["b1_entities_music_author"]["music"]["videos_by_music_category"]
    groups = {"videos\n(metadata)": {k: v / len(vids) for k, v in vc.items()},
              "long_view pos\n(all rows)": pc["long_view_all"], "is_like pos\n(all rows)": pc["is_like_all"]}
    left = np.zeros(len(groups))
    for c, col in zip(cats, colors):
        vals = np.array([g.get(c, 0.0) for g in groups.values()])
        ax.barh(list(groups), vals, left=left, color=col, label=c)
        for i, (l, w) in enumerate(zip(left, vals)):
            if w > 0.015:
                ax.text(l + w / 2, i, f"{w:.1%}", ha="center", va="center", fontsize=7, color="white")
        left += vals
    ax.set_xlim(0, 1)
    ax.set_xlabel("share")
    ax.set_title("Share by music category")
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, out / "music_videos_per_id.png")


def daily_volume(daily: pd.DataFrame, cands: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    x = pd.to_datetime(daily.index).tz_localize(TZ)
    f0, f1 = daily.file0_rows, daily.rows - daily.file0_rows
    width = pd.Timedelta(hours=20)
    centers = x + pd.Timedelta(hours=12)
    ax.bar(centers, f0, width=width, color="tab:blue", label="log_standard_4_08_to_4_21")
    ax.bar(centers, f1, width=width, bottom=f0, color="tab:green", label="log_standard_4_22_to_5_08")
    ax.set_ylabel("raw log rows per CST day")
    for name, cuts in cands.items():
        col, ls, lab = CAND_STYLE[name]
        for j, c in enumerate(cuts):
            t = pd.Timestamp(c, unit="ms", tz="UTC").tz_convert(TZ)
            ax.axvline(t, color=col, ls=ls, lw=1.4, label=lab if j == 0 else None)
    ax2 = ax.twinx()
    ax2.plot(centers, daily.long_view_rate, "k.-", lw=0.8, label="long_view rate")
    ax2.plot(centers, daily.is_like_rate * 10, "m.-", lw=0.8, label="is_like rate x10")
    ax2.set_ylabel("rate")
    ax2.set_ylim(0, 0.6)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    ax.set_title("Daily interaction volume (standard logs, CST) with candidate 70/85% split boundaries")
    fig.autofmt_xdate()
    _save(fig, out / "daily_volume_splits.png")


def history_length(log: pd.DataFrame, cuts: tuple[int, int], R: dict, out: Path) -> None:
    """History lengths at prediction time for reference-test positives (long_view), uncapped."""
    key = ["user_id", "video_id", "time_ms"]
    ev = log[key].drop_duplicates()
    lvp = log.loc[log.long_view == 1, key].drop_duplicates()
    tp = (log.split.to_numpy() == 2) & (log.long_view.to_numpy() == 1)
    qu, qt = log.user_id.to_numpy()[tp], log.time_ms.to_numpy()[tp]
    users = np.unique(qu)
    series = {}
    for src_name, src in (("all events", ev), ("long_view positives", lvp)):
        hu, ht = src.user_id.to_numpy(), src.time_ms.to_numpy()
        series[f"frozen @ test start | {src_name}"] = stats.count_strictly_before(hu, ht, users, np.full(len(users), cuts[1]))
        series[f"rolling per event | {src_name}"] = stats.count_strictly_before(hu, ht, qu, qt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    names = list(series)
    w = 0.2
    xs = np.arange(len(stats.HISTORY_BUCKETS))
    for i, n in enumerate(names):
        b = pd.Series(stats.history_bucket(series[n])).value_counts(normalize=True).reindex(stats.HISTORY_BUCKETS, fill_value=0)
        bars = ax.bar(xs + (i - 1.5) * w, b.values, width=w, label=f"{n} (n={len(series[n]):,})")
        for r, v in zip(bars, b.values):
            ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.0%}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(xs, ["0 (no history)", "1-5", "6-20", ">20"])
    ax.set_ylabel("share")
    ax.set_title("History-length buckets, ref-test long_view positives")
    ax.legend(fontsize=6.5)

    ax = axes[1]
    bins = np.concatenate([[0, 1], np.unique(np.round(np.logspace(0, np.log10(3000), 40)))[1:]])
    for n in names:
        ax.hist(np.clip(series[n], 0, 3000), bins=bins, histtype="step", density=True, label=n)
    for edge in (1, 6, 21, 100):
        ax.axvline(edge, color="gray", lw=0.6, ls="--")
    ax.text(100, ax.get_ylim()[1] * 0.9, " cap=100", fontsize=7, color="gray")
    ax.set_xscale("symlog", linthresh=1)
    # Lengths are counts >= 0. Autoscaling a symlog axis pads below 0 and shows negative
    # ticks, so pin the left edge at 0 and use explicit non-negative ticks.
    ax.set_xlim(0, bins[-1])
    ticks = [0, 1, 5, 20, 100, 1000]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("history length (uncapped; strict t < prediction time)")
    ax.set_ylabel("density")
    ax.set_title("History-length distribution")
    ax.legend(fontsize=6.5)
    _save(fig, out / "history_length.png")


def action_by_tab(R: dict, out: Path) -> None:
    t = pd.DataFrame(R["b3_labels"]["rates_by_tab"]).T
    t.index = t.index.astype(int)
    t = t.sort_index()
    o = R["b3_labels"]["rates_all_rows"]
    # One panel per quantity, each on its own true-percentage axis. Like rates (~2%) are an
    # order of magnitude below long_view rates (~33%), so they are not drawn on a shared,
    # rescaled axis. Values are the unchanged rates from audit.json.
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, col, color, overall, d in (
        (axes[0], "long_view", "tab:blue", o["long_view"], 0),
        (axes[1], "is_like", "tab:pink", o["is_like"], 1),
    ):
        bars = ax.bar(t.index, t[col], width=0.6, color=color)
        for b, v in zip(bars, t[col]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{100 * v:.{d + 1}f}%", ha="center", va="bottom", fontsize=6.5)
        ax.axhline(overall, color="gray", ls="--", lw=0.8, label=f"overall {100 * overall:.{d + 1}f}%")
        ax.set_ylim(0, t[col].max() * 1.2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=d))
        ax.set_ylabel(f"{col} rate")
        ax.legend(fontsize=7, loc="upper right")
    axes[2].bar(t.index, t.rows, width=0.6, color="gray")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("raw rows (log)")
    axes[2].set_xticks(t.index)
    axes[2].set_xlabel("tab (scenario id)")
    axes[0].set_title("Action rates by tab (raw rows, no filtering)")
    _save(fig, out / "action_by_tab.png")


def make_all(log, vids, R, daily, cands, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    music_sharing(vids, R, out)
    daily_volume(daily, cands, out)
    history_length(log, cands[R["meta"]["reference_split"]], R, out)
    action_by_tab(R, out)
