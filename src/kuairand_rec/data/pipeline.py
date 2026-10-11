"""M2 data pipeline: raw logs -> cleaned, split events + catalogs + train-only vocabs/stats.

Run:  uv run python -m kuairand_rec.data.pipeline --config configs/data.yaml

Outputs (regenerated, git-ignored) in `output_dir` (default data/processed):
  events.parquet          one row per cleaned event, sorted (user_id, time_ms, video_id),
                          with split, labels, feedback, catalog rows and train-vocab indices
  video_catalog.parquet   video catalog rows + encoded metadata features (train-only vocabs)
  music_catalog.parquet   music catalog rows (non-zero metadata music ids)
  vocab_{user,video,author,music}.parquet, vocab_meta.json
  popularity_{video,music}.parquet   train-only counts per catalog row
  meta.json               cut points, config, sizes, git sha
Small, committed report in `report_dir` (default results/data_pipeline):
  data_report.json, DATA_REPORT.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kuairand_rec import paths
from kuairand_rec.data import catalog as catm
from kuairand_rec.data import clean, io, split, train_stats
from kuairand_rec.data.history import HistoryIndex
from kuairand_rec.utils import git_sha, set_seed


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else paths.PROJECT_ROOT / p


def attach_ids(events: pd.DataFrame, video_meta: pd.DataFrame, cat: catm.Catalogs) -> pd.DataFrame:
    """Add author_id / music_id from metadata and catalog rows. Many-to-one join, rows preserved."""
    m = video_meta[["video_id", "author_id", "music_id"]]
    out = events.merge(m, on="video_id", how="left", validate="many_to_one")
    assert len(out) == len(events) and out.author_id.notna().all()
    out["video_row"] = cat.video_row(out.video_id.to_numpy())
    out["music_row"] = cat.music_row(out.music_id.to_numpy())  # -1 for music_id 0
    return out


def encode_events(events: pd.DataFrame, vocabs: catm.Vocabs) -> pd.DataFrame:
    """Train-vocab indices for every event (OOV for train-unseen ids). History tokens use
    exactly these, so a val/test-only video in history is OOV, never a fresh embedding."""
    out = events.copy()
    out["user_idx"] = vocabs.user.encode(out.user_id.to_numpy())
    out["video_idx"] = vocabs.video.encode(out.video_id.to_numpy())
    out["author_idx"] = vocabs.author.encode(out.author_id.to_numpy())
    out["music_idx"] = vocabs.encode_music(out.music_id.to_numpy())
    return out


def fit_condition(events: pd.DataFrame, video_meta: pd.DataFrame, cat: catm.Catalogs,
                  actions: list[str], exclude_video_ids=()) -> dict:
    """Fit every train-only quantity for one training condition, in one place.

    events: cleaned + split + attach_ids'd events (any precomputed *_idx columns are
    overwritten). With `exclude_video_ids` (cold-video H, P1-9) the H events are removed
    *first*, so vocabs, popularity, numeric stats, metadata vocabs and the re-encoded
    event/video indices are all consistent with the H-free train set. Catalog rows are not
    condition-dependent and are unchanged.

    Returns the model-side events (H removed). The evaluation frame (GT / mask / population)
    must still be built from the unfiltered cleaned log.
    """
    ev = train_stats.remove_videos(events, exclude_video_ids) if len(exclude_video_ids) else events
    train = ev[ev.split == 0]
    vocabs = catm.build_vocabs(train, video_meta)
    num_stats = train_stats.numeric_video_stats(train, video_meta)
    ev = encode_events(ev, vocabs)
    pop = train_stats.popularity(ev[ev.split == 0], cat, actions)
    vfeat = catm.encode_video_features(video_meta, cat, vocabs, num_stats)
    mfeat = catm.encode_music_features(video_meta, cat, vocabs)
    return {"events": ev, "vocabs": vocabs, "num_stats": num_stats, "popularity": pop,
            "video_features": vfeat, "music_features": mfeat}


def unseen_summary(events: pd.DataFrame, vocabs: catm.Vocabs, actions: list[str]) -> dict:
    """Val/test entities and positives that fall outside the train vocab."""
    res = {}
    for s, name in [(1, "val"), (2, "test")]:
        d = events[events.split == s]
        r = {
            "users": int(d.user_id.nunique()),
            "users_oov": int(d.loc[d.user_idx == catm.OOV, "user_id"].nunique()),
            "videos": int(d.video_id.nunique()),
            "videos_oov": int(d.loc[d.video_idx == catm.OOV, "video_id"].nunique()),
            "authors_oov": int(d.loc[d.author_idx == catm.OOV, "author_id"].nunique()),
            "music_oov_nonzero": int(d.loc[(d.music_idx == catm.OOV), "music_id"].nunique()),
        }
        for a in actions:
            p = d[d[a] == 1]
            r[f"{a}_pos_events"] = int(len(p))
            r[f"{a}_pos_events_video_oov"] = int((p.video_idx == catm.OOV).sum())
            r[f"{a}_pos_events_music_sentinel"] = int((p.music_id == catm.MUSIC_SENTINEL).sum())
            r[f"{a}_pos_events_music_oov"] = int((p.music_idx == catm.OOV).sum())
            r[f"{a}_pos_users"] = int(p.user_id.nunique())
            r[f"{a}_pos_users_oov"] = int(p.loc[p.user_idx == catm.OOV, "user_id"].nunique())
        res[name] = r
    return res


def frozen_history_summary(hidx: HistoryIndex, events: pd.DataFrame, t: int, split_id: int,
                           max_len: int, train_vocab_users: np.ndarray) -> dict:
    """Frozen history lengths at t for users active in the given split (descriptive only)."""
    users = np.unique(events.loc[events.split == split_id, "user_id"].to_numpy())
    full = hidx.history_length(users, t)
    capped = np.minimum(full, max_len)
    b = pd.Series(pd.cut(capped, [-1, 0, 5, 20, 10**9], labels=["0", "1-5", "6-20", ">20"])).value_counts()
    return {
        "users": int(len(users)),
        "bucket_counts_capped": {k: int(v) for k, v in b.sort_index().items()},
        "share_full_ge_cap": float((full >= max_len).mean()),
        "median_full": float(np.median(full)),
        "users_not_in_train_vocab": int((~np.isin(users, train_vocab_users)).sum()),
    }


def run(cfg: dict) -> dict:
    set_seed(int(cfg["seed"]))
    tz, actions = cfg["timezone"], list(cfg["actions"])
    out_dir, rep_dir = _resolve(cfg["output_dir"]), _resolve(cfg["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    raw = io.load_standard_logs()
    video_meta = io.load_video_features()
    n_users_meta = int(len(io.load_user_features()))

    # A. cleaning
    events, clean_rep, dropped = clean.clean_logs(raw, list(cfg["label_conflict_cols"]), actions,
                                                  return_dropped=True)
    del raw
    events = clean.mark_missing_duration(events)
    clean_rep["duration_ms_zero_events_marked_missing"] = int(events.duration_ms.isna().sum())

    # B. split
    t_val = split.local_to_epoch_ms(cfg["t_val"], tz)
    t_test = split.local_to_epoch_ms(cfg["t_test"], tz)
    events["split"] = split.assign_split(events.time_ms.to_numpy(), t_val, t_test)
    split_rows = split.split_summary(events, actions, tz)
    assert sum(r["events"] for r in split_rows) == len(events)
    on_cut = int(np.isin(events.time_ms.to_numpy(), [t_val, t_test]).sum())
    # Dropped label-conflict events are kept aside only so the evaluator can count GT
    # events lost to cleaning (P1-1). Never used for history, masks or training.
    dropped["split"] = split.assign_split(dropped.time_ms.to_numpy(), t_val, t_test)

    # C. catalogs, ids, train-only vocabs + stats (normal condition: nothing excluded)
    cat = catm.build_catalogs(video_meta)
    events = attach_ids(events, video_meta, cat)
    fit = fit_condition(events, video_meta, cat, actions)
    events, vocabs, num_stats, pop = fit["events"], fit["vocabs"], fit["num_stats"], fit["popularity"]
    vfeat, mfeat = fit["video_features"], fit["music_features"]

    # D. history index (descriptive frozen-history summary; consumers rebuild it from events)
    hidx = HistoryIndex(events)
    max_len = int(cfg["history_max_len"])
    hist = {
        "val_at_T_val": frozen_history_summary(hidx, events, t_val, 1, max_len, vocabs.user.ids),
        "test_at_T_test": frozen_history_summary(hidx, events, t_test, 2, max_len, vocabs.user.ids),
    }

    # Save artifacts.
    keep = ["user_id", "video_id", "time_ms", "split", "author_id", "music_id", "video_row", "music_row",
            "user_idx", "video_idx", "author_idx", "music_idx", *actions,
            *[c for c in cfg["feedback_cols"] if c not in actions], "duration_ms", "tab"]
    hidx.events[keep].to_parquet(out_dir / "events.parquet", index=False)
    vfeat.to_parquet(out_dir / "video_catalog.parquet", index=False)
    mfeat.to_parquet(out_dir / "music_catalog.parquet", index=False)
    dropped.to_parquet(out_dir / "label_conflict_events.parquet", index=False)
    for name in ("user", "video", "author", "music"):
        getattr(vocabs, name).to_frame().to_parquet(out_dir / f"vocab_{name}.parquet", index=False)
    (out_dir / "vocab_meta.json").write_text(json.dumps(
        {c: {"specials": v.specials, "values": [str(x) for x in v.ids], "offset": v.offset}
         for c, v in vocabs.meta.items()}, indent=1))
    for k, df in pop.items():
        df.to_parquet(out_dir / f"popularity_{k}.parquet", index=False)

    sizes = {
        "video_catalog": int(len(cat.video_ids)),
        "music_catalog_nonzero": int(len(cat.music_ids)),
        "videos_with_music_sentinel": int((cat.video_music_row == catm.VIDEO_NO_MUSIC_ROW).sum()),
        "music_ids_shared_by_gt1_video": int((np.bincount(cat.video_music_row[cat.video_music_row >= 0]) > 1).sum()),
        "user_metadata_rows": n_users_meta,
        "vocab_sizes_incl_specials": {n: getattr(vocabs, n).size for n in ("user", "video", "author", "music")},
        "vocab_in_vocab_ids": {n: int(len(getattr(vocabs, n).ids)) for n in ("user", "video", "author", "music")},
        "meta_vocab_sizes": {c: v.size for c, v in vocabs.meta.items()},
        "catalog_rows_with_train_events": {
            "video": int((pop["video"].train_events > 0).sum()), "music": int((pop["music"].train_events > 0).sum())},
        "catalog_rows_with_train_positive": {
            **{f"video_{a}": int((pop["video"][f"train_{a}"] > 0).sum()) for a in actions},
            **{f"music_{a}": int((pop["music"][f"train_{a}"] > 0).sum()) for a in actions}},
    }
    meta = {
        "config": cfg, "git_sha": git_sha(),
        "t_val_ms": t_val, "t_test_ms": t_test,
        "t_val_local": split.epoch_ms_to_local(t_val, tz), "t_test_local": split.epoch_ms_to_local(t_test, tz),
        "special_indices": {"PAD": catm.PAD, "OOV": catm.OOV, "NO_MUSIC": catm.NO_MUSIC,
                            "VIDEO_NO_MUSIC_ROW": catm.VIDEO_NO_MUSIC_ROW},
        "numeric_video_stats_train": num_stats,
        "sizes": sizes,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    checks = {
        "cleaning_row_conservation": clean_rep["row_conservation_ok"],
        "cleaning_event_conservation": clean_rep["event_conservation_ok"],
        "split_partition": sum(r["events"] for r in split_rows) == len(events),
        "no_event_on_cut": on_cut == 0,
        "no_timestamp_spans_splits": bool(events.groupby("time_ms").split.nunique().max() == 1),
        "all_log_videos_in_catalog": bool((events.video_row >= 0).all()),
        "music_row_neg_iff_sentinel": bool(((events.music_row < 0) == (events.music_id == 0)).all()),
        "label_conflict_events_saved": len(dropped) == clean_rep["exclusions"]["label_conflict_events_dropped"],
        "music_catalog_rows_consistent": bool(
            (mfeat.music_id.to_numpy() == cat.music_ids).all()
            and mfeat.n_videos_metadata.sum() == (cat.video_music_row >= 0).sum()),
        "train_events_all_in_vocab": bool(
            (events.loc[events.split == 0, ["user_idx", "video_idx", "author_idx"]] > catm.OOV).all().all()),
    }
    report = {"meta": {k: meta[k] for k in ("git_sha", "t_val_local", "t_test_local")},
              "cleaning": clean_rep, "split": split_rows, "events_on_cut_points": on_cut,
              "catalog_and_vocab": sizes, "train_unseen": unseen_summary(events, vocabs, actions),
              "frozen_history": hist, "checks": checks}
    (rep_dir / "data_report.json").write_text(json.dumps(report, indent=1, default=int))
    (rep_dir / "DATA_REPORT.md").write_text(render_markdown(report, actions))
    return report


def render_markdown(r: dict, actions: list[str]) -> str:
    c = r["cleaning"]
    L = ["# M2 Data Pipeline Report", "",
         f"Generated by `uv run python -m kuairand_rec.data.pipeline`. git `{r['meta']['git_sha']}`. "
         f"Cuts (Asia/Shanghai): T_val = {r['meta']['t_val_local']}, T_test = {r['meta']['t_test_local']}.", "",
         f"Checks: {sum(r['checks'].values())}/{len(r['checks'])} passed"
         + ("" if all(r["checks"].values()) else f" -- FAILED: {[k for k, v in r['checks'].items() if not v]}"), "",
         "## 1. Cleaning (PLAN.md P1-4)", "",
         "| stage | rows | distinct events | users | videos | " + " | ".join(f"{a} pos rows" for a in actions) + " |",
         "|---|---:|---:|---:|---:|" + "---:|" * len(actions)]
    for k in ("before", "after_exact_dedup", "after"):
        s = c[k]
        L.append(f"| {k} | {s['rows']:,} | {s['distinct_events']:,} | {s['users']:,} | {s['videos']:,} | "
                 + " | ".join(f"{s[f'{a}_positive_rows']:,}" for a in actions) + " |")
    L += ["", "Exclusions:", ""] + [f"- {k}: {v}" for k, v in c["exclusions"].items()]
    L += [f"- duration_ms=0 events marked missing (kept): {c['duration_ms_zero_events_marked_missing']:,}", "",
          "## 2. Temporal split (P1-3)", "",
          "| split | events | share | users | videos | first | last | " + " | ".join(f"{a} pos (rate)" for a in actions) + " |",
          "|---|---:|---:|---:|---:|---|---|" + "---:|" * len(actions)]
    for s in r["split"]:
        L.append(f"| {s['split']} | {s['events']:,} | {s['share']:.1%} | {s['users']:,} | {s['videos']:,} | "
                 f"{s['first_event']} | {s['last_event']} | "
                 + " | ".join(f"{s[f'{a}_positive_events']:,} ({s[f'{a}_rate']:.2%})" for a in actions) + " |")
    L += [f"", f"Events exactly on a cut point: {r['events_on_cut_points']}.", "",
          "## 3. Catalogs and train-only vocabularies (P1-5/6/6b)", ""]
    L += [f"- {k}: {v}" for k, v in r["catalog_and_vocab"].items()]
    L += ["", "## 4. Train-unseen entities in val/test", "",
          "| key | val | test |", "|---|---:|---:|"]
    for k in r["train_unseen"]["val"]:
        L.append(f"| {k} | {r['train_unseen']['val'][k]:,} | {r['train_unseen']['test'][k]:,} |")
    L += ["", "## 5. Frozen history (H1, all events, strict t < T; capped at 100 for buckets)", ""]
    for k, h in r["frozen_history"].items():
        L.append(f"- {k}: users {h['users']:,}; buckets {h['bucket_counts_capped']}; "
                 f"share full>=cap {h['share_full_ge_cap']:.1%}; median full {h['median_full']:.0f}; "
                 f"not in train user vocab {h['users_not_in_train_vocab']:,}")
    L += ["", "## 6. Checks", ""] + [f"- {k}: {'ok' if v else 'FAIL'}" for k, v in r["checks"].items()]
    return "\n".join(L) + "\n"


def load_processed(out_dir: Path | str = paths.PROCESSED_DIR) -> dict:
    """Reload the saved artifacts for M3/M4/export. Returns events, catalogs, vocabs, pop, meta.

    Vocab objects are reconstructed from the saved tables, so downstream code uses the
    exact same raw-id -> index mapping as the pipeline run.

    WARNING: every *_idx column and the popularity tables are for the *normal* training
    condition. A cold-video run must not reuse them; call `fit_condition(...,
    exclude_video_ids=H)` on the events instead (raw ids, rows and labels are reusable).
    """
    d = Path(out_dir)
    meta = json.loads((d / "meta.json").read_text())
    vcat = pd.read_parquet(d / "video_catalog.parquet")
    mcat = pd.read_parquet(d / "music_catalog.parquet")
    cat = catm.Catalogs(vcat.video_id.to_numpy(np.int64), mcat.music_id.to_numpy(np.int64),
                        vcat.music_row.to_numpy(np.int64))

    def _v(name, offset, specials):
        f = pd.read_parquet(d / f"vocab_{name}.parquet")
        assert (f["index"].to_numpy() == np.arange(len(f)) + offset).all()
        return catm.Vocab(name, f.raw_id.to_numpy(np.int64), offset, specials)

    vm = json.loads((d / "vocab_meta.json").read_text())
    vocabs = catm.Vocabs(
        user=_v("user", 2, {"PAD": 0, "OOV": 1}), video=_v("video", 2, {"PAD": 0, "OOV": 1}),
        author=_v("author", 2, {"PAD": 0, "OOV": 1}),
        music=_v("music", 3, {"PAD": 0, "OOV": 1, "NO_MUSIC": 2}),
        meta={c: catm.Vocab(c, np.asarray(x["values"], dtype=object), x["offset"], x["specials"])
              for c, x in vm.items()},
    )
    return {
        "events": pd.read_parquet(d / "events.parquet"),
        "label_conflict_events": pd.read_parquet(d / "label_conflict_events.parquet"),
        "catalogs": cat, "video_catalog": vcat, "music_catalog": mcat, "vocabs": vocabs, "meta": meta,
        "popularity": {k: pd.read_parquet(d / f"popularity_{k}.parquet") for k in ("video", "music")},
    }


def prediction_time(meta: dict, split_name: str) -> int:
    """Frozen prediction time for evaluation: val -> T_val (train-only history),
    test/export -> T_test (train + val history)."""
    return {"val": meta["t_val_ms"], "test": meta["t_test_ms"]}[split_name]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.PROJECT_ROOT / "configs" / "data.yaml"))
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    rep = run(cfg)
    failed = [k for k, v in rep["checks"].items() if not v]
    print(json.dumps({"checks_failed": failed, "split": [(s["split"], s["events"]) for s in rep["split"]]}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
