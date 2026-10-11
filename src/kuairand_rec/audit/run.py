"""Deterministic Dataset Audit (Phase C).

Reads ONLY the four agreed raw files (via kuairand_rec.data.io), never modifies them,
and applies NO filtering: every statistic is computed on raw rows unless the field
name says otherwise (e.g. "distinct", "events", "pairs").

Split-dependent sections (B4-B7, B9, B10) use a *provisional reference split* that
exists only for audit purposes. It is NOT a protocol decision (see PLAN.md, P1).

Usage:
    uv run python -m kuairand_rec.audit.run [--out results/audit]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kuairand_rec import paths
from kuairand_rec.audit import stats
from kuairand_rec.data import io

TZ = "Asia/Shanghai"  # inferred: time_ms agrees with `date` in this tz for 99.2% of rows
KEY = ["user_id", "video_id", "time_ms"]
LABELS = ("long_view", "is_like")
SPLITS = ("train", "val", "test")
REF_SPLIT = "row_quantile_raw"
SENTINEL_MUSIC = 0
SENTINEL_INT = -124  # documented by KuaiRand as "unavailable" for category ids


# ----------------------------------------------------------------------------- helpers
def _cst(ms) -> str:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").tz_convert(TZ).strftime("%Y-%m-%d %H:%M:%S")


def _quant(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) -> dict:
    x = np.asarray(x)
    return {str(q): float(np.quantile(x, q)) for q in qs} if len(x) else {}


def _pair(a, b, base: int = 10**6) -> np.ndarray:
    """Encode (a, b) int pairs into one int64 for fast set membership."""
    return np.asarray(a, dtype=np.int64) * base + np.asarray(b, dtype=np.int64)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class Checks:
    """Cross-checks between independent counting paths. All must pass."""

    def __init__(self) -> None:
        self.items: dict[str, bool] = {}

    def add(self, name: str, ok) -> None:
        self.items[name] = bool(ok)

    @property
    def all_ok(self) -> bool:
        return all(self.items.values())

    @property
    def failed(self) -> list[str]:
        return [k for k, v in self.items.items() if not v]


# ----------------------------------------------------------------------------- B0
def audit_integrity(log, users, vids, ck: Checks) -> dict:
    # Metadata primary keys are also asserted inside the io loaders.
    j_u = log[["user_id"]].merge(users[["user_id"]], on="user_id", how="left", validate="many_to_one")
    j_v = log[["video_id"]].merge(vids[["video_id"]], on="video_id", how="left", validate="many_to_one")
    ck.add("join_user_features_preserves_rows", len(j_u) == len(log))
    ck.add("join_video_features_preserves_rows", len(j_v) == len(log))

    ts = pd.to_datetime(log["time_ms"], unit="ms", utc=True).dt.tz_convert(TZ)
    hour_off = ((log["hourmin"] // 100 - ts.dt.hour) % 24).value_counts().sort_index()
    f0 = log[log.src_file == 0].time_ms
    f1 = log[log.src_file == 1].time_ms
    overlap = log[(log.time_ms >= f1.min()) & (log.time_ms <= f0.max())]
    return {
        "rows_per_file": log.src_file.value_counts().sort_index().to_dict(),
        "rows_total": len(log),
        "user_features": {"rows": len(users), "user_id_unique": users.user_id.is_unique},
        "video_features": {"rows": len(vids), "video_id_unique": vids.video_id.is_unique},
        "log_rows_user_missing_in_features": int((~log.user_id.isin(users.user_id)).sum()),
        "log_rows_video_missing_in_features": int((~log.video_id.isin(vids.video_id)).sum()),
        "feature_users_absent_from_logs": int((~users.user_id.isin(log.user_id)).sum()),
        "feature_videos_absent_from_logs": int((~vids.video_id.isin(log.video_id)).sum()),
        "null_cells_in_logs": int(log.isna().sum().sum()),
        "time_range_cst": {f"file{i}": [_cst(s.min()), _cst(s.max())] for i, s in ((0, f0), (1, f1))},
        "file_overlap_window_rows": {
            "file0": int((overlap.src_file == 0).sum()),
            "file1": int((overlap.src_file == 1).sum()),
            "window_cst": [_cst(f1.min()), _cst(f0.max())],
        },
        "date_field_matches_time_ms_cst_share": float((ts.dt.strftime("%Y%m%d").astype(int) == log.date).mean()),
        "hourmin_minus_time_ms_hour_cst": hour_off.to_dict(),
    }


# ----------------------------------------------------------------------------- B2
def audit_duplicates(log, ck: Checks) -> dict:
    value_cols = [c for c in log.columns if c not in KEY + ["src_file"]]
    dup = stats.classify_key_duplicates(log, KEY, value_cols)
    n_dup_pandas = int(log.duplicated(KEY + value_cols).sum())
    ck.add("redundant_exact_rows_eq_pandas_duplicated", dup["n_redundant_exact_rows"] == n_dup_pandas)
    singles = dup["n_key_groups"] - dup["n_multi_row_groups"]
    ck.add(
        "dup_classes_partition_rows",
        singles + dup["n_rows_in_exact_dup_groups"] + dup["n_rows_in_conflict_groups"] == len(log),
    )
    within = sum(int(d[KEY + value_cols].duplicated().sum()) for _, d in log.groupby("src_file"))
    dup["cross_file_exact_copies"] = n_dup_pandas - within

    # Per-row class to see where duplicates live (by tab) and their label rates.
    gsize = log.groupby(KEY, sort=False)["tab"].transform("size").to_numpy()
    ndist = log[KEY + value_cols].drop_duplicates().groupby(KEY).size().rename("nd").reset_index()
    nd = log[KEY].merge(ndist, on=KEY, how="left", validate="many_to_one")["nd"].to_numpy()
    cls = np.where(gsize == 1, "single", np.where(nd == 1, "exact_dup_group", "conflict_group"))
    by_cls = log.groupby(cls).agg(rows=("tab", "size"), long_view=("long_view", "mean"), is_like=("is_like", "mean"))
    dup["rows_by_class"] = by_cls.round(4).to_dict(orient="index")
    tab_ct = pd.crosstab(log["tab"], cls)
    dup["rows_by_tab_and_class"] = tab_ct.to_dict(orient="index")

    # Same user, same timestamp, different videos.
    st = stats.same_timestamp_multi_item(log)
    nitems = log.groupby(["user_id", "time_ms"], sort=False)["video_id"].transform("nunique")
    ck.add("same_ts_rows_two_ways", st["n_rows_in_multi_item_groups"] == int((nitems > 1).sum()))
    st["rows_in_multi_item_groups_by_tab"] = log.loc[nitems > 1, "tab"].value_counts().to_dict()
    st["rows_in_multi_item_groups_share"] = st["n_rows_in_multi_item_groups"] / len(log)

    # Genuine repeats: same (user, video) at different timestamps.
    rep = {"all_rows": stats.repeat_interactions(log)}
    for a in LABELS:
        rep[f"{a}_positive_rows"] = stats.repeat_interactions(log[log[a] == 1])
    ck.add(
        "distinct_events_consistent",
        rep["all_rows"]["n_pairs"] - rep["all_rows"]["n_pairs_repeated"]
        + rep["all_rows"]["n_events_in_repeated_pairs"]
        == dup["n_key_groups"],
    )

    vc_t = log.time_ms.value_counts()
    return {
        "key": KEY,
        "value_cols_compared": value_cols,
        "key_duplicates": dup,
        "same_user_same_timestamp": st,
        "repeat_interactions": rep,
        "global_time_ms_shared_by_multiple_rows": int((vc_t > 1).sum()),
    }


# ----------------------------------------------------------------------------- B3
def audit_labels(log, vids, ck: Checks) -> dict:
    lab_cols = ["is_click", "long_view", "is_like", "is_follow", "is_comment", "is_forward", "is_hate"]
    p, d = log.play_time_ms.to_numpy(), log.duration_ms.to_numpy()
    off = stats.official_long_view(p, d)
    dz = d == 0
    lv = log.long_view.to_numpy()
    mis = (off == 0) & (lv == 1) & ~dz  # labelled long_view although the official rule says no
    shortfall = np.minimum(d, 18000)[mis] - p[mis]
    agree = {
        "official0_label1_profile": {
            "rows": int(mis.sum()),
            "share_duration_le_18s": float((d[mis] <= 18000).mean()) if mis.any() else None,
            "shortfall_ms_quantiles": _quant(shortfall, (0.5, 0.9, 0.99, 1.0)),
            "note": "HYPOTHESIS: a ~1 s tolerance on short videos in the label pipeline (not documented).",
        },
        "all_rows": float((off == lv).mean()),
        "duration_gt_0": float((off[~dz] == lv[~dz]).mean()),
        "duration_eq_0": float((off[dz] == lv[dz]).mean()) if dz.any() else None,
        "mismatch_rows_duration_gt_0": {
            "official1_label0": int(((off == 1) & (lv == 0) & ~dz).sum()),
            "official0_label1": int(((off == 0) & (lv == 1) & ~dz).sum()),
        },
    }
    dz_rows = log[dz]
    dz_videos = dz_rows.video_id.unique()
    vd = vids.set_index("video_id")["video_duration"]
    by_tab = log.groupby("tab").agg(
        rows=("tab", "size"),
        long_view=("long_view", "mean"),
        is_like=("is_like", "mean"),
        is_click=("is_click", "mean"),
        play0=("play_time_ms", lambda s: (s == 0).mean()),
    )
    per_user = log.groupby("user_id").agg(n=("tab", "size"), lv=("long_view", "sum"), like=("is_like", "sum"))
    ck.add("per_user_positive_sums", int(per_user.lv.sum()) == int(log.long_view.sum()))
    like_play0 = log[(log.is_like == 1) & (log.play_time_ms == 0)]
    return {
        "rates_all_rows": log[lab_cols].mean().round(5).to_dict(),
        "positives_rows": {a: int(log[a].sum()) for a in LABELS},
        "rates_by_file": log.groupby("src_file")[list(LABELS)].mean().round(5).to_dict(orient="index"),
        "rates_by_tab": by_tab.round(4).to_dict(orient="index"),
        "like_x_long_view_rows": pd.crosstab(log.is_like, log.long_view).to_dict(orient="index"),
        "share_like_rows_also_long_view": float(log.loc[log.is_like == 1, "long_view"].mean()),
        "official_long_view_rule_agreement": agree,
        "duration_ms_zero": {
            "rows": int(dz.sum()),
            "videos": len(dz_videos),
            "videos_with_null_video_duration_in_metadata": int(vd.loc[dz_videos].isna().sum()),
            "long_view_rate": float(dz_rows.long_view.mean()),
            "is_like_rate": float(dz_rows.is_like.mean()),
        },
        "play_time_zero_share": float((p == 0).mean()),
        "like_with_play_time_zero": {"rows": len(like_play0), "by_tab": like_play0.tab.value_counts().to_dict()},
        "per_user": {
            "users": len(per_user),
            "rows_per_user_quantiles": _quant(per_user.n),
            "long_view_per_user_quantiles": _quant(per_user.lv),
            "like_per_user_quantiles": _quant(per_user.like),
            "users_with_ge1_long_view": int((per_user.lv > 0).sum()),
            "users_with_ge1_like": int((per_user.like > 0).sum()),
        },
    }


# ----------------------------------------------------------------------------- B4
def split_summary(log, split: np.ndarray) -> list[dict]:
    rows = []
    train_users = set(log.user_id[split == 0])
    for s, name in enumerate(SPLITS):
        d = log[split == s]
        r = {
            "split": name,
            "rows": len(d),
            "row_share": len(d) / len(log),
            "distinct_events": int(len(d[KEY].drop_duplicates())),
            "users": d.user_id.nunique(),
            "videos": d.video_id.nunique(),
            "long_view_pos_rows": int(d.long_view.sum()),
            "is_like_pos_rows": int(d.is_like.sum()),
            "long_view_rate": float(d.long_view.mean()),
            "is_like_rate": float(d.is_like.mean()),
            "start_cst": _cst(d.time_ms.min()),
            "end_cst": _cst(d.time_ms.max()),
            "span_days": (d.time_ms.max() - d.time_ms.min()) / 86_400_000,
        }
        if s > 0:
            u = d.user_id.unique()
            r["users_seen_in_train_share"] = float(np.isin(u, list(train_users)).mean())
            for a in LABELS:
                ua = d.loc[d[a] == 1, "user_id"].unique()
                r[f"users_with_{a}_pos"] = len(ua)
        rows.append(r)
    return rows


def audit_split_candidates(log, ck: Checks) -> tuple[dict, dict]:
    t_raw = log.time_ms.to_numpy()
    t_evt = log[KEY].drop_duplicates().time_ms.to_numpy()
    raw = (stats.quantile_timestamp_cut(t_raw, 0.70), stats.quantile_timestamp_cut(t_raw, 0.85))
    cands = {
        # Boundary T at the row quantile of raw rows; rows with t == T go to the later split.
        "row_quantile_raw": raw,
        # Same, but quantiles over distinct (user, video, time) events (dedup-insensitive variant).
        "row_quantile_distinct_events": (
            stats.quantile_timestamp_cut(t_evt, 0.70),
            stats.quantile_timestamp_cut(t_evt, 0.85),
        ),
        # Boundaries snapped to the nearest local (CST) midnight: day-aligned alternative.
        "local_midnight_nearest": tuple(stats.nearest_local_midnight_ms(c, TZ) for c in raw),
    }
    out = {}
    for name, cuts in cands.items():
        sp = stats.assign_split(t_raw, cuts)
        summ = split_summary(log, sp)
        ck.add(f"split_rows_sum_{name}", sum(r["rows"] for r in summ) == len(log))
        out[name] = {
            "cuts_ms": list(cuts),
            "cuts_cst": [_cst(c) for c in cuts],
            "rows_at_exact_cut_ts": [int((t_raw == c).sum()) for c in cuts],
            "rows_within_1s_of_cut": [int((np.abs(t_raw - c) < 1000).sum()) for c in cuts],
            "splits": summ,
        }
    return out, {k: v for k, v in cands.items()}


# ----------------------------------------------------------------------------- B1
def audit_entities_and_music(log, vids, ck: Checks) -> dict:
    vm = vids.set_index("video_id")
    vpm = vids.groupby("music_id").size()
    vpm_nz = vpm.drop(SENTINEL_MUSIC, errors="ignore")
    ck.add("videos_per_music_sums_to_catalog", int(vpm.sum()) == len(vids))
    shared_nz = set(vpm_nz[vpm_nz > 1].index)

    def music_cat(m: pd.Series) -> np.ndarray:
        return np.where(m == SENTINEL_MUSIC, "music_id=0", np.where(m.isin(shared_nz), "shared_nonzero", "unique_nonzero"))

    log_music = log.video_id.map(vm["music_id"])
    log_author = log.video_id.map(vm["author_id"])
    ck.add("music_map_complete", not log_music.isna().any())

    def ent(d, m, a) -> dict:
        return {
            "rows": len(d),
            "users": d.user_id.nunique(),
            "videos": d.video_id.nunique(),
            "authors": a.nunique(),
            "music_ids_incl_0": m.nunique(),
            "music_ids_nonzero": m[m != SENTINEL_MUSIC].nunique(),
        }

    entities = {"all_standard_logs": ent(log, log_music, log_author)}
    for i in (0, 1):
        msk = (log.src_file == i).to_numpy()
        entities[f"file{i}"] = ent(log[msk], log_music[msk], log_author[msk])
    for s, name in enumerate(SPLITS):
        msk = log.split.to_numpy() == s
        entities[f"ref_{name}"] = ent(log[msk], log_music[msk], log_author[msk])
    entities["metadata"] = {
        "videos": len(vids),
        "authors": vids.author_id.nunique(),
        "music_ids_incl_0": vids.music_id.nunique(),
        "music_ids_nonzero": int(len(vpm_nz)),
    }

    m0 = vids[vids.music_id == SENTINEL_MUSIC]
    pos_cat = {}
    for a in LABELS:
        for scope, msk in (("all", np.ones(len(log), bool)), ("ref_train", log.split.to_numpy() == 0),
                           ("ref_test", log.split.to_numpy() == 2)):
            sel = msk & (log[a].to_numpy() == 1)
            pos_cat[f"{a}_{scope}"] = pd.Series(music_cat(log_music[sel])).value_counts(normalize=True).round(4).to_dict()
    vid_cat = pd.Series(music_cat(vids.music_id)).value_counts().to_dict()

    # Within-user positives: distinct music vs distinct videos (how close to 1:1 for users).
    per_user_ratio = {}
    for a in LABELS:
        d = log[log[a] == 1]
        mm = log_music[log[a] == 1]
        nz = mm != SENTINEL_MUSIC
        g = pd.DataFrame({"u": d.user_id[nz], "v": d.video_id[nz], "m": mm[nz]}).groupby("u")
        per_user_ratio[a] = float((g.m.nunique() / g.v.nunique()).mean())

    vpa = vids.groupby("author_id").size()
    multi_auth = set(vpa[vpa > 1].index)
    return {
        "entities": entities,
        "music": {
            "videos_per_music_hist_nonzero": vpm_nz.value_counts().sort_index().to_dict(),
            "videos_per_music_quantiles_nonzero": _quant(vpm_nz, (0.5, 0.9, 0.99, 1.0)),
            "music_ids_nonzero_shared": int((vpm_nz > 1).sum()),
            "videos_on_shared_nonzero_music": int(vpm_nz[vpm_nz > 1].sum()),
            "top_music_by_videos_incl_0": vpm.sort_values(ascending=False, kind="stable").head(10).to_dict(),
            "videos_by_music_category": vid_cat,
            "positive_rows_share_by_music_category": pos_cat,
            "mean_per_user_distinct_music_over_distinct_videos_in_positives_nonzero": per_user_ratio,
            "music_id_0": {
                "videos": len(m0),
                "music_type_values": m0.music_type.value_counts(dropna=False).to_dict(),
                "music_type_null_total_in_metadata": int(vids.music_type.isna().sum()),
                "video_type": m0.video_type.value_counts().to_dict(),
                "upload_type_top5": m0.upload_type.value_counts().head(5).to_dict(),
                "distinct_authors": m0.author_id.nunique(),
                "log_rows": int((log_music == SENTINEL_MUSIC).sum()),
                "evidence_note": (
                    "203/204 music_id=0 videos have null music_type, and these are all 203 "
                    "null music_type values in the metadata. HYPOTHESIS: music_id=0 means "
                    "missing/no background music. The official dictionary does not define it."
                ),
            },
            "music_type_values_all": vids.music_type.value_counts(dropna=False).to_dict(),
        },
        "author": {
            "videos_per_author_hist": vpa.value_counts().sort_index().head(15).to_dict(),
            "authors_with_gt1_video": int((vpa > 1).sum()),
            "videos_on_multi_video_authors": int(vpa[vpa > 1].sum()),
            "max_videos_per_author": int(vpa.max()),
            "positive_rows_share_on_multi_video_authors": {
                a: float(log_author[log[a] == 1].isin(multi_auth).mean()) for a in LABELS
            },
        },
    }


# ----------------------------------------------------------------------------- B5 / B6
def audit_repeats_and_catalog(log, vids, ck: Checks) -> dict:
    vm = vids.set_index("video_id")
    music = log.video_id.map(vm["music_id"]).to_numpy()
    mcode, _ = pd.factorize(music, sort=True)  # compact music index for pair keys
    meta_music = vids.music_id.to_numpy()  # full metadata music set (incl. ids never in logs)
    sp = log.split.to_numpy()
    u, v = log.user_id.to_numpy(), log.video_id.to_numpy()
    uv = _pair(u, v)
    um = _pair(u, mcode)
    m_nz = music != SENTINEL_MUSIC
    out: dict = {"reconsumption": {}, "catalog_ooc": {}, "music_repeats": {}}

    tr, va, te = sp == 0, sp == 1, sp == 2
    uv_train_any = np.unique(uv[tr])
    uv_trval_any = np.unique(uv[tr | va])
    for a in LABELS:
        pos = log[a].to_numpy() == 1
        uv_train_pos = np.unique(uv[tr & pos])
        tp = te & pos
        r = {
            "test_pos_rows": int(tp.sum()),
            "test_pos_distinct_user_video_pairs": int(len(np.unique(uv[tp]))),
            "share_test_pos_rows_pair_seen_in_train_any": float(np.isin(uv[tp], uv_train_any).mean()),
            "share_test_pos_rows_pair_seen_in_train_val_any": float(np.isin(uv[tp], uv_trval_any).mean()),
            "share_test_pos_rows_pair_was_train_pos_same_action": float(np.isin(uv[tp], uv_train_pos).mean()),
        }
        # Within train: positives repeated on the same (user, video) at distinct timestamps.
        rp = stats.repeat_interactions(log[tr & pos])
        r["train_pos_pairs"] = rp["n_pairs"]
        r["train_pos_pairs_repeated_distinct_times"] = rp["n_pairs_repeated"]
        out["reconsumption"][a] = r

        # Music repeats within train positives (nonzero music only).
        trp = tr & pos & m_nz
        dfm = pd.DataFrame({"um": um[trp], "v": v[trp]})
        g = dfm.groupby("um")
        n_rows, n_vid = g.size(), g.v.nunique()
        um_train_pos = np.unique(um[trp])
        uv_train_pos_set = uv_train_pos
        tpm = tp & m_nz
        seen_m = np.isin(um[tpm], um_train_pos)
        same_video = np.isin(uv[tpm], uv_train_pos_set)
        out["music_repeats"][a] = {
            "train_pos_user_music_pairs": int(len(n_rows)),
            "pairs_with_gt1_pos_rows": int((n_rows > 1).sum()),
            "pairs_with_gt1_distinct_videos": int((n_vid > 1).sum()),
            "test_music_pos_rows_nonzero": int(tpm.sum()),
            "share_test_music_pos_user_music_was_train_pos": float(seen_m.mean()),
            "share_test_music_pos_via_same_video_train_pos": float((seen_m & same_video).mean()),
            "share_test_music_pos_only_via_other_video": float((seen_m & ~same_video).mean()),
        }

        # Catalog / OOC under three nested candidate catalogs (metadata ⊇ train_seen ⊇ train_positive).
        # The metadata level comes from the metadata table, not from log rows (see stats.build_catalogs).
        cat_v = stats.build_catalogs(vids.video_id.to_numpy(), v[tr], v[tr & pos])
        cat_m = {
            "incl_0": stats.build_catalogs(meta_music, music[tr], music[tr & pos]),
            "nonzero": stats.build_catalogs(meta_music, music[tr], music[tr & pos], exclude=(SENTINEL_MUSIC,)),
        }
        out["catalog_ooc"][a] = {
            "video": {k: {"catalog_size": len(c), "test_pos_rows_ooc_share": stats.ooc_share(v[tp], c),
                          "val_pos_rows_ooc_share": stats.ooc_share(v[va & pos], c)}
                      for k, c in cat_v.items()},
            # For nonzero catalogs, "all rows" counts music_id=0 positives as OOC; the
            # "nonzero rows" share restricts to test positives whose music_id != 0.
            "music": {f"{k}_{var}": {"catalog_size": len(c),
                                     "test_pos_rows_ooc_share": stats.ooc_share(music[tp], c),
                                     "test_pos_rows_nonzero_music_ooc_share": stats.ooc_share(music[tpm], c)}
                      for var, cats in cat_m.items() for k, c in cats.items()},
            "test_pos_rows_with_music_id_0": int((tp & ~m_nz).sum()),
        }
        for kind, cats in (("video", cat_v), ("music_incl_0", cat_m["incl_0"]), ("music_nonzero", cat_m["nonzero"])):
            ck.add(f"catalogs_nested_{kind}_{a}",
                   np.isin(cats["train_positive"], cats["train_seen"]).all()
                   and np.isin(cats["train_seen"], cats["metadata"]).all())
    ck.add("all_test_videos_in_metadata", out["catalog_ooc"]["long_view"]["video"]["metadata"]["test_pos_rows_ooc_share"] == 0)
    ck.add("metadata_music_catalog_is_metadata_table",
           len(cat_m["incl_0"]["metadata"]) == vids.music_id.nunique()
           and len(cat_m["nonzero"]["metadata"]) == vids.music_id[vids.music_id != SENTINEL_MUSIC].nunique())
    out["catalog_definitions"] = {
        "metadata": "all ids in video_features_basic_pure.csv (music: all music_id values of metadata videos)",
        "train_seen": "ids with >=1 reference-train interaction (any label)",
        "train_positive": "ids with >=1 reference-train positive for the action",
        "music_variants": "incl_0 keeps music_id=0 as an id; nonzero excludes it from every level",
        "metadata_music_ids_absent_from_logs": int((~np.isin(meta_music, music)).sum()),
    }
    return out


# ----------------------------------------------------------------------------- B7
def _hist_summary(lengths: np.ndarray) -> dict:
    b = pd.Series(stats.history_bucket(lengths)).value_counts().reindex(stats.HISTORY_BUCKETS, fill_value=0)
    return {
        "n": int(len(lengths)),
        "bucket_counts": b.to_dict(),
        "bucket_shares": (b / max(len(lengths), 1)).round(4).to_dict(),
        "quantiles": _quant(lengths, (0.1, 0.25, 0.5, 0.75, 0.9)),
        "share_ge_100": float((lengths >= 100).mean()) if len(lengths) else None,
    }


def audit_history(log, cuts, ck: Checks) -> dict:
    t1, t2 = cuts
    sp = log.split.to_numpy()
    ev = log[KEY].drop_duplicates()
    lvp = log.loc[log.long_view == 1, KEY].drop_duplicates()
    sources = {
        "all_rows": (log.user_id.to_numpy(), log.time_ms.to_numpy()),
        "all_events_dedup": (ev.user_id.to_numpy(), ev.time_ms.to_numpy()),
        "long_view_pos_events": (lvp.user_id.to_numpy(), lvp.time_ms.to_numpy()),
    }
    out: dict = {"definitions": {
        "frozen_at_test_start": "per test user with >=1 positive of the action; history = events with t < T_test_start",
        "frozen_at_train_end": "same users; history = events with t < T_val_start (excludes val events)",
        "rolling_per_event": "per test positive row; history = same-user events with t < event time (strict)",
        "note": "Counts are uncapped. Buckets: 0 / 1-5 / 6-20 / >20. share_ge_100 is the share hitting the cap.",
    }}
    for a in LABELS:
        tp = (sp == 2) & (log[a].to_numpy() == 1)
        qu, qt = log.user_id.to_numpy()[tp], log.time_ms.to_numpy()[tp]
        users = np.unique(qu)
        res = {}
        for sname, (hu, ht) in sources.items():
            frozen = stats.count_strictly_before(hu, ht, users, np.full(len(users), t2))
            rolling = stats.count_strictly_before(hu, ht, qu, qt)
            res[sname] = {"frozen_at_test_start": _hist_summary(frozen), "rolling_per_event": _hist_summary(rolling)}
            ck.add(f"hist_bucket_sum_{a}_{sname}", sum(res[sname]["rolling_per_event"]["bucket_counts"].values()) == len(qu))
            if sname == "all_events_dedup":
                fz1 = stats.count_strictly_before(hu, ht, users, np.full(len(users), t1))
                res[sname]["frozen_at_train_end"] = _hist_summary(fz1)
                ck.add(f"hist_frozen_monotone_{a}", bool((fz1 <= frozen).all()))
        # Tie impact: other same-user events at exactly the query time (excluded by the strict rule).
        same_rows = stats.count_same_time(*sources["all_rows"], qu, qt) - 1
        same_evt = stats.count_same_time(*sources["all_events_dedup"], qu, qt) - 1
        res["tie_impact_rolling"] = {
            "test_pos_rows": int(len(qu)),
            "rows_with_other_same_ts_rows": int((same_rows > 0).sum()),
            "rows_with_other_same_ts_distinct_events": int((same_evt > 0).sum()),
            "other_same_ts_distinct_events_total": int(same_evt.clip(min=0).sum()),
        }
        out[a] = res
    return out


# ----------------------------------------------------------------------------- B8
_USER_DOC = {
    "user_active_degree": "activity level; official set {high,full,middle,UNKNOWN}",
    "is_lowactive_period": "whether user is in low-active period",
    "is_live_streamer": "is the user a live streamer",
    "is_video_author": "has the user uploaded any video",
    "follow_user_num": "#users followed", "fans_user_num": "#fans", "friend_user_num": "#friends",
    "register_days": "days since registration",
}
_VIDEO_DOC = {
    "author_id": "author id", "video_type": "NORMAL or AD", "upload_dt": "upload date",
    "upload_type": "upload type", "visible_status": "visible state on the app *now*",
    "video_duration": "duration (ms)", "server_width": "width", "server_height": "height",
    "music_id": "background music id", "music_type": "background music type",
    "tag": "list of key categories",
}


def _col_profile(df: pd.DataFrame, c: str) -> dict:
    s = df[c]
    num = pd.api.types.is_numeric_dtype(s)
    return {
        "column": c, "dtype": str(s.dtype), "n_null": int(s.isna().sum()), "n_unique": int(s.nunique(dropna=True)),
        "min": s.min() if num else None, "max": s.max() if num else None,
        "n_sentinel_-124": int((s == SENTINEL_INT).sum()) if num else 0,
        "constant": bool(s.nunique(dropna=False) <= 1),
    }


def audit_columns(log, users, vids, ck: Checks) -> tuple[dict, pd.DataFrame]:
    rows = []
    for c in users.columns:
        r = {"table": "user_features", **_col_profile(users, c)}
        r["official_doc"] = _USER_DOC.get(c, "encrypted one-hot index" if c.startswith("onehot") else
                                          ("bucketised version of numeric column" if c.endswith("_range") else ""))
        rows.append(r)
    for c in vids.columns:
        r = {"table": "video_features_basic", **_col_profile(vids, c)}
        r["official_doc"] = _VIDEO_DOC.get(c, "")
        rows.append(r)
    table = pd.DataFrame(rows)

    # Range columns vs their numeric source: check that each bucket maps to a coherent numeric interval.
    range_check = {}
    for c in ("follow_user_num", "fans_user_num", "friend_user_num", "register_days"):
        g = users.groupby(f"{c}_range")[c].agg(["min", "max", "size"])
        range_check[c] = g.sort_values("min").to_dict(orient="index")

    # upload_dt vs first observed interaction (CST date).
    first = log.groupby("video_id").time_ms.min()
    first_d = pd.to_datetime(first, unit="ms", utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    up = vids.set_index("video_id").upload_dt.reindex(first_d.index)
    # log duration_ms vs metadata video_duration (per video; log duration is constant per video).
    per_v = log.groupby("video_id").duration_ms.agg(["nunique", "first"])
    vd = vids.set_index("video_id").video_duration.reindex(per_v.index)
    tags = vids.tag.dropna().astype(str).str.split(",")
    sp = log.split.to_numpy()
    tab_by_split = pd.crosstab(log.tab, np.array(SPLITS)[sp], normalize="columns").round(4)

    extra = {
        "user_active_degree_values": users.user_active_degree.value_counts().to_dict(),
        "user_active_degree_values_outside_official_set": sorted(
            set(users.user_active_degree) - {"high_active", "full_active", "middle_active", "UNKNOWN"}),
        "is_live_streamer_values": users.is_live_streamer.value_counts().to_dict(),
        "range_vs_numeric": range_check,
        "onehot_nulls": {c: int(users[c].isna().sum()) for c in users.columns if users[c].isna().any()},
        "onehot_null_rows_overlap_feat12_17": int(users[[f"onehot_feat{i}" for i in range(12, 18)]].isna().all(axis=1).sum()),
        "upload_dt_values": vids.upload_dt.value_counts().to_dict(),
        "videos_first_interaction_before_upload_dt": int((first_d < up).sum()),
        "log_duration_constant_per_video": bool((per_v["nunique"] == 1).all()),
        "log_duration_eq_metadata_share_per_video": float((per_v["first"] == vd).mean()),
        "videos_log_duration0_and_metadata_null": int(((per_v["first"] == 0) & vd.isna()).sum()),
        "tags_per_video": tags.str.len().value_counts().sort_index().to_dict(),
        "distinct_tags": int(tags.explode().nunique()),
        "tag_null_videos": int(vids.tag.isna().sum()),
        "ad_videos": int((vids.video_type == "AD").sum()),
        "ad_log_rows": int(log.video_id.isin(vids.video_id[vids.video_type == "AD"]).sum()),
        "tab_share_by_ref_split": tab_by_split.to_dict(orient="index"),
        "rows_per_tab": log.tab.value_counts().to_dict(),
    }
    ck.add("no_interaction_before_upload_dt", extra["videos_first_interaction_before_upload_dt"] == 0)
    return extra, table


# ----------------------------------------------------------------------------- B9
def _pop_drift(train_ids: np.ndarray, test_ids: np.ndarray) -> dict:
    ct = pd.Series(train_ids).value_counts()
    cs = pd.Series(test_ids).value_counts()
    idx = ct.index.union(cs.index)
    rho = spearmanr(ct.reindex(idx, fill_value=0), cs.reindex(idx, fill_value=0)).statistic
    # Deterministic top-k: by count desc, then id asc.
    rank_tr = ct.sort_index().sort_values(ascending=False, kind="stable")
    rank_te = cs.sort_index().sort_values(ascending=False, kind="stable")
    out = {"spearman_counts_union": float(rho), "ids_with_test_pos": int(len(cs)),
           "test_pos_share_on_ids_without_train_pos": float(cs[~cs.index.isin(ct.index)].sum() / cs.sum())}
    for k in (10, 50, 100):
        out[f"top{k}_overlap"] = float(len(set(rank_tr.index[:k]) & set(rank_te.index[:k])) / k)
        out[f"test_pos_share_on_train_top{k}"] = float(cs[cs.index.isin(rank_tr.index[:k])].sum() / cs.sum())
    return out


def audit_popularity_drift(log, vids) -> dict:
    music = log.video_id.map(vids.set_index("video_id").music_id).to_numpy()
    sp = log.split.to_numpy()
    out = {}
    for a in LABELS:
        pos = log[a].to_numpy() == 1
        tr, te = (sp == 0) & pos, (sp == 2) & pos
        nz = music != SENTINEL_MUSIC
        out[a] = {
            "video": _pop_drift(log.video_id.to_numpy()[tr], log.video_id.to_numpy()[te]),
            "music_nonzero": _pop_drift(music[tr & nz], music[te & nz]),
        }
    return out


# ----------------------------------------------------------------------------- B10
def audit_cold_feasibility(log, vids) -> dict:
    vm = vids.set_index("video_id")
    sp = log.split.to_numpy()
    tr_rows = log.video_id[sp == 0].value_counts()
    te_lv = log.video_id[(sp == 2) & (log.long_view == 1)].value_counts()
    te_like = log.video_id[(sp == 2) & (log.is_like == 1)].value_counts()
    # Candidate pool for a controlled holdout: has train interactions (so removing them matters)
    # and has test long_view positives (so it can be evaluated).
    pool = tr_rows.index.intersection(te_lv.index)
    train_vids = vids[vids.video_id.isin(tr_rows.index)]
    auth_train = train_vids.groupby("author_id").size()
    mus_train = train_vids[train_vids.music_id != SENTINEL_MUSIC].groupby("music_id").size()
    pa, pm = vm.loc[pool, "author_id"], vm.loc[pool, "music_id"]
    # "-1" removes the held-out video itself (it is in train_vids by construction).
    author_other = (pa.map(auth_train).fillna(0) - 1) > 0
    music_other = (pm != SENTINEL_MUSIC) & ((pm.map(mus_train).fillna(0) - 1) > 0)
    df = pd.DataFrame({"train_rows": tr_rows.reindex(pool), "test_lv": te_lv.reindex(pool),
                       "test_like": te_like.reindex(pool).fillna(0), "author_other": author_other,
                       "music_other": music_other, "music_nonzero": pm != SENTINEL_MUSIC})
    df["stratum"] = pd.qcut(df.train_rows.rank(method="first"), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    strata = df.groupby("stratum", observed=True).agg(
        videos=("train_rows", "size"), train_rows_min=("train_rows", "min"), train_rows_max=("train_rows", "max"),
        train_rows_sum=("train_rows", "sum"), test_lv_pos=("test_lv", "sum"), test_like_pos=("test_like", "sum"),
        author_has_other_train_video=("author_other", "mean"), music_shared_with_other_train_video=("music_other", "mean"))
    return {
        "pool_definition": "videos with >=1 ref-train row and >=1 ref-test long_view positive",
        "pool_videos": int(len(pool)),
        "pool_share_author_has_other_train_video": float(author_other.mean()),
        "pool_share_music_nonzero": float(df.music_nonzero.mean()),
        "pool_share_music_shared_with_other_train_video": float(music_other.mean()),
        "pool_tag_null": int(vm.loc[pool, "tag"].isna().sum()),
        "strata_by_train_rows": strata.round(4).to_dict(orient="index"),
        "videos_with_test_lv_pos_but_no_train_rows": int(len(te_lv.index.difference(tr_rows.index))),
    }


# ----------------------------------------------------------------------------- main
def run_audit(raw_dir: Path, out_dir: Path) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = Checks()
    log = io.load_standard_logs(raw_dir)
    users = io.load_user_features(raw_dir)
    vids = io.load_video_features(raw_dir)
    n_raw = len(log)

    R: dict = {"meta": {"inputs": list(io.ALLOWED_FILES), "timezone": TZ, "reference_split": REF_SPLIT,
                        "note": "No filtering applied; all counts on raw rows unless stated."}}
    R["b0_integrity"] = audit_integrity(log, users, vids, ck)
    R["b2_duplicates_and_ties"] = audit_duplicates(log, ck)
    R["b3_labels"] = audit_labels(log, vids, ck)
    R["b4_split_candidates"], cands = audit_split_candidates(log, ck)

    # Provisional reference split for downstream audit statistics only (NOT a decision).
    cuts = cands[REF_SPLIT]
    log["split"] = stats.assign_split(log.time_ms.to_numpy(), cuts)
    R["b1_entities_music_author"] = audit_entities_and_music(log, vids, ck)
    R["b5_b6_repeats_catalog"] = audit_repeats_and_catalog(log, vids, ck)
    R["b7_history_length"] = audit_history(log, cuts, ck)
    R["b8_columns"], col_table = audit_columns(log, users, vids, ck)
    R["b9_popularity_drift"] = audit_popularity_drift(log, vids)
    R["b10_cold_video_feasibility"] = audit_cold_feasibility(log, vids)
    ck.add("raw_rows_unchanged", len(log) == n_raw)
    R["consistency_checks"] = ck.items
    R["meta"]["all_checks_passed"] = ck.all_ok
    R["meta"]["failed_checks"] = ck.failed
    R["meta"]["runtime_s"] = round(time.time() - t0, 1)

    # Machine-readable outputs.
    (out_dir / "audit.json").write_text(json.dumps(_jsonable(R), indent=2, ensure_ascii=False))
    col_table.to_csv(out_dir / "column_profile.csv", index=False)
    pd.DataFrame([{"candidate": k, **s} for k, v in R["b4_split_candidates"].items() for s in v["splits"]]
                 ).to_csv(out_dir / "split_candidates.csv", index=False)
    daily = _daily_table(log)
    daily.to_csv(out_dir / "daily_volume.csv")

    from kuairand_rec.audit import figures, report  # matplotlib only needed here

    figures.make_all(log, vids, R, daily, cands, out_dir / "figures")
    (out_dir / "AUDIT.md").write_text(report.render(R))
    return R


def _daily_table(log) -> pd.DataFrame:
    day = pd.to_datetime(log.time_ms, unit="ms", utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    d = log.groupby(day).agg(rows=("tab", "size"), users=("user_id", "nunique"),
                             long_view_rate=("long_view", "mean"), is_like_rate=("is_like", "mean"),
                             file0_rows=("src_file", lambda s: int((s == 0).sum())))
    d.index.name = "date_cst"
    d["cum_row_share"] = d.rows.cumsum() / d.rows.sum()
    return d


def main(argv: list[str] | None = None) -> int:
    """Run the audit. Returns 0 only if every consistency check passed, else 1.

    Outputs are still written on failure (and flagged in audit.json / AUDIT.md) so the
    failure can be inspected, but the non-zero exit code stops scripts/CI that rely on them.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=paths.RAW_DIR)
    ap.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "audit")
    args = ap.parse_args(argv)
    R = run_audit(args.raw_dir, args.out)
    failed = [k for k, v in R["consistency_checks"].items() if not v]
    print(f"audit written to {args.out} in {R['meta']['runtime_s']}s; "
          f"checks: {len(R['consistency_checks']) - len(failed)}/{len(R['consistency_checks'])} passed")
    if failed:
        print(f"ERROR: {len(failed)} consistency check(s) FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
