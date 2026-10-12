"""Final assignment-compliance audit: independent, read-only evidence on real data and formal artifacts.

Run (CPU, never trains, never writes into results/ or the formal artifact dirs):
    uv run python scripts/audit_final_compliance.py checks     # ~2-4 min
    uv run python scripts/audit_final_compliance.py rescore    # ~10 min: re-score the 5 formal exports locally
Outputs: artifacts/final_compliance_review/evidence/{checks,rescore}.json (+ printed summary).
Exit code 1 if any check fails.

"Independent" here means: the expected values are recomputed with plain pandas / numpy from
`events.parquet`, the raw CSVs or the formal result files, not by calling the code under audit.
The objects under audit (BatchBuilder, PosTrain, loss, exports, frames) are only *read* and
compared against those recomputations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kuairand_rec import paths  # noqa: E402

OUT = ROOT / "artifacts" / "final_compliance_review" / "evidence"
RUNS = {"u0": "artifacts/u0/9218a76_seed0_job27", "u1": "artifacts/u1/9218a76_seed0_job28",
        "cold_u0": "artifacts/cold_u0/671dde9_seed0_job33", "abl_nohist": "artifacts/abl_nohist/671dde9_seed0_job34",
        "abl_nomusic": "artifacts/abl_nomusic/671dde9_seed0_job35"}
TAU = 0.05
D_MS = 20 * 60_000


class Checks:
    def __init__(self):
        self.res, self.failed = {}, []

    def add(self, name, ok, **info):
        self.res[name] = {"ok": bool(ok), **info}
        if not ok:
            self.failed.append(name)
        print(("PASS " if ok else "FAIL ") + name, {k: v for k, v in info.items() if not isinstance(v, (list, dict))})


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ============================================================================ A. dataset
def check_dataset(ck: Checks, P: dict):
    raw = paths.RAW_DIR
    logs = [pd.read_csv(raw / f, usecols=["user_id", "video_id", "time_ms", "long_view", "is_like"])
            for f in paths.STANDARD_LOGS]
    log = pd.concat(logs, ignore_index=True)
    vm = pd.read_csv(raw / paths.VIDEO_FEATURES_BASIC)
    uf = pd.read_csv(raw / paths.USER_FEATURES)
    ev = P["events"]
    n_rows = [len(x) for x in logs]
    ck.add("A1_raw_row_counts", sum(n_rows) == 1_436_609, rows_per_file=n_rows, total=sum(n_rows),
           users=int(log.user_id.nunique()), videos=int(log.video_id.nunique()),
           metadata_videos=int(len(vm)), metadata_users=int(len(uf)),
           authors_metadata=int(vm.author_id.nunique()),
           authors_logged=int(vm[vm.video_id.isin(log.video_id.unique())].author_id.nunique()))
    # Music sharing in the metadata (every catalog video).
    nz = vm[vm.music_id != 0]
    per_music = nz.groupby("music_id").size()
    hist = per_music.value_counts().sort_index()
    ck.add("A2_music_sharing", int(per_music.size) == 7201 and int((per_music > 1).sum()) == 112,
           nonzero_music_ids=int(per_music.size), music_ids_shared_gt1=int((per_music > 1).sum()),
           videos_on_shared_music=int(per_music[per_music > 1].sum()), max_videos_per_music=int(per_music.max()),
           videos_music_id_0=int((vm.music_id == 0).sum()), videos_per_music_hist={int(k): int(v) for k, v in hist.items()},
           music_ids_never_logged=int((~per_music.index.isin(vm[vm.video_id.isin(log.video_id.unique())].music_id)).sum()))
    # No silent drops: cleaned keys = distinct raw keys minus the dropped label-conflict keys.
    key = ["user_id", "video_id", "time_ms"]
    raw_keys = log[key].drop_duplicates()
    lc = P["label_conflict_events"]
    exp_n = len(raw_keys) - len(lc)
    merged = ev[key].merge(raw_keys, on=key, how="left", indicator=True)
    lc_in_ev = ev[key].merge(lc[key], on=key, how="inner")
    ck.add("A3_no_silent_row_drop", len(ev) == exp_n and (merged["_merge"] == "both").all() and len(lc_in_ev) == 0
           and ev.user_id.nunique() == log.user_id.nunique() and ev.video_id.nunique() == log.video_id.nunique(),
           raw_distinct_events=int(len(raw_keys)), label_conflict_events_dropped=int(len(lc)),
           cleaned_events=int(len(ev)), users_kept=int(ev.user_id.nunique()), videos_kept=int(ev.video_id.nunique()))
    # Positive counts per split (cleaned events).
    pos = {s: {a: int(ev.loc[ev.split == i, a].sum()) for a in ("long_view", "is_like")}
           for i, s in enumerate(("train", "val", "test"))}
    ck.add("A4_positive_counts", pos["train"] == {"long_view": 339386, "is_like": 18771}
           and pos["test"] == {"long_view": 65812, "is_like": 3610}, by_split=pos,
           action_rates_raw={"long_view": float(log.long_view.mean()), "is_like": float(log.is_like.mean())},
           users_with_like_raw=int(log.loc[log.is_like == 1, "user_id"].nunique()))
    # The monthly statistic file: no column of it reaches any processed artifact / model input.
    stat_cols = set(pd.read_csv(raw / "video_features_statistic_pure.csv", nrows=1).columns) - {"video_id"}
    proc_cols = set()
    for f in sorted(paths.PROCESSED_DIR.glob("*.parquet")):
        proc_cols |= set(pd.read_parquet(f).columns)
    from kuairand_rec.data import io
    refused = False
    try:
        io._read("video_features_statistic_pure.csv", raw)
    except ValueError:
        refused = True
    from kuairand_rec.models import features as F
    vf_fields = [f for f in F.VideoFeatures.__dataclass_fields__]
    ck.add("A5_statistic_file_never_used", refused and not (stat_cols & proc_cols),
           loader_refuses_file=refused, n_statistic_columns=len(stat_cols),
           overlap_with_processed_columns=sorted(stat_cols & proc_cols), video_feature_fields=vf_fields,
           video_metadata_sources=["video_type", "upload_type", "upload_dt", "music_type", "tag",
                                   "video_duration", "server_width", "server_height", "author_id", "music_id"])
    return log, vm, uf


# ============================================================================ B. temporal
def check_temporal(ck: Checks, P: dict, uf: pd.DataFrame):
    from kuairand_rec.data.split import local_to_epoch_ms
    from kuairand_rec.models import data as D
    from kuairand_rec.models import features as F
    ev, meta = P["events"], P["meta"]
    tv = local_to_epoch_ms("2022-04-17 00:00:00", "Asia/Shanghai")
    tt = local_to_epoch_ms("2022-04-26 00:00:00", "Asia/Shanghai")
    sp = np.where(ev.time_ms < tv, 0, np.where(ev.time_ms < tt, 1, 2))
    share = [float((sp == i).mean()) for i in range(3)]
    ck.add("B1_split_recomputed", (sp == ev.split.to_numpy()).all() and tv == meta["t_val_ms"] and tt == meta["t_test_ms"]
           and not ev.time_ms.isin([tv, tt]).any() and ev.groupby("time_ms").split.nunique().max() == 1,
           t_val_ms=tv, t_test_ms=tt, shares=share)

    rng = np.random.default_rng(20261012)
    hcfg = F.HistoryConfig()
    srt = ev.sort_values(["user_id", "time_ms", "video_id"], kind="mergesort")
    # B2: frozen history (export path) at T_val / T_test, 600 random users, vs pandas.
    users = np.sort(rng.choice(ev.user_id.unique(), 600, replace=False))
    sub = srt[srt.user_id.isin(users)]
    bad = 0
    n_hidden = 0
    for split, t in (("val", tv), ("test", tt)):
        hidx, t_pred = F.frozen_history_index(ev, split, meta)
        hb = F.build_history_batch(hidx, users, t_pred, hcfg)
        for i, u in enumerate(users):
            h = sub[(sub.user_id == u) & (sub.time_ms < t)].tail(100)
            want = h.video_row.to_numpy()
            got = hb.video_row[i][hb.valid[i]]
            hid_want = (t - h.time_ms.to_numpy()) < D_MS
            hid_got = hb.hidden[i][hb.valid[i]].astype(bool)
            fb_zero = (hb.feedback[i][hb.valid[i]][hid_got] == 0).all()
            n_hidden += int(hid_want.sum())
            if not (np.array_equal(want, got) and np.array_equal(hid_want, hid_got) and fb_zero):
                bad += 1
    ck.add("B2_frozen_history_strict_past_cap100_delay", bad == 0, users_checked=int(len(users)), splits=["val", "test"],
           mismatches=bad, hidden_tokens_seen=n_hidden)
    # B3: rolling training history for 600 random training samples (train events only).
    cond = D.normal_condition(P)
    s = D.build_samples(cond)
    hidx = F.train_history_index(cond.events)
    pick = np.sort(rng.choice(len(s), 600, replace=False))
    hb = F.build_history_batch(hidx, s.user_id[pick], s.time_ms[pick], hcfg)
    trs = srt[srt.split == 0]
    trs_by_u = {u: g for u, g in trs[trs.user_id.isin(np.unique(s.user_id[pick]))].groupby("user_id")}
    bad = 0
    for i, j in enumerate(pick):
        g = trs_by_u[s.user_id[j]]
        want = g[g.time_ms < s.time_ms[j]].tail(100).video_row.to_numpy()
        if not np.array_equal(want, hb.video_row[i][hb.valid[i]]):
            bad += 1
    same_ts = int(sum(((trs_by_u[s.user_id[j]].time_ms == s.time_ms[j]).sum() > 1) for j in pick))
    ck.add("B3_rolling_train_history_strict_past", bad == 0, samples_checked=int(len(pick)), mismatches=bad,
           samples_with_same_timestamp_siblings=same_ts)
    # B4: seen mask in the formal U0 export = ALL pre-T_test interactions (uncapped), vs pandas.
    exp = ROOT / RUNS["u0"] / "export"
    uid = np.load(exp / "user_ids_test.npy")
    bad, n_over_100 = 0, 0
    for tgt, col in (("video", "video_row"), ("music", "music_row")):
        ip, rows = np.load(exp / f"seen_test_{tgt}_indptr.npy"), np.load(exp / f"seen_test_{tgt}_rows.npy")
        r = np.searchsorted(uid, users)
        for i, u in enumerate(users):
            h = sub[(sub.user_id == u) & (sub.time_ms < tt)]
            want = np.unique(h[col].to_numpy())
            want = want[want >= 0]
            got = np.sort(rows[ip[r[i]]:ip[r[i] + 1]])
            if tgt == "video":
                n_over_100 += int(len(want) > 100)
            if not np.array_equal(want, got):
                bad += 1
    ck.add("B4_export_seen_mask_uncapped", bad == 0, users_checked=int(len(users)), mismatches=bad,
           users_with_more_than_100_seen_videos=n_over_100)
    # B5: every fitted quantity comes from train only.
    tr = ev[ev.split == 0]
    pv = P["popularity"]["video"]
    ok_pop = all((np.bincount(tr.video_row, weights=tr[a], minlength=len(pv)).astype(int) == pv[f"train_{a}"]).all()
                 for a in ("long_view", "is_like"))
    voc = P["vocabs"]
    vmeta = pd.read_csv(paths.RAW_DIR / paths.VIDEO_FEATURES_BASIC)
    seen_v = np.unique(tr.video_id)
    ok_voc = (np.array_equal(voc.user.ids, np.unique(tr.user_id)) and np.array_equal(voc.video.ids, seen_v)
              and np.array_equal(voc.author.ids, np.unique(vmeta[vmeta.video_id.isin(seen_v)].author_id)))
    num = P["meta"]["numeric_video_stats_train"]
    x = np.log1p(vmeta[vmeta.video_id.isin(seen_v)].video_duration.to_numpy(float))
    ok_num = abs(np.nanmean(x) - num["video_duration"]["mean"]) < 1e-12
    from kuairand_rec.baselines.als import train_fingerprint
    sel = json.loads((ROOT / "results/baselines/run_meta.json").read_text())["als_selected"]
    ok_als = all(train_fingerprint(tr, k.split("_", 1)[0], k.split("/")[0].split("_", 1)[1]) == v["fingerprint"]
                 for k, v in sel.items())
    ck.add("B5_train_only_fits", ok_pop and ok_voc and ok_num and ok_als, popularity_equals_train_bincount=ok_pop,
           vocabs_equal_train_ids=ok_voc, numeric_scaler_equals_train_videos=ok_num,
           als_fit_fingerprints_equal_train_view=ok_als, als_models_checked=len(sel))
    # B6: user snapshot date bound (P1-8, U1 leakage caveat).
    first = ev.groupby("user_id").time_ms.min()
    u = uf.set_index("user_id").reindex(first.index)
    bound = pd.to_datetime(first, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.normalize() \
        + pd.to_timedelta(u.register_days, unit="D")
    t_test_d = pd.Timestamp("2022-04-26", tz="Asia/Shanghai")
    end_d = pd.Timestamp("2022-05-08", tz="Asia/Shanghai")
    ck.add("B6_user_snapshot_date_upper_bound", True, earliest_upper_bound=str(bound.min().date()),
           user_of_earliest_bound=int(bound.idxmin()), users_with_bound_before_log_end=int((bound < end_d).sum()),
           users_with_bound_before_T_test=int((bound < t_test_d).sum()),
           note="snapshot_date <= first_event_date + register_days (registration precedes the first event); "
                "no lower bound exists, so the snapshot cannot be shown to precede T_val / T_test")
    return cond, s


# ============================================================================ D. loss / FN mask / balance
def check_loss(ck: Checks, P: dict, cond, base):
    import torch
    from kuairand_rec.models import artifacts as A
    from kuairand_rec.models import data as D
    from kuairand_rec.models import features as F
    from kuairand_rec.models import loss as L
    from kuairand_rec.models import train as TR
    from kuairand_rec.models.towers import TASKS
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    ev = P["events"]
    tr = cond.train_events()
    pos = D.PosTrain.build(base, cond.vocabs.user.size)
    builder = D.BatchBuilder(cond, base, pos, F.HistoryConfig())
    batches = D.epoch_batches(len(base), 1024, 0, 0)          # the formal epoch-0 order (seed 0, B = 1024)
    pick = [0, 1, 150, len(batches) - 2]
    model, _ = A.load_model(ROOT / RUNS["u0"] / "best.pt", cond, None, "cpu")
    items = TR.ItemInputs.from_condition(cond)
    with torch.no_grad():
        enc = model.encode_items(items.vf, items.mf)
        ie = model.item_embeddings(enc)
    # Independent per-(user, task) positive sets, from train and from val/test (raw events).
    def col_sets(frame, task):
        tgt, act = task.split("_", 1)
        f = frame[frame[act] == 1]
        if tgt == "video":
            c = cond.video_col[f.video_row.to_numpy()]
        else:
            f = f[f.music_row >= 0]
            c = cond.music_col[f.music_row.to_numpy()]
        return pd.Series(c, index=f.user_id.to_numpy()).groupby(level=0).agg(lambda x: set(x.tolist())).to_dict()
    tr_sets = {t: col_sets(tr, t) for t in TASKS}
    fut_sets = {t: col_sets(ev[ev.split > 0], t) for t in TASKS}
    stats = {t: dict(rows=0, valid=0, fn_pairs=0, dup_label_rows=0, dup_label_cols=0, same_user_multi_rows=0,
                     shared_music_label_rows=0, rows_with_future_only_pos=0, future_only_pos_masked=0,
                     max_loss_abs_diff=0.0, label_masked=0) for t in TASKS}
    n_vid_on_music = np.bincount(cond.video_catalog.music_row[cond.video_catalog.music_row >= 0],
                                 minlength=len(cond.music_catalog))
    sev = tr.reset_index(drop=True).iloc[base.event_pos]
    bad_label = bad_mask = 0
    for bi in pick:
        idx = batches[bi]
        b = builder(idx)
        with torch.no_grad():
            ue = model.user_embeddings(model.encode_users(b.user_idx, b.history, enc, None))
            _, info = L.multitask_loss(ue, ie, items.cand_rows, b.targets, TAU)
        for t in TASKS:
            tgt = t.split("_", 1)[0]
            tt = b.targets[t]
            valid = tt.valid.numpy()
            lab = tt.label.numpy()
            fr, fc = tt.fn_rows.numpy(), tt.fn_cols.numpy()
            st = stats[t]
            st["rows"] += len(idx)
            st["valid"] += int(valid.sum())
            st["fn_pairs"] += int(len(fr))
            users = base.user_id[idx]
            rows_ev = sev.iloc[idx]
            exp_lab = (cond.video_col[rows_ev.video_row.to_numpy()] if tgt == "video"
                       else np.where(rows_ev.music_row.to_numpy() >= 0,
                                     cond.music_col[np.maximum(rows_ev.music_row.to_numpy(), 0)], -1))
            if not np.array_equal(np.where(valid, exp_lab, -1), lab):
                bad_label += 1
            masks = [set() for _ in idx]
            for r, c in zip(fr, fc):
                masks[r].add(int(c))
            vr = np.flatnonzero(valid)
            for r in vr:
                want = tr_sets[t].get(users[r], set()) - {int(lab[r])}
                if masks[r] != want:
                    bad_mask += 1
                if int(lab[r]) in masks[r]:
                    st["label_masked"] += 1
                fut_only = fut_sets[t].get(users[r], set()) - tr_sets[t].get(users[r], set())
                if fut_only:
                    st["rows_with_future_only_pos"] += 1
                    st["future_only_pos_masked"] += len(fut_only & masks[r])
            lv = pd.Series(lab[vr])
            vc = lv.value_counts()
            st["dup_label_cols"] += int((vc > 1).sum())
            st["dup_label_rows"] += int(vc[vc > 1].sum())
            uc = pd.Series(users[vr]).value_counts()
            st["same_user_multi_rows"] += int(uc[uc > 1].sum())
            if tgt == "music":
                mrow = cond.cand_music_rows[lab[vr]]
                st["shared_music_label_rows"] += int((n_vid_on_music[mrow] > 1).sum())
            # Independent float64 loss over the same embeddings (full softmax over candidates).
            U = ue[t].numpy().astype(np.float64)[vr]
            C = ie[t][items.cand_rows[tgt]].numpy().astype(np.float64)
            Z = U @ C.T / TAU
            for k, r in enumerate(vr):
                Z[k, list(masks[r])] = -np.inf
            zl = Z[np.arange(len(vr)), lab[vr]]
            m = Z.max(1, keepdims=True)
            ce = (np.log(np.exp(Z - m).sum(1)) + m[:, 0]) - zl
            p = np.exp(Z - m)
            masked_weight = max((p[k, list(masks[r])].sum() if masks[r] else 0.0) for k, r in enumerate(vr)) if len(vr) else 0.0
            if info[t].get("loss") is not None:
                st["max_loss_abs_diff"] = max(st["max_loss_abs_diff"], abs(float(ce.mean()) - info[t]["loss"]))
            st["max_masked_softmax_weight"] = max(st.get("max_masked_softmax_weight", 0.0), float(masked_weight))
            st["n_fn_masked_reported_equals_pairs"] = st.get("n_fn_masked_reported_equals_pairs", True) and \
                info[t]["n_fn_masked"] == len(fr)
    ok = bad_label == 0 and bad_mask == 0 and all(
        s["label_masked"] == 0 and s["future_only_pos_masked"] == 0 and s["max_loss_abs_diff"] < 1e-4
        and s["max_masked_softmax_weight"] == 0.0 and s["n_fn_masked_reported_equals_pairs"] for s in stats.values())
    ck.add("D1_real_batch_labels_fn_mask_duplicates_loss", ok, batches_checked=pick, label_mismatch_tasks=bad_label,
           fn_mask_mismatch_rows=bad_mask, per_task=stats,
           definition="label = candidate column of the row's own video / non-zero music; FN mask(row) = "
                      "Pos_train(user, task) - {label} recomputed from raw train events; loss recomputed in "
                      "float64 from the formal U0 checkpoint embeddings (eval mode)")
    # D2: action balance as actually trained (formal U0 train log + sample counts).
    tl = pd.read_json(ROOT / RUNS["u0"] / "train_log.jsonl", lines=True)
    n = {t: float(tl[f"n_{t}"].mean()) for t in TASKS}
    meta = json.loads((ROOT / RUNS["u0"] / "run_meta.json").read_text())
    cfg = meta["config"]["train"]
    ck.add("D2_action_balance_as_trained", cfg["like_oversample"] == 1, like_oversample=cfg["like_oversample"],
           samples=meta["samples"]["base"], mean_valid_rows_per_logged_batch=n,
           per_row_weight_ratio_like_vs_lv_video=n["video_long_view"] / n["video_is_like"],
           loss="sum over 4 tasks of the per-task mean CE (equal task weights)")


# ============================================================================ E. formal results provenance
def check_results(ck: Checks):
    mo = pd.read_csv(ROOT / "results/m5/main_formal/bootstrap_models.csv")
    mo = mo[(mo.stratum == "all")]
    src = {"pop": "results/baselines/per_user/pop", "als": "results/baselines/per_user/als",
           "als_like_aux": "results/baselines/per_user/als_like_aux",
           **{k: RUNS[k] + "/per_user" for k in ("u0", "u1", "abl_nohist", "abl_nomusic")}}
    maxdev, n = 0.0, 0
    for _, r in mo.iterrows():
        f = ROOT / src[r.model] / f"test_{r.task}_{r.frame_kind}.parquet"
        v = pd.read_parquet(f)[r.metric].mean()
        maxdev = max(maxdev, abs(v - r["mean"]))
        n += 1
    ck.add("E1_main_formal_point_estimates_equal_local_per_user_means", maxdev <= 1e-12, rows=n, max_abs_dev=maxdev)
    cm = pd.read_csv(ROOT / "results/m5/cold_video/eval_671dde9_job33/cold_metrics.csv")
    d = ROOT / "results/m5/cold_video/eval_671dde9_job33/per_user"
    maxdev = 0.0
    for _, r in cm.iterrows():
        pu = pd.read_parquet(d / f"{r.model}_{r.task}_{r.frame}.parquet")
        maxdev = max(maxdev, abs(pu["recall@20"].mean() - r["recall@20"]))
    ck.add("E2_cold_metrics_equal_per_user_means", maxdev <= 1e-12, rows=int(len(cm)), max_abs_dev=maxdev)
    # E3: no reported three-tower number used test labels for selection; parity ok; SHAs.
    info = {}
    ok = True
    for k, r in RUNS.items():
        meta = json.loads((ROOT / r / "run_meta.json").read_text())
        sel = json.loads((ROOT / r / "selection.json").read_text())
        par = json.loads((ROOT / r / "parity.json").read_text())
        info[k] = {"git_sha": meta["git_sha"], "condition_id": meta["condition_id"], "best_epoch": sel["best_epoch"],
                   "selection_frames": sel.get("selection_frames", {}).get("kind", "normal_val_masked (pre-M5 schema)"),
                   "test_used_for_selection": sel["test_used_for_selection"], "parity_ok": par["ok"],
                   "parity_max_abs": par["check1_2_scores_topk"]["max_abs"]}
        ok &= sel["test_used_for_selection"] is False and par["ok"]
    ck.add("E3_formal_run_provenance", ok, runs=info)


# ============================================================================ F. local re-scoring of formal exports
def rescore(ck: Checks):
    from kuairand_rec.data.pipeline import load_processed
    from kuairand_rec.models.evaluation import build_eval_frames, contexts, evaluate_frames
    from kuairand_rec.models.export import ExportedScorer
    from kuairand_rec.models.towers import TASKS
    P = load_processed()
    frames = build_eval_frames(P)
    ctx = contexts(P)
    keys = [("val", t, "masked") for t in TASKS] + [("test", t, k) for t in TASKS for k in ("masked", "cold")]
    cols = [f"{p}@{k}" for k in (10, 20, 50) for p in ("recall", "ndcg")]
    out = {}
    for name, r in RUNS.items():
        t0 = time.time()
        sc = ExportedScorer(ROOT / r / "export")
        pu, _ = evaluate_frames(sc, frames, ctx, keys)
        rec = {}
        for key in keys:
            ref = pd.read_parquet(ROOT / r / "per_user" / f"{'_'.join(key)}.parquet")
            a, b = pu[key][["user_id", *cols]].reset_index(drop=True), ref[["user_id", *cols]].reset_index(drop=True)
            diff = (a[cols].to_numpy() != b[cols].to_numpy()).any(1)
            rec["/".join(key)] = {"n_users": int(len(a)), "users_differ": int(diff.sum()),
                                  "max_abs_mean_diff": float(np.abs(a[cols].mean() - b[cols].mean()).max())}
        out[name] = {"frames": rec, "max_users_differ": max(v["users_differ"] for v in rec.values()),
                     "max_abs_mean_diff": max(v["max_abs_mean_diff"] for v in rec.values()),
                     "seconds": round(time.time() - t0, 1)}
        print(name, out[name]["max_users_differ"], out[name]["max_abs_mean_diff"], out[name]["seconds"])
    ck.add("F1_local_rescore_of_formal_exports", True, runs=out,
           note="descriptive: formal metrics were scored on the Linux server; this re-scores the same export files "
                "with the same code on macOS. Differences come only from float32 BLAS rounding at exact/near ties.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["checks", "rescore"])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    ck = Checks()
    t0 = time.time()
    if a.step == "checks":
        from kuairand_rec.data.pipeline import load_processed
        P = load_processed()
        _, _, uf = check_dataset(ck, P)
        cond, base = check_temporal(ck, P, uf)
        check_loss(ck, P, cond, base)
        check_results(ck)
    else:
        rescore(ck)
    from kuairand_rec.utils import git_sha
    doc = {"step": a.step, "git_sha": git_sha(), "runtime_s": round(time.time() - t0, 1),
           "failed": ck.failed, "checks": ck.res}
    (OUT / f"{a.step}.json").write_text(json.dumps(doc, indent=1, default=_jsonable))
    print(json.dumps({"failed": ck.failed, "n_checks": len(ck.res), "runtime_s": doc["runtime_s"]}))
    return 1 if ck.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
