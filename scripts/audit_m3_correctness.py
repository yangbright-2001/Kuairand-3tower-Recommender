"""Independent M3 correctness audit -> results/baselines/m3_audit/audit_m3.json
(summarised in results/baselines/M3_CORRECTNESS_AUDIT.md).

Oracle side (pandas / numpy / scipy / implicit only; NO kuairand_rec.eval, data.history,
baselines or catalog code):
  * catalog rows rebuilt from the raw metadata CSV; split recomputed from literal cut times
  * GT / seen mask / population / exclusion counts via per-user Python sets
  * Pop counts by bincount; ALS confidence matrix, implicit fit and closed-form fold-in
    written from scratch; ranks by *counting* (no sort): rank(g) = #{s > s_g} +
    #{s == s_g, row < g}; Recall/NDCG in plain numpy
Official side (kuairand_rec): only used to load the audited artifacts and to obtain the
official model objects / score matrices for numerical comparison.

Run:  uv run python scripts/audit_m3_correctness.py      (about 3 min on the M1; exit 1 on any FAIL)
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("KUAIRAND_RAW_DIR", ROOT / "data/raw/KuaiRand-Pure/data"))
PROC = ROOT / "data/processed"
RES = ROOT / "results/baselines"
OUT = RES / "m3_audit"

# Literal cut points (P1-3), computed here rather than read from meta.json.
T_VAL = int(pd.Timestamp("2022-04-17 00:00:00", tz="Asia/Shanghai").value // 10**6)
T_TEST = int(pd.Timestamp("2022-04-26 00:00:00", tz="Asia/Shanghai").value // 10**6)
TASKS = [("video", "long_view"), ("video", "is_like"), ("music", "long_view"), ("music", "is_like")]
KS = (10, 20, 50)
CAP = 100
TOL_METRIC = 1e-12        # recall/ndcg (both sides float64 from identical integer ranks)
TOL_FOLDIN_OURS = 1e-5    # our float64 closed form vs official float64 closed form (float32 out)
TOL_FOLDIN_LIB = 1e-4     # vs implicit.recalculate_user (float32 Cholesky)

R: dict = {"checks": [], "meta": {}}
t0 = time.time()


def check(name, ok, **info):
    R["checks"].append({"check": name, "ok": bool(ok), **info})
    print(("PASS " if ok else "FAIL ") + name, json.dumps(info, default=str)[:300] if info else "", flush=True)


def sets_by_user(u, i) -> dict:
    """{user: frozenset(items)} from parallel arrays (duplicates collapse)."""
    df = pd.DataFrame({"u": np.asarray(u, np.int64), "i": np.asarray(i, np.int64)}).drop_duplicates()
    df = df.sort_values(["u", "i"])
    uu, ii = df.u.to_numpy(), df.i.to_numpy()
    if not len(uu):
        return {}
    cut = np.flatnonzero(np.diff(uu)) + 1
    return {int(k): frozenset(g.tolist()) for k, g in zip(uu[np.r_[0, cut]], np.split(ii, cut))}


# =============================================================================== load
def load():
    ev = pd.read_parquet(PROC / "events.parquet",
                         columns=["user_id", "video_id", "time_ms", "split", "music_id", "video_row",
                                  "music_row", "long_view", "is_like"])
    vm = pd.read_csv(RAW / "video_features_basic_pure.csv", usecols=["video_id", "music_id"])
    vids = np.sort(vm.video_id.to_numpy(np.int64))
    mids = np.sort(vm.music_id[vm.music_id != 0].unique().astype(np.int64))
    vidx, midx = pd.Index(vids), pd.Index(mids)
    v2m = vm.set_index("video_id").music_id.astype(np.int64)
    cat_music = v2m.reindex(vids).to_numpy()
    cat = {"vids": vids, "mids": mids, "vidx": vidx, "midx": midx, "v2m": v2m,
           "video_music_row": np.where(cat_music == 0, -1, midx.get_indexer(cat_music))}
    mid = v2m.reindex(ev.video_id.to_numpy()).to_numpy()
    vrow = vidx.get_indexer(ev.video_id.to_numpy())
    mrow = np.where(mid == 0, -1, midx.get_indexer(mid))
    t = ev.time_ms.to_numpy(np.int64)
    split = np.where(t < T_VAL, 0, np.where(t < T_TEST, 1, 2))
    meta = json.loads((PROC / "meta.json").read_text())
    check("cut_points_equal_meta", meta["t_val_ms"] == T_VAL and meta["t_test_ms"] == T_TEST,
          t_val=T_VAL, t_test=T_TEST)
    check("catalog_sizes", len(vids) == 7583 and len(mids) == 7201, n_video=len(vids), n_music=len(mids))
    check("event_video_row_matches_raw_metadata", (vrow == ev.video_row.to_numpy()).all())
    check("event_music_id_and_row_match_raw_metadata",
          (mid == ev.music_id.to_numpy()).all() and (mrow == ev.music_row.to_numpy()).all())
    check("split_column_matches_literal_cuts", (split == ev.split.to_numpy()).all())
    check("no_event_on_cut", not np.isin(t, [T_VAL, T_TEST]).any())
    E = pd.DataFrame({"u": ev.user_id.to_numpy(np.int64), "t": t, "v": vrow, "m": mrow, "mid": mid,
                      "long_view": ev.long_view.to_numpy(np.int64), "is_like": ev.is_like.to_numpy(np.int64)})
    lc = pd.read_parquet(PROC / "label_conflict_events.parquet")
    return E, cat, lc


# =============================================================================== part 1: frames
def oracle_frames(E, cat, lc, split, target, action) -> dict:
    tp = T_VAL if split == "val" else T_TEST
    te = T_TEST if split == "val" else np.iinfo(np.int64).max
    col = "v" if target == "video" else "m"
    music = target == "music"
    per = E[(E.t >= tp) & (E.t < te)]
    pos = per[per[action] == 1]
    sent = (pos.mid == 0).to_numpy() if music else np.zeros(len(pos), bool)
    ooc = (pos[col].to_numpy() < 0) & ~sent
    inc = pos[pos[col] >= 0]
    gt_raw = sets_by_user(inc.u, inc[col])
    past = E[E.t < tp]
    hist = past.groupby("u").size().to_dict()
    pc = past[past[col] >= 0]
    seen = sets_by_user(pc.u, pc[col])
    trn = E[E.t < T_VAL]
    train_users = set(trn.u.tolist())
    train_items = set(trn.loc[trn[col] >= 0, col].tolist())
    # label-conflict events: period by time, item via raw metadata
    l = lc[(lc.time_ms >= tp) & (lc.time_ms < te) & (lc[f"{action}_any"] == 1)]
    lrow = cat["vidx"].get_indexer(l.video_id.to_numpy())
    if music:
        lm = cat["v2m"].reindex(l.video_id.to_numpy()).to_numpy()
        lrow = np.where(lm == 0, -1, cat["midx"].get_indexer(lm))
    lpairs = {(int(u), int(i)) for u, i in zip(l.user_id, lrow) if i >= 0}
    lc_possible = sum(1 for (u, i) in lpairs if i not in gt_raw.get(u, ()))

    gt_eff, pop_u, cold_u, all_masked, masked_pairs = {}, [], [], 0, 0
    for u, g in gt_raw.items():
        s = seen.get(u, frozenset())
        e = g - s
        masked_pairs += len(g & s)
        h = hist.get(u, 0)
        if h == 0:
            cold_u.append(u)
        elif e:
            pop_u.append(u)
            gt_eff[u] = e
        else:
            all_masked += 1
    pop_u, cold_u = sorted(pop_u), sorted(cold_u)
    base = {"pos_events": int(len(pos)), "pos_events_sentinel": int(sent.sum()),
            "pos_events_ooc": int(ooc.sum()), "pos_events_in_catalog": int(len(inc)),
            "gt_pairs_raw": int(sum(map(len, gt_raw.values()))),
            "users_with_raw_gt": len(gt_raw),
            "label_conflict_events_any_positive": int(len(l)),
            "label_conflict_possible_gt_pairs": int(lc_possible),
            "gt_pairs_masked_seen": int(masked_pairs), "users_all_gt_masked": int(all_masked),
            "users_cold_no_history": len(cold_u),
            "cold_gt_pairs": int(sum(len(gt_raw[u]) for u in cold_u)),
            "evaluated_users": len(pop_u),
            "evaluated_users_train_oov": int(sum(u not in train_users for u in pop_u))}
    base["dedup_events_collapsed"] = base["pos_events_in_catalog"] - base["gt_pairs_raw"]
    kinds = {"masked": (pop_u, gt_eff, {u: seen.get(u, frozenset()) for u in pop_u}),
             "nomask": (pop_u, {u: gt_raw[u] for u in pop_u}, None),
             "cold": (cold_u, {u: gt_raw[u] for u in cold_u}, None)}
    out = {}
    for k, (users, gt, mask) in kinds.items():
        n_eff = int(sum(len(gt[u]) for u in users))
        unseen = int(sum(sum(i not in train_items for i in gt[u]) for u in users))
        out[k] = {"users": users, "gt": gt, "mask": mask, "hist": hist, "train_users": train_users,
                  "t_pred": tp, "counts": {**base, "n_users": len(users), "gt_pairs_effective": n_eff,
                                           "gt_pairs_train_unseen_item": unseen}}
    # extra context for categories
    out["_raw_videos_per_music_gt"] = None
    if music:
        # (user, music) GT pairs whose positives came from >= 2 distinct videos (collapsed)
        pv = inc.groupby(["u", "m"]).v.nunique()
        out["_raw_videos_per_music_gt"] = {k: int(v) for k, v in pv[pv >= 2].items()}
    return out


def _arr_hash(h, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype="<i8"))
    h.update(str(a.shape).encode())
    h.update(a.tobytes())


def ids_from_spec(split, target, action, t_pred, catalog_ids, users, gt, mask, kind):
    """Re-implementation of the PLAN §3c identity spec on oracle data."""
    h = hashlib.sha256()
    h.update(json.dumps({"schema": "m3-frame-v1", "split": split, "target": target, "action": action,
                         "t_pred": int(t_pred)}, sort_keys=True).encode())
    _arr_hash(h, catalog_ids)
    _arr_hash(h, users)
    pid = h.hexdigest()
    gptr = np.r_[0, np.cumsum([len(gt[u]) for u in users])].astype(np.int64)
    gidx = np.array([i for u in users for i in sorted(gt[u])], np.int64)
    if mask is None:
        mptr, midx = np.zeros(len(users) + 1, np.int64), np.zeros(0, np.int64)
    else:
        mptr = np.r_[0, np.cumsum([len(mask[u]) for u in users])].astype(np.int64)
        midx = np.array([i for u in users for i in sorted(mask[u])], np.int64)
    h = hashlib.sha256()
    h.update(json.dumps({"population_id": pid, "kind": kind}, sort_keys=True).encode())
    for a in (gptr, gidx, mptr, midx):
        _arr_hash(h, a)
    return pid, h.hexdigest()


def bucket(c):
    return "0" if c == 0 else "1-5" if c <= 5 else "6-20" if c <= 20 else ">20"


def audit_frames(E, cat, lc):
    fsum = json.loads((RES / "frames_summary.json").read_text())
    O = {}
    diffs = []
    for split in ("val", "test"):
        for target, action in TASKS:
            fr = oracle_frames(E, cat, lc, split, target, action)
            O[(split, target, action)] = fr
            cids = cat["vids"] if target == "video" else cat["mids"]
            for kind in ("masked", "nomask", "cold"):
                o = fr[kind]
                key = f"{split}/{target}_{action}/{kind}"
                off = fsum[key]
                d = {k: (off.get(k), v) for k, v in o["counts"].items() if off.get(k) != v}
                pid, fid = ids_from_spec(split, target, action, o["t_pred"], cids, o["users"], o["gt"],
                                         o["mask"], kind)
                if pid != off["population_id"]:
                    d["population_id"] = "differs"
                if fid != off["frame_id"]:
                    d["frame_id"] = "differs"
                # per-user frame table
                ft = pd.read_parquet(RES / "frames" / f"{split}_{target}_{action}_{kind}.parquet")
                users = o["users"]
                if ft.user_id.tolist() != users:
                    d["frame_table_users"] = "differs"
                else:
                    n_bad = {}
                    exp = {
                        "n_gt": [len(o["gt"][u]) for u in users],
                        "gt_rows": [sorted(o["gt"][u]) for u in users],
                        "n_masked_candidates": [len(o["mask"][u]) if o["mask"] is not None else 0 for u in users],
                        "hist_len_full": [o["hist"].get(u, 0) for u in users],
                        "hist_len_capped": [min(o["hist"].get(u, 0), CAP) for u in users],
                        "bucket": [bucket(min(o["hist"].get(u, 0), CAP)) for u in users],
                        "is_train_oov_user": [u not in o["train_users"] for u in users],
                    }
                    for c, v in exp.items():
                        got = ft[c].tolist() if c != "gt_rows" else [list(x) for x in ft[c]]
                        nb = sum(a != b for a, b in zip(got, v))
                        if nb:
                            n_bad[c] = nb
                    if n_bad:
                        d["frame_table_columns"] = n_bad
                if d:
                    diffs.append({"frame": key, "diff": d})
                R.setdefault("frames", {})[key] = o["counts"]
    check("frames_oracle_counts_ids_tables_all_24", not diffs, n_frames=24, diffs=diffs)
    return O


# =============================================================================== part 2: metrics
def idcg_table(maxn=max(KS)):
    out = {}
    for k in KS:
        disc = 1.0 / np.log2(np.arange(2, k + 2))
        out[k] = np.array([disc[:min(n, k)].sum() if n else np.nan for n in range(maxn + 1)])
    return out


IDCG = idcg_table()


def indep_metrics(S, users, gt, mask):
    """Count-based ranks (no sort). S: float32 [len(users), n_items], modified in place."""
    n_items = S.shape[1]
    if mask is not None:
        for j, u in enumerate(users):
            m = mask[u]
            if m:
                S[j, np.fromiter(m, np.int64)] = -np.inf
    pu = np.array([j for j, u in enumerate(users) for _ in gt[u]], np.int64)
    pg = np.array([g for u in users for g in sorted(gt[u])], np.int64)
    sg = S[pu, pg]
    rank = np.empty(len(pu), np.int64)
    ties = np.empty(len(pu), np.int64)
    cols = np.arange(n_items)
    for a in range(0, len(pu), 512):
        Sb, s = S[pu[a:a + 512]], sg[a:a + 512, None]
        eq = Sb == s
        rank[a:a + 512] = (Sb > s).sum(1) + (eq & (cols[None, :] < pg[a:a + 512, None])).sum(1)
        ties[a:a + 512] = eq.sum(1) - 1
    fin = np.isfinite(sg)
    n_gt = np.bincount(pu, minlength=len(users))
    out = {"user_id": np.asarray(users, np.int64), "n_gt": n_gt}
    for k in KS:
        hit = fin & (rank < k)
        hits = np.bincount(pu, weights=hit, minlength=len(users)).astype(np.int64)
        dcg = np.bincount(pu, weights=np.where(hit, 1.0 / np.log2(rank + 2.0), 0.0), minlength=len(users))
        out[f"hits@{k}"] = hits
        out[f"recall@{k}"] = hits / n_gt
        out[f"ndcg@{k}"] = dcg / IDCG[k][np.minimum(n_gt, k)]
    aux = {"gt_pairs": int(len(pu)), "gt_pairs_neginf": int((~fin).sum()),
           "gt_pairs_tied_in_top50": int((fin & (ties > 0) & (rank < 50)).sum())}
    return pd.DataFrame(out), aux, (pu, pg, rank, fin)


def evaluate_indep(users, gt, mask, score_fn, scorable, signal, chunk=1024):
    parts, aux_tot = [], {}
    for a in range(0, len(users), chunk):
        uc = users[a:a + chunk]
        S = score_fn(uc)
        df, aux, _ = indep_metrics(S, uc, gt, mask)
        df["n_gt_unscorable"] = [sum(not scorable[i] for i in gt[u]) for u in uc]
        df["has_signal"] = [signal(u) for u in uc]
        parts.append(df)
        for k, v in aux.items():
            aux_tot[k] = aux_tot.get(k, 0) + v
    return pd.concat(parts, ignore_index=True), aux_tot


def compare_per_user(mine: pd.DataFrame, path: Path, fsum_entry: dict, mrow: pd.Series | None):
    off = pd.read_parquet(path)
    res = {"n_users": int(len(mine))}
    if off.user_id.tolist() != mine.user_id.tolist():
        res["users_differ"] = True
        return res, False
    bad = {}
    for c in ["n_gt", "n_gt_unscorable", "has_signal"] + [f"hits@{k}" for k in KS]:
        nb = int((off[c].to_numpy() != mine[c].to_numpy()).sum())
        if nb:
            bad[c] = nb
    mx = 0.0
    for k in KS:
        for p in ("recall", "ndcg"):
            mx = max(mx, float(np.abs(off[f"{p}@{k}"].to_numpy() - mine[f"{p}@{k}"].to_numpy()).max()))
    res["max_abs_metric_diff"] = mx
    if mx > TOL_METRIC:
        bad["metric_values"] = mx
    if off.frame_id.nunique() != 1 or off.frame_id.iloc[0] != fsum_entry["frame_id"]:
        bad["frame_id"] = True
    if mrow is not None:   # metrics.csv row = mean of independent per-user values
        mxm = max(abs(float(mrow[f"{p}@{k}"]) - float(mine[f"{p}@{k}"].mean())) for k in KS for p in ("recall", "ndcg"))
        res["max_abs_metrics_csv_diff"] = mxm
        if mxm > 1e-12 or int(mrow.n_users) != len(mine) or int(mrow.n_no_signal) != int((~mine.has_signal).sum()) \
                or int(mrow.gt_unscorable) != int(mine.n_gt_unscorable.sum()) or int(mrow.gt_pairs) != int(mine.n_gt.sum()):
            bad["metrics_csv"] = mxm
    if bad:
        # first few mismatching users for diagnosis
        m = np.zeros(len(off), bool)
        for k in KS:
            m |= off[f"hits@{k}"].to_numpy() != mine[f"hits@{k}"].to_numpy()
        res["example_users"] = off.user_id[m].head(10).tolist()
        res["bad"] = bad
    return res, not bad


# =============================================================================== ALS (oracle)
def my_conf(df, target, action, aux_w, alpha):
    """r = 1[any action] + aux_w 1[any long_view] per (user, item row); c = 1 + alpha r."""
    col = "v" if target == "video" else "m"
    d = df[df[col] >= 0]
    a = d.loc[d[action] == 1, ["u", col]].drop_duplicates().assign(ra=1.0)
    if aux_w:
        lv = d.loc[d.long_view == 1, ["u", col]].drop_duplicates().assign(rl=1.0)
        a = a.merge(lv, on=["u", col], how="outer")
    else:
        a["rl"] = 0.0
    a = a.fillna({"ra": 0.0, "rl": 0.0})
    out = pd.DataFrame({"u": a.u.to_numpy(np.int64), "i": a[col].to_numpy(np.int64),
                        "r": a.ra.to_numpy(np.float64) + aux_w * a.rl.to_numpy(np.float64)})
    out = out[out.r > 0].sort_values(["u", "i"]).reset_index(drop=True)
    out["c"] = (1.0 + alpha * out.r).astype(np.float32)
    return out


def my_fit(conf, cfg):
    from implicit.cpu.als import AlternatingLeastSquares
    users, items = np.unique(conf.u.to_numpy()), np.unique(conf.i.to_numpy())
    ui = pd.Index(users).get_indexer(conf.u.to_numpy())
    ii = pd.Index(items).get_indexer(conf.i.to_numpy())
    C = sp.csr_matrix((conf.c.to_numpy(np.float32), (ui, ii)), shape=(len(users), len(items)))
    m = AlternatingLeastSquares(factors=cfg["factors"], regularization=cfg["regularization"], alpha=1.0,
                                iterations=cfg["iterations"], random_state=cfg["seed"], num_threads=1)
    m.fit(C, show_progress=False)
    return {"users": users, "items": items, "C": C, "model": m,
            "Y": np.asarray(m.item_factors, np.float32).copy(), "X": np.asarray(m.user_factors, np.float32).copy()}


def my_foldin(fit, hist, target, action, cfg):
    conf = my_conf(hist, target, action, cfg["aux_w"], cfg["alpha"])
    col = pd.Index(fit["items"]).get_indexer(conf.i.to_numpy())
    keep = col >= 0
    stats = {"history_pairs": int(len(conf)), "history_pairs_unscorable_dropped": int((~keep).sum())}
    conf, col = conf[keep].reset_index(drop=True), col[keep]
    Y = fit["Y"].astype(np.float64)
    G = Y.T @ Y + cfg["regularization"] * np.eye(Y.shape[1])
    c_all = conf.c.to_numpy(np.float64)
    groups = conf.groupby("u").indices
    users = np.array(sorted(groups), np.int64)
    X = np.zeros((len(users), Y.shape[1]), np.float32)
    for j, u in enumerate(users):
        ix = groups[u]
        Yi, c = Y[col[ix]], c_all[ix]
        X[j] = np.linalg.solve(G + Yi.T @ (Yi * (c - 1.0)[:, None]), Yi.T @ c)
    # implicit's own solver on the same rows (independent numerical reference)
    ui = pd.Index(users).get_indexer(conf.u.to_numpy())
    Cu = sp.csr_matrix((conf.c.to_numpy(np.float32), (ui, col)), shape=(len(users), len(fit["items"])))
    X_lib = fit["model"].recalculate_user(np.arange(len(users)), Cu)
    return users, X, np.asarray(X_lib, np.float32), stats, conf


def als_score_fn(users_vec, X, Y, item_rows, n_items):
    pos = {int(u): j for j, u in enumerate(users_vec)}

    def f(uc):
        S = np.full((len(uc), n_items), -np.inf, np.float32)
        j = np.array([pos.get(int(u), -1) for u in uc])
        has = j >= 0
        if has.any():
            sub = np.full((has.sum(), n_items), -np.inf, np.float32)
            sub[:, item_rows] = X[j[has]] @ Y.T
            S[has] = sub
        return S
    return f, (lambda u: int(u) in pos)


# =============================================================================== part 3a: semantics
def semantics_checks(E):
    from implicit.cpu.als import AlternatingLeastSquares
    out = {}
    tr = E[E.t < T_VAL]
    conf = my_conf(tr, "video", "long_view", 0.0, 10.0)
    # (i) recalculate_user on a real-data fit under three hypotheses
    fit = my_fit(conf, {"factors": 32, "regularization": 0.1, "iterations": 10, "seed": 0})
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(fit["users"]), 300, replace=False))
    Y = fit["Y"].astype(np.float64)
    G = Y.T @ Y + 0.1 * np.eye(32)
    lib = fit["model"].recalculate_user(rows, fit["C"][rows])
    err = {"C=d,p=1": 0.0, "C=1+d,p=1": 0.0, "C=d,p=d": 0.0}
    for k, r in enumerate(rows):
        row = fit["C"][r]
        Yi, d = Y[row.indices], row.data.astype(np.float64)
        hyp = {"C=d,p=1": (G + Yi.T @ (Yi * (d - 1)[:, None]), Yi.T @ d),
               "C=1+d,p=1": (G + Yi.T @ (Yi * d[:, None]), Yi.T @ (1 + d)),
               "C=d,p=d": (G + Yi.T @ (Yi * (d - 1)[:, None]), Yi.T @ (d * d))}
        for h, (A_, b) in hyp.items():
            err[h] = max(err[h], float(np.abs(np.linalg.solve(A_, b) - lib[k]).max()))
    out["recalculate_user_max_abs_err_by_hypothesis"] = err
    out["stored_values_unique"] = sorted(set(np.unique(fit["C"].data).tolist()))
    # (ii) training-side stationarity: the final item step solves the item normal equations
    sub_u = np.unique(conf.u)[:3000]
    cs = conf[conf.u.isin(sub_u)]
    for use_cg in (False, True):
        users, items = np.unique(cs.u), np.unique(cs.i)
        C = sp.csr_matrix((cs.c.to_numpy(np.float32), (pd.Index(users).get_indexer(cs.u), pd.Index(items).get_indexer(cs.i))),
                          shape=(len(users), len(items)))
        m = AlternatingLeastSquares(factors=16, regularization=0.1, alpha=1.0, iterations=60, random_state=0,
                                    num_threads=1, use_cg=use_cg)
        m.fit(C, show_progress=False)
        X, Yf = m.user_factors.astype(np.float64), m.item_factors.astype(np.float64)
        XtX = X.T @ X + 0.1 * np.eye(16)
        Ciu = C.T.tocsr()
        res = {"C=d,p=1": [], "C=1+d,p=1": []}
        # Items whose users all have zero factors (tiny disconnected components of the
        # subsample collapse to 0 under Cholesky) have b = 0 and carry no evidence: skip.
        live = np.flatnonzero([np.abs(X[Ciu[i].indices]).max() > 0 for i in range(len(items))])
        for i in rng.choice(live, 300, replace=False):
            row = Ciu[i]
            Xi, d = X[row.indices], row.data.astype(np.float64)
            for h, (A_, b) in {"C=d,p=1": (XtX + Xi.T @ (Xi * (d - 1)[:, None]), Xi.T @ d),
                               "C=1+d,p=1": (XtX + Xi.T @ (Xi * d[:, None]), Xi.T @ (1 + d))}.items():
                res[h].append(np.linalg.norm(A_ @ Yf[i] - b) / np.linalg.norm(b))
        out[f"fit_item_stationarity_rel_residual_{'cg' if use_cg else 'cholesky'}"] = {
            h: {"median": float(np.median(v)), "max": float(np.max(v))} for h, v in res.items()}
    chol, cg = (out[f"fit_item_stationarity_rel_residual_{s}"] for s in ("cholesky", "cg"))
    ok = (err["C=d,p=1"] < TOL_FOLDIN_LIB and err["C=1+d,p=1"] > 100 * TOL_FOLDIN_LIB
          and err["C=d,p=d"] > 100 * TOL_FOLDIN_LIB
          and chol["C=d,p=1"]["max"] < 1e-4 and chol["C=1+d,p=1"]["median"] > 1e-2
          and cg["C=d,p=1"]["median"] < 0.1 * cg["C=1+d,p=1"]["median"])
    check("implicit_0.7.3_confidence_semantics_C_eq_stored_value", ok, **out)
    R["semantics"] = out


# =============================================================================== main
def main() -> int:
    import implicit

    from kuairand_rec.baselines import als as A
    from kuairand_rec.baselines.popularity import PopularityScorer
    from kuairand_rec.data.pipeline import load_processed
    from kuairand_rec.eval import metrics as M
    from kuairand_rec.eval.evaluator import EvalContext
    from kuairand_rec.eval.views import history_view, train_view

    OUT.mkdir(parents=True, exist_ok=True)
    R["meta"] = {"implicit": implicit.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                 "tolerances": {"metric": TOL_METRIC, "foldin_ours_vs_official": TOL_FOLDIN_OURS,
                                "foldin_vs_recalculate_user": TOL_FOLDIN_LIB}}
    check("implicit_version", implicit.__version__ == "0.7.3", version=implicit.__version__)

    E, cat, lc = load()
    print(f"[{time.time() - t0:.0f}s] loaded", flush=True)
    O = audit_frames(E, cat, lc)
    print(f"[{time.time() - t0:.0f}s] frames done", flush=True)
    semantics_checks(E)
    print(f"[{time.time() - t0:.0f}s] semantics done", flush=True)

    d = load_processed(PROC)
    ev_off, cat_off, meta = d["events"], d["catalogs"], d["meta"]
    tr_off = train_view(ev_off)
    hv_off = {s: history_view(ev_off, s, meta) for s in ("val", "test")}
    fsum = json.loads((RES / "frames_summary.json").read_text())
    run_meta = json.loads((RES / "run_meta.json").read_text())
    mcsv = pd.read_csv(RES / "metrics.csv")
    n_items = {"video": len(cat["vids"]), "music": len(cat["mids"])}
    tr = E[E.t < T_VAL]
    hist = {"val": E[E.t < T_VAL], "test": E[E.t < T_TEST]}
    rng = np.random.default_rng(0)

    # ---- tuning selection re-derived from als_tuning.csv (strict '>' keeps first grid point)
    tun = pd.read_csv(RES / "als_tuning.csv")
    sel_bad = []
    for (task, var), g in tun.groupby(["task", "variant"], sort=False):
        best = None
        for _, r in g.iterrows():
            key = (r["val_recall@20"], r["val_ndcg@20"])
            if best is None or key > best[0]:
                best = (key, r)
        b, s = best[1], run_meta["als_selected"][f"{task}/{var}"]
        if (int(b.factors), float(b.regularization), float(b.alpha)) != (s["factors"], s["regularization"], s["alpha"]):
            sel_bad.append(f"{task}/{var}")
        tgt, act = task.split("_", 1)
        mv = mcsv[(mcsv.model == var) & (mcsv.split == "val") & (mcsv.target == tgt) & (mcsv.action == act)].iloc[0]
        if abs(mv["recall@20"] - b["val_recall@20"]) > 1e-12:
            sel_bad.append(f"{task}/{var}: val row != tuning row")
    check("als_selection_rederived_from_tuning_csv", not sel_bad, bad=sel_bad, n_groups=int(tun.groupby(["task", "variant"]).ngroups))

    parity, als_num, conf_checks, examples = [], {}, [], []

    def run_parity(model, split, tgt, act, kind, score_fn, scorable, signal):
        fr = O[(split, tgt, act)][kind]
        mine, aux = evaluate_indep(fr["users"], fr["gt"], fr["mask"], score_fn, scorable, signal)
        path = RES / "per_user" / model / f"{split}_{tgt}_{act}_{kind}.parquet"
        mrow = mcsv[(mcsv.model == model) & (mcsv.split == split) & (mcsv.target == tgt) & (mcsv.action == act)
                    & (mcsv.kind == kind)]
        res, ok = compare_per_user(mine, path, fsum[f"{split}/{tgt}_{act}/{kind}"], mrow.iloc[0] if len(mrow) else None)
        # population composition actually covered by this comparison
        users = fr["users"]
        res.update(aux)
        # -inf GT accounting: every -inf GT pair is explained by a no-signal user or an
        # item-level unscorable GT (masked items are already out of GT).
        hs = mine.has_signal.to_numpy()
        res["neginf_gt_explained"] = aux["gt_pairs_neginf"] == int(mine.n_gt[~hs].sum() + mine.n_gt_unscorable[hs].sum())
        if not res["neginf_gt_explained"]:
            ok = False
        res["n_train_oov"] = int(sum(u not in fr["train_users"] for u in users))
        res["n_no_signal"] = int((~mine.has_signal).sum())
        res["n_multi_gt"] = int((mine.n_gt >= 2).sum())
        if tgt == "music":
            col = cat["video_music_row"]
            nv = np.bincount(col[col >= 0], minlength=n_items["music"])
            res["n_users_gt_on_shared_music"] = int(sum(any(nv[i] >= 2 for i in fr["gt"][u]) for u in users))
            coll = O[(split, tgt, act)]["_raw_videos_per_music_gt"]
            us = set(users)
            res["n_users_gt_from_collapsed_videos"] = int(len({u for (u, m) in coll if u in us}))
        parity.append({"model": model, "split": split, "task": f"{tgt}_{act}", "kind": kind, "ok": ok, **res})
        print(f"  parity {model:24s} {split} {tgt}_{act} {kind:6s} ok={ok} users={res['n_users']} "
              f"maxdiff={res.get('max_abs_metric_diff')}", flush=True)
        return mine

    def sample_users(tgt, act, split, kind, mine, signal_set):
        """Representative users for explicit sort-based cross-checks."""
        fr = O[(split, tgt, act)][kind]
        users = np.array(fr["users"])
        cats = {
            "normal": [u for u in users if u in fr["train_users"] and len(fr["gt"][u]) == 1 and signal_set(u)],
            "multi_gt": [u for u in users if len(fr["gt"][u]) >= 3],
            "train_oov": [u for u in users if u not in fr["train_users"]],
            "no_signal": [u for u in users if not signal_set(u)],
            "max_mask": [max(users, key=lambda u: len(fr["mask"][u]) if fr["mask"] else 0)],
            "max_gt": [max(users, key=lambda u: len(fr["gt"][u]))],
        }
        if tgt == "music":
            col = cat["video_music_row"]
            nv = np.bincount(col[col >= 0], minlength=n_items["music"])
            cats["shared_music"] = [u for u in users if any(nv[i] >= 2 for i in fr["gt"][u])]
            coll = O[(split, tgt, act)]["_raw_videos_per_music_gt"]
            us = set(users.tolist())
            cats["collapsed_videos"] = sorted({u for (u, m) in coll if u in us})
        out = {}
        for c, lst in cats.items():
            if lst:
                out[c] = sorted(rng.choice(lst, min(5, len(lst)), replace=False).tolist())
        return out

    def explicit_check(tag, tgt, act, split, kind, samples, my_fn, off_scores_fn, mine):
        """Sorted top-50 (np.lexsort on our scores) vs official rank_topk on official scores,
        both under the oracle mask; and our scores vs official scores (catalog alignment)."""
        fr = O[(split, tgt, act)][kind]
        flat = sorted({u for v in samples.values() for u in v})
        S_my = my_fn(flat)
        S_off = off_scores_fn(np.array(flat, np.int64))
        fin_eq = np.array_equal(np.isneginf(S_my), np.isneginf(S_off))
        f = np.isfinite(S_my) & np.isfinite(S_off)
        score_diff = float(np.abs(S_my[f] - S_off[f]).max()) if f.any() else 0.0
        if fr["mask"] is not None:
            for j, u in enumerate(flat):
                if fr["mask"][u]:
                    idx = np.fromiter(fr["mask"][u], np.int64)
                    S_my[j, idx] = -np.inf
                    S_off[j, idx] = -np.inf
        top_off, valid_off = M.rank_topk(S_off.copy(), 50)
        n_list_bad, n_rank_bad = 0, 0
        cols = np.arange(S_my.shape[1])
        _, _, (pu, pg, rank, fin) = indep_metrics(S_my.copy(), flat, fr["gt"], None)
        my = mine.set_index("user_id")
        for j, u in enumerate(flat):
            order = np.lexsort((cols, -S_my[j]))[:50]
            valid = np.isfinite(S_my[j, order])
            if not (np.array_equal(order[valid], top_off[j][valid_off[j]])):
                n_list_bad += 1
            pos = {int(r): p for p, r in enumerate(order) if valid[p]}
            for q in np.flatnonzero(pu == j):
                exp_r = pos.get(int(pg[q]))
                got = int(rank[q]) if (fin[q] and rank[q] < 50) else None
                if exp_r != got:
                    n_rank_bad += 1
        for c, lst in samples.items():
            u = lst[0]
            examples.append({"model": tag, "task": f"{tgt}_{act}", "split": split, "kind": kind, "category": c,
                             "user_id": int(u), "n_gt": int(my.loc[u, "n_gt"]),
                             "hist_len": int(fr["hist"].get(u, 0)),
                             "n_masked": len(fr["mask"][u]) if fr["mask"] else 0,
                             "has_signal": bool(my.loc[u, "has_signal"]),
                             "hits@20": int(my.loc[u, "hits@20"]), "ndcg@20": round(float(my.loc[u, "ndcg@20"]), 4)})
        r = {"model": tag, "task": f"{tgt}_{act}", "kind": kind, "n_sampled_users": len(flat),
             "categories": {c: len(v) for c, v in samples.items()},
             "neginf_pattern_equal": bool(fin_eq), "max_abs_score_diff_finite": score_diff,
             "top50_list_mismatch_users": n_list_bad, "gt_rank_mismatch": n_rank_bad}
        R.setdefault("explicit_samples", []).append(r)
        return fin_eq and n_list_bad == 0 and n_rank_bad == 0 and score_diff <= 1e-5

    # ---- Pop
    pop_off = PopularityScorer(tr_off, cat_off, ["long_view", "is_like"])
    pop_cnt = {}
    for tgt in ("video", "music"):
        col = "v" if tgt == "video" else "m"
        for act in ("long_view", "is_like"):
            p = tr[(tr[act] == 1) & (tr[col] >= 0)]
            cnt = np.bincount(p[col].to_numpy(), minlength=n_items[tgt])
            pop_cnt[(tgt, act)] = cnt.astype(np.float32)
            m2 = d["popularity"][tgt][f"train_{act}"].to_numpy()
            check(f"pop_counts_{tgt}_{act}", np.array_equal(cnt, m2) and np.array_equal(
                pop_off.counts(tgt, act), cnt.astype(np.float32)), n_nonzero=int((cnt > 0).sum()),
                n_zero=int((cnt == 0).sum()))
    explicit_ok = []
    for tgt, act in TASKS:
        cnt = pop_cnt[(tgt, act)]
        fn = lambda uc, c=cnt: np.tile(c, (len(uc), 1))  # noqa: E731
        scorable = np.ones(n_items[tgt], bool)
        for split, kinds in (("val", ("masked",)), ("test", ("masked", "nomask", "cold"))):
            for kind in kinds:
                mine = run_parity("pop", split, tgt, act, kind, fn, scorable, lambda u: True)
                if split == "test" and kind == "masked":
                    smp = sample_users(tgt, act, split, kind, mine, lambda u: True)
                    ctx = EvalContext("test", T_TEST, hv_off["test"])
                    explicit_ok.append(explicit_check("pop", tgt, act, split, kind, smp, fn,
                                                      lambda us, t=tgt, a=act, c=ctx: pop_off.score(us, t, a, c), mine))
    print(f"[{time.time() - t0:.0f}s] pop done", flush=True)

    # ---- ALS
    for tgt, act in TASKS:
        variants = ["als"] if act == "long_view" else ["als", "als_like_aux"]
        for var in variants:
            s = run_meta["als_selected"][f"{tgt}_{act}/{var}"]
            cfg = {k: s[k] for k in ("factors", "regularization", "alpha", "aux_w", "iterations", "seed")}
            key = f"{tgt}_{act}/{var}"
            # signal construction: ours vs official build_confidence on train / val-hist / test-hist
            for name, mine_df, off_df in (("train", tr, tr_off), ("val_hist", hist["val"], hv_off["val"]),
                                          ("test_hist", hist["test"], hv_off["test"])):
                a_ = my_conf(mine_df, tgt, act, cfg["aux_w"], cfg["alpha"])
                b_ = A.build_confidence(off_df, tgt, act, cfg["aux_w"], cfg["alpha"])
                ok = (len(a_) == len(b_) and np.array_equal(a_.u.to_numpy(), b_.user_id.to_numpy())
                      and np.array_equal(a_.i.to_numpy(), b_.item.to_numpy())
                      and np.array_equal(a_.c.to_numpy(), b_.c.to_numpy()))
                vals = {str(k): int(v) for k, v in a_.c.value_counts().sort_index().items()}
                conf_checks.append({"model": key, "data": name, "pairs": int(len(a_)), "c_values": vals, "ok": bool(ok)})
            conf_tr = my_conf(tr, tgt, act, cfg["aux_w"], cfg["alpha"])
            ch = hashlib.sha256(np.ascontiguousarray(conf_tr[["u", "i"]].to_numpy(np.int64)).tobytes()
                                + conf_tr.c.to_numpy(np.float32).tobytes()).hexdigest()
            fit = my_fit(conf_tr, cfg)
            m_off = A.fit_als(tr_off, n_items[tgt], tgt, act, A.ALSConfig(num_threads=1, **cfg))
            num = {"confidence_hash_equals_run_meta": ch == s["confidence_hash"],
                   "n_scorable_items": int(len(fit["items"])),
                   "n_scorable_equals_run_meta": int(len(fit["items"])) == s["n_scorable_items"],
                   "item_rows_equal": bool(np.array_equal(fit["items"], m_off.item_rows)),
                   "train_users_equal": bool(np.array_equal(fit["users"], m_off.train_users)),
                   "Y_bitwise_equal": bool(np.array_equal(fit["Y"], m_off.Y)),
                   "X_train_bitwise_equal": bool(np.array_equal(fit["X"], m_off.X_train)),
                   "fingerprint_equals_run_meta": m_off.fingerprint == s["fingerprint"],
                   "n_train_users": int(len(fit["users"]))}
            Y = fit["Y"]
            # Mapping semantics (not just equality with the official arrays): refold sampled train
            # users from their own train row with frozen Y; X_train[k] must be closest to user k's
            # refold, and the user's train positives must rank near the top of X_train[k] @ Y.T.
            ks_ = np.sort(rng.choice(len(fit["users"]), min(500, len(fit["users"])), replace=False))
            Yd = Y.astype(np.float64)
            G = Yd.T @ Yd + cfg["regularization"] * np.eye(Yd.shape[1])
            Xr = np.zeros((len(ks_), Yd.shape[1]))
            pct = []
            for j, k in enumerate(ks_):
                row = fit["C"][k]
                Yi, c = Yd[row.indices], row.data.astype(np.float64)
                Xr[j] = np.linalg.solve(G + Yi.T @ (Yi * (c - 1)[:, None]), Yi.T @ c)
                sc_ = fit["X"][k] @ Y.T
                pct.append(float(np.mean([(sc_ > sc_[i]).mean() for i in row.indices])))
            Xt = fit["X"][ks_].astype(np.float64)
            nrm = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)  # noqa: E731
            cos = nrm(Xt) @ nrm(Xr).T
            num["mapping_cos_own_refold_median"] = float(np.median(np.diag(cos)))
            num["mapping_cos_other_refold_median"] = float(np.median(cos[~np.eye(len(ks_), dtype=bool)]))
            # Descriptive only: single-interaction users' stale X_train can be closer to another
            # single-item refold, so restrict the argmax share to users with >= 3 train items.
            nnz = np.diff(fit["C"].indptr)[ks_]
            sel = nnz >= 3
            num["mapping_own_refold_is_argmax_share_nnz_ge3"] = float(
                (np.diag(cos)[sel] >= cos[sel].max(1) - 1e-3).mean())
            num["mapping_train_pos_mean_percentile_rank"] = float(np.mean(pct))   # 0 = top, 0.5 = random
            # Exact reconstruction (pass criterion): X_train = one implicit CG user step
            # (cg_steps=3, warm start) on (C, X_{T-1}, Y_{T-1}) from a fit with T-1 iterations.
            # This proves both the compact user mapping and the "half an iteration older" note.
            from implicit.cpu import _als
            prev = my_fit(conf_tr, {**cfg, "iterations": cfg["iterations"] - 1})
            Xn = prev["model"].user_factors.copy()
            _als.least_squares_cg(fit["C"], Xn, prev["model"].item_factors, cfg["regularization"],
                                  num_threads=1, cg_steps=3)
            num["nofoldin_X_equals_one_cg_user_step_on_prev_Y_max_abs"] = float(np.abs(Xn - fit["X"]).max())
            num["nofoldin_X_vs_exact_refold_on_final_Y_median_cos"] = float(np.median(np.diag(cos)))
            for split in ("val", "test"):
                users, X, X_lib, st, _ = my_foldin(fit, hist[split], tgt, act, cfg)
                all_u = np.unique(hv_off[split].user_id.to_numpy())
                X_off, has_off, st_off = A.fold_in(m_off, hv_off[split], all_u)
                sig_off = all_u[has_off]
                X_off_s = X_off[np.searchsorted(all_u, users)] if np.array_equal(sig_off, users) else None
                num[f"{split}_signal_users"] = int(len(users))
                num[f"{split}_signal_users_equal"] = bool(np.array_equal(sig_off, users))
                num[f"{split}_foldin_stats_equal_run_meta"] = st == run_meta["als_foldin_stats"][key][split] == st_off
                num[f"{split}_foldin_stats"] = st
                if X_off_s is not None:
                    num[f"{split}_max_abs_X_ours_vs_official"] = float(np.abs(X - X_off_s).max())
                    num[f"{split}_n_X_bitwise_equal_rows"] = int((X == X_off_s).all(1).sum())
                num[f"{split}_max_abs_X_ours_vs_recalculate_user"] = float(np.abs(X - X_lib).max())
                num[f"{split}_max_rel_X_ours_vs_recalculate_user"] = float(
                    (np.abs(X - X_lib).max(1) / np.maximum(np.abs(X).max(1), 1e-12)).max())
                if split == "test":
                    # power check: folding in with test-period events too would change vectors
                    u_leak, X_leak, _, _, _ = my_foldin(fit, E, tgt, act, cfg)
                    common = np.intersect1d(users, u_leak)
                    a1 = X[np.searchsorted(users, common)]
                    a2 = X_leak[np.searchsorted(u_leak, common)]
                    num["test_power_users_changed_if_test_events_leaked"] = int((np.abs(a1 - a2).max(1) > 1e-6).sum())
                fn, sig = als_score_fn(users, X, Y, fit["items"], n_items[tgt])
                scorable = np.zeros(n_items[tgt], bool)
                scorable[fit["items"]] = True
                kinds = ("masked",) if split == "val" else ("masked", "nomask")
                for kind in kinds:
                    mine = run_parity(var, split, tgt, act, kind, fn, scorable, sig)
                    if split == "test" and kind == "masked":
                        smp = sample_users(tgt, act, split, kind, mine, sig)
                        sc = A.ALSScorer({(tgt, act): m_off}, "foldin", var)
                        ctx = EvalContext("test", T_TEST, hv_off["test"])
                        explicit_ok.append(explicit_check(var, tgt, act, split, kind, smp, fn,
                                                          lambda us, sc=sc, c=ctx, t=tgt, a=act: sc.score(us, t, a, c), mine))
            # no-fold-in: raw train factors of our own fit
            fn, sig = als_score_fn(fit["users"], fit["X"], Y, fit["items"], n_items[tgt])
            for kind in ("masked", "nomask"):
                mine = run_parity(f"{var}_nofoldin", "test", tgt, act, kind, fn, scorable, sig)
                if kind == "masked":
                    smp = sample_users(tgt, act, "test", kind, mine, sig)
                    sc = A.ALSScorer({(tgt, act): m_off}, "nofoldin", f"{var}_nofoldin")
                    ctx = EvalContext("test", T_TEST, hv_off["test"])
                    explicit_ok.append(explicit_check(f"{var}_nofoldin", tgt, act, "test", kind, smp, fn,
                                                      lambda us, sc=sc, c=ctx, t=tgt, a=act: sc.score(us, t, a, c), mine))
            als_num[key] = num
            print(f"[{time.time() - t0:.0f}s] {key}: {json.dumps(num)[:400]}", flush=True)

    # shared-music aux combination: (u, music) with like on one video and long_view only on another
    for split_name, df in (("train", tr), ("test_hist", hist["test"])):
        dm = df[df.m >= 0]
        g = dm.groupby(["u", "m"]).agg(like=("is_like", "max"), lv=("long_view", "max"))
        both_same_video = dm[(dm.is_like == 1) & (dm.long_view == 1)][["u", "m"]].drop_duplicates()
        combo = g[(g.like == 1) & (g.lv == 1)].reset_index().merge(both_same_video, how="left", indicator=True)
        R.setdefault("shared_music_aux", {})[split_name] = {
            "pairs_like_and_lv": int(len(combo)),
            "pairs_like_and_lv_from_different_videos_only": int((combo["_merge"] == "left_only").sum())}

    # ---- negative controls: inject one bug at a time into the *oracle* and confirm the
    # per-user comparison flags it (i.e. the parity checks above have power).
    neg = {}
    for model, tgt, act in (("pop", "video", "long_view"), ("als", "video", "is_like"), ("pop", "music", "is_like")):
        fr = O[("test", tgt, act)]["masked"]
        users = fr["users"]
        if model == "pop":
            base_fn = lambda uc, c=pop_cnt[(tgt, act)]: np.tile(c, (len(uc), 1))  # noqa: E731
        else:
            s = run_meta["als_selected"][f"{tgt}_{act}/als"]
            cfg = {k: s[k] for k in ("factors", "regularization", "alpha", "aux_w", "iterations", "seed")}
            fit = my_fit(my_conf(tr, tgt, act, 0.0, cfg["alpha"]), cfg)
            uu, X, _, _, _ = my_foldin(fit, hist["test"], tgt, act, cfg)
            base_fn, _ = als_score_fn(uu, X, fit["Y"], fit["items"], n_items[tgt])
        path = RES / "per_user" / model / f"test_{tgt}_{act}_masked.parquet"
        off = pd.read_parquet(path)

        def n_diff(df):
            return int(sum((off[f"hits@{k}"].to_numpy() != df[f"hits@{k}"].to_numpy()).sum() for k in KS)
                       + sum((np.abs(off[f"ndcg@{k}"].to_numpy() - df[f"ndcg@{k}"].to_numpy()) > TOL_METRIC).sum()
                             for k in KS)
                       + sum((np.abs(off[f"recall@{k}"].to_numpy() - df[f"recall@{k}"].to_numpy()) > TOL_METRIC).sum()
                             for k in KS))

        def run(fn, mask=fr["mask"], gt=fr["gt"]):
            parts = []
            for a in range(0, len(users), 1024):
                uc = users[a:a + 1024]
                parts.append(indep_metrics(fn(uc), uc, gt, mask)[0])
            return pd.concat(parts, ignore_index=True)

        ctrl = {"baseline_no_bug": n_diff(run(base_fn))}
        # Tie-break reversed (row descending): rank on reversed columns with GT/mask mapped
        # to the reversed index, so only the tie order changes.
        last = n_items[tgt] - 1
        rev_gt = {u: frozenset(last - i for i in g) for u, g in fr["gt"].items()}
        rev_mask = {u: frozenset(last - i for i in m) for u, m in fr["mask"].items()}
        ctrl["tiebreak_row_descending"] = n_diff(run(lambda uc: base_fn(uc)[:, ::-1].copy(), rev_mask, rev_gt))
        ctrl["mask_not_applied"] = n_diff(run(base_fn, None))
        ctrl["user_rows_shifted_by_one"] = n_diff(run(lambda uc: np.roll(base_fn(uc), 1, axis=0)))
        ctrl["catalog_cols_shifted_by_one"] = n_diff(run(lambda uc: np.roll(base_fn(uc), 1, axis=1)))
        ctrl["neginf_replaced_by_min_finite"] = n_diff(run(
            lambda uc: (lambda S: np.where(np.isneginf(S), np.float32(-1e30), S))(base_fn(uc))))
        drop1 = {u: (frozenset(sorted(g)[1:]) if len(g) > 1 else g) for u, g in fr["gt"].items()}
        ctrl["gt_denominator_drop_one_item"] = n_diff(run(base_fn, fr["mask"], drop1))
        neg[f"{model}/{tgt}_{act}"] = ctrl
    R["negative_controls"] = neg
    # Expected power: Pop is user-independent (row shift is a no-op) and has no -inf;
    # ALS scores are continuous (ties ~ absent, so a tie-break flip is a no-op).
    must = {"pop": ("tiebreak_row_descending", "mask_not_applied", "catalog_cols_shifted_by_one",
                    "gt_denominator_drop_one_item"),
            "als": ("mask_not_applied", "user_rows_shifted_by_one", "catalog_cols_shifted_by_one",
                    "neginf_replaced_by_min_finite", "gt_denominator_drop_one_item")}
    check("negative_controls_detect_injected_bugs",
          all(c["baseline_no_bug"] == 0 and all(c[k] > 0 for k in must[m.split("/")[0]])
              for m, c in neg.items()), **neg)

    # ---- descriptive: Pop sensitivity to the (protocol-fixed) row-ascending tie-break.
    # Re-rank with ties broken by a random row permutation (5 seeds); report metric ranges.
    tb = {}
    for tgt, act in TASKS:
        fr = O[("test", tgt, act)]["masked"]
        users, n = fr["users"], n_items[tgt]
        vals = []
        for seed in range(5):
            perm = np.random.default_rng(100 + seed).permutation(n)   # new position of each row
            inv = np.argsort(perm)
            c = pop_cnt[(tgt, act)][inv]
            pgt = {u: frozenset(int(perm[i]) for i in g) for u, g in fr["gt"].items()}
            pm = {u: frozenset(int(perm[i]) for i in m) for u, m in fr["mask"].items()}
            parts = [indep_metrics(np.tile(c, (len(users[a:a + 1024]), 1)), users[a:a + 1024], pgt, pm)[0]
                     for a in range(0, len(users), 1024)]
            df = pd.concat(parts)
            vals.append({f"{p}@{k}": float(df[f"{p}@{k}"].mean()) for k in KS for p in ("recall", "ndcg")})
        off = mcsv[(mcsv.model == "pop") & (mcsv.split == "test") & (mcsv.target == tgt) & (mcsv.action == act)
                   & (mcsv.kind == "masked")].iloc[0]
        tb[f"{tgt}_{act}"] = {m: {"official": round(float(off[m]), 4),
                                  "random_tiebreak_min": round(min(v[m] for v in vals), 4),
                                  "random_tiebreak_max": round(max(v[m] for v in vals), 4)}
                              for m in ("recall@20", "ndcg@20", "recall@50")}
    R["pop_tiebreak_sensitivity"] = tb

    files = sorted(str(p.relative_to(RES / "per_user")) for p in (RES / "per_user").rglob("*.parquet"))
    covered = {f"{p['model']}/{p['split']}_{p['task']}_{p['kind']}.parquet" for p in parity}
    check("per_user_files_all_covered", set(files) == covered, n_files=len(files), n_covered=len(covered),
          missing=sorted(set(files) - covered))
    bad = [p for p in parity if not p["ok"]]
    check("per_user_parity_all_files", not bad, n_files=len(parity),
          n_user_rows=int(sum(p["n_users"] for p in parity)), bad=[{k: p.get(k) for k in ("model", "split", "task", "kind", "bad", "example_users")} for p in bad])
    check("signal_construction_ours_equals_official", all(c["ok"] for c in conf_checks), n=len(conf_checks))
    for key, num in als_num.items():
        ok = (num["confidence_hash_equals_run_meta"] and num["n_scorable_equals_run_meta"] and num["item_rows_equal"]
              and num["train_users_equal"] and num["Y_bitwise_equal"] and num["X_train_bitwise_equal"]
              and num["fingerprint_equals_run_meta"]
              and all(num[f"{s}_signal_users_equal"] and num[f"{s}_foldin_stats_equal_run_meta"]
                      and num.get(f"{s}_max_abs_X_ours_vs_official", 1) <= TOL_FOLDIN_OURS
                      and num[f"{s}_max_abs_X_ours_vs_recalculate_user"] <= TOL_FOLDIN_LIB for s in ("val", "test"))
              and num["test_power_users_changed_if_test_events_leaked"] > 0
              and num["nofoldin_X_equals_one_cg_user_step_on_prev_Y_max_abs"] <= 1e-6
              and num["mapping_train_pos_mean_percentile_rank"] < 0.1)
        check(f"als_numerics_{key}", ok)
    check("explicit_sorted_topk_and_score_alignment_samples", all(explicit_ok), n=len(explicit_ok))

    R["parity"] = parity
    R["als_numerics"] = als_num
    R["confidence_checks"] = conf_checks
    R["examples"] = examples
    R["meta"]["runtime_sec"] = round(time.time() - t0, 1)
    R["meta"]["n_checks"] = len(R["checks"])
    R["meta"]["n_failed"] = int(sum(not c["ok"] for c in R["checks"]))
    (OUT / "audit_m3.json").write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps({"failed": [c["check"] for c in R["checks"] if not c["ok"]], "runtime": R["meta"]["runtime_sec"]}))
    return 1 if R["meta"]["n_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
