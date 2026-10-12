"""Strict Direct vs Export parity (PLAN.md §3d D1-7, checks 1-4).

Run in its own process (it reloads everything from disk: config, M2 artifacts, checkpoint,
export), so the direct path shares no in-memory state with training or export:

    uv run python -m kuairand_rec.models.parity --run_dir <run_dir>

Checks (all must pass; no near-tie exemption):
  1. Direct vs Export scores, >= 2048 users (train-OOV, history buckets 1-5 / 6-20 / >20 and cold
     users all covered), every catalog row, 4 tasks, val and test: max |d| <= 1e-5 (max, p99 reported).
  2. Direct vs Export top-50 identical (order included), with and without the seen mask. Any
     difference fails and dumps (user, task, rank, both scores, gap, boundary).
  3. retrieve() vs ExportedScorer: bitwise-equal scores and identical top-K; retrieve's seen mask
     equals the M3 test masked frame CSR row by row; mask_seen=False returns seen items too. Plus
     an independent Recall/NDCG recomputation from retrieve()'s top-50 against the frame GT.
  4. Direct vs Export per-user metrics identical on the val and test masked frames (all users),
     both through the shared M3 evaluator.

What is compared: DirectScorer recomputes item / user embeddings from the reloaded checkpoint
and the condition; ExportedScorer reads them from the export files. Both apply the same
canonical `score_rows` dot product, so any score difference comes from the embeddings (the
thing under test), not from two BLAS backends rounding float32 differently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kuairand_rec.eval import metrics as M
from kuairand_rec.eval.frame import history_bucket
from kuairand_rec.models import artifacts as A
from kuairand_rec.models.evaluation import build_eval_frames, contexts, evaluate_frames, task_parts
from kuairand_rec.models.export import ExportedScorer, retrieve
from kuairand_rec.models.inference import DirectScorer
from kuairand_rec.models.towers import TASKS

TOL = 1e-5
TOPK = 50
METRIC_COLS = [f"{p}@{k}" for k in (10, 20, 50) for p in ("hits", "recall", "ndcg")] + ["n_gt"]


def select_parity_users(frames: dict, n: int = 2048, seed: int = 0, require_all_strata: bool = True):
    """Deterministic stratified users from the test frames (cold, train-OOV, 1-5 / 6-20 / >20)."""
    rng = np.random.default_rng(seed)
    rows = []
    for (s, task, kind), f in frames.items():
        if s != "test":
            continue
        rows.append(pd.DataFrame({"user_id": f.users, "oov": f.is_train_oov_user,
                                  "bucket": history_bucket(f.hist_len_capped)}))
    df = pd.concat(rows).drop_duplicates("user_id").sort_values("user_id").reset_index(drop=True)
    strata = {"cold": df.bucket == "0", "train_oov_with_history": df.oov & (df.bucket != "0"),
              "1-5": df.bucket == "1-5", "6-20": df.bucket == "6-20", ">20": df.bucket == ">20"}
    quota = {"cold": 256, "train_oov_with_history": 256, "1-5": 384, "6-20": 384, ">20": 384}
    chosen = set()
    for k, m in strata.items():
        pool = df.user_id[m].to_numpy()
        if len(pool) == 0:
            if require_all_strata:
                raise ValueError(f"no test users in stratum {k}")
            continue
        chosen.update(rng.choice(pool, min(quota[k], len(pool)), replace=False).tolist())
    rest = np.setdiff1d(df.user_id.to_numpy(), np.fromiter(chosen, np.int64))
    need = max(0, n - len(chosen))
    chosen.update(rng.choice(rest, min(need, len(rest)), replace=False).tolist())
    users = np.array(sorted(chosen), dtype=np.int64)
    cover = {k: int(np.isin(users, df.user_id[m].to_numpy()).sum()) for k, m in strata.items()}
    return users, {"n_users": int(len(users)), "strata": cover}


_topk = M.rank_topk


def check_scores_and_topk(direct: DirectScorer, exported: ExportedScorer, users: np.ndarray, frames: dict) -> dict:
    out = {"tol": TOL, "tasks": {}, "topk_mismatches": []}
    ok = True
    from kuairand_rec.eval.evaluator import EvalContext
    for split in ("val", "test"):
        ctx_users = users
        for task in TASKS:
            tgt, act = task_parts(task)
            ctx = EvalContext(split, exported.t_pred[split], pd.DataFrame({"time_ms": []}))
            sd = direct.score(ctx_users, tgt, act, ctx)
            se = exported.score(ctx_users, tgt, act, ctx)
            d = np.abs(sd.astype(np.float64) - se.astype(np.float64))
            rec = {"max_abs": float(d.max()), "p99_abs": float(np.quantile(d, 0.99)),
                   "bitwise_equal": bool(np.array_equal(sd, se)), "n_users": int(len(ctx_users)),
                   "n_items": int(sd.shape[1])}
            if not (d.max() <= TOL):
                ok = False
            # top-50 with and without the seen mask (test: exported mask; val: none exists in the
            # export, so only the unmasked ranking is compared there).
            variants = {"nomask": (sd.copy(), se.copy())}
            if split == "test":
                ip, rows = exported.seen[tgt]
                r = exported.rows(ctx_users, "test")
                n = ip[r + 1] - ip[r]
                sub_ip = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
                sub_rows = np.concatenate([rows[ip[i]:ip[i + 1]] for i in r]) if len(r) else np.zeros(0, np.int64)
                a, b = sd.copy(), se.copy()
                M.apply_mask(a, sub_ip, sub_rows)
                M.apply_mask(b, sub_ip, sub_rows)
                variants["masked"] = (a, b)
            for vname, (a, b) in variants.items():
                ta, va = _topk(a, TOPK)
                tb, vb = _topk(b, TOPK)
                diff = np.flatnonzero(~((ta == tb) & (va == vb)).all(1))
                rec[f"top{TOPK}_{vname}_mismatch_users"] = int(len(diff))
                for i in diff[:20]:
                    j = int(np.flatnonzero((ta[i] != tb[i]) | (va[i] != vb[i]))[0])
                    srt = np.sort(b[i])[::-1]
                    out["topk_mismatches"].append({
                        "split": split, "task": task, "variant": vname, "user_id": int(ctx_users[i]), "rank": j,
                        "direct_row": int(ta[i, j]), "export_row": int(tb[i, j]),
                        "direct_scores": [float(a[i, ta[i, j]]), float(b[i, ta[i, j]])],
                        "export_scores": [float(a[i, tb[i, j]]), float(b[i, tb[i, j]])],
                        "gap_at_rank": float(srt[j] - srt[j + 1]) if j + 1 < len(srt) else None,
                        "boundary_gap_k": float(srt[TOPK - 1] - srt[TOPK]) if len(srt) > TOPK else None})
                if len(diff):
                    ok = False
            out["tasks"][f"{split}/{task}"] = rec
    out["max_abs"] = max(r["max_abs"] for r in out["tasks"].values())
    out["p99_abs"] = max(r["p99_abs"] for r in out["tasks"].values())
    out["ok"] = ok
    return out


def check_retrieve(exported: ExportedScorer, users: np.ndarray, frames: dict, k: int = TOPK) -> dict:
    ok, n_checked, notes = True, 0, []
    metric_rows = []
    for task in TASKS:
        tgt, act = task_parts(task)
        f = frames[("test", task, "masked")]
        pos = {u: i for i, u in enumerate(f.users.tolist())}
        S = exported.score_users(users, tgt, act, "test")
        for ui, u in enumerate(users.tolist()):
            res = retrieve(exported, u, tgt, act, k)
            rows_r = np.searchsorted(exported.catalog_ids[tgt], [i for i, _ in res])
            s = S[ui:ui + 1].copy()
            seen = exported.seen_rows(u, tgt)
            M.apply_mask(s, np.array([0, len(seen)]), seen)
            top, valid = M.rank_topk(s, k)
            want_rows = top[0][valid[0]]
            if not (np.array_equal(rows_r, want_rows)
                    and np.array_equal(np.array([v for _, v in res], np.float32), s[0, want_rows])):
                ok = False
                notes.append({"user_id": u, "task": task, "issue": "retrieve != ExportedScorer"})
            if np.isin(rows_r, seen).any():
                ok = False
                notes.append({"user_id": u, "task": task, "issue": "masked item returned"})
            if u in pos:
                i = pos[u]
                fm = f.mask_idx[f.mask_indptr[i]:f.mask_indptr[i + 1]]
                if not np.array_equal(np.sort(fm), np.sort(seen)):
                    ok = False
                    notes.append({"user_id": u, "task": task, "issue": "retrieve mask != frame mask CSR"})
                # Independent metric recomputation from retrieve()'s ranked ids vs frame GT.
                gt = set(f.gt_rows(i).tolist())
                rel = np.array([r in gt for r in rows_r] + [False] * (k - len(rows_r)), bool)
                disc = 1.0 / np.log2(np.arange(2, k + 2))
                row = {"task": task, "user_id": u}
                for kk in (10, 20, 50):
                    idcg = disc[:min(len(gt), kk)].sum()
                    row[f"recall@{kk}"] = rel[:kk].sum() / len(gt)
                    row[f"ndcg@{kk}"] = (rel[:kk] * disc[:kk]).sum() / idcg
                metric_rows.append(row)
            n_checked += 1
        # mask_seen=False must be allowed to return seen items: verify on a user whose unmasked
        # top-K contains one.
        found = False
        for ui, u in enumerate(users.tolist()):
            seen = exported.seen_rows(u, tgt)
            if len(seen) == 0:
                continue
            nm = retrieve(exported, u, tgt, act, k, mask_seen=False)
            rr = np.searchsorted(exported.catalog_ids[tgt], [i for i, _ in nm])
            top, valid = M.rank_topk(S[ui:ui + 1].copy(), k)
            if not np.array_equal(rr, top[0][valid[0]]):         # padding (catalog < k) is invalid
                ok = False
                notes.append({"user_id": u, "task": task, "issue": "mask_seen=False ranking differs"})
            if np.isin(rr, seen).any():
                found = True
                break
        if not found:
            notes.append({"task": task, "issue": "no sampled user had a seen item in its unmasked top-K"})
    return {"ok": ok, "n_user_task_checked": n_checked, "issues": notes[:50],
            "_metric_rows": pd.DataFrame(metric_rows)}


def check_metrics(direct: DirectScorer, exported: ExportedScorer, frames: dict, ctxs: dict,
                  kinds=("masked",)) -> tuple[dict, dict]:
    keys = [(s, t, k) for s in ("val", "test") for t in TASKS for k in kinds]
    pd_e, sum_e = evaluate_frames(exported, frames, ctxs, keys)
    pd_d, sum_d = evaluate_frames(direct, frames, ctxs, keys)
    res, ok = {}, True
    for key in keys:
        a, b = pd_e[key][["user_id", *METRIC_COLS]], pd_d[key][["user_id", *METRIC_COLS]]
        eq = a.reset_index(drop=True).equals(b.reset_index(drop=True))
        n_diff = int((a.to_numpy() != b.to_numpy()).any(1).sum()) if a.shape == b.shape else -1
        res["/".join(key)] = {"identical": bool(eq), "n_users": int(len(a)), "n_users_differ": n_diff}
        ok &= bool(eq)
    return {"ok": ok, "frames": res}, pd_e


def parity_checks(model, cond, P: dict, export_dir, ckpt_path, user_table=None, n_users: int = 2048,
                  seed: int = 0, frames: dict | None = None, require_all_strata: bool = True) -> dict:
    """Checks 1-4 for one (model, condition, export). `model` must be a fresh reload of the
    checkpoint (`artifacts.load_model`), not the training instance."""
    t0 = time.time()
    exported = ExportedScorer(export_dir, expect_condition_id=cond.condition_id)
    if exported.manifest["checkpoint_sha256"] != A.sha256_file(ckpt_path):
        raise ValueError("export was produced from a different checkpoint")
    if exported.manifest["data_fingerprint"] != A.data_fingerprint(P["events"], P["meta"]):
        raise ValueError("export was produced from different data")
    threads = int(exported.manifest["inference"]["threads"])
    direct = DirectScorer(model, cond, P["meta"], user_table, threads=threads)
    frames = build_eval_frames(P) if frames is None else frames
    ctxs = contexts(P)
    users, cover = select_parity_users(frames, n_users, seed, require_all_strata)
    r12 = check_scores_and_topk(direct, exported, users, frames)
    r3 = check_retrieve(exported, users, frames)
    metric_rows = r3.pop("_metric_rows")
    r4, pu_e = check_metrics(direct, exported, frames, ctxs)
    # Independent metric path (retrieve top-50 vs frame GT) == shared evaluator on those users.
    max_dev = 0.0
    for task, g in metric_rows.groupby("task"):
        ref = pu_e[("test", task, "masked")].set_index("user_id").loc[g.user_id]
        for c in [c for c in g.columns if "@" in c]:
            max_dev = max(max_dev, float(np.abs(g[c].to_numpy() - ref[c].to_numpy()).max()))
    indep_ok = len(metric_rows) > 0 and max_dev <= 1e-12
    r3["independent_metrics_max_abs_dev"] = max_dev
    r3["ok"] = r3["ok"] and indep_ok
    return {"condition_id": cond.condition_id, "users": cover, "check1_2_scores_topk": r12,
            "check3_retrieve": r3, "check4_metrics": r4, "ok": bool(r12["ok"] and r3["ok"] and r4["ok"]),
            "parity_s": round(time.time() - t0, 1)}


def rebuild_condition_from_raw(cfg: dict, P: dict, cond):
    """D1-7: rebuild vocabs / scalers / feature tables / event encodings from raw ids with the
    M2 `fit_condition` (raw metadata CSV + raw-id events) and require the same fingerprint as
    the condition the checkpoint was trained on."""
    from kuairand_rec.data import io
    from kuairand_rec.data.pipeline import fit_condition
    from kuairand_rec.models import data as D

    raw_cols = [c for c in P["events"].columns if not c.endswith("_idx")]
    ev = P["events"][raw_cols]
    fit = fit_condition(ev, io.load_video_features(), P["catalogs"], list(D.ACTIONS), cond.exclude_video_ids)
    rebuilt = D.condition_from_fit(cond.name, fit, cond.exclude_video_ids)
    A.check_fingerprint(A.condition_fingerprint(cond), A.condition_fingerprint(rebuilt), "raw-id rebuild")
    return rebuilt


def run_parity(run_dir, n_users: int = 2048, seed: int = 0, out_name: str = "parity.json",
               rebuild_from_raw: bool = True) -> dict:
    """Reload config / data / checkpoint / export from `run_dir` and run checks 1-4 (CPU)."""
    from kuairand_rec.data.pipeline import load_processed
    from kuairand_rec.models import data as D
    from kuairand_rec.models.train import _resolve, user_features_for

    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    sel = json.loads((run_dir / "selection.json").read_text())
    ckpt = run_dir / sel["checkpoint"]
    P = load_processed(_resolve(cfg.get("processed_dir", "data/processed")))
    cond = D.build_condition(cfg.get("condition", "normal"), P)
    if rebuild_from_raw:
        cond = rebuild_condition_from_raw(cfg, P, cond)
    user_vocab, user_table = user_features_for(cfg, cond)
    model, _ = A.load_model(ckpt, cond, user_vocab, device="cpu")
    res = parity_checks(model, cond, P, run_dir / "export", ckpt, user_table, n_users, seed)
    res.update(run_dir=str(run_dir), checkpoint=sel["checkpoint"], rebuilt_from_raw=rebuild_from_raw)
    A.write_json(run_dir / out_name, res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_users", type=int, default=2048)
    a = ap.parse_args(argv)
    r = run_parity(a.run_dir, a.n_users)
    s = r["check1_2_scores_topk"]
    print(json.dumps({"ok": r["ok"], "max_abs": s["max_abs"], "p99_abs": s["p99_abs"],
                      "topk_mismatches": len(s["topk_mismatches"]), "retrieve_ok": r["check3_retrieve"]["ok"],
                      "metrics_ok": r["check4_metrics"]["ok"], "users": r["users"], "parity_s": r["parity_s"]}))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
