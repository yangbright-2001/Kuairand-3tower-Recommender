"""S9 train/val performance diagnostics for the formal three-tower runs (read-only).

Reproduces the diagnostics D1-D12 used in the S9 research plan
(.cursor/plans/三塔优化研究方案_b178e03c.plan.md §1) and the low-cost probes P1-P4 added in
its Revision 1. Every number written here is a *diagnostic*, not a reported metric.

Data scope (enforced):
  * train events (`train_view`) and the normal-condition validation masked frames at T_val
    (`build_eval_frames(P, ("val",), ("masked",))`, frame_ids asserted equal to M3).
  * formal artifacts are opened read-only: checkpoints, `selection.json`, `train_log.jsonl`
    and `per_user/val_*.parquet`. `_val_parquet` refuses any per-user file that is not val.
  * test labels are never read: no test frame is built and no test per-user / metrics file is
    opened. (`events.parquet` is loaded whole, exactly as the formal Validator does.)
Determinism: fixed numpy / torch seeds per diagnostic, fixed torch threads, CPU, FP32.

Run (from the explore worktree):
    uv run python scripts/explore/diagnose_u0.py \
        --formal_root /Users/yangliang/Desktop/kuairand-threetower-recsys
Outputs: results/explore/diagnostics/{D*,P*}.json + DIAGNOSTICS_RUN.json (inputs, sha256, timing).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kuairand_rec import paths
from kuairand_rec.data.pipeline import load_processed
from kuairand_rec.models import data as D
from kuairand_rec.models import features as F
from kuairand_rec.models.artifacts import load_model
from kuairand_rec.models.evaluation import build_eval_frames, contexts, evaluate_frames
from kuairand_rec.models.inference import EmbeddingScorer, inference_settings
from kuairand_rec.models.towers import TASKS, ItemEncodings
from kuairand_rec.models.train import ItemInputs, forward_loss
from kuairand_rec.utils import git_sha

RUNS = {  # formal runs (read-only), relative to <formal_root>/artifacts
    "u0": "u0/9218a76_seed0_job27",
    "u1": "u1/9218a76_seed0_job28",
    "abl_nohist": "abl_nohist/671dde9_seed0_job34",
    "abl_nomusic": "abl_nomusic/671dde9_seed0_job35",
}
BASELINES = ("pop", "als", "als_like_aux")     # M3 per-user val files (like tasks only for aux)
FULL_BINS = [0, 5, 20, 50, 100, 200, 400, 10**9]
HIST_FIELDS = ("video_row", "feedback", "hidden", "play_bucket", "recency_bucket", "valid", "length")


# ----------------------------------------------------------------------------- helpers
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _val_parquet(p: Path) -> pd.DataFrame:
    if not p.name.startswith("val_"):
        raise PermissionError(f"diagnostics may only read val per-user files, got {p}")
    return pd.read_parquet(p)


def eff_rank(X: np.ndarray) -> float:
    """exp(entropy) of the normalised squared singular values of the centred matrix."""
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = s**2 / (s**2).sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def r(x, k=4):
    return None if x is None else round(float(x), k)


class Ctx:
    """Shared, lazily built state (data, condition, val frames, checkpoints)."""

    def __init__(self, a):
        self.a = a
        self.root = Path(a.formal_root)
        self.P = load_processed(paths.PROJECT_ROOT / "data" / "processed")
        self.cond = D.normal_condition(self.P)
        self.frames = build_eval_frames(self.P, ("val",), ("masked",))   # asserts M3 frame_ids
        self.keys = sorted(self.frames, key=lambda k: TASKS.index(k[1]))
        self.vctx = contexts(self.P, ("val",))
        self.items = ItemInputs.from_condition(self.cond, "cpu")
        self.hidx_val, self.t_val = F.frozen_history_index(self.cond.events, "val", self.P["meta"])
        self.users = np.unique(np.concatenate([self.frames[k].users for k in self.keys]))
        self._models = {}

    def run_dir(self, name):
        return self.root / "artifacts" / RUNS[name]

    def model(self, name, ckpt="best.pt"):
        key = (name, ckpt)
        if key not in self._models:
            m, ck = load_model(self.run_dir(name) / ckpt, self.cond, None, "cpu")
            self._models[key] = (m.eval(), ck)
        return self._models[key]

    def frame(self, task):
        return self.frames[("val", task, "masked")]

    # -- encoders (validation-style batched encoding; diagnostics only, not the canonical export)
    def item_embs(self, model, music_override=None):
        """(ItemEncodings, {task: np [n_catalog, 64]}). `music_override(video_music_base)` lets P1
        replace the music input of the video tower; m_base / music heads are left unchanged."""
        with inference_settings(self.a.threads):
            if music_override is None:
                enc = model.encode_items(self.items.vf, self.items.mf)
            else:
                m_base, vm = model._music_forward(self.items.vf, self.items.mf, None)
                vm2 = music_override(vm)
                v_base = model.video_tower(self.items.vf, {}, vm2)
                enc = ItemEncodings(v_base, m_base, vm2)
            I = {t: v.numpy().copy() for t, v in model.item_embeddings(enc).items()}
        return enc, I

    def history(self, users):
        return F.build_history_batch(self.hidx_val, users, self.t_val, F.HistoryConfig())

    def user_embs(self, model, enc, users, hb=None, bs=512):
        ui = torch.as_tensor(self.cond.vocabs.user.encode(users), dtype=torch.long)
        use_h = model.cfg.use_history
        if use_h and hb is None:
            hb = self.history(users)
        ht = hb.to_torch("cpu") if use_h else None
        U = {t: np.empty((len(users), model.cfg.head_dim), np.float32) for t in TASKS}
        with inference_settings(self.a.threads):
            for a in range(0, len(users), bs):
                h = F.HistoryTensors(*(getattr(ht, f)[a:a + bs] for f in HIST_FIELDS)) if use_h else None
                e = model.user_embeddings(model.encode_users(ui[a:a + bs], h, enc))
                for t in TASKS:
                    U[t][a:a + bs] = e[t].numpy()
        return U

    def evaluate(self, U, I, users, tasks=TASKS):
        sc = EmbeddingScorer("diag", {"val": users}, {"val": U}, I, {"val": self.t_val})
        keys = [k for k in self.keys if k[1] in tasks]
        _, s = evaluate_frames(sc, {k: self.frames[k] for k in keys}, self.vctx, keys)
        return {f"{x.target}_{x.action}": {"recall@20": r(x["recall@20"], 6), "ndcg@20": r(x["ndcg@20"], 6)}
                for _, x in s.iterrows()}


def mean_r20(res):
    return r(np.mean([v["recall@20"] for v in res.values()]), 6)


# ----------------------------------------------------------------------------- D1 / D12
def d1_buckets(c: Ctx) -> dict:
    """Per-user val R@20 of every formal model and M3 baseline, grouped by the P1-1 bucket
    (capped length) and by finer full-length bins. D12: per-user standard error."""
    out = {}
    base_dir = Path(c.a.baselines_per_user)
    for task in TASKS:
        f = c.frame(task)
        df = pd.DataFrame({"user_id": f.users, "len_capped": f.hist_len_capped, "len_full": f.hist_len_full})
        models = list(RUNS) + [b for b in BASELINES if b != "als_like_aux" or "is_like" in task]
        for m in models:
            p = (c.run_dir(m) / "per_user" if m in RUNS else base_dir / m) / f"val_{task}_masked.parquet"
            pu = _val_parquet(p)
            if not (np.array_equal(pu.user_id.to_numpy(), f.users) and (pu.frame_id == f.frame_id).all()):
                raise ValueError(f"{m}/{task}: per-user file not aligned with the M3 val frame")
            df[m] = pu["recall@20"].to_numpy()
        df["bucket"] = pd.cut(df.len_capped, [0, 5, 20, 100], labels=["1-5", "6-20", ">20"])
        df["full_bin"] = pd.cut(df.len_full, FULL_BINS).astype(str)
        res = {"n_users": int(len(df)), "frame_id": f.frame_id,
               "overall": {m: r(df[m].mean(), 6) for m in models},
               "se_per_user_unpaired": {m: r(df[m].std(ddof=1) / math.sqrt(len(df)), 6) for m in models}}
        for col in ("bucket", "full_bin"):
            g = df.groupby(col, observed=True)
            res[col] = {str(k): {"n": int(len(x)), **{m: r(x[m].mean(), 4) for m in models},
                                 "u0_minus_als": r(x.u0.mean() - x.als.mean(), 4),
                                 "u0_minus_nohist": r(x.u0.mean() - x.abl_nohist.mean(), 4)}
                        for k, x in g}
        out[task] = res
    return {"method": "join M3 val masked frame (bucket = P1-1 capped length; full_bin = uncapped "
                      "length) with per-user val recall@20 of each model; group means. SE = std/sqrt(n), "
                      "unpaired, magnitude only.", "tasks": out}


# ----------------------------------------------------------------------------- D2 / D3
def d2_trajectories(c: Ctx) -> dict:
    out, per = {}, {}
    for m in RUNS:
        sel = json.loads((c.run_dir(m) / "selection.json").read_text())
        if sel.get("test_used_for_selection") is not False:
            raise ValueError(f"{m}: selection.json does not certify test_used_for_selection = false")
        hist = sel["val_history"]
        per[m] = {h["epoch"]: h for h in hist}
        out[m] = {"best_epoch": sel["best_epoch"], "best_score": r(sel["best_score"], 6),
                  "epochs": [{"epoch": h["epoch"], "score": r(h["score"], 4), "tie": r(h["tie"], 4),
                              **{t: r(h[f"{t}/recall@20"], 4) for t in TASKS}} for h in hist]}
    cmp = {}
    for other in ("abl_nomusic", "u1"):
        common = sorted(set(per["u0"]) & set(per[other]))
        cmp[f"{other}_vs_u0"] = {t: {"epochs": len(common),
                                     "other_higher": int(sum(per[other][e][f"{t}/recall@20"] > per["u0"][e][f"{t}/recall@20"]
                                                             for e in common))} for t in TASKS}
    return {"method": "selection.json val_history (D1-8 validation path). Epoch-wise counts share only "
                      "the data order (epoch_batches(seed, epoch)); model init / dropout RNG consumption "
                      "and code SHA differ, and epochs are autocorrelated -> not independent trials.",
            "runs": out, "epochwise_comparison": cmp}


def d3_train_logs(c: Ctx) -> dict:
    out = {}
    for m in RUNS:
        L = [json.loads(x) for x in (c.run_dir(m) / "train_log.jsonl").read_text().splitlines()]
        gn = np.array([x["grad_norm"] for x in L])
        res = {"records": len(L), "log_every": "step 1 + every 50 steps",
               "grad_norm_pre_clip": {"min": r(gn.min(), 3), "median": r(np.median(gn), 3), "max": r(gn.max(), 3),
                                      "share_gt_clip_1.0": r((gn > 1.0).mean(), 3),
                                      "median_after_warmup": r(np.median(gn[[x["step"] >= 300 for x in L]]), 3)},
               "lr_after_warmup": sorted({x["lr"] for x in L if x["step"] >= 300})}
        for t in TASKS:
            lo = [x[f"loss_{t}"] for x in L if x[f"loss_{t}"] is not None]
            n = np.array([x[f"n_{t}"] for x in L])
            fn = np.array([x[f"fn_{t}"] for x in L])
            res[t] = {"loss_first": r(lo[0], 3), "loss_last": r(lo[-1], 3), "loss_last5_mean": r(np.mean(lo[-5:]), 3),
                      "valid_rows_per_batch": [int(n.min()), int(n.max())],
                      "fn_masked_cols_per_valid_row": r((fn / np.maximum(n, 1)).mean(), 2)}
        out[m] = res
    return {"method": "train_log.jsonl (pre-clip total grad norm, per-task loss / valid rows / FN pairs).",
            "runs": out}


# ----------------------------------------------------------------------------- D4 / P2
def d4_fusion(c: Ctx) -> dict:
    """First-layer contribution of each video-tower input block (D4) + P2 normalisation and a
    variance decomposition of the first-layer pre-activation over train-vocab candidates."""
    rows = np.asarray(c.cond.cand_video_rows)
    out = {}
    for m in ("u0", "abl_nomusic"):
        model, ck = c.model(m)
        vt, vf = model.video_tower, c.items.vf
        with inference_settings(c.a.threads):
            enc = model.encode_items(vf, c.items.mf)
            parts = {"video_id": vt.video_emb(vf.video_idx)}
            if vt.author_emb is not None:
                parts["author"] = vt.author_emb(vf.author_idx)
            if vt.cfg.use_music:
                parts["music_base"] = enc.video_music_base
            for j, (col, emb) in enumerate(zip(("video_type", "upload_type", "upload_dt", "music_type"), vt.cat_embs)):
                parts[col] = emb(vf.cat[:, j])
            tv = (vf.tags != 0).float()
            parts["tags"] = (vt.tag_emb(vf.tags) * tv.unsqueeze(-1)).sum(1) / tv.sum(1, keepdim=True).clamp_min(1)
            parts["num"], parts["num_missing"] = vf.num, vf.num_missing
            W = vt.mlp[0].weight
            off, contrib = 0, {}
            for k, x in parts.items():
                d = x.shape[1]
                contrib[k] = (x @ W[:, off:off + d].T)[rows].detach().numpy().astype(np.float64)
                off += d
            assert off == W.shape[1], "input blocks do not tile the first layer"
            xs = {k: x[rows].numpy() for k, x in parts.items()}
        Z = sum(contrib.values())
        var_total = float(Z.var(0).sum())
        blocks = {}
        for k, C in contrib.items():
            d = xs[k].shape[1]
            nrm = np.linalg.norm(C, axis=1).mean()
            blocks[k] = {"dim": d, "mean_input_norm": r(np.linalg.norm(xs[k], axis=1).mean(), 3),
                         "mean_contrib_norm": r(nrm, 3), "contrib_norm_per_sqrt_dim": r(nrm / math.sqrt(d), 3),
                         "var_share": r(C.var(0).sum() / var_total, 3)}
        cross = var_total - sum(C.var(0).sum() for C in contrib.values())
        out[m] = {"epoch": ck.get("epoch"), "blocks": blocks, "var_total": r(var_total, 3),
                  "cross_term_share": r(cross / var_total, 3)}
    return {"method": "eval mode, train-vocab candidate videos. contrib_k = W_k x_k with W_k the first-layer "
                      "columns of block k (pre-activation only; GELU, layer 2 and LayerNorm follow, so this does "
                      "NOT measure influence on v_base). var_share = tr Cov(contrib_k) / tr Cov(sum); "
                      "cross_term_share = covariance between blocks.", "runs": out}


# ----------------------------------------------------------------------------- D5 / P4
def _grad_stats(c: Ctx, model, n_batches: int, train_mode: bool, seed: int) -> dict:
    groups = {"music_tower": model.music_tower, "video_tower": model.video_tower,
              "user_tower": model.user_tower, "history_encoder": model.user_tower.history}
    base = D.build_samples(c.cond)
    bb = D.BatchBuilder(c.cond, base, D.PosTrain.build(base, c.cond.vocabs.user.size), F.HistoryConfig(), True, None)
    rng = np.random.default_rng(seed)
    cos = {g: [] for g in groups}
    norms = {g: {t: [] for t in TASKS} for g in groups}
    torch.set_num_threads(c.a.threads)
    for it in range(n_batches):
        batch = bb(rng.choice(len(base), 1024, replace=False), "cpu")
        G = {}
        for ti, t in enumerate(TASKS):
            model.train(train_mode)
            torch.manual_seed(seed * 1000 + it * 10 + ti)        # train mode: reproducible dropout draw
            model.zero_grad(set_to_none=True)
            loss, _ = forward_loss(model, c.items, batch.only([t]), 0.05)
            loss.backward()
            G[t] = {g: torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                                  for p in mod.parameters()]) for g, mod in groups.items()}
        for g in groups:
            M = torch.stack([G[t][g] for t in TASKS])
            for i, t in enumerate(TASKS):
                norms[g][t].append(float(M[i].norm()))
            Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cos[g].append((Mn @ Mn.T).numpy())
    model.eval()
    model.zero_grad(set_to_none=True)
    out = {}
    for g in groups:
        A = np.stack(cos[g])                                  # [batches, 4, 4]
        iu = np.triu_indices(4, 1)
        pair = A[:, iu[0], iu[1]]
        out[g] = {"grad_norm_mean": {t: r(np.mean(v), 3) for t, v in norms[g].items()},
                  "cos_mean": np.round(A.mean(0), 3).tolist(),
                  "cos_min_over_batches_and_pairs": r(pair.min(), 3),
                  "share_negative_pair_batches": r((pair < 0).mean(), 3)}
    return out


def d5_gradients(c: Ctx) -> dict:
    model, _ = c.model("u0")
    return {"method": "U0 best.pt, 4 random train batches (B=1024, rng 0), single-task loss (tau 0.05) "
                      "backward in TRAIN mode (dropout on; per-task draws differ). Only the final checkpoint: "
                      "says nothing about gradient conflict during training.",
            "tasks_order": list(TASKS), "groups": _grad_stats(c, model, 4, True, 0)}


def p4_gradients(c: Ctx) -> dict:
    out = {}
    for ck in ("best.pt", "last.pt"):
        model, meta = c.model("u0", ck)
        out[ck] = {"epoch_or_step": meta.get("epoch", meta.get("step")),
                   "groups": _grad_stats(c, model, c.a.p4_batches, False, 1)}
    return {"method": f"U0 best.pt and last.pt, {c.a.p4_batches} random train batches (rng 1), EVAL mode "
                      "(no dropout / ID dropout -> deterministic), single-task backward. Two checkpoints only: "
                      "does not prove the absence of conflict during training.", "tasks_order": list(TASKS),
            "checkpoints": out}


# ----------------------------------------------------------------------------- D6 / P3
def d6_p3_pooling(c: Ctx) -> dict:
    model, _ = c.model("u0")
    H = model.user_tower.history
    res = {}
    with inference_settings(c.a.threads):
        enc = model.encode_items(c.items.vf, c.items.mf)
        for scope, users in (("first3000_video_lv", c.frame("video_long_view").users[:3000]),
                             ("all_val_users", c.users)):
            ent_pool, mass, att_ent, n_full = [], [], {0: [], 1: []}, 0
            for a in range(0, len(users), 1024):
                h = c.history(users[a:a + 1024]).to_torch("cpu")
                valid = h.valid
                vfm = valid.unsqueeze(-1).float()
                item = enc.v_base[h.video_row.clamp_min(0)] * vfm
                x = (H.item_proj(item) * vfm + H.fb_proj(torch.cat([h.feedback, h.hidden.unsqueeze(-1)], -1))
                     + H.play_emb(h.play_bucket) + H.rec_emb(h.recency_bucket) + H.pos_emb.unsqueeze(0))
                has = h.length > 0
                key_ok = valid.clone()
                key_ok[:, -1] |= ~has
                full = (h.length == H.max_len)
                for li, blk in enumerate(H.blocks):
                    B, L, d = x.shape
                    q, k, _ = blk.qkv(blk.ln1(x)).view(B, L, 3, blk.h, blk.dh).permute(2, 0, 3, 1, 4)
                    att = ((q @ k.transpose(-1, -2)) / math.sqrt(blk.dh)).masked_fill(
                        ~key_ok[:, None, None, :], float("-inf")).softmax(-1)          # [B, h, L, L]
                    e = -(att * torch.log(att.clamp_min(1e-12))).sum(-1)              # [B, h, L]
                    if full.any():
                        att_ent[li].append(e[full].mean(dim=(1, 2)).numpy())          # full users only
                    x = blk(x, key_ok)
                x = H.ln_f(x)
                s = (H.pool_key(x) @ H.pool_query) / math.sqrt(x.shape[-1])
                pa = s.masked_fill(~key_ok, float("-inf")).softmax(-1)
                if full.any():
                    A = pa[full].numpy()
                    n_full += int(full.sum())
                    ent_pool.append(-(A * np.log(A + 1e-12)).sum(1))
                    mass.append(np.stack([A[:, -k:].sum(1) for k in (1, 5, 10, 20, 50)], 1))
            ep = np.concatenate(ent_pool)
            res[scope] = {"n_users": int(len(users)), "n_full_100": n_full,
                          "pool_entropy_mean": r(ep.mean(), 4), "uniform_entropy": r(math.log(H.max_len), 4),
                          "pool_eff_tokens": r(np.exp(ep).mean(), 1),
                          "pool_mass_last_k": dict(zip(("1", "5", "10", "20", "50"),
                                                       np.round(np.concatenate(mass).mean(0), 3).tolist())),
                          "self_attention_entropy_mean_by_layer": {str(li): r(np.concatenate(v).mean(), 4)
                                                                   for li, v in att_ent.items() if v}}
    # P3: replace the learned pooling query by 0 -> exactly uniform weights over the valid keys.
    enc, I = c.item_embs(model)
    orig = c.evaluate(c.user_embs(model, enc, c.users), I, c.users)
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        m2.user_tower.history.pool_query.zero_()
    enc2, _ = c.item_embs(m2)
    unif = c.evaluate(c.user_embs(m2, enc2, c.users), I, c.users)
    return {"method": "U0 best.pt at T_val. D6: final pooling weights on users with 100 tokens; self-attention "
                      "entropy per layer (mean over heads / query rows). P3: pool_query := 0 (uniform pooling) "
                      "with everything else fixed, re-evaluated on the 4 val frames. Near-uniform pooling does "
                      "NOT make the encoder equivalent to mean pooling (the self-attention layers still mix).",
            "D6": res, "P3": {"original": orig, "uniform_pooling": unif,
                              "mean_r20": {"original": mean_r20(orig), "uniform_pooling": mean_r20(unif)}}}


# ----------------------------------------------------------------------------- D7 / P1
def d7_recency(c: Ctx) -> dict:
    model, _ = c.model("u0")
    enc, I = c.item_embs(model)
    hb0 = c.history(c.users)
    rb, v = hb0.recency_bucket, hb0.valid
    top = F.HistoryConfig().recency_max_bucket + 1          # stored row of the largest bucket
    variants = {"original": rb, "shift_minus2_fresher": np.where(v, np.maximum(rb - 2, 1), 0),
                "shift_minus4_fresher": np.where(v, np.maximum(rb - 4, 1), 0),
                "all_max_bucket": np.where(v, top, 0)}
    out = {}
    for name, b in variants.items():
        res = c.evaluate(c.user_embs(model, enc, c.users, dataclasses.replace(hb0, recency_bucket=b)), I, c.users)
        out[name] = {"mean_r20": mean_r20(res), "tasks": res}
    return {"method": "U0 best.pt; T_val histories; only recency_bucket replaced at inference (out-of-"
                      "distribution perturbation of a trained model; says nothing about the value of recency "
                      "for a model trained differently).", "variants": out}


def p1_music(c: Ctx) -> dict:
    model, _ = c.model("u0")
    rng = np.random.default_rng(0)
    perm = torch.as_tensor(rng.permutation(len(c.items.vf)))
    variants = {"original": None, "music_input_zero": lambda vm: torch.zeros_like(vm),
                "music_input_permuted": lambda vm: vm[perm]}
    out = {}
    for name, fn in variants.items():
        enc, I = c.item_embs(model, fn)
        res = c.evaluate(c.user_embs(model, enc, c.users), I, c.users)
        out[name] = {"mean_r20": mean_r20(res), "tasks": res}
    return {"method": "U0 best.pt, eval mode. The video tower's music input (gathered m_base) is replaced by "
                      "0 or by a random permutation over videos (rng 0); m_base / music candidates are unchanged; "
                      "history tokens use the modified v_base. Measures dependence of the TRAINED U0 on the "
                      "music path, not negative transfer.", "variants": out}


# ----------------------------------------------------------------------------- D8 / D9 / D10 / D11
def d8_untrained_probe(c: Ctx) -> dict:
    model, _ = c.model("u0")
    _, I = c.item_embs(model)
    users = c.frame("video_long_view").users
    hb = c.history(users)
    vr, valid, lv = hb.video_row, hb.valid, hb.feedback[..., 1]
    out = {}
    Iv = I["video_long_view"]
    for n in (10, 100):
        for pos in (False, True):
            m = valid.copy()
            m[:, :-n] = False
            if pos:
                m &= (lv > 0) | (hb.hidden > 0)
            E = np.where(m[..., None], Iv[np.maximum(vr, 0)], 0).sum(1)
            E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
            U = {t: E.astype(np.float32) for t in TASKS}
            res = c.evaluate(U, I, users, ("video_long_view",))
            out[f"last{n}_{'lv_or_hidden' if pos else 'all'}"] = res["video_long_view"]
    return {"method": "no user tower: user vector = L2(sum of U0 video_long_view item embeddings of the last N "
                      "history videos), video_long_view val only. Low-information reference.", "variants": out}


def d9_recency_distribution(c: Ctx) -> dict:
    base = D.build_samples(c.cond)
    idx = np.random.default_rng(0).choice(len(base), 20000, replace=False)
    hc = F.HistoryConfig()
    tr = F.build_history_batch(F.train_history_index(c.cond.events), base.user_id[idx], base.time_ms[idx], hc)
    va = c.history(c.frame("video_long_view").users)

    def dist(hb):
        last = hb.recency_bucket[hb.valid[:, -1], -1]
        allb = hb.recency_bucket[hb.valid]
        n = hc.n_recency_rows
        return {"empty_share": r((hb.length == 0).mean(), 4),
                "last_token_bucket_share": np.round(np.bincount(last, minlength=n)[1:] / len(last), 4).tolist(),
                "all_token_bucket_share": np.round(np.bincount(allb, minlength=n)[1:] / len(allb), 4).tolist(),
                "hidden_token_share": r(hb.hidden[hb.valid].mean(), 4),
                "length_quantiles": np.quantile(hb.length, [.1, .25, .5, .75, .9]).tolist()}
    t, v = dist(tr), dist(va)
    tv = 0.5 * float(np.abs(np.array(t["all_token_bucket_share"]) - np.array(v["all_token_bucket_share"])).sum())
    return {"method": "train: 20,000 base samples (rng 0), rolling history at t_event (Q2a). val: video_long_view "
                      "val users at T_val. Bucket b (index 0 = stored row 1) covers age_min in [2^b - 1, 2^(b+1) - 1); "
                      "last row is clipped (>= 8191 min).", "bucket_lower_edge_minutes": [2**b - 1 for b in range(14)],
            "train_rolling": t, "val_T_val": v, "total_variation_all_tokens": r(tv, 4)}


def d10_eff_rank(c: Ctx) -> dict:
    rows, mrows = np.asarray(c.cond.cand_video_rows), np.asarray(c.cond.cand_music_rows)
    out = {}
    for name, ck in (("u0", "best.pt"), ("u0", "last.pt"), ("abl_nomusic", "best.pt"), ("abl_nohist", "best.pt")):
        model, meta = c.model(name, ck)
        enc, I = c.item_embs(model)
        out[f"{name}/{ck}"] = {"v_base": r(eff_rank(enc.v_base.numpy()[rows]), 2),
                               "m_base": r(eff_rank(enc.m_base.numpy()[mrows]), 2),
                               **{t: r(eff_rank(I[t][rows if t.startswith("video") else mrows]), 2) for t in TASKS}}
    return {"method": "train-vocab candidates; effective rank = exp(entropy of s_i^2 / sum s^2) of the centred "
                      "matrix (head dim 64, base 128). Descriptive only: no causal link to metrics is claimed.",
            "runs": out}


def d11_feedback_shares(c: Ctx) -> dict:
    out = {}
    for task in ("video_long_view", "video_is_like"):
        f = c.frame(task)
        hb = c.history(f.users)
        b = pd.cut(f.hist_len_capped, [0, 5, 20, 100], labels=["1-5", "6-20", ">20"]).astype(str)
        res = {}
        for k in ("1-5", "6-20", ">20"):
            m = np.asarray(b == k)
            vv = hb.valid[m]
            res[k] = {"users": int(m.sum()), "long_view_token_share": r(hb.feedback[m][..., 1][vv].mean(), 4),
                      "like_token_share": r(hb.feedback[m][..., 2][vv].mean(), 4),
                      "hidden_token_share": r(hb.hidden[m][vv].mean(), 4)}
        out[task] = res
    return {"method": "T_val histories of the val masked frame users, token shares by P1-1 bucket.", "tasks": out}


DIAGS = {"D1_D12_buckets": d1_buckets, "D2_trajectories": d2_trajectories, "D3_train_logs": d3_train_logs,
         "D4_P2_fusion": d4_fusion, "D5_gradients": d5_gradients, "D6_P3_pooling": d6_p3_pooling,
         "D7_recency": d7_recency, "D8_untrained_probe": d8_untrained_probe,
         "D9_recency_distribution": d9_recency_distribution, "D10_eff_rank": d10_eff_rank,
         "D11_feedback_shares": d11_feedback_shares, "P1_music": p1_music, "P4_gradients": p4_gradients}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal_root", required=True, help="repo holding the formal artifacts/ (read-only)")
    ap.add_argument("--baselines_per_user", default=None,
                    help="M3 per-user dir (default <formal_root>/results/baselines/per_user)")
    ap.add_argument("--out_dir", default=str(paths.PROJECT_ROOT / "results" / "explore" / "diagnostics"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--p4_batches", type=int, default=16)
    a = ap.parse_args(argv)
    a.baselines_per_user = a.baselines_per_user or str(Path(a.formal_root) / "results" / "baselines" / "per_user")
    torch.manual_seed(0)
    np.random.seed(0)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = Ctx(a)
    run = {"git_sha": git_sha(), "formal_root": str(a.formal_root), "threads": a.threads, "torch": torch.__version__,
           "condition_id": c.cond.condition_id,
           "val_frame_ids": {k[1]: c.frames[k].frame_id for k in c.keys},
           "checkpoints_sha256": {f"{m}/{ck}": sha256(c.run_dir(m) / ck) for m in RUNS for ck in ("best.pt", "last.pt")},
           "scope": "train events + val masked frames only; no test frame or test file opened", "timing_s": {}}
    for name, fn in DIAGS.items():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        res = fn(c)
        run["timing_s"][name] = round(time.time() - t0, 1)
        (out / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({"done": name, "s": run["timing_s"][name]}), flush=True)
    (out / "DIAGNOSTICS_RUN.json").write_text(json.dumps(run, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
