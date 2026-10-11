"""Shared evaluator: every model (Pop, ALS, three-tower) is scored through `evaluate`.

A scorer only receives the evaluated user ids and an `EvalContext` whose `history` comes
from `history_view` (t < t_pred). It never sees GT and never applies the mask: masking,
ranking, tie-break and metrics all live here, so they are identical across models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from kuairand_rec.eval import metrics as M
from kuairand_rec.eval.frame import EvalFrame

KS = (10, 20, 50)


@dataclass(frozen=True)
class EvalContext:
    split: str
    t_pred: int
    history: pd.DataFrame   # history_view(events, split, meta); never GT


class Scorer(Protocol):
    name: str

    def score(self, users: np.ndarray, target: str, action: str, ctx: EvalContext) -> np.ndarray:
        """float32 [len(users), n_items]; -inf = unscorable. Rows follow `users` order."""
        ...

    def signal(self, users: np.ndarray, target: str, action: str, ctx: EvalContext) -> np.ndarray:
        """bool [len(users)]: whether the model has any usable signal for the user."""
        ...

    def item_scorable(self, target: str, action: str) -> np.ndarray:
        """bool [n_items]: items the model can score at all (user-independent)."""
        ...


def evaluate(scorer: Scorer, frame: EvalFrame, ctx: EvalContext, ks=KS, chunk: int = 2048) -> pd.DataFrame:
    """Per-user metrics for one model on one frame. Returns one row per frame user."""
    if ctx.split != frame.split or ctx.t_pred != frame.t_pred:
        raise ValueError("context does not match frame (split / t_pred)")
    if len(ctx.history) and int(ctx.history.time_ms.max()) >= frame.t_pred:
        raise ValueError("history contains events at or after t_pred")
    k_max = max(ks)
    item_ok = np.asarray(scorer.item_scorable(frame.target, frame.action), bool)
    if item_ok.shape != (frame.n_items,):
        raise ValueError("item_scorable shape mismatch")
    parts = []
    n = len(frame.users)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        u = frame.users[a:b]
        s = scorer.score(u, frame.target, frame.action, ctx)
        M.check_scores(s, len(u), frame.n_items)
        s = s.copy()
        gptr = frame.gt_indptr[a:b + 1] - frame.gt_indptr[a]
        gidx = frame.gt_idx[frame.gt_indptr[a]:frame.gt_indptr[b]]
        # Unscorable GT = GT on items the model cannot score for anyone (item-level, so it is
        # not confused with no-signal users). Kept in the denominator, counted.
        g_rows = np.repeat(np.arange(b - a), np.diff(gptr))
        n_unscorable = np.bincount(g_rows[~item_ok[gidx]], minlength=b - a)
        # Consistency: an item declared scorable must not be -inf for a user with signal.
        sig = np.asarray(scorer.signal(u, frame.target, frame.action, ctx), bool)
        if np.isneginf(s[sig][:, item_ok]).any():
            raise AssertionError("scorer returned -inf on a declared-scorable item for a user with signal")
        if frame.masked:
            mptr = frame.mask_indptr[a:b + 1] - frame.mask_indptr[a]
            midx = frame.mask_idx[frame.mask_indptr[a]:frame.mask_indptr[b]]
            M.apply_mask(s, mptr, midx)
        top, valid = M.rank_topk(s, k_max)
        m = M.per_user_metrics(top, valid, gptr, gidx, frame.n_items, ks)
        m["user_id"] = u
        m["n_gt_unscorable"] = n_unscorable
        m["has_signal"] = sig
        parts.append(pd.DataFrame(m))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out["frame_id"] = frame.frame_id
    out["population_id"] = frame.population_id
    out["model"] = scorer.name
    _sanity(out, frame, ks)
    cols = ["user_id", "n_gt"] + [f"{p}@{k}" for k in ks for p in ("hits", "recall", "ndcg")] + [
        "n_gt_unscorable", "has_signal", "frame_id", "population_id", "model"]
    return out[cols] if len(out) else out


def _sanity(df: pd.DataFrame, frame: EvalFrame, ks) -> None:
    if len(df) != len(frame.users) or not (df.user_id.to_numpy() == frame.users).all():
        raise AssertionError("per-user rows do not match frame users")
    for k in ks:
        for p in ("recall", "ndcg"):
            v = df[f"{p}@{k}"].to_numpy()
            if not ((v >= 0) & (v <= 1 + 1e-9)).all():
                raise AssertionError(f"{p}@{k} out of [0,1]")
    for k1, k2 in zip(ks, ks[1:]):
        if (df[f"recall@{k2}"] < df[f"recall@{k1}"] - 1e-12).any():
            raise AssertionError("Recall@K decreasing in K")
    # No-signal users must score 0 (strict handling; a signal-less model cannot hit).
    ns = ~df.has_signal.to_numpy()
    if ns.any() and (df.loc[ns, [f"hits@{k}" for k in ks]].to_numpy() > 0).any():
        raise AssertionError("no-signal users have hits")


def summarize(per_user: pd.DataFrame, frame: EvalFrame, ks=KS) -> dict:
    r = {"model": per_user.model.iloc[0] if len(per_user) else None, "split": frame.split,
         "target": frame.target, "action": frame.action, "kind": frame.kind,
         "n_users": int(len(per_user)), "n_no_signal": int((~per_user.has_signal).sum()),
         "gt_pairs": int(per_user.n_gt.sum()), "gt_unscorable": int(per_user.n_gt_unscorable.sum()),
         "frame_id": frame.frame_id, "population_id": frame.population_id}
    for k in ks:
        r[f"recall@{k}"] = float(per_user[f"recall@{k}"].mean()) if len(per_user) else float("nan")
        r[f"ndcg@{k}"] = float(per_user[f"ndcg@{k}"].mean()) if len(per_user) else float("nan")
    return r
