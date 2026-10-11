"""Deterministic full-catalog top-K and per-user Recall/NDCG (PLAN.md §3b P1-1).

Conventions (correctness-critical, all tested by hand in tests/test_eval_metrics.py):
* Ranking = score descending, then catalog row ascending. Implemented as a *stable* full
  argsort of -score (no argpartition: its order at the K boundary is arbitrary under ties).
* A position whose score is -inf (masked or unscorable) is never a hit, even if K exceeds
  the number of finite scores.
* Recall@K = hits@K / |GT|;  NDCG@K = DCG@K / IDCG@K with binary gain and
  IDCG over min(|GT|, K) relevant items. Every user with |GT| >= 1 counts (0-hit users too).
"""

from __future__ import annotations

import numpy as np

NEG_INF = np.float32(-np.inf)


def apply_mask(scores: np.ndarray, indptr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Set masked catalog rows to -inf in place. CSR rows align with score rows."""
    n = scores.shape[0]
    counts = np.diff(indptr)
    r = np.repeat(np.arange(n), counts)
    scores[r, idx] = NEG_INF
    return scores


def check_scores(scores: np.ndarray, n_users: int, n_items: int) -> None:
    if scores.dtype != np.float32:
        raise TypeError(f"scores must be float32, got {scores.dtype}")
    if scores.shape != (n_users, n_items):
        raise ValueError(f"scores shape {scores.shape} != {(n_users, n_items)}")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("scores contain NaN or +inf")


def rank_topk(scores: np.ndarray, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k_max catalog rows per user and a validity flag (score > -inf)."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k_max]
    valid = np.take_along_axis(scores, order, axis=1) > NEG_INF
    if order.shape[1] < k_max:  # catalog smaller than K: pad with never-hit positions
        pad = k_max - order.shape[1]
        order = np.pad(order, ((0, 0), (0, pad)))
        valid = np.pad(valid, ((0, 0), (0, pad)), constant_values=False)
    return order, valid


def per_user_metrics(top: np.ndarray, valid: np.ndarray, gt_indptr: np.ndarray, gt_idx: np.ndarray,
                     n_items: int, ks=(10, 20, 50)) -> dict[str, np.ndarray]:
    n, k_max = top.shape
    if max(ks) > k_max:
        raise ValueError("k_max smaller than the largest K")
    n_gt = np.diff(gt_indptr)
    if (n_gt < 1).any():
        raise ValueError("every evaluated user needs |GT| >= 1")
    gt_dense = np.zeros((n, n_items), dtype=bool)
    gt_dense[np.repeat(np.arange(n), n_gt), gt_idx] = True
    rel = np.take_along_axis(gt_dense, top, axis=1) & valid
    disc = 1.0 / np.log2(np.arange(2, k_max + 2))
    cum_disc = np.cumsum(disc)
    out = {"n_gt": n_gt}
    for k in ks:
        hits = rel[:, :k].sum(1)
        dcg = (rel[:, :k] * disc[:k]).sum(1)
        idcg = cum_disc[np.minimum(n_gt, k) - 1]
        out[f"hits@{k}"] = hits
        out[f"recall@{k}"] = hits / n_gt
        out[f"ndcg@{k}"] = dcg / idcg
    return out
