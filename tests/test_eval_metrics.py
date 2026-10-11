"""Hand-computed metric / ranking tests (PLAN.md §3b P1-1)."""

from __future__ import annotations

import numpy as np
import pytest

from kuairand_rec.eval import metrics as M


def _csr(rows):
    indptr = np.r_[0, np.cumsum([len(r) for r in rows])].astype(np.int64)
    idx = np.concatenate([np.asarray(r, np.int64) for r in rows]) if rows else np.zeros(0, np.int64)
    return indptr, idx


def _run(scores, gt, ks, mask=None):
    s = np.asarray(scores, np.float32).copy()
    if mask is not None:
        M.apply_mask(s, *_csr(mask))
    top, valid = M.rank_topk(s, max(ks))
    return M.per_user_metrics(top, valid, *_csr(gt), s.shape[1], ks)


def test_recall_ndcg_hand_computed():
    # ranking [5, 1, 2, 0, 3, 4]; GT = {2, 5}: hits at ranks 1 and 3
    s = [[0.5, 0.8, 0.7, 0.2, 0.1, 0.9]]
    m = _run(s, [[2, 5]], ks=(1, 2, 3))
    assert m["recall@1"][0] == 0.5 and m["recall@2"][0] == 0.5 and m["recall@3"][0] == 1.0
    assert m["ndcg@1"][0] == pytest.approx(1.0)
    # NDCG@2 = 1 / (1 + 1/log2 3)
    assert m["ndcg@2"][0] == pytest.approx(1 / (1 + 1 / np.log2(3)))
    # NDCG@3 = (1 + 1/log2 4) / (1 + 1/log2 3) = 1.5 / 1.63093 = 0.91972
    assert m["ndcg@3"][0] == pytest.approx(0.919721, abs=1e-6)


def test_idcg_uses_min_gt_k_and_recall_uses_gt():
    # |GT| = 5, K = 2, both top-2 hit -> NDCG@2 = 1 but Recall@2 = 0.4
    s = [[9, 8, 7, 6, 5, 0, 0]]
    m = _run(s, [[0, 1, 2, 3, 4]], ks=(2, 5))
    assert m["ndcg@2"][0] == pytest.approx(1.0)
    assert m["recall@2"][0] == pytest.approx(0.4)
    assert m["recall@5"][0] == 1.0 and m["ndcg@5"][0] == pytest.approx(1.0)


def test_zero_hit_user_kept_in_average():
    s = [[3, 2, 1, 0], [3, 2, 1, 0]]
    m = _run(s, [[0], [3]], ks=(1,))
    assert list(m["recall@1"]) == [1.0, 0.0]
    assert m["recall@1"].mean() == 0.5


def test_ties_break_by_row_ascending():
    s = [[1.0] * 6]
    top, _ = M.rank_topk(np.asarray(s, np.float32), 6)
    assert top[0].tolist() == [0, 1, 2, 3, 4, 5]
    m = _run(s, [[3]], ks=(3, 4))
    assert m["hits@3"][0] == 0 and m["hits@4"][0] == 1
    assert m["ndcg@4"][0] == pytest.approx(1 / np.log2(5))
    # partial tie: higher score first, ties among the rest by row
    top, _ = M.rank_topk(np.asarray([[1, 2, 1, 2, 0]], np.float32), 5)
    assert top[0].tolist() == [1, 3, 0, 2, 4]


def test_neg_inf_never_hits_even_when_k_exceeds_finite():
    # Only rows 0, 1 are finite; GT row 3 is -inf (unscorable) -> no hit at any K.
    ninf = -np.inf
    s = [[0.2, 0.1, ninf, ninf]]
    m = _run(s, [[3]], ks=(4,))
    assert m["hits@4"][0] == 0 and m["recall@4"][0] == 0.0


def test_mask_removes_candidate_and_does_not_occupy_rank():
    # Row 0 has the top score but is masked; GT row 2 moves from rank 3 to rank 2.
    s = [[9, 5, 4, 1]]
    m = _run(s, [[2]], ks=(2,), mask=[[0]])
    assert m["hits@2"][0] == 1
    assert m["ndcg@2"][0] == pytest.approx(1 / np.log2(3))
    m0 = _run(s, [[2]], ks=(2,))
    assert m0["hits@2"][0] == 0


def test_check_scores_guards():
    with pytest.raises(TypeError):
        M.check_scores(np.zeros((1, 3), np.float64), 1, 3)
    with pytest.raises(ValueError):
        M.check_scores(np.zeros((2, 3), np.float32), 1, 3)
    bad = np.zeros((1, 3), np.float32)
    bad[0, 1] = np.nan
    with pytest.raises(ValueError):
        M.check_scores(bad, 1, 3)


def test_empty_gt_rejected():
    s = np.zeros((1, 3), np.float32)
    top, valid = M.rank_topk(s, 2)
    with pytest.raises(ValueError):
        M.per_user_metrics(top, valid, np.array([0, 0]), np.zeros(0, np.int64), 3, ks=(1,))
