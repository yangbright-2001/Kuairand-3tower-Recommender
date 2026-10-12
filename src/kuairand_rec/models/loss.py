"""Train-seen full-softmax contrastive loss (PLAN.md §3d D1-3, Q1 / Q6 / Q7a).

Per task (video / music x long_view / is_like), for every *valid* training row r of the batch:

    logits_rc = cos(u_r, i_c) / tau          over c in the condition's train-vocab candidates
    logits_rc = -inf  if c != y_r and c in Pos_train(user_r, task)      (false-negative mask)
    L_task    = mean_r [ logsumexp_c logits_rc - logits_r,y_r ]
    L         = sum over tasks with >= 1 valid row (equal weights)

* Exact softmax over a fixed candidate set: no sampling, so no logQ correction.
* Duplicate positives: candidate columns are unique items, so rows with the same label item
  (same or different users, or videos sharing one music) point at the same column.
* The FN mask is per (user, task) and never touches the row's own label column, so the label
  logit is always finite; an all-masked row degenerates to CE = 0 (finite, zero gradient).
* A task with no valid rows contributes nothing (never 0/0).

This module is pure tensor math; the sample / mask construction lives in `models/data.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kuairand_rec.models.towers import TASKS


@dataclass(frozen=True)
class TaskTargets:
    """Per-task targets of one batch, in *candidate-column* space.

    valid:   [B] bool, the row is a positive for this task (music tasks: music_id != 0).
    label:   [B] long, candidate column of the label (-1 where not valid).
    fn_rows: [P] long, batch row of each false-negative pair (valid rows only).
    fn_cols: [P] long, candidate column to mask; never equal to label[fn_rows].
    """

    valid: torch.Tensor
    label: torch.Tensor
    fn_rows: torch.Tensor
    fn_cols: torch.Tensor

    def to(self, device) -> "TaskTargets":
        return TaskTargets(*(t.to(device) for t in (self.valid, self.label, self.fn_rows, self.fn_cols)))


def masked_logits(user_emb: torch.Tensor, cand_emb: torch.Tensor, tt: TaskTargets, tau: float):
    """(valid batch rows [n], label columns [n], logits [n, C] with FN columns at -inf).

    user_emb [B, d] and cand_emb [C, d] are L2-normalised head outputs; n may be 0.
    """
    vi = torch.nonzero(tt.valid, as_tuple=False).squeeze(1)
    n = int(vi.numel())
    label = tt.label[vi]
    C = cand_emb.shape[0]
    if (label < 0).any() or (label >= C).any():
        raise ValueError("label outside the candidate set")
    # Batch row -> position among the valid rows (FN pairs are only built for valid rows).
    local = torch.full((tt.valid.shape[0],), -1, dtype=torch.long, device=vi.device)
    local[vi] = torch.arange(n, device=vi.device)
    fr = local[tt.fn_rows]
    if (fr < 0).any():
        raise ValueError("false-negative pair on an invalid row")
    if (tt.fn_cols == tt.label[tt.fn_rows]).any():
        raise ValueError("false-negative mask would remove the label column")

    logits = (user_emb[vi] @ cand_emb.T) / tau                    # [n, C]
    mask = torch.zeros((n, C), dtype=torch.bool, device=logits.device)
    mask[fr, tt.fn_cols] = True
    return vi, label, logits.masked_fill(mask, float("-inf"))


def task_loss(user_emb: torch.Tensor, cand_emb: torch.Tensor, tt: TaskTargets, tau: float):
    """Mean full-softmax CE over the valid rows of one task.

    Returns (loss or None if no valid row, info dict).
    """
    vi, label, logits = masked_logits(user_emb, cand_emb, tt, tau)
    n = int(vi.numel())
    if n == 0:
        return None, {"n_valid": 0, "n_fn_masked": 0}
    label_logit = logits[torch.arange(n, device=logits.device), label]
    if not torch.isfinite(label_logit).all():
        raise FloatingPointError("non-finite label logit")
    # -inf entries get softmax weight exactly 0 (and zero gradient); the label is finite,
    # so logsumexp is finite and >= label_logit.
    loss = (torch.logsumexp(logits, dim=1) - label_logit).mean()
    return loss, {"n_valid": n, "n_fn_masked": int(torch.isneginf(logits).sum())}


def multitask_loss(user_embs: dict, item_embs: dict, cand_rows: dict, targets: dict, tau: float):
    """Equal-weight sum of the per-task losses (D1-3 Q6).

    user_embs / item_embs: task -> [B, 64] / [n_catalog, 64] (model head outputs).
    cand_rows: {"video": LongTensor, "music": LongTensor} catalog rows of the candidates;
               column c of a task = catalog row cand_rows[target][c].
    targets:   task -> TaskTargets.
    Returns (total or None if every task is empty, {task: info}).
    """
    total, info = None, {}
    for task in TASKS:
        target = task.split("_", 1)[0]
        cand = item_embs[task][cand_rows[target]]
        l, inf = task_loss(user_embs[task], cand, targets[task], tau)
        if l is not None:
            inf["loss"] = float(l.detach())
            total = l if total is None else total + l
        info[task] = inf
    return total, info
