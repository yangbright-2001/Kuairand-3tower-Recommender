"""Inference-time encoders and scorers (PLAN.md §3d D1-7).

Shared inference settings: eval mode (no dropout / ID dropout), FP32, TF32 off, deterministic
algorithms, a fixed torch thread count, no autograd.

Batch independence (correctness-critical for parity): float32 GEMM results depend on the
number of rows in the call and on a row's position in it (blocked kernels; measured ~1e-7 on
the M1). So the *canonical* user encoding runs each user in its own forward (batch size 1), and
the canonical score of a user is one matrix-vector product `I @ u` (`score_rows`). Then a user's
exported embedding and scores do not depend on which other users are in the batch, so the
exporter, `DirectScorer`, `ExportedScorer` and `retrieve()` see identical numbers by construction
instead of by luck. Batched user encoding (`batch_size > 1`) is used only for in-training
validation (checkpoint selection), never for exported / reported numbers.

Canonical scoring backend: every scorer (Direct, Exported, retrieve) computes final scores with
`score_rows` (numpy float32, one gemv per user). Different BLAS libraries do not agree bitwise on
float32 dot products: on Linux x86 torch (MKL) and numpy (OpenBLAS) `I @ u` differ by ~1e-7,
which flips near-tied items in a strict top-50. So DirectScorer must NOT score with torch; it
still runs its own towers (independent embeddings) and only shares this final dot product.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch

from kuairand_rec.eval.evaluator import EvalContext
from kuairand_rec.eval.views import prediction_time
from kuairand_rec.models import features as F
from kuairand_rec.models.towers import TASKS, ThreeTowerModel

HISTORY_CHUNK = 4096        # users per vectorised history build (row-independent numpy gathers)
TARGET_OF = {t: t.split("_", 1)[0] for t in TASKS}


def task_name(target: str, action: str) -> str:
    t = f"{target}_{action}"
    if t not in TASKS:
        raise ValueError(f"unknown task {target}/{action}")
    return t


@contextlib.contextmanager
def inference_settings(threads: int | None = None):
    """FP32 / TF32 off / deterministic / fixed threads / no grad. Restores the thread count."""
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if threads is not None:
        torch.set_num_threads(int(threads))
    try:
        with torch.inference_mode():
            yield
    finally:
        torch.set_num_threads(prev_threads)
        torch.use_deterministic_algorithms(prev_det)


def score_rows(U: np.ndarray, I: np.ndarray) -> np.ndarray:
    """Canonical float32 cosine scores [n_users, n_items]: one numpy `I @ u` per user (batch
    independent; see module docstring). Both inputs are L2-normalised head outputs. This is the
    single scoring function of DirectScorer, ExportedScorer and retrieve()."""
    U = np.asarray(U, dtype=np.float32)
    I = np.ascontiguousarray(I, dtype=np.float32)
    out = np.empty((U.shape[0], I.shape[0]), dtype=np.float32)
    for r in range(U.shape[0]):
        out[r] = I @ U[r]
    return out


def encode_items(model: ThreeTowerModel, vf: F.VideoFeatures, mf: F.MusicFeatures):
    """(ItemEncodings, {task: [n_catalog, 64]}) in eval mode. Must run inside inference_settings."""
    model.eval()
    enc = model.encode_items(vf, mf)
    return enc, model.item_embeddings(enc)


def encode_users(model: ThreeTowerModel, enc, hidx, t_pred: int, users: np.ndarray, vocabs,
                 hcfg: F.HistoryConfig, user_table: F.UserFeatureTable | None = None,
                 batch_size: int = 1, device="cpu") -> dict[str, np.ndarray]:
    """{task: float32 [len(users), 64]} at frozen t_pred. `hidx` is the frozen history index of
    the model's condition (`frozen_history_index`). batch_size = 1 is the canonical encoding."""
    model.eval()
    users = np.asarray(users, dtype=np.int64)
    out = {t: np.empty((len(users), model.cfg.head_dim), dtype=np.float32) for t in TASKS}
    uidx_all = vocabs.user.encode(users)
    use_hist = model.cfg.use_history
    for a in range(0, len(users), HISTORY_CHUNK):
        b = min(a + HISTORY_CHUNK, len(users))
        hb = F.build_history_batch(hidx, users[a:b], t_pred, hcfg).to_torch(device) if use_hist else None
        uf = (torch.as_tensor(user_table.lookup(users[a:b]), dtype=torch.long, device=device)
              if user_table is not None else None)
        ui = torch.as_tensor(uidx_all[a:b], dtype=torch.long, device=device)
        for s in range(0, b - a, batch_size):
            e = min(s + batch_size, b - a)
            h = None if hb is None else F.HistoryTensors(*(getattr(hb, f)[s:e] for f in _HIST_FIELDS))
            emb = model.user_embeddings(model.encode_users(ui[s:e], h, enc, None if uf is None else uf[s:e]))
            for t in TASKS:
                out[t][a + s:a + e] = emb[t].detach().cpu().numpy()
    return out


_HIST_FIELDS = ("video_row", "feedback", "hidden", "play_bucket", "recency_bucket", "valid", "length")


# ----------------------------------------------------------------------------- scorers
class EmbeddingScorer:
    """M3 `Scorer` over precomputed embeddings (export files or in-memory validation).

    U[split][task]: [n_users, 64] aligned with user_ids[split]; I[task]: [n_catalog, 64].
    Every catalog row is scorable (content path, P1-6b / D1-4) and every user has a
    representation (empty-history + zero-ID path for cold / OOV users), so signal is all True.
    """

    def __init__(self, name: str, user_ids: dict, U: dict, I: dict, t_pred: dict):
        self.name = name
        self.user_ids = {s: np.asarray(v, dtype=np.int64) for s, v in user_ids.items()}
        for s, ids in self.user_ids.items():
            if not (np.diff(ids) > 0).all():
                raise ValueError("user_ids must be strictly ascending")
        self.U, self.I, self.t_pred = U, I, {s: int(t) for s, t in t_pred.items()}

    def rows(self, users, split: str) -> np.ndarray:
        ids = self.user_ids[split]
        users = np.asarray(users, dtype=np.int64)
        pos = np.searchsorted(ids, users)
        ok = (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == users)
        if not ok.all():
            raise KeyError(f"unknown user_id(s) for split {split}: {users[~ok][:10].tolist()}")
        return pos

    def _check_ctx(self, ctx: EvalContext) -> None:
        if ctx.split not in self.t_pred or self.t_pred[ctx.split] != ctx.t_pred:
            raise ValueError(f"embeddings were not computed at {ctx.split} t_pred = {ctx.t_pred}")

    def score_users(self, users, target, action, split) -> np.ndarray:
        t = task_name(target, action)
        return score_rows(self.U[split][t][self.rows(users, split)], self.I[t])

    def score(self, users, target, action, ctx: EvalContext) -> np.ndarray:
        self._check_ctx(ctx)
        return self.score_users(users, target, action, ctx.split)

    def signal(self, users, target, action, ctx: EvalContext) -> np.ndarray:
        return np.ones(len(users), dtype=bool)

    def item_scorable(self, target, action) -> np.ndarray:
        return np.ones(self.I[task_name(target, action)].shape[0], dtype=bool)


class DirectScorer:
    """Scores straight from a model (D1-7): item and user towers are run here, on features
    rebuilt by the caller from raw ids. User history comes from `frozen_history_index` on the
    model condition's events (= `history_view`); embeddings use the canonical batch-1 encoding
    and are cached per (split, user), which cannot change their values. Item and user
    embeddings are produced here by the model; export files are never read. Only the final dot
    product is shared with `ExportedScorer`: the canonical `score_rows` (see module docstring),
    so parity compares embeddings, not two BLAS backends."""

    name = "threetower_direct"

    def __init__(self, model: ThreeTowerModel, cond, meta: dict, user_table=None, threads: int | None = 1,
                 device="cpu"):
        from kuairand_rec.models.train import ItemInputs

        self.model, self.cond, self.meta, self.user_table = model.eval(), cond, meta, user_table
        self.threads, self.device = threads, device
        self.items = ItemInputs.from_condition(cond, device)
        with inference_settings(threads):
            self.enc, I = encode_items(model, self.items.vf, self.items.mf)
        # Item matrices as contiguous CPU float32 numpy, catalog row order (computed once).
        self.I = {t: np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32) for t, v in I.items()}
        self._hidx: dict = {}
        self._cache: dict = {}

    def _hist(self, split: str):
        if split not in self._hidx:
            self._hidx[split] = F.frozen_history_index(self.cond.events, split, self.meta)
        return self._hidx[split]

    def user_embeddings(self, users, split: str) -> dict[str, np.ndarray]:
        users = np.asarray(users, dtype=np.int64)
        cache = self._cache.setdefault(split, {})
        missing = np.unique([u for u in users.tolist() if u not in cache]).astype(np.int64)
        if len(missing):
            hidx, t = self._hist(split)
            with inference_settings(self.threads):
                e = encode_users(self.model, self.enc, hidx, t, missing, self.cond.vocabs, self.model.hcfg,
                                 self.user_table, batch_size=1, device=self.device)
            for i, u in enumerate(missing.tolist()):
                cache[u] = {k: v[i] for k, v in e.items()}
        return {k: np.stack([cache[u][k] for u in users.tolist()]) if len(users)
                else np.zeros((0, self.model.cfg.head_dim), np.float32) for k in TASKS}

    def score(self, users, target, action, ctx: EvalContext) -> np.ndarray:
        if ctx.t_pred != prediction_time(self.meta, ctx.split):
            raise ValueError("context t_pred does not match the frozen prediction time")
        t = task_name(target, action)
        return score_rows(self.user_embeddings(users, ctx.split)[t], self.I[t])

    def signal(self, users, target, action, ctx) -> np.ndarray:
        return np.ones(len(users), dtype=bool)

    def item_scorable(self, target, action) -> np.ndarray:
        return np.ones(self.I[task_name(target, action)].shape[0], dtype=bool)


@dataclass
class InferenceSettings:
    threads: int = 1
    user_batch_size: int = 1        # canonical (export / direct); validation may use more

    def to_dict(self) -> dict:
        return {"threads": self.threads, "user_batch_size": self.user_batch_size, "device": "cpu", "fp32": True,
                "tf32": False, "deterministic": True, "torch": torch.__version__, "numpy": np.__version__,
                "score": "score_rows: numpy float32 per-user I @ u (shared by Direct / Exported / retrieve)"}
