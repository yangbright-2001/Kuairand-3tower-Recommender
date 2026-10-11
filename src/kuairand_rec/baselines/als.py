"""Implicit-feedback ALS baseline (PLAN.md §3b P1-1, P1-6b; §3c Q1-Q3, Q5).

Signal (one definition, shared by fit and fold-in -- `build_confidence`):
    r_ui = 1[any action-positive event on (u,i)] + aux_w * 1[any long_view event on (u,i)]
    p_ui = 1[r_ui > 0],   C_ui = 1 + alpha * r_ui   (C = 1, p = 0 for unobserved pairs)
For music, (u, i) = (user, music row) and "any" ranges over all videos carrying the music;
music_id 0 has no music row and contributes nothing.

implicit 0.7.3 semantics (verified numerically in tests/test_baselines.py): a stored
entry d is used *as the confidence itself* (p = 1, C = d); unstored entries have p = 0,
C = 1. With the library's alpha = 1 we therefore store d = C_ui = 1 + alpha * r_ui, and the
closed-form fold-in below solves the same objective:
    x_u = (YᵀY + Yᵀ (C_u - I) Y + λI)⁻¹ Yᵀ C_u p_u

Train-only fit: only users / items with >= 1 non-zero train entry enter `implicit.fit`,
via an explicit compact index (catalog rows and user ids are not contiguous). Items with
no train signal under the current config are *unscorable* (-inf): their exact solution is
the zero vector, which P1-6b forbids, and with CG + random init they would not even be
exactly zero and would pollute YᵀY for every user.

User vectors:
  foldin    -- frozen Y, user vector solved from history_view (val: train, test: train+val)
  nofoldin  -- the raw train user factors returned by `fit` (no recomputation). Train-OOV
               users have none -> no signal. Note: implicit updates users then items per
               iteration, so these factors are half an iteration older than Y (documented).
Users without signal score -inf everywhere (strict, Q2): zero vectors would rank by catalog
row (the tie-break) and produce spurious hits.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from kuairand_rec.data.catalog import Catalogs, assert_train_only
from kuairand_rec.eval.evaluator import EvalContext

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit warns otherwise; also deterministic


@dataclass(frozen=True)
class ALSConfig:
    factors: int = 64
    regularization: float = 0.1
    alpha: float = 10.0
    aux_w: float = 0.0
    iterations: int = 20
    seed: int = 0
    num_threads: int = 1

    def tag(self) -> str:
        return f"f{self.factors}_r{self.regularization}_a{self.alpha}_w{self.aux_w}"


def _item_col(target: str) -> str:
    return {"video": "video_row", "music": "music_row"}[target]


def build_confidence(events: pd.DataFrame, target: str, action: str, aux_w: float, alpha: float) -> pd.DataFrame:
    """One row per (user_id, item_row) with r > 0: columns user_id, item, r, c (= 1 + alpha r).
    Sorted by (user_id, item). Used identically for the train matrix and for fold-in."""
    if action == "long_view" and aux_w != 0:
        raise ValueError("aux_w only applies to non-long_view actions")
    col = _item_col(target)
    labels = list(dict.fromkeys([action, "long_view"]))   # no duplicate column when action == long_view
    d = events.loc[events[col] >= 0, ["user_id", col, *labels]]
    g = d.groupby(["user_id", col], sort=True)[labels].max().reset_index()
    r = g[action].to_numpy(np.float64)
    if aux_w:
        r = r + aux_w * g["long_view"].to_numpy(np.float64)
    out = pd.DataFrame({"user_id": g.user_id.to_numpy(np.int64), "item": g[col].to_numpy(np.int64), "r": r})
    out = out[out.r > 0].reset_index(drop=True)
    out["c"] = (1.0 + alpha * out.r).astype(np.float32)
    return out


def train_fingerprint(train: pd.DataFrame, target: str, action: str) -> str:
    """sha256 over every train field that affects the fit, deterministically sorted."""
    cols = list(dict.fromkeys(["user_id", "video_id", "time_ms", _item_col(target), action, "long_view", "split"]))
    d = train[cols].sort_values(["user_id", "time_ms", "video_id"], kind="mergesort")
    h = hashlib.sha256()
    h.update(",".join(cols).encode())
    for c in cols:
        h.update(np.ascontiguousarray(d[c].to_numpy(np.int64)).astype("<i8").tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ALSModel:
    cfg: ALSConfig
    target: str
    action: str
    n_items: int
    item_rows: np.ndarray        # [n_s] scorable catalog rows (ascending) = compact col -> row
    Y: np.ndarray                # [n_s, f] item factors of the scorable rows
    train_users: np.ndarray      # [n_u] user ids with a non-zero train row (ascending)
    X_train: np.ndarray          # [n_u, f] raw `fit` user factors (no-fold-in reference)
    fingerprint: str
    confidence_hash: str

    @property
    def scorable(self) -> np.ndarray:
        m = np.zeros(self.n_items, bool)
        m[self.item_rows] = True
        return m


def fit_als(train: pd.DataFrame, n_items: int, target: str, action: str, cfg: ALSConfig) -> ALSModel:
    from implicit.cpu.als import AlternatingLeastSquares

    assert_train_only(train)
    conf = build_confidence(train, target, action, cfg.aux_w, cfg.alpha)
    users = np.unique(conf.user_id.to_numpy())
    items = np.unique(conf.item.to_numpy())
    ui = np.searchsorted(users, conf.user_id.to_numpy())
    ii = np.searchsorted(items, conf.item.to_numpy())
    C = sp.csr_matrix((conf.c.to_numpy(np.float32), (ui, ii)), shape=(len(users), len(items)))
    model = AlternatingLeastSquares(factors=cfg.factors, regularization=cfg.regularization, alpha=1.0,
                                    iterations=cfg.iterations, random_state=cfg.seed,
                                    num_threads=cfg.num_threads)
    model.fit(C, show_progress=False)
    Y = np.asarray(model.item_factors, dtype=np.float32).copy()
    X = np.asarray(model.user_factors, dtype=np.float32).copy()
    for a in (Y, X, users, items):
        a.setflags(write=False)
    ch = hashlib.sha256(np.ascontiguousarray(conf[["user_id", "item"]].to_numpy(np.int64)).tobytes()
                        + conf.c.to_numpy(np.float32).tobytes()).hexdigest()
    return ALSModel(cfg, target, action, n_items, items, Y, users, X,
                    train_fingerprint(train, target, action), ch)


def fold_in(model: ALSModel, history: pd.DataFrame, users: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Closed-form user vectors from `history` with frozen Y. Returns (X [n, f], has_signal, stats).
    Interactions on unscorable items have no factor and are dropped (counted)."""
    cfg = model.cfg
    conf = build_confidence(history, model.target, model.action, cfg.aux_w, cfg.alpha)
    conf = conf[conf.user_id.isin(users)]
    col = np.searchsorted(model.item_rows, conf.item.to_numpy())
    col_c = np.minimum(col, len(model.item_rows) - 1)
    ok = (col < len(model.item_rows)) & (model.item_rows[col_c] == conf.item.to_numpy())
    stats = {"history_pairs": int(len(conf)), "history_pairs_unscorable_dropped": int((~ok).sum())}
    conf = conf[ok]
    col = col[ok]
    Y = model.Y.astype(np.float64)
    YtY = Y.T @ Y
    lam = cfg.regularization * np.eye(Y.shape[1])
    if len(users) > 1 and not (np.diff(users) > 0).all():
        raise ValueError("users must be sorted ascending and unique")
    X = np.zeros((len(users), Y.shape[1]), dtype=np.float32)
    has = np.zeros(len(users), bool)
    upos = np.searchsorted(users, conf.user_id.to_numpy())   # conf is sorted by user
    cvals = conf.c.to_numpy(np.float64)
    starts = np.flatnonzero(np.r_[True, np.diff(upos) != 0]) if len(upos) else np.zeros(0, int)
    ends = np.r_[starts[1:], len(upos)]
    for s0, e0 in zip(starts, ends):
        Yi, c = Y[col[s0:e0]], cvals[s0:e0]
        A = YtY + (Yi.T * (c - 1.0)) @ Yi + lam
        X[upos[s0]] = np.linalg.solve(A, Yi.T @ c).astype(np.float32)   # p_ui = 1 on stored pairs
        has[upos[s0]] = True
    return X, has, stats


class ALSScorer:
    """mode = 'foldin' (main) or 'nofoldin' (reference: raw train user factors)."""

    def __init__(self, models: dict, mode: str, name: str):
        if mode not in ("foldin", "nofoldin"):
            raise ValueError(mode)
        self.models, self.mode, self.name = models, mode, name
        self._cache: dict = {}
        self.foldin_stats: dict = {}

    def _vectors(self, users, target, action, ctx: EvalContext):
        m = self.models[(target, action)]
        if self.mode == "nofoldin":
            pos = np.searchsorted(m.train_users, users)
            pos_c = np.minimum(pos, max(len(m.train_users) - 1, 0))
            has = (pos < len(m.train_users)) & (m.train_users[pos_c] == users)
            X = np.where(has[:, None], m.X_train[pos_c], 0).astype(np.float32)
            return X, has
        key = (ctx.split, ctx.t_pred, target, action)
        hit = self._cache.get(key)
        if hit is None or hit[0] is not ctx.history:
            # Fold in once for every user with history at t_pred; chunks just look up.
            all_u = np.unique(ctx.history.user_id.to_numpy())
            X, has, st = fold_in(m, ctx.history, all_u)
            self._cache[key] = (ctx.history, all_u, X, has)
            self.foldin_stats[(ctx.split, target, action)] = st
        _, all_u, X, has = self._cache[key]
        pos = np.searchsorted(all_u, users)
        pos_c = np.minimum(pos, max(len(all_u) - 1, 0))
        found = (pos < len(all_u)) & (all_u[pos_c] == users) if len(all_u) else np.zeros(len(users), bool)
        return X[pos_c] * found[:, None], has[pos_c] & found

    def score(self, users, target, action, ctx: EvalContext) -> np.ndarray:
        m = self.models[(target, action)]
        X, has = self._vectors(np.asarray(users, np.int64), target, action, ctx)
        s = np.full((len(users), m.n_items), -np.inf, dtype=np.float32)
        s[:, m.item_rows] = X @ m.Y.T
        s[~has] = -np.inf   # strict no-signal: never rank by the tie-break
        return s

    def signal(self, users, target, action, ctx: EvalContext) -> np.ndarray:
        return self._vectors(np.asarray(users, np.int64), target, action, ctx)[1]

    def item_scorable(self, target, action) -> np.ndarray:
        return self.models[(target, action)].scorable
