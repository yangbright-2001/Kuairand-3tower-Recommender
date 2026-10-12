"""Paired user-level bootstrap (PLAN.md §3b P1-1: 1000 user resamples, same indices for every model).

Statistic = the equal-weight mean over evaluated users of a per-user metric, exactly as
`eval.evaluator.summarize` computes the reported numbers. One replicate draws n users with
replacement (n = population size); the replicate statistic of every model is computed on the
*same* draw, so model differences are paired.

Determinism: the random stream of a population depends only on (seed, population key), never
on which models / metrics are analysed or in which order frames are processed. Replicates are
drawn one at a time (`integers(0, n, n)`) and each replicate mean is its own matrix-vector
product, so neither chunking nor the model / metric set changes any replicate value.
CI = percentile interval (numpy `quantile`, method "linear").
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

CI_METHOD = "percentile (numpy.quantile, method='linear')"
_CHUNK = 128


def stream(seed: int, key: str) -> np.random.Generator:
    """Generator for one population: SeedSequence([seed, 64-bit digest of key])."""
    k = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([int(seed), k]))


def replicate_counts(n: int, n_boot: int, rng: np.random.Generator):
    """Yield (start, counts [c, n] float64): how often each user is drawn in each replicate."""
    if n < 1:
        raise ValueError("bootstrap needs at least one user")
    for a in range(0, n_boot, _CHUNK):
        c = min(_CHUNK, n_boot - a)
        C = np.empty((c, n), dtype=np.float64)
        for j in range(c):
            C[j] = np.bincount(rng.integers(0, n, size=n), minlength=n)
        yield a, C


@dataclass(frozen=True)
class BootstrapResult:
    key: str                 # population key (population_id [+ stratum])
    seed: int
    n_users: int
    n_boot: int
    columns: list            # [(model, metric)] column order of point / boot
    point: np.ndarray        # [M] full-sample means
    boot: np.ndarray         # [n_boot, M] replicate means

    def col(self, model: str, metric: str) -> int:
        return self.columns.index((model, metric))


def paired_bootstrap(per_user: dict, metrics, n_boot: int, seed: int, key: str) -> BootstrapResult:
    """per_user: {model: DataFrame} whose rows are the *same users in the same order* (checked)."""
    models = list(per_user)
    if not models:
        raise ValueError("no models")
    users = per_user[models[0]].user_id.to_numpy()
    for m in models[1:]:
        if not np.array_equal(per_user[m].user_id.to_numpy(), users):
            raise ValueError(f"model {m} is not aligned with {models[0]} (users / order differ)")
    cols = [(m, k) for m in models for k in metrics]
    X = np.stack([per_user[m][k].to_numpy(np.float64) for m, k in cols], axis=1)   # [n, M]
    if not np.isfinite(X).all():
        raise ValueError("non-finite per-user metric")
    n = X.shape[0]
    boot = np.empty((n_boot, X.shape[1]), dtype=np.float64)
    Xt = np.ascontiguousarray(X.T)
    for a, C in replicate_counts(n, n_boot, stream(seed, key)):
        # One 1-D dot product per (replicate, column): a blocked GEMM / GEMV would round
        # differently depending on the chunk size or on which other columns are present (the
        # same effect as M4's batch dependence), breaking bitwise reproducibility.
        for j in range(len(C)):
            c = C[j]
            boot[a + j] = [np.dot(x, c) / n for x in Xt]
    return BootstrapResult(key, int(seed), int(n), int(n_boot), cols, X.mean(axis=0), boot)


def _interval(x: np.ndarray, ci: float) -> tuple[float, float]:
    lo, hi = np.quantile(x, [(1 - ci) / 2, 1 - (1 - ci) / 2], method="linear")
    return float(lo), float(hi)


def model_rows(res: BootstrapResult, ci: float = 0.95) -> list[dict]:
    out = []
    for j, (m, k) in enumerate(res.columns):
        lo, hi = _interval(res.boot[:, j], ci)
        out.append({"model": m, "metric": k, "n_users": res.n_users, "mean": float(res.point[j]),
                    "boot_se": float(res.boot[:, j].std(ddof=1)), "ci_lo": lo, "ci_hi": hi})
    return out


def diff_rows(res: BootstrapResult, pairs, metrics, ci: float = 0.95) -> list[dict]:
    """a - b for each (a, b) pair, on the paired replicates.

    p_two_sided = min(1, 2 * min(P*(d <= 0), P*(d >= 0))) over replicates: a descriptive
    bootstrap tail share (not a multiplicity-corrected test)."""
    out = []
    for a, b in pairs:
        for k in metrics:
            d = res.boot[:, res.col(a, k)] - res.boot[:, res.col(b, k)]
            lo, hi = _interval(d, ci)
            p = min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))
            out.append({"model_a": a, "model_b": b, "metric": k, "n_users": res.n_users,
                        "mean_a": float(res.point[res.col(a, k)]), "mean_b": float(res.point[res.col(b, k)]),
                        "diff": float(res.point[res.col(a, k)] - res.point[res.col(b, k)]),
                        "boot_se": float(d.std(ddof=1)), "ci_lo": lo, "ci_hi": hi,
                        "ci_excludes_0": bool(lo > 0 or hi < 0), "p_two_sided": p})
    return out


def naive_bootstrap_means(x: np.ndarray, n_boot: int, seed: int, key: str) -> np.ndarray:
    """Reference implementation (explicit index resampling) for tests."""
    rng = stream(seed, key)
    n = len(x)
    return np.array([x[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])


def to_frame(rows: list[dict], **const) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for k, v in const.items():
        df.insert(0, k, v)
    return df
