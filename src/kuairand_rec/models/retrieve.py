"""Assignment-facing retrieval interface: `retrieve(user_id, target, action, k)` (PLAN.md §3d D1-7).

A thin wrapper around `export.retrieve` / `ExportedScorer`, added after the formal runs: it binds
one export directory so the call has exactly the assignment's signature. It adds no scoring logic;
scores, the T_test seen mask and the tie-break are those of `export.retrieve`, which the strict
parity check (check 3) compares against `ExportedScorer` and the M3 frames.

Python:
    from kuairand_rec.models.retrieve import load_retriever
    retrieve = load_retriever("artifacts/u0/9218a76_seed0_job27/export")
    retrieve(user_id=1, target="video", action="long_view", k=10)   # [(video_id, cosine score), ...]

CLI:
    uv run python -m kuairand_rec.models.retrieve --export_dir artifacts/u0/9218a76_seed0_job27/export \
        --user_id 1 --target video --action long_view --k 10 [--no_mask_seen]

Semantics: user embedding frozen at T_test (history = train + val events, t < T_test); items =
the full metadata catalog (every row scorable through the content path); by default every item
the user interacted with before T_test is excluded (`mask_seen=False` keeps them). Unknown users
raise KeyError. Results: (raw item id, cosine score), score descending, ties by catalog row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from kuairand_rec.models.export import ExportedScorer
from kuairand_rec.models.export import retrieve as _retrieve


class Retriever:
    """Holds one export (sha256-verified on load) and answers `retrieve(user_id, target, action, k)`."""

    def __init__(self, export_dir, expect_condition_id: str | None = None, verify_files: bool = True):
        self.export_dir = Path(export_dir)
        self.scorer = ExportedScorer(self.export_dir, expect_condition_id=expect_condition_id,
                                     verify_files=verify_files)

    @property
    def manifest(self) -> dict:
        return self.scorer.manifest

    def retrieve(self, user_id: int, target: str, action: str, k: int,
                 mask_seen: bool = True) -> list[tuple[int, float]]:
        return _retrieve(self.scorer, int(user_id), target, action, int(k), mask_seen=mask_seen)

    __call__ = retrieve


def load_retriever(export_dir, expect_condition_id: str | None = None) -> Callable[..., list[tuple[int, float]]]:
    """Return a function with the assignment signature `retrieve(user_id, target, action, k)`."""
    return Retriever(export_dir, expect_condition_id).retrieve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Top-k retrieval from an exported three-tower model (T_test users).")
    ap.add_argument("--export_dir", required=True, help="<run_dir>/export of a finished run")
    ap.add_argument("--user_id", type=int, required=True)
    ap.add_argument("--target", choices=["video", "music"], required=True)
    ap.add_argument("--action", choices=["long_view", "is_like"], required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no_mask_seen", action="store_true", help="also return items seen before T_test")
    a = ap.parse_args(argv)
    r = Retriever(a.export_dir)
    res = r.retrieve(a.user_id, a.target, a.action, a.k, mask_seen=not a.no_mask_seen)
    print(json.dumps({"user_id": a.user_id, "target": a.target, "action": a.action, "k": a.k,
                      "mask_seen": not a.no_mask_seen, "condition_id": r.manifest["condition_id"],
                      "t_pred_ms": r.manifest["t_pred"]["test"],
                      "items": [{"id": i, "score": s} for i, s in res]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
