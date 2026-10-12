"""Embedding export, ExportedScorer and retrieve() (PLAN.md §3d D1-7).

Export layout (<run_dir>/export/):
  items_{task}.npy           float32 [n_catalog(target), 64], every catalog row (incl. OOV items)
  users_{split}_{task}.npy   float32 [n_users(split), 64], split in {val, test}
  user_ids_{split}.npy       int64 ascending raw user ids (row order of users_{split}_*)
  video_ids.npy / music_ids.npy   catalog row -> raw id (the M2 catalog order)
  seen_test_{target}_{indptr,rows}.npy   T_test seen-item CSR over user_ids_test (uncapped,
                             history_view = train + val events, the same `seen_rows` as M3 frames)
  manifest.json              git sha, config hash, checkpoint sha256, t_pred, frame_ids, data
                             and condition fingerprints, per-file sha256, inference settings

Users exported = every user with >= 1 event in the cleaned log (so every M3 frame user is
covered, incl. cold and train-OOV users). Embeddings use the canonical batch-1 encoding.

Reported metrics come from `ExportedScorer` (numpy, export files only). `retrieve()` shares its
scoring function (`score_rows`), the M3 `apply_mask` and the M3 `rank_topk`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from kuairand_rec.data.history import HistoryIndex
from kuairand_rec.eval import metrics as M
from kuairand_rec.eval.views import history_view, prediction_time
from kuairand_rec.models import artifacts as A
from kuairand_rec.models import features as F
from kuairand_rec.models.inference import (
    TARGET_OF, EmbeddingScorer, InferenceSettings, encode_items, encode_users, inference_settings, score_rows,
    task_name,
)
from kuairand_rec.models.towers import TASKS
from kuairand_rec.utils import git_sha

SPLITS = ("val", "test")
SCHEMA = "threetower-export-v1"


def export_users(P: dict) -> np.ndarray:
    return np.unique(P["events"].user_id.to_numpy(np.int64))


def export_embeddings(model, cond, P: dict, out_dir, checkpoint_path, config: dict | None = None,
                      user_vocab=None, user_table=None, frame_ids: dict | None = None,
                      users: np.ndarray | None = None,
                      settings: InferenceSettings = InferenceSettings(), device="cpu") -> dict:
    """Write the export directory and return the manifest."""
    from kuairand_rec.models.train import ItemInputs

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    users = export_users(P) if users is None else np.unique(np.asarray(users, dtype=np.int64))
    items = ItemInputs.from_condition(cond, device)
    files = {}

    def _save(name, arr):
        p = out / name
        np.save(p, arr)
        files[name] = A.sha256_file(p)

    with inference_settings(settings.threads):
        enc, I = encode_items(model, items.vf, items.mf)
        I = {t: v.detach().cpu().numpy().astype(np.float32) for t, v in I.items()}
        for t in TASKS:
            _save(f"items_{t}.npy", I[t])
        t_pred = {}
        for s in SPLITS:
            hidx, t = F.frozen_history_index(cond.events, s, P["meta"])
            t_pred[s] = int(t)
            U = encode_users(model, enc, hidx, t, users, cond.vocabs, model.hcfg, user_table,
                             batch_size=settings.user_batch_size, device=device)
            _save(f"user_ids_{s}.npy", users)
            for task in TASKS:
                _save(f"users_{s}_{task}.npy", U[task])
    _save("video_ids.npy", P["catalogs"].video_ids.astype(np.int64))
    _save("music_ids.npy", P["catalogs"].music_ids.astype(np.int64))
    # Evaluation-side seen mask at T_test: unfiltered log (P1-9: the mask never depends on the
    # training condition), same `seen_rows` as the M3 frames.
    hv = HistoryIndex(history_view(P["events"], "test", P["meta"]))
    for tgt in ("video", "music"):
        ip, rows = hv.seen_rows(users, prediction_time(P["meta"], "test"), tgt)
        _save(f"seen_test_{tgt}_indptr.npy", ip.astype(np.int64))
        _save(f"seen_test_{tgt}_rows.npy", rows.astype(np.int64))
    manifest = {
        "schema": SCHEMA, "git_sha": git_sha(), "config_hash": A.hash_json(config or {}),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": A.sha256_file(checkpoint_path),
        "condition_id": cond.condition_id, "condition_fingerprint": A.condition_fingerprint(cond, user_vocab),
        "data_fingerprint": A.data_fingerprint(P["events"], P["meta"]),
        "t_pred": t_pred, "n_users": int(len(users)),
        "n_items": {"video": int(len(P["catalogs"].video_ids)), "music": int(len(P["catalogs"].music_ids))},
        "head_dim": int(model.cfg.head_dim), "tasks": list(TASKS), "frame_ids": frame_ids or {},
        "inference": settings.to_dict(), "files": files, "export_s": round(time.time() - t0, 2),
    }
    A.write_json(out / "manifest.json", manifest)
    return manifest


class ExportedScorer(EmbeddingScorer):
    """M3 Scorer reading only the export directory (numpy float32 `I @ u`)."""

    def __init__(self, export_dir, expect_condition_id: str | None = None, verify_files: bool = True,
                 name: str = "threetower"):
        d = Path(export_dir)
        man = json.loads((d / "manifest.json").read_text())
        if man.get("schema") != SCHEMA:
            raise ValueError("unknown export schema")
        if expect_condition_id is not None and man["condition_id"] != expect_condition_id:
            raise ValueError(f"export condition_id {man['condition_id']} != expected {expect_condition_id}")
        if verify_files:
            for f, h in man["files"].items():
                if A.sha256_file(d / f) != h:
                    raise ValueError(f"export file {f} does not match its manifest sha256")
        load = lambda f: np.load(d / f, allow_pickle=False)
        user_ids = {s: load(f"user_ids_{s}.npy") for s in SPLITS}
        U = {s: {t: load(f"users_{s}_{t}.npy") for t in TASKS} for s in SPLITS}
        I = {t: load(f"items_{t}.npy") for t in TASKS}
        for t in TASKS:
            if I[t].shape != (man["n_items"][TARGET_OF[t]], man["head_dim"]) or I[t].dtype != np.float32:
                raise ValueError(f"bad item matrix for {t}")
        super().__init__(name, user_ids, U, I, man["t_pred"])
        self.manifest, self.dir = man, d
        self.catalog_ids = {"video": load("video_ids.npy"), "music": load("music_ids.npy")}
        self.seen = {tgt: (load(f"seen_test_{tgt}_indptr.npy"), load(f"seen_test_{tgt}_rows.npy"))
                     for tgt in ("video", "music")}
        if not np.array_equal(self.user_ids["val"], self.user_ids["test"]):
            raise ValueError("val / test user id lists differ")

    def seen_rows(self, user_id: int, target: str) -> np.ndarray:
        r = int(self.rows([user_id], "test")[0])
        ip, rows = self.seen[target]
        return rows[ip[r]:ip[r + 1]]


def retrieve(scorer: ExportedScorer, user_id: int, target: str, action: str, k: int,
             mask_seen: bool = True) -> list[tuple[int, float]]:
    """Top-k (raw item id, cosine score) for one user at T_test, from the exported embeddings.

    Same scoring (`score_rows`), M2 seen mask (exported T_test CSR = M3 frame mask) and
    `rank_topk` (score desc, catalog row asc) as `ExportedScorer` + the shared evaluator.
    Masked items are never returned. Unknown users raise KeyError.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    task_name(target, action)                                            # validates target / action
    s = scorer.score_users([user_id], target, action, "test")          # [1, n_items], float32
    if mask_seen:
        seen = scorer.seen_rows(user_id, target)
        M.apply_mask(s, np.array([0, len(seen)], dtype=np.int64), seen)
    top, valid = M.rank_topk(s, k)
    rows = top[0][valid[0]]
    ids = scorer.catalog_ids[target][rows]
    return [(int(i), float(v)) for i, v in zip(ids, s[0, rows])]
