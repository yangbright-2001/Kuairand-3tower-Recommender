"""Provenance of training conditions, checkpoints and exports (PLAN.md §3d D1-5, D1-7, D1-11).

A checkpoint stores a *condition fingerprint*: hashes of every artifact that defines the model's
inputs (vocabs, metadata scalers, encoded video / music feature tables, model-side events, U1
user vocab). `load_model` rebuilds that fingerprint from the caller's condition and refuses any
mismatch, so a cold-video checkpoint can never be scored with normal-condition vocabs / features
(or vice versa), and a checkpoint can never silently meet a different data build.

Float columns are rounded to 1e-6 before hashing, so a rebuild on another machine (same data,
same code) still matches while any real change does not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kuairand_rec.models import features as F

EVENT_COLS = ("user_id", "video_id", "time_ms", "split", "video_row", "music_row", "user_idx", "video_idx",
              "author_idx", "music_idx", "long_view", "is_like", *F.FEEDBACK_BINARY_COLS, "play_time_ms",
              "duration_ms")


def _series_bytes(s: pd.Series) -> bytes:
    if pd.api.types.is_float_dtype(s):
        s = s.astype(np.float64).round(6)
    return pd.util.hash_pandas_object(s.reset_index(drop=True), index=False).to_numpy().tobytes()


def hash_frame(df: pd.DataFrame, cols=None) -> str:
    cols = list(df.columns if cols is None else cols)
    h = hashlib.sha256(json.dumps([len(df), cols]).encode())
    for c in cols:
        h.update(_series_bytes(df[c]))
    return h.hexdigest()


def hash_array(a) -> str:
    a = np.asarray(a)
    if a.dtype.kind == "f":
        a = np.round(a.astype(np.float64), 6)
    if a.dtype == object:
        return hashlib.sha256(json.dumps([str(x) for x in a]).encode()).hexdigest()
    return hashlib.sha256(str(a.shape).encode() + np.ascontiguousarray(a).tobytes()).hexdigest()


def hash_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def vocab_hashes(vocabs) -> dict:
    out = {n: hash_array(getattr(vocabs, n).ids) for n in ("user", "video", "author", "music")}
    out.update({f"meta_{c}": hash_array(v.ids) for c, v in vocabs.meta.items()})
    return out


def user_vocab_hash(user_vocab: F.UserFeatureVocab | None) -> str | None:
    if user_vocab is None:
        return None
    return hash_json({"fields": list(user_vocab.fields), "min_count": user_vocab.min_count,
                      "values": {f: [str(x) for x in v.ids] for f, v in user_vocab.vocabs.items()}})


def condition_fingerprint(cond, user_vocab: F.UserFeatureVocab | None = None) -> dict:
    """Everything that defines the model inputs of one condition (D1-5)."""
    cols = [c for c in EVENT_COLS if c in cond.events.columns]
    return {
        "condition_id": cond.condition_id,
        "condition_name": cond.name,
        "n_exclude_video_ids": int(len(cond.exclude_video_ids)),
        "vocabs": vocab_hashes(cond.vocabs),
        "num_stats": hash_json(cond.num_stats) if cond.num_stats is not None else None,
        "video_features": hash_frame(cond.video_catalog),
        "music_features": hash_frame(cond.music_catalog),
        "model_events": hash_frame(cond.events, cols),
        "candidates": hash_array(np.concatenate([cond.cand_video_rows, [-1], cond.cand_music_rows])),
        "user_vocab": user_vocab_hash(user_vocab),
    }


def data_fingerprint(events: pd.DataFrame, meta: dict) -> dict:
    """Evaluation-side data identity (unfiltered cleaned log + cut points)."""
    cols = [c for c in ("user_id", "video_id", "time_ms", "split", "video_row", "music_row", "long_view", "is_like")
            if c in events.columns]
    return {"events": hash_frame(events, cols), "n_events": int(len(events)),
            "t_val_ms": int(meta["t_val_ms"]), "t_test_ms": int(meta["t_test_ms"]),
            "data_git_sha": meta.get("git_sha")}


def check_fingerprint(stored: dict, current: dict, what: str) -> None:
    bad = [k for k in sorted(set(stored) | set(current)) if stored.get(k) != current.get(k)]
    if bad:
        raise ValueError(f"{what}: artifact mismatch on {bad} (condition / vocab / feature mixing refused)")


def load_model(path, cond, user_vocab: F.UserFeatureVocab | None = None, device="cpu"):
    """Rebuild the model from a checkpoint after verifying it belongs to `cond` (D1-5)."""
    from kuairand_rec.models.towers import ThreeTowerModel, TowerConfig

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("condition_id") != cond.condition_id:
        raise ValueError(f"checkpoint condition_id {ck.get('condition_id')} != {cond.condition_id}")
    check_fingerprint(ck["fingerprint"], condition_fingerprint(cond, user_vocab), f"checkpoint {path}")
    sizes = F.FeatureSizes.from_vocabs(cond.vocabs, user_vocab)
    if sizes.to_dict() != ck["sizes"]:
        raise ValueError("checkpoint embedding sizes differ from the condition's vocabs")
    model = ThreeTowerModel(sizes, TowerConfig.from_dict(ck["tower_config"]),
                            F.HistoryConfig.from_dict(ck["history_config"]))
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval(), ck


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, default=str))
