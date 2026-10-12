"""Model-side feature tensors and the strict-past history batch (PLAN.md §3d D1-1, D1-2, D1-6).

This module only *consumes* M2 artifacts; it never re-encodes raw ids:
  * video / music features come from `encode_video_features` / `encode_music_features`
    (the `video_features` / `music_features` of `fit_condition`, or the saved
    `video_catalog` / `music_catalog` tables). Row = catalog row.
  * history uses `HistoryIndex.model_history` (strict t < t_pred, same-timestamp events
    excluded, cap = max_len, right-aligned). The event set is exactly P1-2; this module only
    decides which *fields* of those events are visible.
  * U1 user-feature vocabs reuse `catalog.Vocab` (PAD = 0, OOV = 1, values from 2).

Feedback availability (D1-2, Q8): a token with age = t_pred - t_event < D has all feedback
fields zeroed (8 binaries and play_ratio) and `hidden = 1`. The rule reads only t_pred and
t_event -- never `play_time_ms` or labels -- and the same function serves train, val and test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd
import torch

from kuairand_rec.data import catalog as catm
from kuairand_rec.data.history import HistoryIndex
from kuairand_rec.data.train_stats import NUMERIC_VIDEO_COLS
from kuairand_rec.eval.views import history_view, prediction_time, train_view

# ----------------------------------------------------------------------------- history
# Binary feedback fields of a past event (configs/data.yaml `feedback_cols` minus play_time_ms).
FEEDBACK_BINARY_COLS = ("is_click", "long_view", "is_like", "is_follow", "is_comment",
                        "is_forward", "is_hate", "is_profile_enter")

# play_ratio = play_time_ms / duration_ms, bucketised. Row 0 is a fixed-zero "no feedback"
# row shared by PAD and hidden tokens (D1-2: hidden play_ratio is *zeroed*, not mapped to the
# real "played 0 ms" bucket). Row 1 = duration missing. Rows 2.. = ratio buckets.
PLAY_NONE, PLAY_DURATION_MISSING, PLAY_ZERO = 0, 1, 2
PLAY_RATIO_EDGES = (0.1, 0.25, 0.5, 0.75, 1.0, 2.0)   # (0,.1) [.1,.25) ... [1,2) [2,inf)
N_PLAY_BUCKETS = 3 + len(PLAY_RATIO_EDGES) + 1        # = 10
RECENCY_PAD = 0                                       # recency row 0 = PAD (fixed zero)
MINUTE_MS = 60_000


@dataclass(frozen=True)
class HistoryConfig:
    max_len: int = 100
    feedback_delay_ms: int = 20 * MINUTE_MS   # D (D1-2): a conservative assumption, not tuned
    # Recency bucket b = floor(log2(1 + age_minutes)) clipped to `recency_max_bucket`.
    # The clip is needed for P1-6b: rolling train histories have age <= ~7.5 days (train
    # span), so b <= 13, but frozen T_test histories reach ~16.5 days (b = 14). Without the
    # clip, bucket 14 would be a never-trained embedding row influencing test scores.
    # 13 is the largest bucket reachable in train ([5.7 d, inf) after clipping).
    recency_max_bucket: int = 13

    @property
    def n_recency_rows(self) -> int:
        return self.recency_max_bucket + 2    # PAD + buckets 0..max (shifted by one)

    @classmethod
    def from_dict(cls, d: dict | None) -> "HistoryConfig":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown history config keys: {sorted(unknown)}")
        return cls(**d)


def recency_bucket(age_ms: np.ndarray, max_bucket: int) -> np.ndarray:
    """1 + min(floor(log2(1 + age_min)), max_bucket) for age_ms > 0 (0 is reserved for PAD)."""
    age_min = np.asarray(age_ms, dtype=np.float64) / MINUTE_MS
    b = np.floor(np.log2(1.0 + np.maximum(age_min, 0.0))).astype(np.int64)
    return 1 + np.minimum(b, max_bucket)


def play_bucket(play_ms: np.ndarray, duration_ms: np.ndarray) -> np.ndarray:
    """Bucket of play_time / duration (event-level fields); duration NaN -> its own bucket."""
    play = np.asarray(play_ms, dtype=np.float64)
    dur = np.asarray(duration_ms, dtype=np.float64)
    missing = ~(dur > 0)                   # NaN (M2 marks duration_ms = 0 as missing) or <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(missing, 0.0, play / np.where(missing, 1.0, dur))
    b = np.where(r <= 0, PLAY_ZERO, PLAY_ZERO + 1 + np.searchsorted(PLAY_RATIO_EDGES, r, side="right"))
    return np.where(missing, PLAY_DURATION_MISSING, b).astype(np.int64)


@dataclass(frozen=True)
class HistoryBatch:
    """Right-aligned [B, L] history (column L-1 = most recent). PAD positions: valid = False,
    video_row = -1, all features / buckets 0."""

    users: np.ndarray            # [B] raw user_id
    t_pred: np.ndarray           # [B] prediction time (ms)
    video_row: np.ndarray        # [B, L] video catalog row, -1 = PAD
    feedback: np.ndarray         # [B, L, 8] float32, zeroed for PAD and hidden tokens
    hidden: np.ndarray           # [B, L] float32, 1 = feedback hidden (age < D)
    play_bucket: np.ndarray      # [B, L] int64, PLAY_NONE for PAD and hidden tokens
    recency_bucket: np.ndarray   # [B, L] int64, RECENCY_PAD for PAD
    valid: np.ndarray            # [B, L] bool
    length: np.ndarray           # [B] number of real tokens (<= max_len)

    def to_torch(self, device=None) -> "HistoryTensors":
        return HistoryTensors(
            video_row=torch.as_tensor(self.video_row, dtype=torch.long, device=device),
            feedback=torch.as_tensor(self.feedback, dtype=torch.float32, device=device),
            hidden=torch.as_tensor(self.hidden, dtype=torch.float32, device=device),
            play_bucket=torch.as_tensor(self.play_bucket, dtype=torch.long, device=device),
            recency_bucket=torch.as_tensor(self.recency_bucket, dtype=torch.long, device=device),
            valid=torch.as_tensor(self.valid, dtype=torch.bool, device=device),
            length=torch.as_tensor(self.length, dtype=torch.long, device=device),
        )


@dataclass(frozen=True)
class HistoryTensors:
    video_row: torch.Tensor
    feedback: torch.Tensor
    hidden: torch.Tensor
    play_bucket: torch.Tensor
    recency_bucket: torch.Tensor
    valid: torch.Tensor
    length: torch.Tensor


def build_history_batch(hidx: HistoryIndex, users, t_pred, cfg: HistoryConfig) -> HistoryBatch:
    """The single history builder for train (t_pred = target time, hidx on train events),
    val (T_val) and test / export (T_test, hidx on `history_view`)."""
    users = np.asarray(users, dtype=np.int64)
    t = np.array(np.broadcast_to(np.asarray(t_pred, dtype=np.int64), users.shape))
    pos, length = hidx.model_history(users, t, cfg.max_len)
    valid = pos >= 0
    t_ev = hidx.gather(pos, "time_ms", pad=0).astype(np.int64)
    age = t[:, None] - t_ev
    # Defence in depth on top of model_history's searchsorted(side="left").
    if not (age[valid] > 0).all():
        raise AssertionError("history token at or after t_pred")

    # Q8: visibility depends on (t_pred, t_event) only; hidden fields are overwritten by
    # constants below, so their raw values cannot reach the output (test L3).
    hidden = valid & (age < cfg.feedback_delay_ms)
    visible = valid & ~hidden

    fb = np.stack([hidx.gather(pos, c, pad=0) for c in FEEDBACK_BINARY_COLS], axis=-1)
    if not np.isin(fb[valid], (0, 1)).all():
        raise ValueError("feedback fields must be binary")
    fb = np.where(visible[..., None], fb, 0).astype(np.float32)
    pb = play_bucket(hidx.gather(pos, "play_time_ms", pad=0), hidx.gather(pos, "duration_ms", pad=np.nan))
    pb = np.where(visible, pb, PLAY_NONE).astype(np.int64)
    rb = np.where(valid, recency_bucket(np.where(valid, age, 1), cfg.recency_max_bucket), RECENCY_PAD)
    return HistoryBatch(
        users=users, t_pred=t,
        video_row=hidx.gather(pos, "video_row", pad=-1).astype(np.int64),
        feedback=fb, hidden=hidden.astype(np.float32), play_bucket=pb,
        recency_bucket=rb.astype(np.int64), valid=valid, length=length.astype(np.int64),
    )


def train_history_index(events: pd.DataFrame) -> HistoryIndex:
    """History index for training targets: train events only (D1-2)."""
    return HistoryIndex(train_view(events))


def frozen_history_index(events: pd.DataFrame, split_name: str, meta: dict) -> tuple[HistoryIndex, int]:
    """(index, t_pred) for frozen scoring: val -> train events at T_val, test / export ->
    train + val events at T_test. Built from `history_view`, the M3 information boundary."""
    return HistoryIndex(history_view(events, split_name, meta)), prediction_time(meta, split_name)


# ----------------------------------------------------------------------------- items
VIDEO_CAT_COLS = tuple(f"{c}_idx" for c in catm.META_CAT_COLS)   # video_type, upload_type, upload_dt, music_type
TAG_COLS = tuple(f"tag{j}_idx" for j in range(catm.MAX_TAGS))


def _t(x, dtype=torch.long) -> torch.Tensor:
    # Copy: pandas may hand out read-only buffers, and the tensors must not alias the frames.
    return torch.tensor(np.array(x), dtype=dtype)


@dataclass(frozen=True)
class VideoFeatures:
    """Per video-catalog row (row i = video_catalog row i): train-vocab indices + numerics."""

    video_idx: torch.Tensor      # [n]
    author_idx: torch.Tensor     # [n]
    music_idx: torch.Tensor      # [n] music vocab index (NO_MUSIC for music_id 0)
    music_type_idx: torch.Tensor # [n] music_type of the video's music
    music_row: torch.Tensor      # [n] music catalog row, VIDEO_NO_MUSIC_ROW (-1) for music_id 0
    cat: torch.Tensor            # [n, 4] VIDEO_CAT_COLS
    tags: torch.Tensor           # [n, 3] tag vocab indices, 0 = PAD
    num: torch.Tensor            # [n, 3] train-standardised log1p numerics (0 if missing)
    num_missing: torch.Tensor    # [n, 3] float missing flags

    @classmethod
    def from_catalog(cls, video_catalog: pd.DataFrame) -> "VideoFeatures":
        vc = video_catalog
        if not (vc.video_row.to_numpy() == np.arange(len(vc))).all():
            raise ValueError("video_catalog must be in catalog-row order")
        return cls(
            video_idx=_t(vc.video_idx), author_idx=_t(vc.author_idx), music_idx=_t(vc.music_idx),
            music_type_idx=_t(vc.music_type_idx), music_row=_t(vc.music_row),
            cat=_t(vc[list(VIDEO_CAT_COLS)].to_numpy()), tags=_t(vc[list(TAG_COLS)].to_numpy()),
            num=_t(vc[[f"{c}_z" for c in NUMERIC_VIDEO_COLS]].to_numpy(np.float32), torch.float32),
            num_missing=_t(vc[[f"{c}_missing" for c in NUMERIC_VIDEO_COLS]].to_numpy(np.float32), torch.float32),
        )

    def __len__(self) -> int:
        return int(self.video_idx.shape[0])

    def to(self, device) -> "VideoFeatures":
        return VideoFeatures(**{k: v.to(device) for k, v in asdict_shallow(self).items()})


@dataclass(frozen=True)
class MusicFeatures:
    """Per music-catalog row (non-zero music ids)."""

    music_idx: torch.Tensor
    music_type_idx: torch.Tensor

    @classmethod
    def from_catalog(cls, music_catalog: pd.DataFrame) -> "MusicFeatures":
        mc = music_catalog
        if not (mc.music_row.to_numpy() == np.arange(len(mc))).all():
            raise ValueError("music_catalog must be in catalog-row order")
        return cls(music_idx=_t(mc.music_idx), music_type_idx=_t(mc.music_type_idx))

    def __len__(self) -> int:
        return int(self.music_idx.shape[0])

    def to(self, device) -> "MusicFeatures":
        return MusicFeatures(**{k: v.to(device) for k, v in asdict_shallow(self).items()})


def asdict_shallow(obj) -> dict:
    # dataclasses.asdict deep-copies tensors; we only need the field references.
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def check_shared_music_inputs(video_catalog: pd.DataFrame, music_catalog: pd.DataFrame) -> None:
    """A video with non-zero music reuses the music candidate's base embedding (gathered by
    `music_row`), so the video's own (music_idx, music_type_idx) must equal that catalog
    row's inputs; music_id 0 videos must carry NO_MUSIC."""
    vc = video_catalog[video_catalog.music_row >= 0]
    mc = music_catalog.set_index("music_row").loc[vc.music_row]
    if not ((vc.music_idx.to_numpy() == mc.music_idx.to_numpy()).all()
            and (vc.music_type_idx.to_numpy() == mc.music_type_idx.to_numpy()).all()):
        raise ValueError("video -> music inputs disagree with the music catalog")
    zero = video_catalog[video_catalog.music_row < 0]
    if not (zero.music_idx == catm.NO_MUSIC).all():
        raise ValueError("music_id 0 videos must use NO_MUSIC")


# ----------------------------------------------------------------------------- sizes
@dataclass(frozen=True)
class FeatureSizes:
    """Embedding-table sizes (incl. specials) and the first in-vocab index of each ID table."""

    n_user: int
    n_video: int
    n_author: int
    n_music: int
    id_offset: dict          # {"user": 2, "video": 2, "author": 2, "music": 3}
    meta: dict               # {"video_type", "upload_type", "upload_dt", "music_type", "tag"} -> size
    user_fields: dict | None = None   # U1 field -> size; None for U0

    @classmethod
    def from_vocabs(cls, vocabs: catm.Vocabs, user_vocab: "UserFeatureVocab | None" = None) -> "FeatureSizes":
        ids = {n: getattr(vocabs, n) for n in ("user", "video", "author", "music")}
        return cls(
            n_user=ids["user"].size, n_video=ids["video"].size, n_author=ids["author"].size,
            n_music=ids["music"].size, id_offset={n: v.offset for n, v in ids.items()},
            meta={c: v.size for c, v in vocabs.meta.items()},
            user_fields=None if user_vocab is None else {f: v.size for f, v in user_vocab.vocabs.items()},
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- U1 user features
U1_FIELDS = (*(f"onehot_feat{i}" for i in range(18)), "is_live_streamer", "is_video_author",
             "register_days_range")


def _as_category(s: pd.Series) -> np.ndarray:
    # Same string normalisation as the M2 metadata vocabs; NaN -> "<NA>" (its own category),
    # is_live_streamer = -124 stays "-124" (its own category).
    return s.astype("string").fillna("<NA>").to_numpy(dtype=object)


@dataclass(frozen=True)
class UserFeatureVocab:
    fields: tuple
    vocabs: dict             # field -> catalog.Vocab (PAD 0, OOV 1, kept values from 2)
    min_count: int
    summary: dict            # field -> {n_values_train_users, n_kept, n_train_users_on_rare}


def fit_user_feature_vocab(user_features: pd.DataFrame, train_events: pd.DataFrame,
                           fields_: tuple = U1_FIELDS, min_count: int = 5) -> UserFeatureVocab:
    """U1 vocabs (D1-6), fitted on the snapshot rows of users with >= 1 train event.

    A value is kept iff >= `min_count` such users have it; rare and unseen values map to the
    fixed-zero OOV row, so no value without enough training signal gets its own embedding.
    """
    catm.assert_train_only(train_events)
    if user_features.user_id.duplicated().any():
        raise ValueError("user_features must have unique user_id")
    uf = user_features[user_features.user_id.isin(np.unique(train_events.user_id.to_numpy()))]
    vocabs, summary = {}, {}
    for f in fields_:
        counts = pd.Series(_as_category(uf[f])).value_counts()
        kept = sorted(counts[counts >= min_count].index)
        vocabs[f] = catm.Vocab(f, np.asarray(kept, dtype=object), 2, {"PAD": catm.PAD, "OOV": catm.OOV})
        summary[f] = {"n_values_train_users": int(len(counts)), "n_kept": int(len(kept)),
                      "n_train_users_on_rare": int(counts[counts < min_count].sum())}
    return UserFeatureVocab(tuple(fields_), vocabs, int(min_count), summary)


@dataclass(frozen=True)
class UserFeatureTable:
    """Encoded U1 codes for every user in the snapshot table; lookup by raw user_id."""

    user_ids: np.ndarray     # sorted
    codes: np.ndarray        # [n_users, n_fields]
    fields: tuple

    def lookup(self, users) -> np.ndarray:
        """[B, n_fields] codes; users absent from the snapshot table get OOV in every field."""
        r = catm._lookup(self.user_ids, np.asarray(users, dtype=np.int64), missing=-1)
        out = np.full((len(r), len(self.fields)), catm.OOV, dtype=np.int64)
        out[r >= 0] = self.codes[r[r >= 0]]
        return out


def build_user_feature_table(user_features: pd.DataFrame, vocab: UserFeatureVocab) -> UserFeatureTable:
    uf = user_features.sort_values("user_id")
    codes = np.stack([vocab.vocabs[f].encode(_as_category(uf[f])) for f in vocab.fields], axis=1)
    return UserFeatureTable(uf.user_id.to_numpy(np.int64), codes.astype(np.int64), vocab.fields)
