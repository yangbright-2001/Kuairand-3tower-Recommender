"""Retrieval catalogs and train-only ID vocabularies (PLAN.md §3b P1-5, P1-6, P1-6b).

Two different index spaces, kept deliberately separate:

* **Catalog rows** (evaluation / export / retrieve()): every candidate item, train-seen or
  not. Video catalog = all metadata videos; music catalog = all non-zero metadata music
  ids. Row order = ascending raw id, so the mapping is deterministic. `music_id = 0` is a
  "no music" sentinel and has no music catalog row (VIDEO_NO_MUSIC_ROW).

* **Vocab indices** (model embedding tables): only ids present in *train* data. Every
  other id maps to OOV. Index layout per vocab:
      0 = PAD, 1 = OOV, [music only: 2 = NO_MUSIC], then in-vocab ids by ascending raw id.
  The vocab is frozen once built; the cold-video condition rebuilds it from H-removed
  train data via `exclude_video_ids`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MUSIC_SENTINEL = 0
VIDEO_NO_MUSIC_ROW = -1  # video -> music catalog row for music_id == 0
PAD, OOV, NO_MUSIC = 0, 1, 2


# ----------------------------------------------------------------------------- catalogs
@dataclass(frozen=True)
class Catalogs:
    video_ids: np.ndarray            # [n_video] raw video_id, ascending; row = position
    music_ids: np.ndarray            # [n_music] raw non-zero music_id, ascending
    video_music_row: np.ndarray      # [n_video] music catalog row or VIDEO_NO_MUSIC_ROW

    def video_row(self, video_ids) -> np.ndarray:
        return _lookup(self.video_ids, video_ids, missing=-1)

    def music_row(self, music_ids) -> np.ndarray:
        # music_id 0 and unknown ids both return -1 (not a music target).
        return _lookup(self.music_ids, music_ids, missing=-1)


def _lookup(sorted_ids: np.ndarray, q, missing: int) -> np.ndarray:
    q = np.asarray(q)
    if len(sorted_ids) == 0:  # e.g. an empty vocab: everything is missing
        return np.full(q.shape, missing, dtype=np.int64)
    pos = np.searchsorted(sorted_ids, q)
    pos_c = np.minimum(pos, len(sorted_ids) - 1)
    ok = (pos < len(sorted_ids)) & (sorted_ids[pos_c] == q)
    return np.where(ok, pos_c, missing).astype(np.int64)


def build_catalogs(video_meta: pd.DataFrame) -> Catalogs:
    """Catalogs come from the metadata table, never from logs (logs miss never-seen items)."""
    vm = video_meta.sort_values("video_id")
    vids = vm.video_id.to_numpy(np.int64)
    music = vm.music_id.to_numpy(np.int64)
    music_ids = np.unique(music[music != MUSIC_SENTINEL])
    vmr = _lookup(music_ids, music, missing=VIDEO_NO_MUSIC_ROW)
    assert ((vmr == VIDEO_NO_MUSIC_ROW) == (music == MUSIC_SENTINEL)).all()
    return Catalogs(vids, music_ids, vmr)


# ----------------------------------------------------------------------------- vocabs
@dataclass(frozen=True)
class Vocab:
    """Frozen raw-id -> embedding index mapping. Unknown ids -> OOV."""

    name: str
    ids: np.ndarray                  # in-vocab raw ids, ascending
    offset: int                      # index of ids[0]
    specials: dict = field(default_factory=dict)

    @property
    def size(self) -> int:           # embedding-table size incl. specials
        return self.offset + len(self.ids)

    def encode(self, raw) -> np.ndarray:
        r = _lookup(self.ids, raw, missing=-1)
        return np.where(r >= 0, r + self.offset, OOV).astype(np.int64)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"raw_id": self.ids, "index": np.arange(len(self.ids)) + self.offset})


def _vocab(name: str, ids, music: bool = False) -> Vocab:
    ids = np.unique(np.asarray(ids))
    if music:
        ids = ids[ids != MUSIC_SENTINEL]
        return Vocab(name, ids, 3, {"PAD": PAD, "OOV": OOV, "NO_MUSIC": NO_MUSIC})
    return Vocab(name, ids, 2, {"PAD": PAD, "OOV": OOV})


@dataclass(frozen=True)
class Vocabs:
    user: Vocab
    video: Vocab
    author: Vocab
    music: Vocab
    # Categorical video-metadata vocabs, also train-only so no category without training
    # signal gets its own (untrained) embedding (P1-6b).
    meta: dict

    def encode_music(self, music_ids) -> np.ndarray:
        """music_id 0 -> NO_MUSIC (trained like any id: 0-music videos appear in train)."""
        m = np.asarray(music_ids)
        return np.where(m == MUSIC_SENTINEL, NO_MUSIC, self.music.encode(m)).astype(np.int64)


META_CAT_COLS = ("video_type", "upload_type", "upload_dt", "music_type")


def train_seen_video_ids(train_events: pd.DataFrame, exclude_video_ids=()) -> np.ndarray:
    v = np.unique(train_events.video_id.to_numpy())
    return v[~np.isin(v, np.asarray(exclude_video_ids))] if len(exclude_video_ids) else v


def assert_train_only(train: pd.DataFrame) -> None:
    """Guard for every train-fitted quantity: input must carry `split` and be all train (0).

    Without the column the input cannot be proven train-only, so it is refused as well;
    otherwise passing train+val/test events would silently admit future ids/statistics.
    """
    if "split" not in train.columns:
        raise ValueError("train-only fit needs a `split` column to verify its input")
    if (train.split != 0).any():
        raise ValueError("train-only fit received non-train events (split != 0)")


def build_vocabs(train_events: pd.DataFrame, video_meta: pd.DataFrame, exclude_video_ids=()) -> Vocabs:
    """All vocabs from train events only (asserted: `split` present and all 0).

    - user: users with >=1 train event (after exclusions).
    - video: videos with >=1 train event, minus `exclude_video_ids` (cold-video H).
    - author / music / metadata categories: attributes of those train-seen videos only.
      An author or music that only occurs on excluded or val/test-only videos is OOV.
    """
    assert_train_only(train_events)
    ex = np.asarray(exclude_video_ids)
    tr = train_events[~train_events.video_id.isin(ex)] if len(ex) else train_events
    seen = np.unique(tr.video_id.to_numpy())
    vm = video_meta[video_meta.video_id.isin(seen)]
    meta = {}
    for c in META_CAT_COLS:
        vals = vm[c].astype("string").fillna("<NA>").to_numpy(dtype=object)
        meta[c] = _vocab(c, np.asarray(sorted(set(vals)), dtype=object))
    tags = sorted({t for s in vm.tag.dropna().astype(str) for t in s.split(",") if t})
    meta["tag"] = _vocab("tag", np.asarray(tags, dtype=object))
    return Vocabs(
        user=_vocab("user", tr.user_id.to_numpy()),
        video=_vocab("video", seen),
        author=_vocab("author", vm.author_id.to_numpy()),
        music=_vocab("music", vm.music_id.to_numpy(), music=True),
        meta=meta,
    )


def encode_music_features(video_meta: pd.DataFrame, cat: Catalogs, vocabs: Vocabs) -> pd.DataFrame:
    """One row per music catalog row: train-vocab music index + music_type index.

    music_type is constant within a non-zero music_id in this dataset (checked here), so it
    is a property of the music itself. `n_videos_metadata` counts catalog videos per music
    (metadata only, no interactions), useful for shared-music handling.
    """
    nz = video_meta[video_meta.music_id != MUSIC_SENTINEL]
    if (nz.groupby("music_id").music_type.nunique(dropna=False) > 1).any():
        raise ValueError("music_type differs within a music_id; music features are ambiguous")
    first = nz.sort_values("video_id").drop_duplicates("music_id").set_index("music_id").loc[cat.music_ids]
    out = pd.DataFrame({"music_row": np.arange(len(cat.music_ids)), "music_id": cat.music_ids})
    out["music_idx"] = vocabs.music.encode(cat.music_ids)
    out["music_type_idx"] = vocabs.meta["music_type"].encode(
        first.music_type.astype("string").fillna("<NA>").to_numpy(dtype=object))
    out["n_videos_metadata"] = np.bincount(cat.video_music_row[cat.video_music_row >= 0],
                                           minlength=len(cat.music_ids))
    return out


# ----------------------------------------------------------------------------- video features
MAX_TAGS = 3  # audit: per-video tag counts are 1..3


def encode_video_features(video_meta: pd.DataFrame, cat: Catalogs, vocabs: Vocabs, num_stats: dict) -> pd.DataFrame:
    """One row per video catalog row: vocab indices + train-standardised numerics.

    `visible_status` (constant, "now" state) is dropped; the statistic file is never read.
    Missing duration (NaN in metadata) gets value 0 plus an explicit missing flag.
    """
    vm = video_meta.set_index("video_id").loc[cat.video_ids].reset_index()
    out = pd.DataFrame({"video_row": np.arange(len(vm)), "video_id": vm.video_id.to_numpy()})
    out["video_idx"] = vocabs.video.encode(vm.video_id.to_numpy())
    out["author_idx"] = vocabs.author.encode(vm.author_id.to_numpy())
    out["music_idx"] = vocabs.encode_music(vm.music_id.to_numpy())
    out["music_row"] = cat.video_music_row
    for c in META_CAT_COLS:
        out[f"{c}_idx"] = vocabs.meta[c].encode(vm[c].astype("string").fillna("<NA>").to_numpy(dtype=object))
    tag_lists = vm.tag.fillna("").astype(str).str.split(",")
    tag_mat = np.zeros((len(vm), MAX_TAGS), dtype=np.int64)  # 0 = PAD
    for i, tl in enumerate(tag_lists):
        tl = [t for t in tl if t][:MAX_TAGS]
        if tl:
            tag_mat[i, : len(tl)] = vocabs.meta["tag"].encode(np.asarray(tl, dtype=object))
    for j in range(MAX_TAGS):
        out[f"tag{j}_idx"] = tag_mat[:, j]
    for c, s in num_stats.items():
        x = np.log1p(vm[c].to_numpy(dtype=float))
        out[f"{c}_missing"] = np.isnan(x).astype(np.int8)
        out[f"{c}_z"] = np.nan_to_num((x - s["mean"]) / s["std"], nan=0.0).astype(np.float32)
    return out
