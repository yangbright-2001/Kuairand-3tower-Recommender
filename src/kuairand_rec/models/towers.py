"""Three-tower architecture (PLAN.md §3d D1-1, D1-2, D1-4, D1-6).

    MusicTower : music_id emb + music_type emb                         -> MLP + LN -> m_base [128]
    VideoTower : video_id + author + shared MusicTower(m_base) + meta  -> MLP + LN -> v_base [128]
    UserTower  : user_id + HistoryEncoder(v_base tokens) [+ U1 fields] -> MLP + LN -> u_base [128]
    Heads      : Linear(128 -> 64) + L2 norm; user 4, video 2, music 2. Score = cos.

Correctness-critical pieces:
  * Fixed-zero rows (Q5b): PAD / OOV rows of every embedding table are multiplied by a constant
    buffer mask in the forward pass, so they are exactly 0, receive exactly 0 gradient (hence
    0 Adam update, and decoupled weight decay of a zero row is 0) and perturbing their storage
    cannot change any score. NO_MUSIC is learnable.
  * Shared music tower, shared *output*: `encode_items` runs `MusicTower` exactly once per
    catalog forward, on the music catalog rows plus one NO_MUSIC input per distinct music_type
    of the music_id-0 videos. The video tower gathers its music input from that output by
    `music_row`, so, even in train mode with dropout, a video's music input is the identical
    tensor as its music candidate's `m_base`, and videos sharing a music share one sample.
    Video losses back-propagate into the music tower through the gather. (Calling the tower
    twice would share parameters but draw independent dropout masks.)
  * History tokens use `v_base` of the shared video tower, gathered from one catalog forward,
    so OOV videos in a history are represented by content (Q3a).
  * Empty history is NaN-safe: one key is unmasked before the softmax and the pooled output
    is multiplied by (len > 0); NaN never appears, so NaN * 0 cannot occur.
  * ID dropout (D1-3) is drawn per *vocab entry* once per item forward, so a dropped video /
    author / music id is dropped consistently in candidates, history tokens and the video's
    music input. Active only in train mode; dropping an id is exactly the OOV (zero-row) path.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn

from kuairand_rec.data import catalog as catm
from kuairand_rec.models.features import (
    FEEDBACK_BINARY_COLS, N_PLAY_BUCKETS, PLAY_NONE, RECENCY_PAD, FeatureSizes, HistoryConfig,
    HistoryTensors, MusicFeatures, VideoFeatures,
)

TASKS = ("video_long_view", "video_is_like", "music_long_view", "music_is_like")
VIDEO_TASKS = TASKS[:2]
MUSIC_TASKS = TASKS[2:]


@dataclass(frozen=True)
class TowerConfig:
    base_dim: int = 128          # D1-1
    head_dim: int = 64           # D1-1
    id_dim: int = 64
    meta_dim: int = 16
    user_field_dim: int = 16     # D1-6
    mlp_hidden: int = 256
    n_layers: int = 2            # D1-2 (Q4a)
    n_heads: int = 4
    ff_dim: int = 512
    dropout: float = 0.1         # Transformer / MLP dropout (train mode only)
    id_dropout: float = 0.1      # D1-3, per ID slot
    emb_init_std: float = 0.02
    use_music: bool = True       # ablation flags (D1-1)
    use_author: bool = True
    use_history: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "TowerConfig":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown tower config keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- building blocks
class MaskedEmbedding(nn.Module):
    """Embedding whose `zero_rows` are fixed at exactly 0 via a constant buffer mask."""

    def __init__(self, num: int, dim: int, zero_rows=(catm.PAD, catm.OOV), init_std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num, dim) * init_std)
        keep = torch.ones(num, 1)
        zr = [r for r in zero_rows if r < num]
        keep[zr] = 0.0
        self.register_buffer("keep", keep, persistent=True)
        with torch.no_grad():
            self.weight.mul_(self.keep)
        self.zero_rows = tuple(zr)

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.keep

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.effective_weight())


def _mlp(d_in: int, cfg: TowerConfig) -> nn.Sequential:
    # MLP + LayerNorm, no BatchNorm (D1-1).
    return nn.Sequential(nn.Linear(d_in, cfg.mlp_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                         nn.Linear(cfg.mlp_hidden, cfg.base_dim), nn.LayerNorm(cfg.base_dim))


def _keep_mask(n: int, p: float, device, training: bool) -> torch.Tensor | None:
    """Per-vocab-entry keep mask [n] for ID dropout, or None when inactive."""
    if not training or p <= 0:
        return None
    return (torch.rand(n, device=device) >= p).to(torch.float32)


def _drop(emb: torch.Tensor, idx: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
    return emb if keep is None else emb * keep[idx].unsqueeze(-1)


class Heads(nn.Module):
    """One Linear(base -> head) + L2 norm per task."""

    def __init__(self, tasks, cfg: TowerConfig):
        super().__init__()
        self.tasks = tuple(tasks)
        self.proj = nn.ModuleDict({t: nn.Linear(cfg.base_dim, cfg.head_dim) for t in self.tasks})

    def forward(self, base: torch.Tensor) -> dict[str, torch.Tensor]:
        return {t: F.normalize(self.proj[t](base), dim=-1) for t in self.tasks}


# ----------------------------------------------------------------------------- item towers
class MusicTower(nn.Module):
    def __init__(self, sizes: FeatureSizes, cfg: TowerConfig):
        super().__init__()
        # music vocab: PAD 0, OOV 1 fixed zero; NO_MUSIC 2 learnable.
        self.music_emb = MaskedEmbedding(sizes.n_music, cfg.id_dim, init_std=cfg.emb_init_std)
        self.type_emb = MaskedEmbedding(sizes.meta["music_type"], cfg.meta_dim, init_std=cfg.emb_init_std)
        self.mlp = _mlp(cfg.id_dim + cfg.meta_dim, cfg)

    def forward(self, music_idx, music_type_idx, keep_music: torch.Tensor | None = None) -> torch.Tensor:
        m = _drop(self.music_emb(music_idx), music_idx, keep_music)
        return self.mlp(torch.cat([m, self.type_emb(music_type_idx)], dim=-1))


class VideoTower(nn.Module):
    """Takes the per-video music base as an input (gathered from the single MusicTower output
    by `ThreeTowerModel.encode_items`); it never calls the music tower itself."""

    def __init__(self, sizes: FeatureSizes, cfg: TowerConfig):
        super().__init__()
        self.cfg = cfg
        self.video_emb = MaskedEmbedding(sizes.n_video, cfg.id_dim, init_std=cfg.emb_init_std)
        self.author_emb = (MaskedEmbedding(sizes.n_author, cfg.id_dim, init_std=cfg.emb_init_std)
                           if cfg.use_author else None)
        cat_cols = ("video_type", "upload_type", "upload_dt", "music_type")   # = features.VIDEO_CAT_COLS
        self.cat_embs = nn.ModuleList(
            [MaskedEmbedding(sizes.meta[c], cfg.meta_dim, init_std=cfg.emb_init_std) for c in cat_cols])
        self.tag_emb = MaskedEmbedding(sizes.meta["tag"], cfg.meta_dim, init_std=cfg.emb_init_std)
        d_in = (cfg.id_dim + (cfg.id_dim if cfg.use_author else 0) + (cfg.base_dim if cfg.use_music else 0)
                + len(cat_cols) * cfg.meta_dim + cfg.meta_dim + 2 * 3)
        self.mlp = _mlp(d_in, cfg)

    def forward(self, vf: VideoFeatures, keep: dict | None = None,
                music_base: torch.Tensor | None = None) -> torch.Tensor:
        keep = keep or {}
        parts = [_drop(self.video_emb(vf.video_idx), vf.video_idx, keep.get("video"))]
        if self.author_emb is not None:
            parts.append(_drop(self.author_emb(vf.author_idx), vf.author_idx, keep.get("author")))
        if self.cfg.use_music:
            if music_base is None or music_base.shape != (len(vf), self.cfg.base_dim):
                raise ValueError("use_music=True needs the per-video music base [n_video, base_dim]")
            parts.append(music_base)
        elif music_base is not None:
            raise ValueError("use_music=False but a music base was passed")
        parts += [emb(vf.cat[:, j]) for j, emb in enumerate(self.cat_embs)]
        # Mean over non-PAD tags; an OOV tag counts but contributes a zero vector.
        tag_valid = (vf.tags != catm.PAD).to(torch.float32)
        tag_sum = (self.tag_emb(vf.tags) * tag_valid.unsqueeze(-1)).sum(1)
        parts.append(tag_sum / tag_valid.sum(1, keepdim=True).clamp_min(1.0))
        parts += [vf.num, vf.num_missing]
        return self.mlp(torch.cat(parts, dim=-1))


# ----------------------------------------------------------------------------- history encoder
class _Block(nn.Module):
    """Pre-norm Transformer block with explicit attention math.

    nn.TransformerEncoderLayer switches to a fused "fast path" under no_grad / eval, which
    computes with different kernels than the training path. Writing the math out keeps one
    code path for training, direct scoring and export (D1-7 parity)."""

    def __init__(self, d: int, n_heads: int, ff: int, dropout: float):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_ok: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)            # [B, h, L, L]
        att = att.masked_fill(~key_ok[:, None, None, :], float("-inf"))  # every row has >= 1 key
        att = self.drop(att.softmax(-1))
        y = (att @ v).transpose(1, 2).reshape(B, L, d)
        x = x + self.drop(self.out(y))
        return x + self.drop(self.ff(self.ln2(x)))


class HistoryEncoder(nn.Module):
    """Tokens = proj(v_base) + proj([8 feedback, hidden]) + play_ratio emb + recency emb + position
    emb -> 2-layer Transformer -> learnable-query attention pooling -> [B, 128]."""

    def __init__(self, cfg: TowerConfig, hcfg: HistoryConfig):
        super().__init__()
        d = cfg.base_dim
        self.max_len = hcfg.max_len
        self.item_proj = nn.Linear(d, d)
        self.fb_proj = nn.Linear(len(FEEDBACK_BINARY_COLS) + 1, d)
        # Row 0 of play / recency is PAD-or-hidden / PAD: fixed zero, never trained.
        self.play_emb = MaskedEmbedding(N_PLAY_BUCKETS, d, zero_rows=(PLAY_NONE,), init_std=cfg.emb_init_std)
        self.rec_emb = MaskedEmbedding(hcfg.n_recency_rows, d, zero_rows=(RECENCY_PAD,), init_std=cfg.emb_init_std)
        self.pos_emb = nn.Parameter(torch.randn(hcfg.max_len, d) * cfg.emb_init_std)
        self.blocks = nn.ModuleList([_Block(d, cfg.n_heads, cfg.ff_dim, cfg.dropout) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.pool_query = nn.Parameter(torch.randn(d) * cfg.emb_init_std)
        self.pool_key = nn.Linear(d, d)

    def forward(self, h: HistoryTensors, v_base: torch.Tensor) -> torch.Tensor:
        B, L = h.video_row.shape
        if L != self.max_len:
            raise ValueError(f"history length {L} != encoder max_len {self.max_len}")
        valid = h.valid
        if not torch.equal(valid, h.video_row >= 0) or not torch.equal(valid.sum(1), h.length):
            raise ValueError("inconsistent history batch (valid / video_row / length)")
        vf = valid.unsqueeze(-1).to(v_base.dtype)
        item = v_base[h.video_row.clamp_min(0)] * vf          # PAD rows gather row 0, then zeroed
        x = (self.item_proj(item) * vf
             + self.fb_proj(torch.cat([h.feedback, h.hidden.unsqueeze(-1)], -1))
             + self.play_emb(h.play_bucket) + self.rec_emb(h.recency_bucket) + self.pos_emb.unsqueeze(0))
        # Empty history: unmask the last key so no softmax row is all -inf (NaN-safe), then
        # zero the pooled output below. Right-aligned, so the last column is the newest slot.
        has = h.length > 0
        key_ok = valid.clone()
        key_ok[:, -1] |= ~has
        for blk in self.blocks:
            x = blk(x, key_ok)
        x = self.ln_f(x)
        s = (self.pool_key(x) @ self.pool_query) / math.sqrt(x.shape[-1])   # [B, L]
        a = s.masked_fill(~key_ok, float("-inf")).softmax(-1)
        pooled = (a.unsqueeze(-1) * x).sum(1)
        return pooled * has.unsqueeze(-1).to(pooled.dtype)


# ----------------------------------------------------------------------------- user tower
class UserTower(nn.Module):
    def __init__(self, sizes: FeatureSizes, cfg: TowerConfig, hcfg: HistoryConfig):
        super().__init__()
        self.cfg = cfg
        self.user_emb = MaskedEmbedding(sizes.n_user, cfg.id_dim, init_std=cfg.emb_init_std)
        self.history = HistoryEncoder(cfg, hcfg) if cfg.use_history else None
        self.u1_fields = tuple(sizes.user_fields) if sizes.user_fields else ()
        self.field_embs = nn.ModuleList(
            [MaskedEmbedding(sizes.user_fields[f], cfg.user_field_dim, init_std=cfg.emb_init_std)
             for f in self.u1_fields])
        d_in = cfg.id_dim + (cfg.base_dim if cfg.use_history else 0) + len(self.u1_fields) * cfg.user_field_dim
        self.mlp = _mlp(d_in, cfg)

    @property
    def is_u1(self) -> bool:
        return bool(self.u1_fields)

    def forward(self, user_idx, hist: HistoryTensors | None, v_base: torch.Tensor | None,
                user_feats: torch.Tensor | None = None) -> torch.Tensor:
        e = self.user_emb(user_idx)
        if self.training and self.cfg.id_dropout > 0:     # per-row user-ID dropout -> OOV path
            e = e * (torch.rand(e.shape[0], 1, device=e.device) >= self.cfg.id_dropout).to(e.dtype)
        parts = [e]
        if self.history is not None:
            if hist is None or v_base is None:
                raise ValueError("use_history=True needs a history batch and catalog v_base")
            if hist.video_row.shape[0] != user_idx.shape[0]:
                raise ValueError("history batch size != user batch size")
            parts.append(self.history(hist, v_base))
        if self.is_u1:
            if user_feats is None or user_feats.shape != (user_idx.shape[0], len(self.u1_fields)):
                raise ValueError(f"U1 tower needs user_feats of shape [B, {len(self.u1_fields)}]")
            parts += [emb(user_feats[:, j]) for j, emb in enumerate(self.field_embs)]
        elif user_feats is not None:
            raise ValueError("U0 tower received user features")
        return self.mlp(torch.cat(parts, dim=-1))


# ----------------------------------------------------------------------------- full model
@dataclass
class ItemEncodings:
    v_base: torch.Tensor   # [n_video_catalog, 128]
    m_base: torch.Tensor   # [n_music_catalog, 128]
    # [n_video_catalog, 128] music input actually fed to the video tower (None if use_music
    # is False). Rows with non-zero music are gathered from m_base.
    video_music_base: torch.Tensor | None = None


class ThreeTowerModel(nn.Module):
    def __init__(self, sizes: FeatureSizes, cfg: TowerConfig, hcfg: HistoryConfig):
        super().__init__()
        self.sizes, self.cfg, self.hcfg = sizes, cfg, hcfg
        self.music_tower = MusicTower(sizes, cfg)
        self.video_tower = VideoTower(sizes, cfg)
        self.user_tower = UserTower(sizes, cfg, hcfg)
        self.user_heads = Heads(TASKS, cfg)
        self.video_heads = Heads(VIDEO_TASKS, cfg)
        self.music_heads = Heads(MUSIC_TASKS, cfg)

    def _item_keep(self, device) -> dict:
        p, tr = self.cfg.id_dropout, self.training
        return {"video": _keep_mask(self.sizes.n_video, p, device, tr),
                "author": _keep_mask(self.sizes.n_author, p, device, tr),
                "music": _keep_mask(self.sizes.n_music, p, device, tr)}

    def encode_items(self, vf: VideoFeatures, mf: MusicFeatures) -> ItemEncodings:
        """One full-catalog item forward. In train mode a single ID-dropout draw is shared by
        the video candidates, the history tokens (gathered from v_base) and the music path."""
        keep = self._item_keep(vf.video_idx.device)
        m_base, video_music = self._music_forward(vf, mf, keep["music"])
        v_base = self.video_tower(vf, keep, video_music if self.cfg.use_music else None)
        return ItemEncodings(v_base, m_base, video_music if self.cfg.use_music else None)

    def _music_forward(self, vf: VideoFeatures, mf: MusicFeatures, keep_music):
        """Single MusicTower call -> (m_base [n_music], per-video music base [n_video]).

        Inputs = the music catalog rows, followed by (NO_MUSIC, t) for each distinct music_type t
        of the music_id-0 videos. A music_id-0 video therefore keeps its original input
        (learnable NO_MUSIC + its own music_type) and is never mapped to a catalog / OOV row.
        """
        n_music = len(mf)
        has_music = vf.music_row >= 0
        # Guard the gather: a video's own music inputs must be those of its catalog row.
        r = vf.music_row[has_music]
        if not (torch.equal(vf.music_idx[has_music], mf.music_idx[r])
                and torch.equal(vf.music_type_idx[has_music], mf.music_type_idx[r])):
            raise ValueError("video music inputs disagree with the music catalog row")
        if not (vf.music_idx[~has_music] == catm.NO_MUSIC).all():
            raise ValueError("music_id 0 videos must use NO_MUSIC")
        no_types, inv = torch.unique(vf.music_type_idx[~has_music], sorted=True, return_inverse=True)
        idx = torch.cat([mf.music_idx, torch.full_like(no_types, catm.NO_MUSIC)])
        typ = torch.cat([mf.music_type_idx, no_types])
        out = self.music_tower(idx, typ, keep_music)
        gather = vf.music_row.clone()
        gather[~has_music] = n_music + inv
        return out[:n_music], out[gather]

    def item_embeddings(self, enc: ItemEncodings) -> dict[str, torch.Tensor]:
        return {**self.video_heads(enc.v_base), **self.music_heads(enc.m_base)}

    def encode_users(self, user_idx, hist: HistoryTensors | None, enc: ItemEncodings | None,
                     user_feats: torch.Tensor | None = None) -> torch.Tensor:
        return self.user_tower(user_idx, hist, None if enc is None else enc.v_base, user_feats)

    def user_embeddings(self, u_base: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.user_heads(u_base)

    @staticmethod
    def scores(user_emb: dict, item_emb: dict, task: str) -> torch.Tensor:
        """Cosine scores [B, n_items] (both sides are L2-normalised); training uses / tau."""
        return user_emb[task] @ item_emb[task].T

    def fixed_zero_embeddings(self) -> list[tuple[str, MaskedEmbedding]]:
        return [(n, m) for n, m in self.named_modules() if isinstance(m, MaskedEmbedding)]
