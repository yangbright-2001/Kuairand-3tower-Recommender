"""M5 fixture: the M4 export fixture plus one user whose only pre-T_test history is on H.

Catalog (m4_fixtures): v10 a1 m100, v11 a1 m100, v12 a2 m0, v13 a3 m101, v14 a4 m102, v15 a9 m103.
Cuts: T_val = 100 min, T_test = 200 min.

Events relevant to the cold-video tests (H = {13, 14}, both train-seen):
  u2: 130 train events on v10..v14 (capped history 100, ">20"); test lv+like on v15 (not H).
  u4: train v13 @99 (H), val v11 @150 lv+like (non-H); test lv+like v14 @220 (H), lv v10 @225.
  u5: train v13 @30 lv (H only); test lv v14 @250 (H)  -> GT_H non-empty but no non-H history.
  u1: has seen every video before T_test -> all test GT masked (not in the population).
  u3: test-only user -> cold-user frame (long_view); u6: test-only lv+like -> cold frames (both).
Hand-computed test video_long_view masked frame (unfiltered): u2 {v15}, u4 {v10, v14}, u5 {v14}.
Cold frames (H = {13, 14}): u4 {v14} only; u5 excluded (only-H history).

`run_m4(cfg, P, ...)` runs the *unchanged* M4 runner `models.train.run` on the fixture. Only the
I/O boundary is patched: `load_processed` returns the fixture P, raw video metadata is the fixture
metadata, frames are not compared with the real M3 frames_summary.json (fixture frames differ),
and parity runs in-process (`parity_checks`) instead of in a subprocess that would reload real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import m4_fixtures as fx
from kuairand_rec.baselines import als as ALS
from kuairand_rec.baselines.popularity import PopularityScorer
from kuairand_rec.eval.evaluator import evaluate, summarize
from kuairand_rec.eval.views import train_view

MIN = fx.MIN
ROWS = fx.raw_rows() + [fx._row(4, 14, 220 * MIN, lv=1, like=1), fx._row(4, 10, 225 * MIN, lv=1),
                        fx._row(2, 15, 240 * MIN, lv=1, like=1), fx._row(4, 11, 150 * MIN, lv=1, like=1),
                        fx._row(5, 13, 30 * MIN, lv=1), fx._row(5, 14, 250 * MIN, lv=1),
                        fx._row(6, 10, 260 * MIN, lv=1, like=1)]
H = [13, 14]
ALS_CFG = {"factors": 4, "regularization": 0.1, "alpha": 10.0, "iterations": 3, "seed": 0, "num_threads": 1}


def make_P(rows=ROWS) -> dict:
    fit = fx.make(rows)
    lc = pd.DataFrame({"user_id": pd.Series([], dtype=np.int64), "video_id": pd.Series([], dtype=np.int64),
                       "time_ms": pd.Series([], dtype=np.int64), "split": pd.Series([], dtype=np.int8),
                       "long_view_any": pd.Series([], dtype=np.int64), "is_like_any": pd.Series([], dtype=np.int64)})
    return {"events": fit["events"], "label_conflict_events": lc, "catalogs": fit["catalogs"], "meta": dict(fx.META),
            "video_catalog": fit["video_features"], "music_catalog": fit["music_features"], "vocabs": fit["vocabs"],
            "popularity": fit["popularity"]}


def base_cfg(exp: str, run_root: Path, **over) -> dict:
    cfg = {"exp": exp, "processed_dir": "unused", "run_root": str(run_root), "condition": "normal",
           "user_features": "u0", "device": "cpu", "require_clean_git": False, "history": {}, "model": {},
           "train": {"batch_size": 16, "max_epochs": 2, "warmup_steps": 0, "lr": 1e-2, "seed": 0, "log_every": 100},
           "eval": {"threads": 1, "val_user_batch_size": 4, "chunk": 64, "parity_users": 16}}
    cfg.update(over)
    return cfg


def run_m4(cfg: dict, P: dict, monkeypatch, out_dir: Path) -> Path:
    """The real M4 runner on the fixture (train -> selection -> export -> ExportedScorer metrics),
    then the real parity checks 1-4 in-process; writes parity.json like the CLI does."""
    from kuairand_rec.data import io
    from kuairand_rec.models import artifacts as A
    from kuairand_rec.models import evaluation as E
    from kuairand_rec.models import parity as PR
    from kuairand_rec.models import train as TR
    from kuairand_rec.models import data as D

    real_build = E.build_eval_frames

    def build_no_ref(P_, splits=("val", "test"), kinds=("masked", "nomask", "cold"), frames_summary=None, hidx=None):
        return real_build(P_, splits, kinds, None, hidx)

    monkeypatch.setattr(TR, "load_processed", lambda *_a, **_k: P)
    monkeypatch.setattr(io, "load_video_features", lambda *_a, **_k: fx.VIDEO_META)
    monkeypatch.setattr(E, "build_eval_frames", build_no_ref)
    TR.run(cfg, out_dir=str(out_dir), run_parity=False)
    cond = D.build_condition(cfg["condition"], P, fx.VIDEO_META)
    model, _ = A.load_model(out_dir / "best.pt", cond, None, "cpu")
    res = PR.parity_checks(model, cond, P, out_dir / "export", out_dir / "best.pt", None, n_users=16,
                           frames=build_no_ref(P), require_all_strata=False)
    A.write_json(out_dir / "parity.json", res)
    assert res["ok"], res
    return out_dir


def write_baselines(P: dict, frames: dict, out: Path) -> Path:
    """M3-layout baseline dir for the fixture: per_user/{pop,als}/..., metrics.csv, run_meta.json."""
    from kuairand_rec.models.evaluation import contexts

    tr = train_view(P["events"])
    cat = P["catalogs"]
    ctxs = contexts(P)
    pop = PopularityScorer(tr, cat, ["long_view", "is_like"])
    als_models, selected = {}, {}
    for tgt, n in (("video", len(cat.video_ids)), ("music", len(cat.music_ids))):
        for act in ("long_view", "is_like"):
            c = ALS.ALSConfig(**ALS_CFG)
            als_models[(tgt, act)] = ALS.fit_als(tr, n, tgt, act, c)
            selected[f"{tgt}_{act}/als"] = dict(c.__dict__)
    c_aux = ALS.ALSConfig(**{**ALS_CFG, "aux_w": 0.3})
    aux = {("video", "is_like"): ALS.fit_als(tr, len(cat.video_ids), "video", "is_like", c_aux),
           ("music", "is_like"): ALS.fit_als(tr, len(cat.music_ids), "music", "is_like", c_aux)}
    selected["video_is_like/als_like_aux"] = dict(c_aux.__dict__)
    selected["music_is_like/als_like_aux"] = dict(c_aux.__dict__)
    scorers = {"pop": pop, "als": ALS.ALSScorer(als_models, "foldin", "als"),
               "als_like_aux": ALS.ALSScorer(aux, "foldin", "als_like_aux")}
    rows = []
    for (s, task, kind), f in frames.items():
        if len(f.users) == 0:            # the M3 evaluator does not accept empty frames
            continue
        for name, sc in scorers.items():
            if name == "als_like_aux" and not task.endswith("is_like"):
                continue
            if kind == "cold" and name != "pop":
                continue
            pu = evaluate(sc, f, ctxs[s])
            pu["model"] = name
            p = out / "per_user" / name / f"{s}_{task}_{kind}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            pu.to_parquet(p, index=False)
            rows.append({**summarize(pu, f), "model": name})
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    (out / "run_meta.json").write_text(json.dumps({"als_selected": selected}))
    return out
