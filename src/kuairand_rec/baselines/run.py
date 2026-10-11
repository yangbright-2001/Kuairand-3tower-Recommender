"""M3 runner: frames -> Pop -> ALS val tuning -> test evaluation -> results files.

Run:  uv run python -m kuairand_rec.baselines.run --config configs/baselines.yaml

Models (model column in every results file):
  pop                    train positive counts (P1-6b)
  als                    per-action ALS, fold-in at t_pred. For like tasks this is
                         als_like_pure (aux_w = 0), the headline (O1).
  als_nofoldin           same fitted model, raw train user factors (reference, test only)
  als_like_aux           like tasks only: aux_w = 0.3 long_view auxiliary signal (O2)
  als_like_aux_nofoldin  its no-fold-in reference
Each task is evaluated on the masked main frame and the no-mask frame (same users); Pop also
on the cold-user frame.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kuairand_rec import paths
from kuairand_rec.baselines import als as A
from kuairand_rec.baselines.popularity import PopularityScorer
from kuairand_rec.data.history import HistoryIndex
from kuairand_rec.data.pipeline import load_processed
from kuairand_rec.eval import frame as F
from kuairand_rec.eval.evaluator import EvalContext, evaluate, summarize
from kuairand_rec.eval.views import history_view, prediction_time, train_view
from kuairand_rec.utils import git_sha, set_seed

ACTIONS = ["long_view", "is_like"]


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else paths.PROJECT_ROOT / p


def _grid(cfg) -> list[dict]:
    g = cfg["als"]["grid"]
    return [dict(factors=f, regularization=r, alpha=a)
            for f, r, a in itertools.product(g["factors"], g["regularization"], g["alpha"])]


def run(cfg: dict) -> dict:
    t0 = time.time()
    set_seed(int(cfg["seed"]))
    ks = tuple(cfg["ks"])
    out = _resolve(cfg["output_dir"])
    for sub in ("frames", "per_user"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    d = load_processed(_resolve(cfg["processed_dir"]))
    ev, cat, meta, lc = d["events"], d["catalogs"], d["meta"], d["label_conflict_events"]
    hidx = HistoryIndex(ev)
    tr = train_view(ev)
    ctx = {s: EvalContext(s, prediction_time(meta, s), history_view(ev, s, meta)) for s in ("val", "test")}
    tasks = [tuple(t) for t in cfg["tasks"]]
    n_items = {"video": len(cat.video_ids), "music": len(cat.music_ids)}

    # ---- frames (built once; every model is evaluated on these exact objects)
    frames, fsum = {}, {}
    for s in ("val", "test"):
        for tgt, act in tasks:
            fr = F.build_frames(ev, lc, cat, meta, hidx, s, tgt, act)
            for kind, f in fr.items():
                frames[(s, tgt, act, kind)] = f
                F.frame_table(f).to_parquet(out / "frames" / f"{s}_{tgt}_{act}_{kind}.parquet", index=False)
                fsum[f"{s}/{tgt}_{act}/{kind}"] = {"frame_id": f.frame_id, "population_id": f.population_id,
                                                  "n_users": int(len(f.users)), **f.counts}

    rows, per_user_paths = [], {}

    def _eval(scorer, model, s, tgt, act, kind, extra=None):
        f = frames[(s, tgt, act, kind)]
        pu = evaluate(scorer, f, ctx[s], ks, int(cfg["chunk"]))
        pu["model"] = model
        p = out / "per_user" / model / f"{s}_{tgt}_{act}_{kind}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pu.to_parquet(p, index=False)
        r = summarize(pu, f, ks)
        r["model"] = model
        r.update(extra or {})
        rows.append(r)
        return r

    # ---- Pop (train counts; cross-checked against the M2 artifact)
    pop = PopularityScorer(tr, cat, ACTIONS)
    for tgt in ("video", "music"):
        for a in ACTIONS:
            m2 = d["popularity"][tgt][f"train_{a}"].to_numpy()
            assert np.array_equal(pop.counts(tgt, a).astype(np.int64), m2), f"pop mismatch {tgt}/{a}"
    for tgt, act in tasks:
        _eval(pop, "pop", "val", tgt, act, "masked")
        for kind in ("masked", "nomask", "cold"):
            _eval(pop, "pop", "test", tgt, act, kind)

    # ---- ALS: val tuning per (task, variant), then test with the selected config
    tuning, selected, foldin_stats = [], {}, {}
    acfg = cfg["als"]
    for tgt, act in tasks:
        variants = {"als": 0.0} if act == "long_view" else {"als": 0.0, "als_like_aux": float(acfg["like_aux_w"])}
        for vname, aux_w in variants.items():
            best = None
            for g in _grid(cfg):
                c = A.ALSConfig(aux_w=aux_w, iterations=int(acfg["iterations"]), seed=int(cfg["seed"]),
                                num_threads=int(acfg["num_threads"]), **g)
                t1 = time.time()
                m = A.fit_als(tr, n_items[tgt], tgt, act, c)
                sc = A.ALSScorer({(tgt, act): m}, "foldin", vname)
                f = frames[("val", tgt, act, "masked")]
                pu = evaluate(sc, f, ctx["val"], ks, int(cfg["chunk"]))
                r = summarize(pu, f, ks)
                rec = {"task": f"{tgt}_{act}", "variant": vname, **g, "aux_w": aux_w,
                       "n_scorable_items": int(len(m.item_rows)), "n_train_users": int(len(m.train_users)),
                       "val_n_users": r["n_users"], "val_n_no_signal": r["n_no_signal"],
                       **{f"val_{k}": r[k] for k in r if k.startswith(("recall", "ndcg"))},
                       "fit_eval_sec": round(time.time() - t1, 1)}
                tuning.append(rec)
                key = (r[acfg["select_metric"]], r[acfg["tie_break_metric"]])
                if best is None or key > best[0]:   # strict '>' keeps the first grid point on exact ties
                    best = (key, c, m)
            _, c, m = best
            selected[f"{tgt}_{act}/{vname}"] = {**c.__dict__, "fingerprint": m.fingerprint,
                                               "confidence_hash": m.confidence_hash,
                                               "n_scorable_items": int(len(m.item_rows))}
            extra = {"aux_w": c.aux_w, "config": c.tag()}
            sc = A.ALSScorer({(tgt, act): m}, "foldin", vname)
            _eval(sc, vname, "val", tgt, act, "masked", extra)
            for kind in ("masked", "nomask"):
                _eval(sc, vname, "test", tgt, act, kind, extra)
            foldin_stats[f"{tgt}_{act}/{vname}"] = {f"{k[0]}": v for k, v in sc.foldin_stats.items()}
            nof = A.ALSScorer({(tgt, act): m}, "nofoldin", f"{vname}_nofoldin")
            for kind in ("masked", "nomask"):
                _eval(nof, f"{vname}_nofoldin", "test", tgt, act, kind, extra)

    # ---- fairness check: one frame per (split, task, kind) shared by every model
    res = pd.DataFrame(rows)
    for (s, t, a, k), g in res.groupby(["split", "target", "action", "kind"]):
        assert g.frame_id.nunique() == 1 and g.frame_id.iloc[0] == frames[(s, t, a, k)].frame_id
    for (s, t, a), g in res[res.kind != "cold"].groupby(["split", "target", "action"]):
        assert g.population_id.nunique() == 1   # masked and nomask share users

    res.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(tuning).to_csv(out / "als_tuning.csv", index=False)
    (out / "frames_summary.json").write_text(json.dumps(fsum, indent=1))
    run_meta = {"config": cfg, "git_sha": git_sha(), "data_git_sha": meta.get("git_sha"),
                "t_val_ms": meta["t_val_ms"], "t_test_ms": meta["t_test_ms"],
                "als_selected": selected, "als_foldin_stats": foldin_stats,
                "runtime_sec": round(time.time() - t0, 1)}
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=1, default=str))
    (out / "BASELINES.md").write_text(render_markdown(res, fsum, selected, ks))
    return {"metrics": res, "frames": fsum, "meta": run_meta}


def render_markdown(res: pd.DataFrame, fsum: dict, selected: dict, ks) -> str:
    L = ["# M3 Baselines: Popularity and implicit ALS", "",
         "Generated by `uv run python -m kuairand_rec.baselines.run`. Protocol: PLAN.md §3b (P1) and §3c (M3).",
         "Main = test, masked frame (frozen user at T_test, GT minus seen items, full metadata catalog).",
         "`als` = per-action ALS with fold-in from train+val history (for like tasks = als_like_pure, headline). "
         "`*_nofoldin` = raw train user factors (reference). `als_like_aux` = like + 0.3·long_view signal.", "",
         "## 1. Main results (test, masked)", ""]
    hdr = "| task | model | users | no-signal | " + " | ".join(f"R@{k}" for k in ks) + " | " + \
          " | ".join(f"N@{k}" for k in ks) + " | GT unscorable |"
    L += [hdr, "|---|---|---:|---:|" + "---:|" * (2 * len(ks)) + "---:|"]
    order = ["pop", "als", "als_nofoldin", "als_like_aux", "als_like_aux_nofoldin"]
    main = res[(res.split == "test") & (res.kind == "masked")].copy()
    main["o"] = main.model.map({m: i for i, m in enumerate(order)})
    for _, r in main.sort_values(["target", "action", "o"]).iterrows():
        L.append(f"| {r.target}/{r.action} | {r.model} | {r.n_users:,} | {r.n_no_signal:,} | "
                 + " | ".join(f"{r[f'recall@{k}']:.4f}" for k in ks) + " | "
                 + " | ".join(f"{r[f'ndcg@{k}']:.4f}" for k in ks) + f" | {r.gt_unscorable:,} |")
    L += ["", "## 2. Evaluation frames (test): users and exclusions", "",
          "Counts in positive events (`pos_events*`) or deduplicated (user, item) pairs. Label-conflict pairs are an "
          "upper bound of GT lost to M2 cleaning (`{a}_any` = some dropped raw row was positive).", ""]
    keys = ["pos_events", "pos_events_sentinel", "pos_events_ooc", "dedup_events_collapsed", "gt_pairs_raw",
            "gt_pairs_masked_seen", "label_conflict_possible_gt_pairs", "users_with_raw_gt", "users_cold_no_history",
            "users_all_gt_masked", "evaluated_users", "evaluated_users_train_oov", "gt_pairs_effective",
            "gt_pairs_train_unseen_item"]
    tasks = sorted({k.split("/")[1] for k in fsum})
    L += ["| count | " + " | ".join(tasks) + " |", "|---|" + "---:|" * len(tasks)]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{fsum[f'test/{t}/masked'][k]:,}" for t in tasks) + " |")
    L.append("| nomask gt_pairs_effective | " + " | ".join(f"{fsum[f'test/{t}/nomask']['gt_pairs_effective']:,}" for t in tasks) + " |")
    L.append("| cold-frame users | " + " | ".join(f"{fsum[f'test/{t}/cold']['n_users']:,}" for t in tasks) + " |")
    L += ["", "## 3. Secondary views (test)", "", "| task | model | kind | users | R@20 | N@20 |", "|---|---|---|---:|---:|---:|"]
    sec = res[(res.split == "test") & (res.kind != "masked")]
    for _, r in sec.sort_values(["target", "action", "kind", "model"]).iterrows():
        L.append(f"| {r.target}/{r.action} | {r.model} | {r.kind} | {r.n_users:,} | {r['recall@20']:.4f} | {r['ndcg@20']:.4f} |")
    L += ["", "## 4. Validation (masked, T_val, train-only history) and selected ALS configs", "",
          "| task | model | users | no-signal | R@20 | N@20 | config |", "|---|---|---:|---:|---:|---:|---|"]
    for _, r in res[res.split == "val"].sort_values(["target", "action", "model"]).iterrows():
        L.append(f"| {r.target}/{r.action} | {r.model} | {r.n_users:,} | {r.n_no_signal:,} | "
                 f"{r['recall@20']:.4f} | {r['ndcg@20']:.4f} | {r.get('config', '') if isinstance(r.get('config', ''), str) else ''} |")
    L += ["", "Selected (val masked Recall@20, tie-break NDCG@20; full grid in `als_tuning.csv`):", ""]
    L += [f"- {k}: factors {v['factors']}, reg {v['regularization']}, alpha {v['alpha']}, aux_w {v['aux_w']}, "
          f"scorable items {v['n_scorable_items']:,}" for k, v in selected.items()]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.PROJECT_ROOT / "configs" / "baselines.yaml"))
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    r = run(cfg)
    print(json.dumps({"runtime_sec": r["meta"]["runtime_sec"], "rows": len(r["metrics"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
