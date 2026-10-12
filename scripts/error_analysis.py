"""Report error analysis: descriptive breakdowns of the formal results (read-only).

Run:  uv run python scripts/error_analysis.py
Inputs: formal per-user metric tables (M3 baselines, the five formal three-tower runs), the M3
frame tables, the formal exports, data/processed. Nothing is trained, tuned or re-scored for a
reported metric; every number here is a post-hoc *descriptive* breakdown of the test results and
was never used for any model, config or epoch choice.
Output: results/final_analysis/error_analysis.json and ERROR_ANALYSIS.md

Breakdowns (paired user bootstrap, 1000 resamples, seed 0, keyed by population_id/subgroup):
  E1  like tasks: U0 - pure ALS and U0 - aux ALS on users with / without a pure-ALS signal
      (= at least one like before T_test).
  E2  long-view >20 bucket: U0 - pure ALS split by whether the full pre-T_test history exceeds the
      100-event cap (i.e. whether the model input was truncated).
  E3  cold users (no history): how many distinct user embeddings each model exports for them, and
      Spearman correlation of that ranking with train popularity.
  E4  cold-video condition: share of top-20 slots that the cold model gives to H videos (full catalog)
      vs H's share of the catalog, and the mean rank of the best GT_H item.
  E5  music vs video: per-user correlation of U0 R@20 between video/long-view and music/long-view.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kuairand_rec.experiments import bootstrap as B  # noqa: E402

OUT = ROOT / "results" / "final_analysis"
BASE = ROOT / "results" / "baselines"
RUNS = {"u0": "artifacts/u0/9218a76_seed0_job27", "u1": "artifacts/u1/9218a76_seed0_job28",
        "abl_nohist": "artifacts/abl_nohist/671dde9_seed0_job34",
        "abl_nomusic": "artifacts/abl_nomusic/671dde9_seed0_job35",
        "cold_u0": "artifacts/cold_u0/671dde9_seed0_job33"}
N_BOOT, SEED, CI = 1000, 0, 0.95


def per_user(model: str, task: str, kind: str = "masked") -> pd.DataFrame:
    if model in RUNS:
        return pd.read_parquet(ROOT / RUNS[model] / "per_user" / f"test_{task}_{kind}.parquet")
    return pd.read_parquet(BASE / "per_user" / model / f"test_{task}_{kind}.parquet")


def frame(task: str, kind: str = "masked") -> pd.DataFrame:
    return pd.read_parquet(BASE / "frames" / f"test_{task}_{kind}.parquet")


def sub_diff(models: dict, mask: np.ndarray, pairs, key: str, metric="recall@20") -> list[dict]:
    sub = {m: df[mask].reset_index(drop=True) for m, df in models.items()}
    res = B.paired_bootstrap(sub, [metric], N_BOOT, SEED, key)
    rows = B.diff_rows(res, pairs, [metric], CI)
    return [{k: r[k] for k in ("model_a", "model_b", "n_users", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi",
                               "ci_excludes_0")} for r in rows]


def e1_like_signal() -> dict:
    out = {}
    for task in ("video_is_like", "music_is_like"):
        m = {k: per_user(k, task) for k in ("u0", "als", "als_like_aux")}
        pid = m["u0"].population_id.iloc[0]
        sig = m["als"].has_signal.to_numpy()
        out[task] = {}
        for name, mask in (("als_signal", sig), ("als_no_signal", ~sig)):
            out[task][name] = sub_diff(m, mask, [("u0", "als"), ("u0", "als_like_aux")], f"{pid}/E1/{name}")
    return out


def e2_truncation() -> dict:
    out = {}
    for task in ("video_long_view", "music_long_view"):
        f = frame(task)
        m = {k: per_user(k, task) for k in ("u0", "als", "pop")}
        assert np.array_equal(f.user_id.to_numpy(), m["u0"].user_id.to_numpy())
        pid = m["u0"].population_id.iloc[0]
        hf = f.hist_len_full.to_numpy()
        out[task] = {}
        for name, mask in (("21-100 (input not truncated)", (hf > 20) & (hf <= 100)),
                           (">100 (input truncated to last 100)", hf > 100)):
            out[task][name] = sub_diff(m, mask, [("u0", "als")], f"{pid}/E2/{name}")
    return out


def e3_cold_users() -> dict:
    from scipy.stats import spearmanr
    pop = pd.read_parquet(ROOT / "data/processed/popularity_video.parquet")
    out = {}
    f = frame("video_long_view", "cold")
    for name in ("u0", "u1", "abl_nohist", "abl_nomusic"):
        d = ROOT / RUNS[name] / "export"
        uid = np.load(d / "user_ids_test.npy")
        U = np.load(d / "users_test_video_long_view.npy")[np.searchsorted(uid, f.user_id.to_numpy())]
        I = np.load(d / "items_video_long_view.npy")
        s = (I @ U.mean(0)).astype(np.float64)
        rho = spearmanr(s, pop.train_long_view.to_numpy()).statistic
        top20 = np.argsort(-s, kind="stable")[:20]
        out[name] = {"cold_users": int(len(f)), "distinct_user_embeddings": int(len(np.unique(U.round(7), axis=0))),
                     "spearman_mean_cold_score_vs_train_lv_popularity": float(rho),
                     "top20_of_mean_cold_user_in_train_top100_popular": int(
                         np.isin(top20, np.argsort(-pop.train_long_view.to_numpy(), kind="stable")[:100]).sum())}
    # Training exposure of the exact cold path (empty history AND user-ID dropped).
    from kuairand_rec.data.pipeline import load_processed
    from kuairand_rec.models import data as D
    from kuairand_rec.models import features as F
    P = load_processed()
    cond = D.normal_condition(P)
    s = D.build_samples(cond)
    n_empty = int((F.train_history_index(cond.events).history_length(s.user_id, s.time_ms) == 0).sum())
    out["training_exposure"] = {
        "train_samples": int(len(s)), "samples_with_empty_history": n_empty,
        "expected_rows_per_epoch_on_u0_cold_path (empty history x id_dropout 0.1)": 0.1 * n_empty,
        "expected_rows_per_epoch_on_nohist_cold_path (id_dropout 0.1)": 0.1 * len(s)}
    return out


def e4_cold_video() -> dict:
    H = np.array([int(x) for x in (ROOT / "results/m5/cold_video/H_video_ids.txt").read_text().split()])
    vids = np.load(ROOT / RUNS["cold_u0"] / "export" / "video_ids.npy")
    is_h = np.isin(vids, H)
    out = {"h_share_of_catalog": float(is_h.mean())}
    cdir = ROOT / "results/m5/cold_video/eval_671dde9_job33/per_user"
    for task in ("video_long_view", "video_is_like"):
        fr = pd.read_parquet(cdir / f"frame_{task}_cold_video_full.parquet")
        res = {}
        for name, run in (("cold_threetower", "cold_u0"), ("seen_threetower", "u0")):
            d = ROOT / RUNS[run] / "export"
            uid = np.load(d / "user_ids_test.npy")
            U = np.load(d / f"users_test_{task}.npy")[np.searchsorted(uid, fr.user_id.to_numpy())]
            I = np.load(d / f"items_{task}.npy")
            ip, rows = np.load(d / "seen_test_video_indptr.npy"), np.load(d / "seen_test_video_rows.npy")
            r = np.searchsorted(uid, fr.user_id.to_numpy())
            h_top, best_rank = [], []
            for i in range(len(fr)):
                # Full-catalog ranking with the evaluation seen mask (same rule as the evaluator).
                sc = I @ U[i]
                sc[rows[ip[r[i]]:ip[r[i] + 1]]] = -np.inf
                order = np.argsort(-sc, kind="stable")
                rank = np.empty(len(order), np.int64)
                rank[order] = np.arange(1, len(order) + 1)
                h_top.append(is_h[order[:20]].mean())
                best_rank.append(rank[np.asarray(fr.gt_rows.iloc[i])].min())
            res[name] = {"users": int(len(fr)), "mean_share_of_top20_slots_on_H": float(np.mean(h_top)),
                         "median_rank_of_best_gt_h_item": float(np.median(best_rank)),
                         "share_users_best_gt_h_in_top20": float(np.mean(np.array(best_rank) <= 20))}
        out[task] = res
    # ID coverage of H in the cold condition: an H video whose author and music are also OOV is
    # encoded from metadata (+ music_type) alone; in training, ID dropout (p = 0.1 per slot)
    # removes all three IDs of a video at once with probability 0.1^3 only.
    from kuairand_rec.data import catalog as catm
    from kuairand_rec.data.pipeline import load_processed
    from kuairand_rec.models import data as D
    P = load_processed()
    cond = D.build_condition({"name": "cold_u0", "exclude_video_ids": H.tolist()}, P)
    vc = cond.video_catalog.set_index("video_id").loc[H]
    a_oov, m_oov = vc.author_idx == catm.OOV, vc.music_idx == catm.OOV
    out["h_id_coverage_in_cold_condition"] = {
        "n_h": int(len(H)), "author_oov": int(a_oov.sum()), "music_oov_nonzero": int(m_oov.sum()),
        "music_zero": int((vc.music_idx == catm.NO_MUSIC).sum()),
        "video_author_music_all_oov": int((a_oov & m_oov).sum()), "share_all_oov": float((a_oov & m_oov).mean()),
        "train_prob_all_three_ids_dropped": 0.1 ** 3}
    return out


def e5_music_vs_video() -> dict:
    v, m = per_user("u0", "video_long_view"), per_user("u0", "music_long_view")
    j = v[["user_id", "recall@20"]].merge(m[["user_id", "recall@20"]], on="user_id", suffixes=("_video", "_music"))
    return {"users_in_both": int(len(j)),
            "pearson_r_u0_recall20_video_lv_vs_music_lv": float(np.corrcoef(j["recall@20_video"], j["recall@20_music"])[0, 1])}


def render(doc: dict) -> str:
    L = ["# Error analysis (descriptive, post-hoc; never used for selection)", "",
         "Generated by `uv run python scripts/error_analysis.py` from the formal per-user tables and exports. "
         "Paired user bootstrap, 1000 resamples, seed 0, 95% percentile CI. R@20 throughout.", ""]
    L += ["## E1 Like tasks by pure-ALS signal (a prior like before T_test)", "",
          "| task | subgroup | a - b | users | a | b | diff [95% CI] |", "|---|---|---|---:|---:|---:|---|"]
    for t, d in doc["E1"].items():
        for g, rows in d.items():
            for r in rows:
                L.append(f"| {t} | {g} | {r['model_a']} - {r['model_b']} | {r['n_users']:,} | {r['mean_a']:.4f} | "
                         f"{r['mean_b']:.4f} | {r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] |")
    L += ["", "## E2 Long view, >20 bucket, by truncation of the model input", "",
          "| task | full history | users | U0 | ALS | U0 - ALS [95% CI] |", "|---|---|---:|---:|---:|---|"]
    for t, d in doc["E2"].items():
        for g, rows in d.items():
            r = rows[0]
            L.append(f"| {t} | {g} | {r['n_users']:,} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | "
                     f"{r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] |")
    L += ["", "## E3 Cold users (no history before T_test), video / long view", "",
          "| model | distinct user embeddings | Spearman(score, train LV popularity) | top-20 in train top-100 |",
          "|---|---:|---:|---:|"]
    for k, v in doc["E3"].items():
        if k == "training_exposure":
            continue
        L.append(f"| {k} | {v['distinct_user_embeddings']} / {v['cold_users']} | "
                 f"{v['spearman_mean_cold_score_vs_train_lv_popularity']:.3f} | {v['top20_of_mean_cold_user_in_train_top100_popular']} |")
    L += ["", "Training exposure: " + json.dumps(doc["E3"]["training_exposure"]), ""]
    L += ["## E4 Cold-video condition, full catalog", "", f"H share of the catalog: {doc['E4']['h_share_of_catalog']:.3f}", "",
          "| task | model | users | mean share of top-20 slots on H | median rank of best GT_H item | users with best GT_H in top 20 |",
          "|---|---|---:|---:|---:|---:|"]
    for t in ("video_long_view", "video_is_like"):
        for k, v in doc["E4"][t].items():
            L.append(f"| {t} | {k} | {v['users']:,} | {v['mean_share_of_top20_slots_on_H']:.4f} | "
                     f"{v['median_rank_of_best_gt_h_item']:.0f} | {v['share_users_best_gt_h_in_top20']:.4f} |")
    L += ["", "ID coverage of H in the cold condition: " + json.dumps(doc["E4"]["h_id_coverage_in_cold_condition"]), ""]
    L += ["", "## E5 Music vs video", "", json.dumps(doc["E5"]), ""]
    return "\n".join(L) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {"inputs": {"runs": RUNS, "baselines": str(BASE.relative_to(ROOT))},
           "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "ci": CI},
           "E1": e1_like_signal(), "E2": e2_truncation(), "E3": e3_cold_users(), "E4": e4_cold_video(),
           "E5": e5_music_vs_video()}
    (OUT / "error_analysis.json").write_text(json.dumps(doc, indent=1, default=float))
    (OUT / "ERROR_ANALYSIS.md").write_text(render(doc))
    print(render(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
