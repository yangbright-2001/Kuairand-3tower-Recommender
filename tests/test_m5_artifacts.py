"""M5 invariants on the real M2 artifacts and committed M5 outputs (skipped if not built)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from kuairand_rec import paths
from kuairand_rec.experiments import cold_video as C
from kuairand_rec.experiments import sources as S

needs_data = pytest.mark.skipif(not (paths.PROCESSED_DIR / "meta.json").exists(),
                                reason="run `python -m kuairand_rec.data.pipeline` first")
CV = yaml.safe_load((paths.PROJECT_ROOT / "configs" / "cold_video.yaml").read_text())
H_FILE, SEL = S.resolve(CV["h_file"]), S.resolve(CV["selection_json"])
needs_h = pytest.mark.skipif(not (H_FILE.exists() and SEL.exists()), reason="run cold_video select first")


@pytest.fixture(scope="module")
def P():
    from kuairand_rec.data.pipeline import load_processed
    return load_processed()


@needs_data
@needs_h
def test_committed_h_is_reproducible_from_train_only(P):
    from kuairand_rec.eval.views import train_view
    H, tab = C.select_holdout(train_view(P["events"]), C.SelectionConfig.from_dict(CV["selection"]))
    assert np.array_equal(H, C.read_h(H_FILE))
    sel = json.loads(SEL.read_text())
    assert sel["h_sha256"] == C.h_sha256(H) and sel["h_ids"] == H.tolist() and sel["n_h"] == len(H)
    # ~10% per popularity stratum, all train-seen.
    assert all(s["n_selected"] == round(0.1 * s["n_videos"]) for s in sel["strata"])
    assert set(H) <= set(P["events"].loc[P["events"].split == 0, "video_id"])
    assert all(sel["removal_checks"].values())


@needs_data
@needs_h
def test_real_cold_frames_match_selection_record_and_m3(P):
    from kuairand_rec.data.history import HistoryIndex
    from kuairand_rec.eval import frame as Fr
    H = C.read_h(H_FILE)
    sel = json.loads(SEL.read_text())
    ref = json.loads((paths.RESULTS_DIR / "baselines" / "frames_summary.json").read_text())
    hidx = HistoryIndex(P["events"])
    for act in ("long_view", "is_like"):
        task = f"video_{act}"
        base = Fr.build_frames(P["events"], P["label_conflict_events"], P["catalogs"], P["meta"], hidx,
                               "test", "video", act)["masked"]
        assert base.frame_id == ref[f"test/{task}/masked"]["frame_id"]           # unfiltered M3 frame
        cf = C.build_cold_frames(base, H, P["catalogs"],
                                 C.non_h_history_length(P["events"], H, base.users, base.t_pred))
        for k in C.FRAME_KINDS:
            assert cf[k].frame_id == sel["cold_frames"][task][k]["frame_id"]
            assert set(cf[k].users) <= set(base.users)
        hrows = P["catalogs"].video_row(H)
        assert np.isin(cf["cold_video_full"].gt_idx, hrows).all()
        assert (cf["non_h_hist_len"] >= 1).all()


@needs_data
@needs_h
def test_real_cold_history_metadata_and_selection_frames(P):
    """Cold frame history = the cold model's frozen T_test input; buckets partition the population;
    cold selection frames (b1) have no H GT, no empty users, and the normal val frames are unchanged."""
    from kuairand_rec.data.history import HistoryIndex
    from kuairand_rec.models import data as D
    from kuairand_rec.models import evaluation as E
    from kuairand_rec.models import features as F
    H = C.read_h(H_FILE)
    sel = json.loads(SEL.read_text())
    cold = D.build_condition({"name": CV["condition_name"], "exclude_video_ids_file": str(H_FILE)}, P)
    hidx = HistoryIndex(P["events"])
    frames, md, extra = C.cold_frames_metadata(P, H, cold, hidx)
    hc, t = F.frozen_history_index(cold.events, "test", P["meta"])
    for (task, k), f in frames.items():
        assert f.frame_id == sel["cold_frames"][task][k]["frame_id"]
        _, ln = hc.model_history(f.users, t, 100)
        assert np.array_equal(f.hist_len_capped, ln) and np.array_equal(f.hist_len_full, hc.history_length(f.users, t))
        bc = md[task][k]["bucket_counts_non_h_history"]
        assert sum(bc.values()) == len(f.users) and bc["0"] == 0
    # Label conflict (13532, 6266) is on H: recorded as an upper bound, never added to GT.
    lca = md["video_long_view"]["cold_video_full"]["label_conflict_h"]
    assert lca["events"] == 1 and lca["details"][0]["video_id"] == 6266
    assert md["video_long_view"]["cold_video_full"]["gt_h_pairs"] == sel["cold_frames"]["video_long_view"]["cold_video_full"]["gt_h_pairs"]
    # Selection frames: no H video GT; mask = unfiltered seen rows; ids recorded.
    sf, smeta = E.build_cold_selection_frames(P, cold, hidx)
    assert smeta == sel["cold_val_selection_frames"]
    hrows = P["catalogs"].video_row(H)
    hc_val, tv = F.frozen_history_index(cold.events, "val", P["meta"])
    for (s, task, k), f in sf.items():
        if task.startswith("video"):
            assert not np.isin(f.gt_idx, hrows).any()
        assert (np.diff(f.gt_indptr) >= 1).all() and (hc_val.history_length(f.users, tv) >= 1).all()
        ip, rows = hidx.seen_rows(f.users, tv, task.split("_")[0])
        assert np.array_equal(f.mask_indptr, ip) and np.array_equal(f.mask_idx, rows)
    # Normal val selection frames: bitwise the M3 ones (build_eval_frames asserts against M3).
    normal = E.build_eval_frames(P, ("val",), ("masked",))
    ref = json.loads((paths.RESULTS_DIR / "baselines" / "frames_summary.json").read_text())
    assert all(f.frame_id == ref[f"val/{k[1]}/masked"]["frame_id"] for k, f in normal.items())


@needs_data
@needs_h
def test_real_cold_condition_vocab_and_scalers_refit(P):
    from kuairand_rec.models import data as D
    H = C.read_h(H_FILE)
    cold = D.build_condition({"name": CV["condition_name"], "exclude_video_ids_file": str(H_FILE)}, P)
    assert cold.condition_id == json.loads(SEL.read_text())["cold_condition_id"]
    assert len(cold.vocabs.video.ids) == len(P["vocabs"].video.ids) - len(H)
    assert cold.num_stats != P["meta"]["numeric_video_stats_train"]                # refitted scalers
    assert not cold.events.video_id.isin(H).any() and P["events"].video_id.isin(H).any()


MAIN = paths.RESULTS_DIR / "m5" / "main"


@pytest.mark.skipif(not (MAIN / "analysis_meta.json").exists(), reason="run experiments.analysis first")
def test_main_analysis_outputs_are_consistent():
    meta = json.loads((MAIN / "analysis_meta.json").read_text())
    ref = json.loads((paths.RESULTS_DIR / "baselines" / "frames_summary.json").read_text())
    m = pd.read_csv(MAIN / "bootstrap_models.csv")
    for (task, kind), g in m.groupby(["task", "frame_kind"]):
        assert g.frame_id.nunique() == 1 and g.frame_id.iloc[0] == ref[f"test/{task}/{kind}"]["frame_id"]
        allr = g[g.stratum == "all"]
        assert (allr.n_users == ref[f"test/{task}/{kind}"]["n_users"]).all()
    for v in meta["inputs"].values():
        assert v["metrics_csv_max_abs_dev"] <= 1e-12
    for v in meta["threetower_provenance"].values():
        assert v["parity_ok"] and len(v["git_sha"]) == 40 and not v["git_sha"].endswith("dirty")
    b = pd.read_csv(MAIN / "bucket_counts.csv")
    for task, g in b.groupby("task"):
        assert g.n_users.sum() == ref[f"test/{task}/masked"]["n_users"]
    assert meta["n_boot"] == 1000 and meta["seed"] == 0
