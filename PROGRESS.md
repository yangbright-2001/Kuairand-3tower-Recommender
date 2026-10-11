# Project Progress

## Current Status

- Stage: **M3 implemented and run, awaiting user review** (2026-10-11). M2 signed off. P1 is in `PLAN.md` §3b. M3 decisions are in §3c.
- Next: user review of M3. D1/M4/M5 are not started and wait for the user.
- Read first: `PLAN.md` (workflow, the P1 decisions in §3b, OPEN items), `results/audit/AUDIT.md` (audit numbers), `docs/assignment.md`.

## Completed

- Workflow agreed with the user and recorded in `PLAN.md`.
- M0 (local):
  - uv 0.13.0 is in `~/.local/bin`. Run `source ~/.local/bin/env` in a new shell. Python 3.12.15.
  - `pyproject.toml` + `uv.lock`: torch 2.14.1, implicit 0.7.3, pandas 3.0.6, numpy 2.5.4, matplotlib.
  - `.gitignore` covers data, artifacts, and secrets.
- Audit Phase A: `scripts/audit_phase_a*.py`. Superseded by Phase C and kept for provenance; their known counting issues are documented in the file headers.
- Audit Phase C:
  - Code:
    - `src/kuairand_rec/data/io.py`: loaders. They refuse the statistic file and the random log, and assert unique metadata keys.
    - `src/kuairand_rec/audit/stats.py`: pure counting functions, unit-tested.
    - `run.py`, `figures.py`, `report.py`.
  - Command: `uv run python -m kuairand_rec.audit.run`. It reads only the 4 agreed CSVs, applies no filtering, and takes about 26 s on the M1.
  - Outputs in `results/audit/`:
    - `AUDIT.md`, `audit.json`, `column_profile.csv`, `split_candidates.csv`, `daily_volume.csv`
    - 4 figures in `figures/`: music sharing, daily volume with split candidates, history length, and action by tab.
  - Split-dependent numbers use a *provisional reference split* (raw-row quantile). It is for the audit only and is not a decision.
- Audit review fixes (2026-10-11, after user review of Phase C):
  - **Music catalog bug.** In `audit_repeats_and_catalog()`, `all_metadata_music_`* was built from music ids found in the logs (7,172) instead of the metadata table (7,202). 30 metadata music ids never occur in the logs.
    - Fix: added `stats.build_catalogs()` with three nested levels, `metadata ⊇ train_seen ⊇ train_positive`, plus `incl_0` and `nonzero` music variants, and `stats.ooc_share()`.
    - Renamed the JSON keys to `metadata` / `train_seen` / `train_positive`.
    - Added `catalog_definitions` and a music-catalog table to `AUDIT.md` §6. That table shows OOC share both over all test positive rows and over rows with `music_id != 0`.
  - **Cross-checks.** `main()` now returns 1 and prints the failed checks to stderr if any check fails. Outputs are still written so the failure can be inspected. `audit.json` gets `meta.failed_checks`, and `AUDIT.md` shows a warning banner on failure.
    - Added 7 checks: catalog nesting per action × {video, music incl_0, music nonzero}, and "metadata music catalog = metadata table".
  - **Figures.**
    - `history_length.png`: the x-axis is pinned at 0 with explicit non-negative ticks. The symlog axis had been padding into negative values.
    - `action_by_tab.png`: now 3 stacked panels (long_view %, is_like %, rows on a log scale), each on its own true-percentage axis with an overall-rate reference line. The ×10 scaling is gone.
  - The ×10 like-rate overlay in `daily_volume_splits.png` was out of scope and left unchanged.
- P1 Experimental Protocol Review (2026-10-11): two rounds of decisions plus three rounds of revisions, all approved by the user and recorded in `PLAN.md` §3b. Highlights:
  - Protocol A, frozen at T_test (val history used as input only). ALS is folded in from train+val. Pop uses train only.
  - H1 history with a cap of 100. The M2 mask uses all pre-t_pred interactions.
  - Midnight cut points. Exact duplicates collapsed, and label-conflict events dropped.
  - Full metadata catalogs. `music_id=0` is excluded as a music target.
  - Explicit no-untrained-parameter rules for train-unseen items and users.
  - U0 is the main user tower and U1 is a required experiment. "usable metadata" in the assignment is not defined; our time-safety reading is our interpretation.
  - The cold-video experiment uses one fixed H and one fixed evaluation frame, with a per-user random reference and the Seen-video Reference labelled as a non-bound.
  - Evidence used for P1-8 (read-only check, not in the audit outputs): user 25929 has `register_days=22` but logs from 04-10, so the snapshot is no later than 05-02. No lower bound exists.
- M2 data pipeline (2026-10-11), following `PLAN.md` §3b with no protocol changes:
  - Command: `uv run python -m kuairand_rec.data.pipeline --config configs/data.yaml` (about 12 s on the M1; deterministic, parquet md5 identical across reruns).
  - Modules in `src/kuairand_rec/data/`:
    - `clean.py`: `clean_logs()` (exact dedup → label-conflict events dropped → max-play representative row), `mark_missing_duration()`.
    - `split.py`: `local_to_epoch_ms()`, `assign_split()` (t < T earlier split), `split_summary()`.
    - `catalog.py`: `build_catalogs()` (metadata video/music catalogs, music 0 → no row), `build_vocabs()` (train-only; PAD=0, OOV=1, music NO_MUSIC=2; `exclude_video_ids` for cold H), `encode_video_features()`.
    - `history.py`: `HistoryIndex` with `window()`, `model_history(max_len=100)`, `gather()`, `seen_rows()` (uncapped mask, CSR), `history_length()`.
    - `train_stats.py`: `popularity()` (video + music = sum over videos), `numeric_video_stats()`, `remove_videos()` (cold-video hook). Both fitters refuse non-train rows.
    - `pipeline.py`: `run()`, `load_processed()`, `prediction_time()`.
  - Outputs: `data/processed/` (git-ignored; events, catalogs, vocabs, popularity, label_conflict_events, meta.json) and `results/data_pipeline/{DATA_REPORT.md, data_report.json}` (small, committable).
  - Implementation choices within P1 (flag for review): representative-row tie-break after max `play_time_ms` = remaining value columns ascending, then `src_file`; video metadata numerics = log1p + train-video z-score with a missing flag; categorical metadata (incl. tags, max 3) also use train-only vocabs; history tokens carry `video_idx` + feedback columns from `configs/data.yaml`.



## Verified Results

- M2 (real data, midnight split):
  - Cleaning: 1,436,609 rows → 1,413,134 events. 21,987 exact redundant rows, 1,486 conflict groups (1,484 collapsed, 2 label conflicts dropped = 4 rows, both long_view, both also play_time conflicts). Positive events: long_view 474,552, is_like 26,361. Users 27,077 and videos 7,551 unchanged. 28,859 `duration_ms=0` events kept with duration missing.
  - Split (events): train 997,078 (70.6%) / val 208,393 (14.7%) / test 207,663 (14.7%). 0 events on a cut; no timestamp spans splits.
  - Positives: long_view 339,386 / 69,354 / 65,812; is_like 18,771 / 3,980 / 3,610.
  - Catalogs: video 7,583, music 7,201 non-zero, 204 videos with music 0, 112 shared music ids. Train vocabs: users 25,734, videos 7,529, authors 6,477, music 7,151.
  - Test: 1,215 of 24,887 users are train-OOV (571 have no history at T_test); 17 videos are train-unseen (4 long_view / 0 like positive events). Music-sentinel test positives: long_view 1,670, like 67.
  - 8/8 pipeline checks pass.
- `uv run pytest`: 51 passed after the review (22 audit + 24 in `tests/test_data_pipeline.py` on hand-built data + 5 in `tests/test_data_artifacts.py` on the real artifacts, skipped if not built). 10/10 pipeline checks pass.
- `uv run pytest`: 22 passed. The original 15 cover the env, loader guards, duplicate and conflict classification (NaN-aware), same-timestamp counts, strict-before history counts, split tie handling, history buckets, and the official long_view rule. The 7 new ones are in `tests/test_audit_catalog_and_checks.py`:
  - the metadata catalog includes ids never seen in the logs
  - excluding the sentinel removes it from all three catalog levels
  - the catalogs are nested
  - `ooc_share`
  - the exit code is 0 or 1 depending on the checks
  - `Checks.failed`
- Audit internal cross-checks: 30/30 pass, exit code 0.
- Re-run vs the previous `audit.json`: every non-catalog value is identical. Only `catalog_ooc` changed:
  - The music metadata catalog grew from 7,172 to 7,202 ids including 0 (7,171 to 7,201 nonzero).
  - Its test OOC shares did not change, because the 30 added ids never occur in the logs.
  - Video catalogs only had their keys renamed. Their values are unchanged.
- Key facts (raw rows; full detail in `AUDIT.md`):
  - Size: 1,436,609 rows, 27,077 users, 7,551 videos. Metadata keys are unique, and joins preserve row counts.
  - Duplicates and conflicts, keyed on (user, video, time_ms):
    - 21,906 exact-duplicate groups (43,884 rows, 21,987 redundant), mostly in tabs 5 and 6.
    - 1,486 conflict groups, which differ mostly in `hourmin`.
  - Same user, same timestamp, multiple videos: 68,683 groups, 143,591 rows (10.0%).
  - Genuine repeats (same pair at distinct times): 34,655 pairs. Repeated positives are about 0.
  - Actions:
    - long_view 33.2% and like 1.85% of rows. Only 6,768 users have at least one like.
    - The official long_view rule matches 99.76% of rows with duration > 0. The 3,420 mismatches are all videos of 18 s or less, short by at most about 1 s.
  - Music:
    - Metadata has 7,202 music ids including 0. 30 of them never occur in the standard logs.
    - `music_id = 0` covers 204 videos.
    - Non-zero music is nearly 1:1 with videos: 112 IDs shared, by at most 15 videos.
  - Reference split:
    - Cuts: 04-16 22:07 and 04-25 21:38 CST.
    - Spans: train 7.4 days, val 9.0, test 13.1.
    - Test positive rows: 66,946 long_view and 3,686 like.
  - About 1% of test positive pairs were seen in train, and 0% were train positives.
  - Cold-start pool: 4,433 videos. In that pool, 22.9% of authors and 3.9% of music IDs have another train video.
- Unverified hypotheses (H1-H5 in `AUDIT.md` and the chat):
  - `music_id = 0` is a missing-music sentinel.
  - A labelling tolerance of about 1 s on long_view.
  - The tab 5 and 6 duplicates are logging artifacts.
  - Same-timestamp multi-video groups come from the double-column UI.
  - User snapshot counts may leak future information. P1-8 treats this as unresolvable, so user snapshot features are kept out of the main model U0.



## Issues / To Fix

- Nothing is committed to git yet. Phase C code, the review fixes, and `results/audit/` are uncommitted. They should be committed after the user approves.
- Minor: in `report.py`, the "Constant columns" line is hard-coded. It should be derived from `column_profile.csv`. Not yet fixed.
- Minor: `daily_volume_splits.png` still overlays the like rate ×10. It is a candidate for the same treatment as the action-by-tab figure, if the user wants.
- Runtime note: inside the Cursor sandbox `~/.matplotlib` is not writable, so matplotlib rebuilds its font cache on every run. The audit took about 118 s instead of about 26 s. Setting `MPLCONFIGDIR` to a writable directory avoids this.
- Server: the SSH alias `gpu228-liang` is not configured on the Mac. The user is setting it up, and it does not block local work. Still to verify: uv/torch CUDA 13, `implicit`, the Zenodo download with checksum, and Slurm vs direct GPU use.



## P1 Decisions

- Approved 2026-10-11. See `PLAN.md` §3b. They are not repeated here.



## M2 correctness review (2026-10-11)

- Independent recomputation on real data (not via the pipeline code), all agree:
  - Cleaning: same 1,413,134 keys; every kept row is a whole raw row, has the max `play_time_ms` of its key, and has the key's unique labels. The 2 dropped label-conflict events are (21537, 7358, train) and (13532, 6266, test); neither pair has any other event.
  - Cut literals: T_val = 2022-04-16 16:00 UTC, T_test = 2022-04-25 16:00 UTC.
  - History + mask brute force for 300 random users at T_val and T_test: model input = the last 100 of the (time_ms, video_id)-sorted events with t < T, video and music masks = all distinct pre-T rows. 0 mismatches.
  - Encoding: no train-seen video has an OOV video/author/music/metadata/tag index. The 54 train-unseen catalog videos are video-OOV (35 author-OOV, 50 music-OOV). Event-level and catalog-level indices agree. The 204 music-0 videos all have NO_MUSIC and music_row -1.
- Fixed (minimal, with tests):
  - [Medium] Label-conflict events were dropped without being saved, so the evaluator could not count "GT dropped as label conflicts" (P1-1). Now `clean_logs(return_dropped=True)` returns them, and they are saved to `label_conflict_events.parquet` with split and `{action}_any`. They are for accounting only. Real data: 1 in train, 1 in test (long_view).
  - [Medium] Cold-video refit risk: `events.parquet` / `video_catalog.parquet` carry normal-condition `*_idx` and popularity. `remove_videos()` alone keeps those stale indices, and `build_vocabs(exclude_video_ids=...)` alone leaves popularity and numeric stats unrefitted. Added `pipeline.fit_condition(events, meta, cat, actions, exclude_video_ids)`. It removes H first, then refits vocabs, numeric stats, popularity, and video/music features, and re-encodes the event indices. The normal pipeline now uses the same function. There is a warning in `load_processed`.
  - [Low] `_assert_train` let a frame without a `split` column through silently. It now raises.
  - [Low] `_lookup` crashed on an empty vocab. It now returns all-missing (OOV).
  - [Medium, from an extra independent review] `build_vocabs()` did not verify that its input was train-only. Train+val/test events would silently add future user/video/author/music ids. Added `catalog.assert_train_only()`: it raises `ValueError` if the `split` column is missing or any row has `split != 0`. `build_vocabs()` calls it, and `train_stats` now reuses the same guard. Call sites checked: `pipeline.fit_condition()` passes `ev[ev.split == 0]`, after H removal on the cold path. Tests call it the same way. New test `test_build_vocabs_train_only_guard`: missing split, val/test rows, and valid train input (incl. exclude) unchanged. Rerun: `events.parquet` is byte-identical, vocab sizes are unchanged (user 25,734, video 7,529, author 6,477, music 7,151), and the pipeline checks pass 10/10.
- Known low-priority items (recorded, not fixed this round):
  - `numeric_video_stats()` returns NaN mean/std if a numeric column is entirely missing over train videos. That is not the case on real data, but it could happen in a tiny cold subset.
  - `HistoryIndex` methods do not handle a completely empty events DataFrame (e.g. `seen_rows` uses `item.max()`).
  - [Low] `music_catalog.parquet` only had ids. It now adds `music_idx`, `music_type_idx` (music_type is checked to be constant per music id), and `n_videos_metadata`, for the music tower and export.
- Verified not a problem / accepted as-is:
  - The representative choice is deterministic: a full-column sort, row-order invariant (tested).
  - `events.parquet` is byte-identical before and after the fixes.
  - `video_type`/`upload_type`/`upload_dt` vocabs are fitted on train-seen videos. The 54 unseen videos have no metadata OOV.
- Noted for D1 (not M2 bugs):
  - Feedback timing: history tokens carry `play_time_ms` and labels of past events, which are only fully known after playback ends. For 0.3% of train targets (3,104), the previous event's `time_ms + play_time_ms` exceeds the target time. At the frozen cuts, only 15 (T_val) and 2 (T_test) events are unfinished. This is standard practice but should be stated in the report; D1 can decide whether it matters.
  - `numeric_video_stats` weights each train-seen video equally (not by interaction count). This is fine, just document it.
  - `user_features` is loaded only for a count. U1 encoding (train-user vocab for one-hot fields) is a D1/M4 task.
  - 208 metadata users never appear in the logs. They are irrelevant to evaluation.



## M3 shared evaluator + Pop/ALS (2026-10-11)

- Design: `.cursor/plans/m3_评估器与基线_fd18bcec.plan.md` (Revision 1). Decisions are recorded in `PLAN.md` §3c, including O1 (`als_like_pure` is the headline) and O2 (aux_w = 0.3 fixed). P1 is unchanged.
- Command: `uv run python -m kuairand_rec.baselines.run --config configs/baselines.yaml`. It takes about 9.8 min on the M1, mostly 48 ALS fit+val evals at about 10 s each.
- New modules:
  - `eval/views.py`: `train_view`, and `history_view` with split/time boundary asserts.
  - `eval/frame.py`: `build_frames` produces masked / nomask / cold frames, with counts, `population_id`, and `frame_id`.
  - `eval/metrics.py`: stable full sort, a -inf validity flag, and Recall/NDCG.
  - `eval/evaluator.py`: the `Scorer` protocol (score / signal / item_scorable), `evaluate`, sanity checks, and `summarize`.
  - `baselines/popularity.py`, `baselines/als.py`, `baselines/run.py`, `configs/baselines.yaml`.
- `implicit` 0.7.3 semantics were verified numerically. A stored CSR value d is used directly as C (p = 1). So `C = 1 + α·r` is stored with the library alpha set to 1. Our closed-form fold-in matches `recalculate_user` (atol 1e-4). The version is pinned by a test.
- Outputs in `results/baselines/`:
  - `BASELINES.md`, `metrics.csv`, `als_tuning.csv`, `frames_summary.json`, `run_meta.json`. These include the selected configs, train fingerprints, and fold-in dropped-pair stats.
  - `frames/*.parquet` and `per_user/{model}/*.parquet` are git-ignored.
- Runtime checks in the runner:
  - Pop counts equal the M2 `popularity_*.parquet`.
  - One `frame_id` per (split, task, kind) across all models.
  - masked and nomask share a `population_id`.
  - Every per-user table: values in [0,1], Recall monotone in K, no hits for no-signal users, scorable items never -inf for users with signal.
- Tests: `uv run pytest` gives **90 passed** (51 M2/audit + 39 new M3):
  - `test_eval_metrics.py`: hand-computed metrics.
  - `test_eval_frame.py`: the hand-built fixture.
  - `test_baselines.py`: ALS semantics, mapping, and leakage.
  - `test_eval_artifacts.py` (real data):
    - an independent pandas recomputation of the masked GT
    - an oracle scorer
    - results-file consistency
- Determinism: a second full run gives byte-identical `metrics.csv` and per-user parquet files.
- Results (test, masked; R@10/20/50 | N@10/20/50):
  - video/long_view (18,752 users):
    - pop .0479/.0806/.1388 | .0348/.0457/.0621
    - als .0759/.1244/.2282 | .0524/.0686/.0969
    - als_nofoldin .0654/.1094/.2026
  - video/is_like (1,924):
    - pop .0466/.0716/.1304 | .0332/.0403/.0532
    - als (pure) .0388/.0555/.1004 | .0234/.0282/.0387
    - als_like_aux .0610/.1008/.1795 | .0356/.0469/.0648
  - music/long_view (18,595):
    - pop .0507/.0856/.1472 | .0362/.0479/.0651
    - als .0772/.1282/.2297 | .0536/.0705/.0982
  - music/is_like (1,895):
    - pop .0480/.0726/.1358 | .0342/.0412/.0552
    - als (pure) .0433/.0586/.0951 | .0256/.0303/.0389
    - als_like_aux .0617/.1054/.2017 | .0338/.0463/.0677
- Anomalies, explained:
  - Pure like ALS is below Pop. 608/1,924 (video) and 606/1,895 (music) evaluated users have no like before T_test, so they are no-signal (strict, Q2) and score 0.
    - On signal users only (descriptive), ALS R@20 is 0.081 vs Pop 0.073 (video) and 0.086 vs 0.075 (music).
    - The aux variant reduces no-signal users to 34 and beats Pop.
  - Pop does much better on cold users (R@20 about 0.14-0.22) than on the main population. New users mostly watch popular items.
  - nomask is below masked for ALS (for example, video/long_view R@20 is 0.124 masked vs 0.074 nomask). ALS ranks already-seen items high, and few of them are re-positive.

## M3 independent correctness audit (2026-10-11)

- Report: `results/baselines/M3_CORRECTNESS_AUDIT.md`. Evidence: `results/baselines/m3_audit/audit_m3.json`.
- Script: `uv run python scripts/audit_m3_correctness.py`. It takes 163 s and exits 1 on any FAIL. The oracle uses pandas/numpy/scipy/implicit only and does not call kuairand_rec eval or baselines code.
- Result: 25/25 checks pass. **No correctness bug found, and no artifact or protocol changed.**
  - All 24 frames: counts, ids and per-user tables, 0 diffs.
  - All 46 per-user files (349,955 user rows): max |Δ| 2.2e-16.
  - ALS: Y, X_train and fold-in are bitwise equal to the oracle. Fold-in matches `recalculate_user` to ≤ 9.1e-5. C = stored value is confirmed for implicit 0.7.3.
  - Negative controls detect every applicable injected bug.
- New descriptive caveat: Pop like-task metrics move by about ±0.001 under a random tie-break. For music/is_like R@20 the official 0.0726 vs random 0.0728–0.0734.

## M3 open items / correctness risks (for user review)

- ALS selection sits on a grid edge: factors = 64 (the smaller value) was selected for every pure task, and val Recall drops at 128. A smaller factor count might do better, but the Q5 grid is fixed. Expanding it is the optional todo.
- No-fold-in factors are half an iteration older than Y (an `implicit` user-then-item update). This is documented and not corrected.
- `gt_unscorable` counts GT on items with no train signal for that config: 191 and 310 pairs for pure video long_view/like. They stay in the denominator. Pop has 0 by definition (P1-6b).
- `label_conflict_possible_gt_pairs` is an upper bound: 1 pair for long_view test, 0 for like.
- Pop ranks every zero-count item by catalog row. That only affects users who exhaust all positive-count items within K, which is negligible for a 7.5k catalog with K ≤ 50.
- The val frame is used both for ALS selection and for the val rows in the report. Test was never used for selection.

## M2 open items (not blocking; for user review)

- Cold-video H selection is not implemented (M5). M2 provides `fit_condition(..., exclude_video_ids=H)`, which is the only supported cold refit path. The "author/music still in cold train" share is computed after H is drawn.
- Training-sample construction (which events are targets, action balancing, rolling vs frozen history) is D1. `HistoryIndex.model_history(users, t_event)` already supports per-event train histories.
- `duration_ms` / `play_time_ms` are per-event log fields; whether they enter history tokens (e.g. bucketised) is a D1 choice. They are saved in `events.parquet`.
- Timing: `seen_rows` (video + music) + `model_history` for all 24,887 test users takes 0.85 s on the M1.



## Next Steps

1. User review of M2. Then M3: shared evaluator + Pop/ALS, reusing `load_processed()`, `prediction_time()`, `HistoryIndex.seen_rows()` and the catalog rows.
2. Tests still to add in M3+ (checklist):
  - The mask never uses test labels (evaluator level).
  - Cold-video interactions are fully removed from every model input and fitted dataset (M5, end to end).
  - ALS scores unseen items as -inf.
  - Score parity covers every catalog row, including unseen and OOV rows.
  - In the cold-video condition, GT, mask, and population are identical to the normal condition.
  - The per-user random-reference expectation matches a simulated random ranking.
  - Hand-computed metric cases (Recall/NDCG, tie-break).
3. D1 (three-tower v1 design review) runs in parallel with M2/M3.

