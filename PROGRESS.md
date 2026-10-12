# Project Progress

## Current Status

- Stage: **M4-3 (validation / selection, export, Direct / Exported scorers, retrieve(), strict parity) implemented and tested, with a CPU end-to-end smoke; awaiting independent review. M4-2 was approved. No GPU training has been run and M5 has not started.**
  - M2 and M3 are signed off, including the M3 correctness audit.
  - P1 is in `PLAN.md` §3b, M3 in §3c, and D1 in §3d.
- Next: start M4 on the user's go. M4 implements `PLAN.md` §3d; the D1-10 minimum deliverable is the acceptance bar.
- Read first:
  - `PLAN.md`: the workflow, §3b/§3c/§3d, and the OPEN items
  - `results/audit/AUDIT.md`
  - `docs/assignment.md`

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

- (Resolved) Everything through M3 has been committed and pushed: `730e643` on `origin/main`. The D1 plan file and these doc updates are not committed yet.
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



## D1 three-tower v1 design (approved 2026-10-11, Revision 2)

- The binding summary is in `PLAN.md` §3d (D1-1 to D1-11). The full design and the R1/R2 revision history are in `.cursor/plans/d1_三塔_v1_设计_1d3a531b.plan.md`. P1 §3b and M3 §3c are unchanged.
- Key decisions:
  - **Loss:** train-seen full softmax with no logQ. A same-user, same-task false-negative mask is applied. In-batch negatives are optional only.
  - **Feedback availability:** fixed delay, D = 20 min, decided by event age only. D is a conservative assumption, not a proven bound, and its residual risks are disclosed.
  - **OOV:** fixed-zero PAD/OOV rows + ID dropout.
  - **U1:** required, using U0's config with no separate tuning.
  - **Parity:** strict, with no near-tie exemption.
- Read-only evidence collected during D1 (real data; no code or artifacts changed):
  - **Train targets:** 345,365 events (18,771 likes); 11,338 have an empty history. Max `play_time_ms` is 965 s.
  - **Time semantics:**
    - Gap ≥ previous play: 99.77%. Gap ≥ next play: 99.41%.
    - Gap − play is within 1 s for only 0.01–0.02% of pairs (median excess about 36 s), because Pure keeps only candidate-pool videos, so sequences are incomplete.
    - 312 likes have `play_time_ms = 0`.
    - So `time_ms` semantics cannot be identified.
  - **D = 20 min hides:** 2.6% of train tokens (42% of targets have at least 1), and a few hundred tokens at T_val / T_test.
  - Official `KuaiRand-Pure.tar.gz` md5: `0820331067a3784d9691136f772b35a7` (from kuairand.com).
- Implementation constraints added at approval:
  - **Cold condition** (D1-5): the cold-video vocabs, scalers, feature mappings, checkpoint and scorer must all come from the same condition, checked via `condition_id`. They are never mixed with normal U0 artifacts.
  - **GPU runs** (D1-11): pinned git SHA with a clean worktree. Outputs go to `RUN_ROOT` outside the clone. Important artifacts are backed up to the Mac after each wave.
  - **Model selection** (D1-8): tuning, early stopping and checkpoint selection use validation only. Test is evaluated after selection is frozen.
- Repo and server:
  - The code source is the private GitHub repo `yangbright-2001/Kuairand-3tower-Recommender`. `main` is at `730e643` (content through M3), in sync with origin.
  - The server will use a read-only Deploy Key generated on the server. The user adds the public key. Personal keys are not used.
  - Fallback: `git archive` + rsync.

## M4-1 features, history, towers (2026-10-11; awaiting review)

- Scope: `models/features.py` and `models/towers.py` only. No loss, training, export or GPU work. No M2/M3 file changed.
- `features.py` consumes M2 outputs as-is: `fit_condition` / `video_catalog` / `music_catalog`, `HistoryIndex.model_history`, `history_view` / `train_view`, `catalog.Vocab`.
  - `build_history_batch(hidx, users, t_pred, HistoryConfig)` is the single builder. It asserts every token has age > 0, and applies the D = 20 min hide rule from (t_pred, t_event) only.
  - `train_history_index` (train events) and `frozen_history_index(split)` (`history_view`) are the only index constructors.
  - U1: `fit_user_feature_vocab` (train users, min_count 5, NaN / −124 as categories, rare + unseen → OOV) and `UserFeatureTable.lookup` (users missing from the snapshot → all OOV).
- `towers.py`: Music / Video / User towers (MLP + LN → 128), 4 + 2 + 2 Linear(128→64) + L2 heads, a shared `MusicTower` instance, a 2-layer pre-norm Transformer (d 128, 4 heads) + learnable-query pooling, and `MaskedEmbedding` (fixed-zero PAD/OOV via a buffer mask; NO_MUSIC learnable). Ablation flags `use_music` / `use_author` / `use_history`.
- Implementation choices within D1 (flagged for review):
  - Recency bucket = 1 + min(floor(log2(1 + age_min)), 13). Real data: rolling train ages reach bucket 13 at most, but frozen T_test ages reach 14. An unclipped bucket 14 would be a never-trained row at test time (P1-6b).
  - play_ratio buckets: row 0 = PAD/hidden (fixed zero), 1 = duration missing, 2 = play 0, then edges .1/.25/.5/.75/1/2.
  - Hand-written attention block instead of `nn.TransformerEncoderLayer`, so its eval fast path cannot change the numerics between training, direct scoring and export.
  - Item ID dropout is drawn per vocab entry once per catalog forward, so one draw is shared by candidates, history tokens and the video's music input. User ID dropout is per row.
  - Dims: id 64, meta 16, U1 field 16, MLP hidden 256, FF 512, dropout 0.1. U0 has 3.77M params, U1 3.88M.
- Tests: 34 new (`test_m4_features.py` L1-L5, buckets, U1; `test_m4_towers.py` OOV, gradients, stability, U0/U1; `test_m4_artifacts.py` real data). `uv run pytest`: **124 passed** (90 before), about 10 s.
- Real-data check (read-only): 345,365 train targets, rolling histories built in 8.3 s, 11,338 empty. 2.64% of tokens hidden; 42.2% of targets have ≥ 1 hidden. Hidden tokens: 819 at T_val, 247 at T_test. These match D1. U1 values kept: feat3 644 / 1,419, feat8 349 / 451. A U0 eval forward on 512 users over the full catalogs takes 0.27 s on CPU, all finite.

### M4-1 review fix (2026-10-11; awaiting confirmation)

- Bug (found by the user's review): `VideoTower` called `MusicTower` a second time. With dropout in train mode, a video's music input therefore differed from its music candidate's `m_base`, and videos sharing a music drew different samples. The two towers shared parameters but not the forward output.
- Fix:
  - `ThreeTowerModel._music_forward` calls `MusicTower` exactly once per catalog forward. Its inputs are the music catalog rows plus `(NO_MUSIC, t)` for each distinct music_type t of the music_id-0 videos.
  - The video tower now receives the per-video music base, gathered by `VideoFeatures.music_row`. The new field comes straight from `video_catalog.music_row`.
  - music_id 0 keeps the learnable NO_MUSIC row and its own music_type. It is never mapped to a catalog row or the OOV row, and it is not a music candidate.
  - A guard checks that a video's music inputs equal those of its catalog row.
  - `ItemEncodings.video_music_base` exposes the gathered tensor.
  - Eval-mode outputs are unchanged.
- Real data: 204 music_id-0 videos, music_type_idx 7 (203) / 6 (1), so 2 NO_MUSIC inputs per forward.
- Tests: +6 in `test_m4_towers.py` (extended fixture with 3 music_id-0 videos over 2 music types):
  - train mode with dropout 0.5: bitwise equality with the candidate and exactly one MusicTower call
  - shared music → one sample
  - video loss → gradients reach the music id / NO_MUSIC rows
  - `use_music=False`
  - OOV music and NO_MUSIC paths
  - the guard rejects inconsistent inputs

  Negative control: re-introducing the double call makes the first two tests fail. Max difference under the old code: 0.61.
- Recency coverage on real data: rows 1-14 are all used by train-target histories (smallest count 10,051). The frozen val / test histories use rows 1-14, all seen in train, whether built for split users or all users. No fix needed. Added `test_frozen_recency_buckets_all_occur_in_train_target_histories`.
- `uv run pytest`: **131 passed**, 0 skipped. No M2/M3/P1/D1 file or protocol changed.

## M4-2 loss, training data, trainer (2026-10-11; awaiting review)

- Scope: `models/loss.py`, `models/data.py`, `models/train.py`, `configs/threetower_smoke.yaml`. No export, retrieve, val selection or GPU run. No P1/M3/D1 protocol, M2/M3 code or baseline result changed. `tests/m4_fixtures.make` gained an `exclude_video_ids` passthrough to `fit_condition`.
- `loss.py`: `masked_logits` / `task_loss` / `multitask_loss`. cos/τ (τ 0.05) over the train-vocab candidate columns; FN pairs set to −inf; the label column is refused as an FN column and its logit is asserted finite; per-task mean over valid rows; equal-weight sum; empty task → skipped (None), never 0/0.
- `data.py`:
  - `TrainCondition` (one condition per artifact set): candidates = catalog rows whose id is in the train vocab (7,529 video / 7,151 music on real data), row↔column maps, and `condition_id` = sha256(name, sorted H). It asserts H is absent from the events, the vocab and the candidates. `normal_condition(P)` / `condition_from_fit(name, fit, H)`.
  - `build_samples`: train events positive for any action (345,365; 18,771 likes); 4 task masks, and the music tasks require music ≠ 0.
  - `PosTrain`: per-task CSR of distinct candidate columns of each user's train positives (music = the non-zero music of those videos, so shared music is merged). Built from the base samples only. `fn_pairs` drops the row's own label.
  - `oversample_likes(k)`: k−1 copies per like sample, with the long_view masks off.
  - `BatchBuilder`: rolling history via `build_history_batch` on `train_history_index` (t_pred = event time), plus targets and FN pairs. `epoch_batches`: one seeded permutation per epoch.
- `train.py`:
  - Each step runs `encode_items` once, and that one output feeds the video candidates, the music candidates and the history tokens.
  - AdamW with 300-step linear warmup, then a constant lr. Weight decay 0.01 applies to `nn.Linear` weights only. Embeddings, biases, LayerNorm, the positional embedding and the pooling query get 0.
  - `clip_grad_norm` 1.0. Determinism: seeds, `use_deterministic_algorithms`, TF32 off.
  - Fail-fast on a non-finite loss, label logit or grad norm. It writes `debug/nonfinite_step*.json` and raises before `opt.step()`.
  - `--max_steps`. A `validate(model, epoch)` hook (D1-8) handles patience and `best.pt`, but no validation is wired in yet (M4-3).
  - The output dir must be outside the git worktree; `require_clean_git` is optional.
  - Writes `run_meta.json`, `train_log.jsonl` and `last.pt` (with `condition_id`).
- Tests: `tests/test_m4_loss.py` (30, hand-built fixture with hand-written Pos_train sets) covers:
  - F1–F6 (including a brute-force FN-mask recomputation and a float64 hand-computed CE for all 4 tasks)
  - the cold condition (H absent from samples, candidates, Pos_train and histories; an inconsistent condition is rejected)
  - per-task head gradients, video loss reaching the music tower (and not with `use_music=False`)
  - one `encode_items` / MusicTower / VideoTower call per step, with shared tensors, in train mode with dropout
  - empty history / no-like batch / OOV user finite; NaN loss and NaN grad fail fast with a dump and no parameter update
  - k = 4 oversampling counts; strict-past rolling history; val/test events changing no training input; weight-decay groups; warmup
  - a 2-epoch fixture run, an overfit (loss halves), determinism, and the worktree guard

  `test_m4_artifacts.py` adds 2 real-data tests: sample and candidate counts plus brute-force Pos_train on 200 users, and one real optimizer step.
- Negative controls (monkeypatched mutations, not committed). Every mutation fails at least one test:
  - no FN mask
  - cross-task mask
  - label masked
  - music columns not merged
  - val events used as samples
  - τ dropped (caught by F6)
- `uv run pytest`: **163 passed** (131 before), 36 s.
- Real-data CPU smoke: `uv run python -m kuairand_rec.models.train --config configs/threetower_smoke.yaml --max_steps 5`.
  - Setup 3.4 s; 3.77M params.
  - B=256: 0.7 s/step. Loss 36.79 → 36.53 (≈ 4 × ln 7.5k at init). Pre-clip grad norm 31–41. No NaN/Inf.
  - B=1024: 3.6 s/step, loss 36.72 → 36.67, grad norm 23–26.
  - Data time is < 0.03 s/step. Reruns are bitwise identical.
- Observations / risks for review:
  - The pre-clip grad norm (~25–40) is far above clip = 1.0 at init, so clipping is active on every early step. With Adam this mostly rescales the moments; watch it in the GPU benchmark.
  - `use_deterministic_algorithms(True)` may raise on CUDA for an op without a deterministic kernel. This is unverified until the GPU smoke.
  - Pos_train includes later train positives of the same user (Q7a, by design; to document in the report).
  - FN-mask share on real data: about 25 masked columns per long_view row (0.3% of 7,529).

## M4-3 validation, export, scorers, retrieve(), parity (2026-10-11; awaiting review)

- New modules:
  - `models/artifacts.py`: condition fingerprint (vocabs, scalers, video/music feature tables, model events, candidates, U1 vocab; floats rounded to 1e-6); `load_model` refuses a checkpoint whose condition_id or fingerprint differs; data fingerprint; sha256.
  - `models/inference.py`: inference settings (eval, FP32, TF32 off, deterministic, fixed threads, no grad); canonical batch-1 user encoding; `score_rows` (one `I @ u` per user); `EmbeddingScorer`; `DirectScorer`.
  - `models/evaluation.py`: `build_eval_frames` (M3 `build_frames` on the unfiltered log, every frame_id asserted equal to `frames_summary.json`); `Validator` (val masked frames at T_val only); `selection_score`.
  - `models/export.py`: `export_embeddings`, `ExportedScorer` (export files only, sha256-verified, condition_id-checked), `retrieve()`.
  - `models/parity.py`: checks 1-4 as a CLI, run in a separate process. It rebuilds the condition from raw ids via `fit_condition` and requires the same fingerprint.
- Changes to M4-2 code:
  - `Trainer.fit`: validation after every epoch, and once more for a partial epoch cut by `max_steps`. The selection key is (mean R@20, mean N@20), compared with a strict `>` so the earliest wins exact ties. `best.pt` is saved on improvement; patience 3; a non-finite val score fails fast.
  - The checkpoint now carries the fingerprint and num_stats.
  - `run()` order: train → frozen `selection.json` → CPU export → val + test metrics through `ExportedScorer` (masked / nomask / cold) → parity subprocess.
  - `TrainCondition` gained `num_stats` and `build_condition(spec)`. The cold spec uses `exclude_video_ids` or a file and refits via `fit_condition`.
  - `user_features_for` added (U0/U1).
- Interface finding and minimal fix (inside D1-7, no protocol change):
  - On the M1, float32 GEMM results depend on how many rows are in the call and on the row's position: user embeddings change by ~1.5e-7 between batch sizes, and a single-row GEMM differs from a batch row by up to ~2e-5.
  - Score gaps inside the top-51 reach 3e-8, so batch-dependent numbers could break strict top-50 parity by chance.
  - Fix: the canonical export/direct path encodes users one at a time and scores per user with `I @ u`.
  - Measured: numpy gemv equals torch mv bitwise; results are identical with 1 or 8 torch threads (they differ under OMP_NUM_THREADS=3, so the thread count is recorded in the manifest).
  - Validation uses batch 256 for speed. That only affects checkpoint selection, never exported or reported numbers.
  - Export is forced to CPU.
- Tests: +24 `test_m4_export.py` (fixture) and +2 `test_m4_artifacts.py` (real data). `uv run pytest`: **189 passed** (163 before), 89 s. Coverage:
  - Validation and selection:
    - the validator reads val masked frames only
    - changing test labels leaves the val score bitwise unchanged
    - selection order, tie-break and patience (snapshot-verified `best.pt`)
    - partial-epoch validation, and NaN validation fails fast
  - Checkpoints and conditions:
    - a checkpoint is rejected under a different condition_id, different feature tables, or different scalers
  - Export:
    - every catalog row and user is covered
    - the export equals an independent forward pass, bitwise
    - the exported seen mask equals the M3 frame mask CSR
    - tampered files and the wrong condition are rejected
    - batch independence
  - `retrieve()`:
    - matches the scorer and the mask
    - mask_seen=False
    - errors are raised for bad input
    - tie-break is catalog row ascending
  - Parity:
    - bitwise on the fixture
    - a negative control detects a perturbed export
    - an export from a different checkpoint is refused
  - Cold condition:
    - its own vocab, checkpoint and export, which cannot be mixed with the normal ones
    - H stays scored and in the catalog
    - H is absent from cold histories
  - Real data:
    - frame_ids match M3
    - parity on 256 stratified users with a random-init checkpoint
- CPU end-to-end smoke (`--max_steps 5`, B 256; total 8.0 min on the M1):
  - Train + val 49 s (val 45 s), export 117 s, ExportedScorer eval 76 s, parity 224 s.
  - Parity on 2,048 users × every catalog row × 4 tasks × {val, test}:
    - max |Δ| = **0.0** (bitwise equal), p99 0.0
    - top-50 mismatches = 0, with and without the mask
    - retrieve(): 8,192 user-task pairs checked, 0 issues; independent metric recomputation max deviation 1.4e-17
    - per-user metrics identical on all 8 val/test masked frames (every user)
    - user strata: cold 265, train-OOV with history 328, 1-5: 499, 6-20: 562, >20: 722
  - Metrics after 5 steps are near random (val mean R@20 = 0.0025). This is a pipeline check, not a result.
- Remaining risks / notes:
  - Export and parity are CPU-only by design. GPU training is fine, because the checkpoint is reloaded on CPU. Bitwise parity relies on the same machine and thread count; the manifest records them.
  - Per-user scoring costs ~75 s per eval pass on the M1 (fine on the server). Validation is about 45 s per epoch on CPU and may need `val_user_batch_size` tuning on GPU.
  - The ExportedScorer cold frames use the empty-history zero-ID path (D1-4); cold users are reported separately.
  - H selection for the cold-video condition is M5. `build_condition` accepts it, but no cold run has been done on real data (fixture only).
  - The parity user sample comes from the test frames. Val-only users are covered by check 4 (every user of the val masked frames).

## M4-4 GPU training readiness (2026-10-11; awaiting confirmation before commit / deploy)

- M4-3 signed off by an independent review (no P0/P1 correctness bug).
- New: `configs/threetower_u0.yaml` / `threetower_u1.yaml`.
  - D1 values, `device: cuda`, normal condition, seed 0, `require_clean_git: true`, batch 1024 (OPEN until the benchmark), `run_root /local-scratch/liang/runs` (overridable by `$RUN_ROOT`).
  - U1 differs from U0 only in `exp`, `user_features` and `u1_min_count` (test-enforced).
- New: `models/benchmark.py`, a training-only benchmark.
  - It reuses `Trainer` + `train_step` (losses are identical to `Trainer.fit` at the same seed, test-enforced).
  - It records steps/s, data/compute time, the epoch estimate, peak GPU memory, loss and the grad-norm state, and optionally times one validation pass (`--val_once`) to give a D1-11 run-time upper bound.
  - It skips selection, export, test and parity. `benchmark.json` is labelled "not a model metric".
- `run()` now writes `selection_vs_export.json`. It compares the best epoch's validation-path metrics with the exported val metrics and flags epochs within the observed difference ("near ties"). This is a report only; the selection is unchanged.
- `.gitignore`: `*_review.zip`.
- Tests: +6 `test_m4_readiness.py`. `uv run pytest`: **195 passed**.
- CPU dry run of the benchmark on real data (B 1024): 0.25 steps/s, data share 0.5%, pre-clip grad norm 16–26 (100% of steps clipped), losses finite. These are CPU numbers, not GPU.
- Open:
  - CUDA deterministic mode is unverified.
  - `scripts/data_fingerprint.py` (D1-11) is not written yet.
  - Batch size is still to be confirmed.

## Next Steps

1. M4 (on the user's go), in order:
   - features + leakage tests (L1-L5)
   - towers + OOV/gradient tests
   - loss/data + F1-F6
   - export/parity
   - train CLI + CPU smoke
   - commit and push
   - server deployment: Deploy Key, fixed SHA, data md5 + fingerprint, frame_id check
   - GPU smoke + benchmark
   - U0 end to end
   - U1
2. Then M5: bootstrap CIs, history buckets, the cold-video experiment, and the required ablations.
3. Tests still to add in M3+ (checklist; the three-tower items are now specified in `PLAN.md` D1-9):
  - The mask never uses test labels (evaluator level).
  - Cold-video interactions are fully removed from every model input and fitted dataset (M5, end to end).
  - ALS scores unseen items as -inf.
  - Score parity covers every catalog row, including unseen and OOV rows.
  - In the cold-video condition, GT, mask, and population are identical to the normal condition.
  - The per-user random-reference expectation matches a simulated random ranking.
  - Hand-computed metric cases (Recall/NDCG, tie-break).
