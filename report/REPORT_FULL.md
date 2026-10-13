---
title: "A Leakage-Controlled Multi-Action Three-Tower Retriever for Videos and Background Music on KuaiRand-Pure"
subtitle: "Extended report (formal v1 results + post-hoc exploratory study)"
date: "2026-10-12"
---

# Abstract

We retrieve both videos and their background-music IDs for two actions, long view (LV) and like, on KuaiRand-Pure. That gives four full-catalog retrieval tasks. Every model is scored by one shared evaluator on identical frozen test frames, with 1,000-resample paired user bootstraps. The models are Popularity, implicit ALS (pure and with an auxiliary LV signal for like) and a three-tower model (user / video / music). We prioritised a leakage-safe protocol:
- a global chronological 70/15/15 split;
- strictly-past histories capped at 100 events;
- train-only fitting of every statistic;
- no monthly statistics file;
- explicit handling of unseen items and users;
- bitwise export/direct score parity.

**Findings (test, Recall@20, 95% paired CI).**
- The main model **U0** (user ID + history) beats Popularity on all four tasks.
- Against pure ALS it is **worse on both LV tasks** (−0.012 each, significant) and **better on both like tasks** (+0.036 / +0.044).
- The like-task gain disappears against ALS with an auxiliary LV signal (CIs include 0).
- **U1** adds a user-metadata snapshot. It reaches ALS level on LV, but it carries an unresolvable snapshot-leakage risk, so U0 is the main model.
- **Controlled cold-video holdout:** among held-out videos only, the content path ranks clearly better than random and the cold baselines. On the full catalog it is **not significantly better than random**.
- **Ablations:** removing history hurts long view. Removing the music input to the video tower **improves** long view, which is a negative result for that design choice.
- **Post-hoc exploratory study** (§2.5; validation only, three seeds; designed after the test results were analysed):
  - **Removing the music-to-video input consistently improved long-view validation Recall@20 across three seeds, without systematically harming like retrieval.** Mean val R@20 .1286 ± .0018 vs U0 .1220 ± .0036 (+.0066); video LV +.0068, music LV +.0072, higher on 3/3 seeds for both. Its long-view R@20 difference from pure ALS is no longer CI-supported (6/6 CIs include 0; not an equivalence claim). Its long-view NDCG@20 is above pure ALS (6/6 nominal CIs exclude 0; secondary metric, uncorrected). The mechanism is not identified.
  - Training-seed variation (mean val R@20 .1220 ± .0036) is larger than user-bootstrap uncertainty.
  - lr 5e-4 raises mean val R@20 on 3/3 seeds (+.0048 on average, about one seed std), mainly through like; long view is seed-dependent.
  - τ = 0.1 trades like for long view without changing the mean.
  - Reproduction: same-config reruns reproduce the formal U0 (job 27) and no-music (job 35) runs bitwise.
  - None of this changes the formal test results.

Supporting material: full tables in `report/tables/`, plotting in `scripts/make_report_tables.py`, error analysis in `results/final_analysis/`, and a requirement-by-requirement audit in `artifacts/final_compliance_review/`.

# 1. Design

## 1.1 Data audit and cleaning

**Inputs.**
- Used: the two standard logs, `user_features_pure.csv` and `video_features_basic_pure.csv`.
- Refused by the loader (`data/io.py` raises on any other file name):
  - the random log (out of scope);
  - `video_features_statistic_pure.csv` (§1.2).

**Raw entities.**
- Logs: 1,436,609 rows, 27,077 users, 7,551 logged videos and 6,489 logged authors.
- Metadata: 7,583 videos, 6,510 authors and 27,285 users.
- Actions:
  - LV is positive on 33.2% of rows and like on 1.85%.
  - Only 6,768 users ever like a video.
  - The official LV rule (play ≥ duration if duration ≤ 18 s, else play ≥ 18 s) reproduces the label on 99.76% of rows with known duration.

**Music sharing.**
- There are 7,201 non-zero music IDs. Music is almost one-to-one with video:
  - 7,089 IDs have exactly one video;
  - only 112 IDs are shared, covering 290 videos, with at most 15 videos per ID;
  - 30 metadata music IDs never occur in the logs.
- `music_id = 0` covers 204 videos. We treat it as a "no music" sentinel; this is a hypothesis, because the dataset dictionary does not define it.
  - Its video interactions are kept for the video tasks.
  - It is never a music candidate and never music ground truth. Its positives are counted as "sentinel excluded" (1,670 LV / 67 like test events).
  - The video tower gives these videos a learnable NO_MUSIC embedding.

**Data quality and cleaning.** Cleaning runs before the split; the event key is (user, video, `time_ms`).
- **Exact duplicates:** 21,987 redundant rows, mostly in tabs 5/6. They are collapsed.
- **Conflict groups:** 1,486, which differ mostly in `hourmin`. Each keeps one *whole* representative row: max `play_time_ms`, then a full-column deterministic tie-break.
- **Label conflicts:** 2 events disagree on the LV label. They are dropped and saved, so the evaluator can report an upper bound of lost ground truth (1 LV test pair).
- **Missing duration:** 28,859 events with `duration_ms = 0` are kept, with duration treated as missing.
- **Not done:** no k-core filtering and no relabelling.
- **Result:** 1,413,134 events, with all 27,077 users and 7,551 videos kept.

An independent pandas recomputation confirms the cleaning:
- cleaned keys = distinct raw keys minus exactly the 2 dropped keys;
- there are no other silent drops;
- the pipeline reruns byte-identically (`results/final_audit/`).

## 1.2 Temporal split and leakage control

**Split.** The split is global and chronological, with cuts at CST midnight: T_val = 2022-04-17 and T_test = 2022-04-26. Events with t < T go to the earlier split. No event lies on a cut, and no timestamp spans two splits.

| split | days (CST) | events | share | LV / like positives |
|---|---|---:|---:|---:|
| train | 04-09 – 04-16 | 997,078 | 70.6% | 339,386 / 18,771 |
| val | 04-17 – 04-25 | 208,393 | 14.7% | 69,354 / 3,980 |
| test | 04-26 – 05-08 | 207,663 | 14.7% | 65,812 / 3,610 |

**Why the monthly video statistics are forbidden.** `video_features_statistic_pure.csv` aggregates engagement over the whole log period (play, like, follow and share counts). Using it at time t would encode test-period behaviour of the same videos.
- The loader refuses the file.
- None of its 51 columns appears in any processed artifact.
- The video tower reads only: `video_type`, `upload_type`, `upload_dt`, `music_type`, tags, log-scaled duration/width/height (each with a missing flag), and the video, author and music IDs.
- `visible_status` is dropped as a constant "now" field.

**Train-only fitting.** The two data views are `train_view` (split = 0) and `history_view`.
- `train_view` is the only input to every fitted quantity, and an assertion guards it:
  - the ID vocabularies;
  - the categorical metadata vocabularies;
  - the numeric scalers;
  - Popularity counts (music = sum over its videos);
  - ALS item factors;
  - the U1 snapshot vocabulary (values with < 5 train users map to OOV).
- `history_view` (t < t_pred) is used only to build user representations and ALS fold-in.
- Validation is used only for ALS grid selection and three-tower epoch selection.
- No test label reaches any fit or choice. Every three-tower run records `test_used_for_selection = false`.

**Strict-past history, cap 100 and the seen mask.** One function builds histories for all three settings:

| setting | t_pred | events used |
|---|---|---|
| training | the target event time (rolling) | train events |
| validation | T_val | train events |
| test / export | T_test | train + val events |

- **Strictness.** The window is `searchsorted(side="left")` on (user, t_pred), so events at exactly t_pred are excluded, including same-timestamp siblings. 10% of rows sit in same-user, same-timestamp groups. The builder also asserts every token has age > 0.
- **Cap.** The model input keeps the most recent 100 events.
- **Seen mask.** A separate function builds the evaluation mask from *all* pre-t_pred interactions, uncapped.
- **Independent check.** For 600 random users, pandas reproduces exactly the frozen histories at T_val and T_test, 600 rolling training histories, and the exported uncapped seen mask. 57 of the 600 users had more than 100 seen videos.

**Feedback delay.** The dataset does not document when likes and other feedback become known. History tokens younger than D = 20 min therefore have all feedback fields zeroed and a "hidden" flag set. This is a conservative assumption, not a verified bound.

**User metadata (U0 / U1 trade-off).** `user_features_pure.csv` is a single snapshot with no date.
- Evidence of a late snapshot: user 25929's `register_days` together with its first event bounds the snapshot at no later than 2022-05-02, which is inside the test period. There is no lower bound.
- So the snapshot cannot be shown to precede T_val or T_test.
- Field-level audit of the 21 U1 fields (`artifacts/final_compliance_review/USER_METADATA_REVIEW.md`):
  - 2 are time-dependent by definition (`register_days_range`, cumulative `is_video_author`);
  - for the other 19 (18 encrypted one-hot fields, `is_live_streamer`), historical availability is unknown;
  - none can be shown to be legally available at the historical prediction time.

  This is not evidence that every field leaks.
- Data-quality finding: one user has the undocumented `register_days_range` value `8-14`; the documented set starts at `15-30`.
- We read "usable metadata" in the assignment as "provably time-safe". That is our interpretation.
- **U0 (main):** user ID + history.
- **U1 (required companion):** U0 plus 21 snapshot fields (`onehot_feat0–17`, `is_live_streamer`, `is_video_author`, `register_days_range`). It uses the same config, seed and selection rule as U0.
- Neither variant alone satisfies the literal assignment wording with certainty:
  - U0 omits user metadata;
  - U1 may leak future information.

## 1.3 Three-tower model

| tower | inputs | output |
|---|---|---|
| Music | music ID + music type | MLP + LN → 128-d base |
| Video | video ID, author, metadata, and the *shared* music-tower output | MLP + LN → 128-d base |
| User | user ID + a history encoder (U1: + snapshot fields) | MLP + LN → 128-d base |

**Music tower inside the video tower.** The music tower runs exactly once per catalog forward. Each video gathers its music base embedding from that output, so in train mode a video's music input is the identical tensor as its music candidate's embedding.

**History encoder.** It runs over the shared video base embeddings of the history videos. The video tower is also run on unseen videos, so a history token for an unseen video is encoded from its content.
- Each token adds 8 feedback bits plus the hidden flag, a play-ratio bucket, a log-recency bucket and a position embedding.
- The encoder is a 2-layer Transformer (d = 128, 4 heads) with learnable-query pooling.

**Heads.** Each target × action has its own Linear(128 → 64) + L2 projection: 4 on the user side, 2 for video and 2 for music. The score is cosine similarity. The U0 model has 3.77M parameters.

**Unseen items and users.** No untrained parameter may influence a score.
- PAD and OOV rows are fixed at zero by a constant mask. ID dropout (0.1) trains the zero-ID path.
- Train-unseen and cold videos are scored from metadata, author and music.
- Users without history use the empty-history + zero-ID path.

The same encoder functions run in training, validation, export and direct scoring. The Transformer is written out explicitly, avoiding PyTorch's eval-mode fast path.

## 1.4 Contrastive objective, duplicate positives, false negatives and action balance

**Samples.** There is one sample per train event that is positive for either action: 345,365 rows, 18,771 of them likes.
- Each sample carries four task masks. The music tasks also require `music_id ≠ 0`.
- Each sample uses its rolling strict-past history.

**Loss.** Per task we use an exact full softmax over all items of the condition's train vocabulary: 7,529 videos and 7,151 music IDs in the normal condition.

$$
\ell_r = -\log \frac{\exp(\cos(u_r, i_{y_r})/\tau)}{\sum_{c \notin \mathrm{FN}(r)} \exp(\cos(u_r, i_c)/\tau)}, \qquad \tau = 0.05
$$

- There is no sampling, so no logQ correction is needed.
- Validation and test still score the full metadata catalog.

**Duplicate positives.** Candidate columns are unique items, so all rows whose label is the same item point at one column.
- This holds for repeated rows of the same user, rows of different users, and videos that share a music ID.
- A row's own label is never masked.

**False negatives.**

$$
\mathrm{FN}(r) = \mathrm{Pos}_{\text{train}}(u_r, \text{task}) \setminus \{y_r\}
$$

- Pos_train is the set of u_r's train positives for the *same* target and action. For music it holds the non-zero music of those videos.
- FN columns are set to −inf, so they get softmax weight 0 and zero gradient.
- Pos_train uses train labels only. It never uses validation or test positives, and it is never a model input.
- It is a *supervision* mask, not history. It also includes u_r's train positives that occur **after** the sample time. About 50% of the masked columns come from the same or later timestamps.
  - This does not leak into the model input.
  - It does mean the loss knows the user's within-train future positives, so they are not pushed down as negatives.

**Verification on real data.** Four real batches of 1,024 rows were taken from the formal epoch-0 order and compared with an independent recomputation from raw train events:
- labels: 0 mismatches;
- FN masks: 0 mismatches;
- validation or test positives masked: 0;
- softmax weight on masked columns: 0;
- the loss, recomputed in float64 from the formal U0 embeddings: |Δ| ≤ 8e-7.

These batches exercise the duplicate-positive cases: 2,031 LV rows shared a label column with another row, and 115 music-LV rows had a label music ID that is shared by several videos.

**Action balance.** The total loss is the equal-weight sum of the four per-task *mean* losses. Like rows are not oversampled (k = 1).
- A batch averages about 1,006 video-LV rows and about 56 video-like rows.
- So each like row carries about 18× the gradient weight of an LV row, and the like and LV tasks get equal weight at the task level.
- The like signal is also shared across actions: the item towers, the user tower and the history are shared, and only the 64-d heads are action-specific.
- Limitations of this balance:
  - like gradients are noisier (about 56 rows per batch);
  - like-heavy users are not reweighted;
  - the 1:1 task weight was fixed a priori, not tuned.

**Optimisation and selection.**
- AdamW, lr 1e-3 with 300 warm-up steps, batch 1,024, grad clip 1.0, FP32 with TF32 off, deterministic kernels.
- At most 30 epochs, with patience 3 on the mean validation-masked R@20 over the 4 tasks (tie-break NDCG@20).
- Any non-finite loss or gradient fails the run immediately (no step is ever skipped).
- One seed (0) per model.
- The formal v1 runs were not tuned: every hyperparameter stayed at its design value. U0 was below ALS on validation LV R@20 (video .1260 vs .1339, music .1308 vs .1361), which is the condition under which the design allows targeted adjustments. Those adjustments are left to separate, clearly labelled exploration runs and are not part of the reported v1.

## 1.5 Evaluation protocol

**Users, ground truth and catalog.**
- **Frozen users.** Each user is represented once at T_test.
- **Ground truth.** GT(u) is the de-duplicated set of u's test positives (shared music collapses to one music row), after removing:
  - out-of-catalog items (0 in practice);
  - sentinel music;
  - items u interacted with before T_test (seen-masked; counts per task in the table below).
- **Catalog.** All 7,583 metadata videos and all 7,201 non-zero music IDs. Train-unseen items stay in the catalog and in GT; Pop scores them 0 and ALS cannot score them (−inf).
- **Ranking.** Score descending, ties by catalog row.

**Metrics.**
- Recall@K = hits / |GT|.
- NDCG@K uses binary gain and IDCG over min(|GT|, K).
- Users are weighted equally.

**Population.** A user is evaluated if GT is non-empty and the user has ≥ 1 history event.

| task | users | GT pairs | seen-excluded GT pairs | OOC positive events | sentinel-excluded events | cold users (separate) | train-OOV users |
|---|---:|---:|---:|---:|---:|---:|---:|
| video LV | 18,752 | 63,281 | 859 | 0 | – | 437 | 499 |
| video like | 1,924 | 3,380 | 63 | 0 | – | 68 | 65 |
| music LV | 18,595 | 61,595 | 913 | 0 | 1,670 | 434 | 496 |
| music like | 1,895 | 3,307 | 72 | 0 | 67 | 67 | 63 |

Source: `results/baselines/frames_summary.json` (test masked frames). The label-conflict upper bound is 1 possible GT pair, on the LV tasks only.

**Frame identity.** Every model on a task must match one `frame_id`, a hash of users, GT and mask, so every comparison is on the same users.

**ALS conventions.**
- Users with no usable signal score −inf, which gives them a metric of 0. They are never given random or zero vectors: 466 such users on video LV, and 608 on video like for pure ALS.
- Items with no train signal for that ALS configuration are unscorable.

**Bootstrap.** 1,000 user resamples, shared by all models on a frame (paired), with 95% percentile CIs.

**Selection hygiene.**
- ALS uses a fixed 8-point grid chosen on validation.
- The three-tower epoch is chosen on validation.
- Test is evaluated once, after selection.

# 2. Results

**Scope.** §2.1–2.4 report the formal v1 **test** results; their numbers come from `results/m5/main_formal/` (bootstrap) and `results/m5/cold_video/eval_671dde9_job33/`. Full R/NDCG@10/20/50 are in `report/tables/t1_main_metrics.md`, and every paired difference is in `t2_paired_diffs.md`. §2.5 is a separate post-hoc exploratory study that reports **validation** metrics only (`results/explore/`); its numbers must not be compared with the test tables. A star (\*) marks a 95% CI that excludes 0.

## 2.1 Main comparison

![Figure 1: test Recall@20 with 95% paired-bootstrap CIs on the full-catalog, seen-masked frames. Popularity, pure ALS, U0 and U1 on all four tasks; ALS with the auxiliary long-view signal exists for the like tasks only. Each task's bars use the same users. Source: results/m5/main_formal/bootstrap_models.csv.](figures/fig1_baselines.png)

**Table 1. Test R@20 / NDCG@20.**

| model | video LV | video like | music LV | music like |
|---|---|---|---|---|
| Pop | .0806 / .0457 | .0716 / .0403 | .0856 / .0479 | .0726 / .0412 |
| ALS (pure) | **.1244** / .0686 | .0555 / .0282 | **.1282** / .0705 | .0586 / .0303 |
| ALS (aux, like only) | – | **.1008 / .0469** | – | **.1054** / .0463 |
| 3T-U0 (main) | .1124 / .0676 | .0913 / .0417 | .1162 / .0691 | .1022 / **.0472** |
| 3T-U1 (+ snapshot) | .1274 / .0744 | .0859 / .0397 | .1321 / .0768 | .0864 / .0407 |

**Table 2. Paired differences in R@20 [95% CI].**

| comparison | video LV | video like | music LV | music like |
|---|---|---|---|---|
| U0 − Pop | +.032\* | +.020\* | +.031\* | +.030\* |
| U0 − ALS pure | **−.012\*** [−.016, −.008] | +.036\* | **−.012\*** [−.016, −.008] | +.044\* |
| U0 − ALS aux | – | −.009 [−.025, +.005] | – | −.003 [−.019, +.014] |
| U1 − ALS pure | +.003 [−.001, +.007] | +.030\* | +.004 [−.000, +.008] | +.028\* |
| U1 − U0 | +.015\* | −.005 | +.016\* | −.016\* |

Other differences (`t2_paired_diffs.md`):
- **U0 vs pure ALS on LV.** The NDCG@20 differences include 0 (−.001 / −.001). U0's NDCG@10 is slightly *higher* on video LV (+.0025\*). U0 is clearly behind at R@50 (−.033\*).
- **U0 vs Pop on like.** NDCG@20 is not significant (+.001 / +.006).
- **U1 vs pure ALS on LV.** NDCG@20 is +.006\* on both tasks. R@20 is not significant, and video-LV R@50 is lower (−.010\*).

**Conclusions.**
- We do not claim that the three-tower model beats ALS.
- U0 is a strong, Pop-beating retriever. It is worse than pure ALS at recall depth on LV, and on like it is statistically tied with the auxiliary-signal ALS.

## 2.2 Controlled cold-video experiment

**Holdout.**
- H = 750 train-seen videos (10%), stratified into 10 train-interaction-count strata, seed 0. Selection uses train data only.
- Every H interaction is removed from training targets, all histories (including pre-T_test history at test time), candidates, false-negative sets, every vocabulary and scaler, and the cold Pop/ALS fits.
- The removal goes through one refit path and is verified by 12 automated checks.
- In the cold condition, 22.8% of H videos still have an in-vocabulary author and 4.0% an in-vocabulary non-zero music ID. For 72.4% of H, video, author and music are *all* OOV.

**Model selection.** The cold model's epoch is selected on validation frames from which all H ground truth is removed.

**Fixed evaluation frame.** It is built once from the unfiltered log and shared by every model.
- GT_H(u) = u's test positives on H that u had not seen before T_test.
- Population: users with ≥ 1 non-H history event.
  - video LV: 5,936 users, 7,536 pairs on 434 H videos;
  - video like: 305 users, 336 pairs.
- Two candidate sets, both seen-masked:
  - (a) the full catalog;
  - (b) H only.

![Figure 2: controlled cold-video experiment, test Recall@20 on GT_H with 95% paired CIs. Left: full-catalog candidates. Right: H-only candidates (note the roughly 10× larger y-scale). Only the cold-trained three-tower model and cold-condition baselines (Pop/ALS refit on H-removed data, analytical random ranking) are shown. Cold ALS is 0 by construction because H has no factors. Source: results/m5/cold_video/eval_671dde9_job33/cold_bootstrap_models.csv.](figures/fig2_cold_video.png)

**Table 3. Cold 3T-U0 minus each cold-condition baseline, R@20 [95% CI]** (`t5_cold_video.md`):

| task | candidates | cold 3T-U0 | − cold Pop | − random |
|---|---|---:|---|---|
| video LV | full catalog | 0.0041 | +.0041\* | +.0014 [−.0000, +.0032] |
| video LV | H only | 0.1045 | +.084\* | +.078\* |
| video like | full catalog | 0.0066 | +.0066 [.0000, .0164] | +.0039 [−.0027, +.0137] |
| video like | H only | 0.0672 | +.059\* | +.040\* |

Notes on the baselines:
- On the full catalog, cold Pop gives every H video a score of 0, and cold ALS cannot score H at all (all GT_H is unscorable). Both are 0 by construction, so "beating" them carries no information.
- On H only, cold Pop is a fixed catalog-order ranking (all scores tie at 0), not a random ranking. The random row is the closed-form per-user expectation.

**What the results show.**
- **H only.** Metadata, author and music carry a real relative-ranking signal among new videos.
- **Full catalog.** R@20 is not distinguishable from random. For video LV, NDCG@10 is even slightly *below* random (−.0003\*), while R@50 is above it (+.0032\*).

**Seen-video reference.** The normally trained U0 scores R@20 0.111 (full) and 0.433 (H only) on the same LV frame. It saw H in training, so it is context only: it is neither a cold baseline nor a guaranteed upper bound.

## 2.3 History-length buckets

The bucket is the number of events actually fed to the model at T_test, capped at 100. The buckets partition each population exactly; for video LV they hold 1,162 / 4,053 / 13,537 users. On LV, 72% of evaluated users have more than 20 events.

![Figure 3: test Recall@20 by history-length bucket (1–5, 6–20, >20 events) with 95% CIs, all four tasks. The same bucket definition and users apply to every model. Source: results/m5/main_formal/bootstrap_models.csv, strata 1-5 / 6-20 / >20.](figures/fig3_buckets.png)

**U0 − ALS (pure) by bucket** (`t3b_bucket_diffs.md`):

| task | 1–5 | 6–20 | >20 |
|---|---|---|---|
| video LV | +.028\* | −.006 | −.017\* |
| music LV | +.037\* | −.012\* | −.016\* |
| video like | +.118\* | +.062\* | +.023\* |
| music like | +.122\* | +.069\* | +.031\* |

The LV deficit is a heavy-user effect: U0 wins in the 1–5 bucket and loses in the >20 bucket.

**Cold users** (no history; separate frame, `t4_cold_users.md`):

| model | video-LV R@20 (437 users) |
|---|---:|
| Pop | 0.142 |
| U0 | 0.039 |
| no-history ablation | 0.147 |

## 2.4 Ablations

Each ablation is a single configuration flag change from U0, with the same data, seed, selection rule and evaluation.

![Figure 4: paired test differences vs U0 (95% CI) for the two ablations. Top: Recall@20; bottom: NDCG@20. Source: results/m5/main_formal/bootstrap_diffs.csv.](figures/fig4_ablations.png)

**No history** (user ID only):
- R@20 drops significantly on video LV (−.012\*), music LV (−.011\*) and music like (−.017\*).
- NDCG@20 on LV drops by −.010\* / −.009\*.
- The video-like difference is not significant.

**No music input to the video tower** (the music tower and heads are still trained by the music tasks):
- LV *improves*: R@20 +.010\* / +.009\* and NDCG@20 +.004\* / +.004\*.
- The like differences are not significant.
- This is a negative result for the music-input design choice. We report it as is and did not tune it away.

## 2.5 Post-hoc optimization and robustness study (validation only)

**Nature of this study.**
- **Post-hoc and exploratory.** It was designed *after* the formal v1 test results above had been analysed, and it runs on a separate branch (`explore/s9-opt`). It does not replace U0 or change any formal result.
- **Validation only.** Every number in this section is a **validation** metric on the M3 val masked frames (T_val, train-only history, frame ids asserted), never a test number. The analysis scripts refuse `metrics.csv` and test per-user files.
- **Same protocol.** Each run changes one setting relative to `configs/threetower_u0.yaml`. Selection is the D1-8 rule: mean val R@20 over the 4 tasks.
- **Two kinds of uncertainty, reported separately:**
  - a paired user-level bootstrap on validation (1,000 resamples), which covers user sampling for a fixed pair of trained models;
  - training-seed variation over three seeds, which is descriptive only.
- **Sources:** `results/explore/` (copied from the explore branch; see `results/explore/PROVENANCE.md`). Full write-up: [`docs/EXPLORATORY_RESULTS.md`](../docs/EXPLORATORY_RESULTS.md).

**Reproduction and correctness.**
- `x_u0_repro` (job 44) reran the formal U0 configuration at the explore commit. It reproduces formal job 27 **bitwise** (`results/explore/wave2/repro_vs_job27.json`):
  - all 68 training-log records, all 10 validation epochs and the selected epoch 6;
  - the `best.pt` state dict, tensor by tensor;
  - the 8 validation-user and item embedding files;
  - every validation per-user metric.

  So formal U0 is a valid same-code seed-0 control.
- `x_nomusic_repro` (job 47) likewise reproduces the formal no-music ablation, job 35, bitwise (`results/explore/wave3/repro_job47_vs_job35.json`).
- Every explore run (jobs 39–49) passes the following checks (`results/explore/wave{1,2,3}/run_verification.json`):
  - exit code 0 and strict Direct/Export parity (max |Δ| = 0.0, 0 top-50 mismatches);
  - the normal condition id `133765eb12b75700`, one data fingerprint, and `test_used_for_selection = false`;
  - validation selection and export frame ids equal to the M3 frames.

### 2.5.1 Random-seed robustness

U0 at seeds 0, 1 and 2 (lr 1e-3, τ 0.05):

| | mean | std (n = 3) | min | max |
|---|---:|---:|---:|---:|
| mean val R@20 | .1220 | .0036 | .1180 | .1252 |
| mean val NDCG@20 | .0672 | .0020 | .0653 | .0693 |

- **Seed variation is larger than user-sampling uncertainty.** Under the user bootstrap, seed 2 alone "differs significantly" from seed 0 (−.0072 [−.0105, −.0038] mean val R@20).
- **Reading guide, not a threshold.** The observed three-seed range (.0072) is a rough scale for training noise, not a statistical threshold; three seeds cannot estimate it precisely. Single-seed differences of that order should therefore be treated as inconclusive, and a CI from the user bootstrap is never evidence of a cross-seed effect.
- **The formal seed is favourable.** Formal seed 0 is the best of the three seeds.

### 2.5.2 Learning-rate sensitivity

The table compares lr 5e-4 with the formal lr 1e-3, three seeds each, at the best checkpoint per run. All values are validation R@20 / NDCG@20.

| lr | seed (job) | best ep | video LV | video like | music LV | music like | mean R@20 | mean N@20 |
|---|---|---:|---|---|---|---|---:|---:|
| 1e-3 | 0 (44 = 27) | 6 | .1260 / .0781 | .1223 / .0589 | .1308 / .0803 | .1218 / .0599 | .1252 | .0693 |
| 1e-3 | 1 (39) | 7 | .1267 / .0770 | .1089 / .0515 | .1344 / .0805 | .1204 / .0586 | .1226 | .0669 |
| 1e-3 | 2 (40) | 5 | .1241 / .0747 | .1129 / .0564 | .1203 / .0728 | .1149 / .0575 | .1180 | .0653 |
| 5e-4 | 0 (41) | 6 | .1304 / .0803 | .1277 / .0635 | .1370 / .0831 | .1277 / .0629 | .1307 | .0724 |
| 5e-4 | 1 (45) | 4 | .1214 / .0727 | .1250 / .0624 | .1242 / .0747 | .1314 / .0631 | .1255 | .0682 |
| 5e-4 | 2 (46) | 9 | .1281 / .0771 | .1160 / .0536 | .1349 / .0793 | .1168 / .0537 | .1239 | .0659 |

Same-seed differences (lr 5e-4 − lr 1e-3) in validation R@20. Brackets give the paired user-bootstrap CI.

| seed | mean R@20 | mean N@20 | video LV | video like | music LV | music like |
|---:|---|---|---:|---:|---:|---:|
| 0 | +.0055 [+.0026, +.0085] | +.0031 [+.0020, +.0043] | +.0044 | +.0054 | +.0062 | +.0059 |
| 1 | +.0029 [−.0014, +.0073] | +.0014 [−.0006, +.0034] | −.0053 | +.0161 | −.0101 | +.0110 |
| 2 | +.0059 [+.0023, +.0099] | +.0006 [−.0013, +.0024] | +.0040 | +.0031 | +.0145 | +.0020 |
| mean | +.0048 | +.0017 | +.0010 | +.0082 | +.0035 | +.0063 |

![Figure 5: mean validation R@20 per seed at lr 1e-3 vs lr 5e-4 (post-hoc exploratory; validation). Dashed lines are the 3-seed means. Source: results/explore/wave2/validation_comparison.csv.](figures/explore_fig_a_lr_mean.png)

![Figure 6: same-seed per-task differences lr 5e-4 − lr 1e-3 in validation R@20 (dots = seeds, bars = 3-seed mean). Source: results/explore/wave2/paired_lr_diffs.csv.](figures/explore_fig_b_lr_task_deltas.png)

**Overall effect.**
- The 3-seed mean val R@20 moves from .1220 ± .0036 to .1267 ± .0035, a mean gain of +.0048.
- The gain is positive on all three seeds (+.0029 to +.0059), but its size is about one seed standard deviation, and the two groups overlap: the lr 5e-4 minimum (.1239) is below the lr 1e-3 maximum (.1252).
- Seed 1's own CI includes 0.

**Per-task pattern.**
- **Like tasks:** numerically higher on all three seeds.
- **Long-view tasks:** seed-dependent. They rise on seeds 0 and 2, but fall clearly on seed 1 (video LV −.0053, music LV −.0101; both CIs exclude 0), where the run stopped early at epoch 4.
- **Source of the gain:** mostly the like tasks.

**Conclusion.** "lr 5e-4 tends to improve validation like retrieval on 3/3 seeds; long view is seed-dependent." This is suggestive, not a demonstrated improvement, and it does not justify replacing the formal U0.

### 2.5.3 Temperature sensitivity

τ = 0.03 and τ = 0.1 versus the formal τ = 0.05. Seed 0 only (jobs 42, 43), paired validation CIs against U0 seed 0.

| Δ val R@20 vs τ 0.05 | video LV | video like | music LV | music like | 4-task mean |
|---|---|---|---|---|---|
| τ 0.03 | +.0005 | −.0028 | −.0002 | −.0003 | −.0007 [−.0030, +.0015] |
| τ 0.10 | **+.0085** [+.0057, +.0114] | **−.0114** [−.0222, −.0004] | **+.0094** [+.0065, +.0122] | −.0083 [−.0188, +.0026] | −.0005 [−.0045, +.0034] |

![Figure 7: temperature, seed 0: per-task validation R@20 difference vs τ = 0.05 with paired user CIs (one seed per setting). Source: results/explore/wave1/validation_bootstrap.csv.](figures/explore_fig_c_tau_tradeoff.png)

**Neither temperature improves the four-task mean.**
- τ = 0.03 changes nothing noticeable.
- τ = 0.1 **reallocates** performance:
  - long view improves on both tasks (CIs exclude 0);
  - likes fall (video like CI excludes 0);
  - the overall level stays the same.
- *Observed:* this long-view versus like trade-off is a single-seed observation (seed 0), with paired user CIs.
- *Possible mechanism (hypothesis, not tested):* a softer softmax (larger τ) penalises hard negatives less, which may favour the dense long-view signal over the sparse like signal. No experiment here isolates this mechanism.

### 2.5.4 Comparison with the ALS baselines on validation

- **Same frames, validation only.** These baselines use the same validation split, catalog, seen mask and frame ids as the models above. They are validation numbers and must not be compared with the test tables in §2.1.
- **Baselines.** Validation R@20: pure ALS .1339 (video LV) / .1361 (music LV); ALS aux .1077 (video like) / .1166 (music like).
- **Source.** `results/explore/wave3_review/longview_vs_als.csv`.

**Long view vs pure ALS (val R@20, paired CI).**

| run | video LV | music LV |
|---|---|---|
| U0 lr 1e-3, seeds 0 / 1 / 2 | −.0079\* / −.0072\* / −.0099\* | −.0053\* / −.0017 [−.0061, +.0024] / −.0158\* |
| lr 5e-4, seeds 0 / 1 / 2 | −.0035 [−.0076, +.0005] / −.0125\* / −.0058\* | +.0009 [−.0035, +.0052] / −.0118\* / −.0012 [−.0053, +.0028] |
| τ 0.10, seed 0 | +.0006 [−.0035, +.0048] | +.0041 [−.0001, +.0081] |
| no-music, seeds 0 / 1 / 2 (Wave 3; seed 0 = formal ablation) | −.0005 / −.0036 / −.0005 (all CIs include 0) | −.0008 / −.0003 / +.0001 (all CIs include 0) |

**Reading the long-view comparison.** Here "tie" means the paired 95% CI includes 0: no significant difference detected. It is **not** a demonstration of equivalence, since no equivalence margin was specified.
- **The best runs tie pure ALS.** A few explore runs tie, or are *numerically* slightly above pure ALS on long view: lr 5e-4 seed 0 on music LV, τ 0.1 on both long-view tasks, and no-music seed 2 on music LV. Every such CI includes 0.
- **No-music at all three seeds.** It shows no significant R@20 difference from pure ALS on both long-view tasks (6/6 CIs include 0), while every U0 seed is significantly below ALS on video LV. On long-view NDCG@20, no-music is above pure ALS on 6/6 seed-task pairs (+.0030 to +.0043; nominal 95% CIs exclude 0, uncorrected for multiple comparisons, secondary metric, validation only).
- **No run has a CI-supported long-view advantage.** Nor are all explore runs below ALS.
- **The deficit shrinks but does not close.** The mean long-view gap to ALS shrinks from .0083 / .0076 (video / music, lr 1e-3 seed mean) to .0073 / .0041 (lr 5e-4), with a large seed spread.

**Like.**
- **Vs pure ALS:** every run is +.053 to +.068 above it, and every CI excludes 0. Pure ALS has no signal for users without a train like.
- **Vs ALS aux:** lr 5e-4 seeds 0 and 1 are above on video like (+.0200, +.0173, CIs exclude 0), and seed 2 ties. On music like all three tie (CIs include 0).

**Conclusion.** These are validation statements. They do not change the formal test result (§2.1): U0 is below pure ALS on long view, and tied with ALS aux on like.

### 2.5.5 No-music ablation at three seeds (Wave 3)

The formal no-music configuration (`model.use_music: false`; everything else as U0) was rerun at
seeds 0/1/2 (jobs 47–49):
- seed 0 reproduces formal job 35 bitwise;
- it is compared with U0 seeds 0/1/2 (jobs 44 = 27, 39, 40);
- same seed numbers are not paired initialisations, because the first-layer width differs.

Source: `results/explore/wave3/`.

| arch | seed (job) | best ep | video LV | video like | music LV | music like | mean R@20 | mean N@20 |
|---|---|---:|---|---|---|---|---:|---:|
| U0 | 0 (44 = 27) | 6 | .1260 / .0781 | .1223 / .0589 | .1308 / .0803 | .1218 / .0599 | .1252 | .0693 |
| U0 | 1 (39) | 7 | .1267 / .0770 | .1089 / .0515 | .1344 / .0805 | .1204 / .0586 | .1226 | .0669 |
| U0 | 2 (40) | 5 | .1241 / .0747 | .1129 / .0564 | .1203 / .0728 | .1149 / .0575 | .1180 | .0653 |
| no-music | 0 (47 = 35) | 6 | .1334 / .0802 | .1177 / .0558 | .1352 / .0812 | .1248 / .0595 | .1278 | .0692 |
| no-music | 1 (48) | 4 | .1303 / .0792 | .1254 / .0608 | .1358 / .0807 | .1311 / .0629 | .1306 | .0709 |
| no-music | 2 (49) | 5 | .1334 / .0801 | .1221 / .0569 | .1362 / .0810 | .1179 / .0583 | .1274 | .0691 |

Cells are validation R@20 / NDCG@20.

Same-seed differences (no-music − U0) in val R@20; brackets give the paired user CI:

| seed | 4-task mean | video LV | music LV | video like | music like |
|---:|---:|---|---|---|---|
| 0 | +.0025 | +.0073 [+.0050, +.0097] | +.0045 [+.0022, +.0070] | −.0047 [−.0131, +.0050] | +.0030 [−.0065, +.0118] |
| 1 | +.0080 | +.0036 [+.0009, +.0062] | +.0014 [−.0013, +.0039] | +.0166 [+.0070, +.0273] | +.0106 [+.0004, +.0226] |
| 2 | +.0094 | +.0094 [+.0069, +.0119] | +.0158 [+.0133, +.0187] | +.0093 [−.0008, +.0197] | +.0031 [−.0060, +.0128] |
| mean | **+.0066** | **+.0068** | **+.0072** | +.0071 | +.0056 |

![Figure 8: same-seed differences no-music − U0 in validation R@20 (dots = seeds, bars = 3-seed mean; post-hoc exploratory). Source: results/explore/wave3/paired_nomusic_minus_u0.csv.](../results/explore/wave3/figures/fig2_nomusic_minus_u0.png)

**Observed.**
- **4-task mean:** .1286 ± .0018 vs U0 .1220 ± .0036, higher on 3/3 seeds. The ranges do not overlap: .1274–.1306 vs .1180–.1252.
- **Both long-view tasks:** higher on 3/3 seeds. On video LV every paired CI excludes 0; on music LV, seed 1's CI includes 0. The no-music ranges lie above the U0 ranges (video LV .1303–.1334 vs .1241–.1267; music LV .1352–.1362 vs .1203–.1344).
- **Like:** 5/6 seed-task differences are positive. The single negative one (video like, seed 0) has a CI including 0, so like is not systematically harmed.
- **Not a selection artefact.**
  - On common epochs, no-music is above U0 on video LV at 10/10, 8/8 and 9/9 epochs, and on music LV at 9/10, 8/8 and 9/9.
  - At each task's own best epoch, the lowest no-music seed is above the highest U0 seed on both LV tasks (video .1343 vs .1304, music .1393 vs .1372).
- **Vs pure ALS:** see §2.5.4.

**Interpretation and limits.**
- **Supported:** a descriptive, cross-seed-consistent architecture-level statement. In the current concatenation fusion, removing the music-tower input from the video tower improves validation long-view retrieval, without a systematic cost on like.
- **Not supported:**
  - With n = 3 per arm and no test over training seeds, it is not a cross-seed significance result.
  - It is post-hoc, motivated by the formal ablation that had already been seen on test.
  - The mechanism is not identified: removing the input also removes the video-loss gradient into the shared music tower (E8).

**Not run.**
- A gated music-fusion variant was reviewed and rejected. It would be a pure re-parameterisation of the first video-tower Linear layer, so its effect could not be interpreted (`results/explore/wave3_review/WAVE3_GATE_REVIEW.md`).
- Stop-gradient fusion is future work (§5).

# 3. Error analysis

**Scope.** E1–E5 are post-hoc breakdowns of the formal **test** results (`results/final_analysis/ERROR_ANALYSIS.md`, from `scripts/error_analysis.py`). E6–E8 draw on the exploratory **validation** runs of §2.5 and the read-only diagnostics in `results/explore/wave3_review/`, plus the formal no-music ablation. None of these numbers was used for any model, configuration or epoch choice of the formal v1. We separate **observed** facts from **hypotheses**.

**The thread through E2 and E5–E8.**
- *The gap.* The formal test result shows a long-view gap to pure ALS concentrated in long histories (E2).
- *Not an optimisation setting.* The exploratory runs then ask whether that gap is an optimisation setting. Learning rate and temperature move it only partly and at a cost to like (E6).
- *Hard to measure.* Seed variation of the same size makes small effects hard to read (§2.5.1).
- *Partly a selection compromise.* Part of the remaining gap is the single four-task checkpoint, which stops before long view peaks (E7).
- *Pointing at fusion.* What is left points at the music-to-video fusion (E5, E8). Removing the music input removes U0's significant validation long-view R@20 deficit to pure ALS on 3/3 seeds, but why it does so is not yet established.

**E1. The like-task gain over pure ALS is mostly a coverage effect.**
- *Observed.* 608 of the 1,924 video-like users have no like before T_test, so pure ALS has no signal for them and scores 0.
  - On those users U0 reaches R@20 .0865: +.086\* over pure ALS, and +.018 [−.008, +.047] over ALS aux.
  - On users that do have a prior like, U0 − pure ALS is +.012 [−.005, +.029], which is not significant.
  - Also on those users, U0 is *below* ALS aux: −.022\* (video) and −.015 [−.037, +.005] (music).
  - The music-like numbers are similar.
- *Interpretation (supported).* The three-tower advantage comes from transferring LV behaviour to like through shared towers and history, which is what ALS aux does explicitly. Among users with a prior like, a dedicated like factorisation still ranks better.

**E2. The long-view deficit concentrates in long histories and grows with truncation.**
- *Observed.* In the >20 bucket, U0 − ALS on video LV is −.016\* for users whose full history is ≤ 100 events. It is −.021\* for users whose input was truncated to the last 100 (2,264 users); music is similar.
- ALS fold-in uses all train + val interactions; U0 sees at most 100.
- *Hypothesis.*
  - The capped, attention-pooled representation under-uses long-range preferences.
  - There is a train/test mismatch: rolling training histories are 0–7 days old, while T_test histories also include the 9-day validation window.
  - We did not test either cause.

**E3. U0's empty-history path is not popularity-calibrated.**
- *Observed.*
  - All 437 cold users receive the identical U0 embedding.
  - Its scores correlate only weakly with train LV popularity (Spearman 0.48).
  - The no-history ablation's embedding correlates at 0.92 and matches Pop (R@20 .147 vs .142).
  - In U0 training, the exact cold path (empty history *and* dropped user ID) is seen for only about 1,100 rows per epoch (11,338 empty-history samples × 0.1 ID dropout). In the no-history model it is about 34,500.
- *Hypothesis (supported by the exposure count).* The fallback is under-trained. A popularity prior or a Pop fallback for empty histories is the obvious fix.

**E4. Cold videos are ranked sensibly among themselves but under-scored against warm videos.**
- *Observed.*
  - H is 9.9% of the catalog, but the cold model gives H only 0.9% of its top-20 slots on video LV (1.7% on like).
  - The best GT_H item has median rank 798.
  - The seen reference gives H 12.0% of its top-20 slots.
  - 72% of H videos have video, author *and* music OOV, so they are encoded from metadata alone. In training, ID dropout removes all three IDs of a training video at once with probability 0.001.
- *Hypothesis.* Content-only embeddings live in a lower-norm or off-distribution region of the head space. They need calibration, or training-time exposure to ID-free items, to compete on the full catalog.

**E5. Music adds little beyond video at this sharing level.**
- *Observed.*
  - Per-user U0 R@20 on music LV and video LV correlates at r = 0.88.
  - Music results track the video results throughout: the same signs and nearly the same levels.
  - Only 290 of 7,583 videos share a music ID.
- *Hypothesis for the no-music gain (unverified).* With near one-to-one music, the music input mostly duplicates the video ID, while the video-task gradients flow into a tower that the music tasks also optimise. It acts as a second, competing ID path rather than a source of collaborative signal. The absence of a cold-video benefit is consistent with this, since only 4% of H have an in-vocabulary music ID.

**E6. Optimisation hyperparameters do not reliably close the long-view gap** (post-hoc exploratory, validation; §2.5). This picks up the formal long-view gap of E2 and asks whether it is an optimisation setting.
- *Observed.* A lower learning rate moves the 4-task mean by about one seed std and is driven by the like tasks; its long-view effect changes sign across seeds. τ = 0.1 lifts long view to a tie with pure ALS but costs like by a similar amount. No run has a CI-supported long-view advantage over ALS.
- *Interpretation.* Within the explored range, lr and τ mostly shift the balance between long view and like. Their effects are of the same order as seed-to-seed variation (§2.5.1). This makes an optimisation-hyperparameter explanation of the gap unlikely, but it does not by itself show what the remaining cause is. E7 and E8 examine two candidates.

**E7. Four-task selection versus per-task optimum** (validation epoch curves, `results/explore/wave3_review/selection_vs_task_best.csv`). This is the first candidate for the remaining gap: the checkpoint choice rather than the model.
- *Observed.* The single D1-8 checkpoint (mean val R@20 over 4 tasks) is a compromise between tasks.
  - In 9 of the 10 runs analysed (U0 seeds, lr, τ and the two formal ablations), the long-view tasks peak 2–3 epochs *after* the selected epoch.
  - Likes peak earlier, at epochs 3–7.
  - The validation R@20 left on the table for long view is 0–.011 per task. For example, formal U0 selects epoch 6 while video LV and music LV peak at epoch 8 (+.0040, +.0064).
  - τ = 0.1 is the exception: it selects epoch 8, where long view already peaks.
- *Interpretation.* Part of the U0 long-view gap is plausibly a multi-task selection trade-off, not only model capacity. E6 and E7 are linked: the τ = 0.1 run, the one with the largest long-view gain, is also the only run whose selected epoch coincides with the long-view peak. Whether that coincidence contributes to its gain is not established.
- *Protocol status.* The single checkpoint is a protocol-level choice (one model for four tasks, chosen on validation), not a correctness issue. It was fixed before any test result was seen.
- *What it does not explain.* The selection trade-off is present in every run, including the no-music ablation, so it cannot explain the no-music advantage. That points to the second candidate (E8).

**E8. The no-music long-view advantage replicates across seeds; its mechanism is open** (validation; `results/explore/wave3_review/nomusic_vs_u0_epochwise.json`, `init_fusion_scale.json`). This is the second candidate: the music-to-video fusion itself, already suggested by the formal ablation (§2.4) and E5.
- *Observed.*
  - On validation video LV, the formal no-music run (one seed) is above U0 at every epoch, against each of the three U0 seeds (10/10, 10/10, 9/9 epochs; mean +.007 to +.010).
  - It is above the per-epoch maximum of the three U0 seeds at 9/9 epochs. On music LV it is above at 9/10, 9/10 and 9/9 epochs (8/9 for the envelope).
  - Likes are mixed.
  - Its training loss is lower on all four tasks, including the music tasks.
  - At initialisation the music-tower output fed into the video tower has mean norm 11.3, against 0.16 for the ID embeddings. It carries about 20% of the first-layer pre-activation variance, against 0.7% each for video ID and author.
- *Hypotheses (not yet distinguished):*
  - (i) the near-ID music path competes with the video ID;
  - (ii) video-loss gradients pull the shared music tower away from what the music tasks need;
  - (iii) the input-scale imbalance dominates early training;
  - (iv) the no-music result is a seed accident.

  Because the no-music lead holds at every epoch, it is not a selection artifact of E7.

  Wave 3 (§2.5.5) addresses (iv): the long-view lead recurs on 3/3 no-music seeds against 3/3 U0 seeds, so a single-seed accident is no longer a plausible explanation (descriptively). A stop-gradient fusion would separate (ii) from (i)/(iii); it has not been run. Until then the mechanism is open.

# 4. Limitations

- **The main LV comparison is negative for our model.** U0 is significantly below pure ALS on LV R@20, and on like it is not significantly above ALS aux. v1 was not tuned at all, on validation or on test, although the validation LV gap to ALS would have allowed targeted adjustments.
- **U1 leakage risk.** The snapshot date is unknown and provably no later than 2022-05-02, inside the test period. U1's LV gain (+.015\*) may partly be future information, and offline data cannot separate the two.
  - U1's `onehot_feat3` (644 kept values) and `feat8` (349) may act as ID proxies.
  - U1 is therefore not the headline, and U0 alone does not use any user metadata.
- **Cold-video full-catalog result.** It is not better than random at R@20. The positive cold-video claim holds only for the H-only candidate set.
- **One seed per model.** CIs reflect user sampling, not training variance. The validation curves were noisy: adjacent epochs differed by up to about 0.01 R@20. Differences of about 0.01 between three-tower variants (U0 vs no-music, U1 vs U0) could change sign under another seed.
- **Selection details.**
  - ALS selection hit the grid edge (64 factors) on most tasks.
  - The cold model's selection population is smaller than U0's (17,976 vs 18,664 video-LV users), so cold and normal *validation* scores are not comparable.
- **Feedback timing.** D = 20 min is an assumption. Later non-play actions, or labels that record the final state, cannot be excluded.
- **False-negative mask scope.** It includes the user's later train positives (supervision only, not input). It does not include validation positives, and it does not cover other users' positives, which are true negatives under our definition.
- **Cross-platform numerics.**
  - The formal three-tower metrics were scored on Linux. Re-scoring the same exports on macOS changes at most 2 users per run and frame, through exact ties at the top-20 boundary.
  - The largest effect is 5.4e-5 on one *validation* frame of no-music; it does not change the selected epoch. Every test-frame mean moves by ≤ 1.1e-7 (`results/final_audit/rescore.json`).
  - Export/direct parity is bitwise on the same platform.
- **Dataset scope.** Pure keeps only candidate-pool videos, so sequences are incomplete and absolute metric levels are not comparable to production. The random log was not used.
- **No popularity-bias correction, logQ or hard negatives.** The full softmax is exact, but it is not debiased.
- **Limits of the post-hoc exploratory study (§2.5).**
  - *Three seeds is a small sample.* Means and standard deviations over n = 3 are descriptive, and different seeds also mean different initialisations, so same-seed pairs share only the data order and the random-number stream.
  - *Selection bias.* The explored settings were chosen after seeing the formal results, and the best of several candidates carries a winner's-curse bias. 3 of the 7 pre-allowed targeted adjustments were used.
  - *Test has already been analysed.* The formal test results were studied before this work, so no explore number is an independent confirmation. That is why §2.5 reports validation only and no explore model is evaluated as a new test result.

# 5. What I'd do next

The items are ordered by expected value per hour, and each targets a measured failure mode.

1. **Variance first (cheap, decisive).** Seeds 1 and 2 for U0 and no-music are done (§2.5.1, §2.5.5); U1 replicates are still to do. Report seed-level CIs before acting on any difference of about 0.01.
   - **Music fusion (E5, E8).** The no-music gain replicates on 3 seeds, so next test a stop-gradient fusion: the video tower still consumes the music output, but video losses no longer update the music tower. This separates gradient interference from the input-competition and scale explanations. Normalising the music input to the ID-embedding scale is a second single-variable test. A per-dimension gate before the first Linear is not worth running: it is a pure re-parameterisation.
   - **Multi-task selection (E7).** Report per-task-selected checkpoints as a diagnostic, or select with task-weighted criteria fixed in advance on validation. A single four-task checkpoint leaves 0–.011 long-view validation R@20 unused.
   - **Learning rate (E6).** lr 5e-4 is a candidate default worth a pre-registered, test-independent confirmation (more seeds, and a new held-out period if available), not a v1 change.
2. **Fix the empty-history path (E3).** Add a learned popularity bias, or a Pop/no-history fallback, for users with an empty history. Expected to close most of the 0.10 R@20 cold-user gap at no cost elsewhere.
3. **Close the heavy-user LV gap (E2).**
   - Train with histories frozen at a per-epoch cut, matching the frozen-T_test regime.
   - Compare a longer cap (200), or a mean-pooled long-term summary concatenated with the attention summary.
   - Validate on the >20 bucket. Ensembling with ALS fold-in is a strong, cheap baseline to report alongside.
4. **Like tasks (E1).** Add an explicit LV → like auxiliary target, or score-level mixing with ALS aux. Tune the like/LV task weight and the like oversampling on validation; the design allowed this, but v1 deliberately shipped untuned.
5. **Cold-video calibration (E4).** Train with whole-item ID dropout (all three IDs together, for example p = 0.1 per item). Add a content-only auxiliary loss so ID-free embeddings share the warm distribution. Measure the share of H in the top 20 and the full-catalog R@20 against random.
6. **Music input (E5).** Keep music as a separate retrieval target, but drop or heavily regularise it as a video-tower input. Revisit only on a dataset with real music sharing.
7. **Evaluation extensions.** A rolling per-event protocol as a robustness check, and a sensitivity row with recent (train + val) popularity.
8. **Pretrained representations and stronger architectures (longer term).** Pretrained content or sequence encoders, for example text, visual or LLM-derived item embeddings for cold items, and stronger sequential user models. Any of these would be compared under the same frozen frames, after the confirmations above.

# Appendix A. Reproducibility and provenance

**Commands.** Every step is a CLI with a YAML config. The complete stage-by-stage commands are in `docs/REPRODUCIBILITY.md`; README.md keeps a short quick start. The exploratory figures are rebuilt with `scripts/make_explore_figures.py`.

**Code versions.**
- U0 and U1 were trained at git `9218a76`.
- The cold, no-history and no-music runs were trained at `671dde9`.
- The formal post-processing (bootstrap, buckets, cold evaluation) ran at `671dde9`.
- The post-hoc exploratory runs (§2.5) ran on branch `explore/s9-opt`:
  - Wave 1, jobs 39–43, at `1051f86`;
  - Wave 2, jobs 44–46, at `a158e5e`;
  - Wave 3, jobs 47–49, at `be5ceb9`.

  All are single-variable configs or seed replicates in `configs/explore/` on that branch. The validation results are copied byte-identically to `results/explore/` from final explore commit `9a1a6b7`; see `PROVENANCE.md` there. Run artifacts are in `05_Exploratory_Experiments.zip`.

**Run records.** Each run records its git SHA, config, condition ID, data fingerprint, checkpoint sha256 and selection log, and the export manifest pins every file hash. All 290 artifact checksums verify locally.

**Tests.** 259 CPU tests cover leakage, masking, metrics, frames, loss, bootstrap and parity, and all pass. A real-data CPU smoke run trains → exports → evaluates → checks parity end to end (bitwise).

**Local reproduction.** At HEAD, the M2 data, M3 baselines, M5 main analysis and cold-video evaluation reproduce the formal outputs: byte-identical, or ≤ 3e-16 on bootstrap CI bounds (`results/final_audit/REPRODUCTION.md`).

# Appendix B. Supporting tables

The full tables are in `report/tables/`:
- `t1_main_metrics.md`: all R/N@10/20/50, all models and tasks;
- `t2_paired_diffs.md`: all paired differences;
- `t3_history_buckets.md` and `t3b_bucket_diffs.md`: buckets;
- `t4_cold_users.md`: cold users;
- `t5_cold_video.md`: the complete cold-video table, including R@50, N@10, cold ALS and the seen references.

**Table B1. Test metrics, all models.** Users per task as in §1.5.

| task | model | R@10 | R@20 | R@50 | N@10 | N@20 | N@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| video LV | Pop | .0479 | .0806 | .1388 | .0348 | .0457 | .0621 |
| video LV | ALS (pure) | .0759 | .1244 | .2282 | .0524 | .0686 | .0969 |
| video LV | 3T-U0 | .0740 | .1124 | .1950 | .0549 | .0676 | .0902 |
| video LV | 3T-U1 | .0829 | .1274 | .2184 | .0597 | .0744 | .0994 |
| video LV | no-history | .0642 | .1003 | .1739 | .0456 | .0578 | .0780 |
| video LV | no-music | .0801 | .1221 | .2131 | .0581 | .0721 | .0970 |
| video like | Pop | .0466 | .0716 | .1304 | .0332 | .0403 | .0532 |
| video like | ALS (pure) | .0388 | .0555 | .1004 | .0234 | .0282 | .0387 |
| video like | ALS (aux) | .0610 | .1008 | .1795 | .0356 | .0469 | .0648 |
| video like | 3T-U0 | .0533 | .0913 | .1757 | .0310 | .0417 | .0606 |
| video like | 3T-U1 | .0601 | .0859 | .1616 | .0324 | .0397 | .0570 |
| video like | no-history | .0571 | .0888 | .1588 | .0343 | .0431 | .0586 |
| video like | no-music | .0562 | .0987 | .1680 | .0311 | .0431 | .0588 |
| music LV | Pop | .0507 | .0856 | .1472 | .0362 | .0479 | .0651 |
| music LV | ALS (pure) | .0772 | .1282 | .2297 | .0536 | .0705 | .0982 |
| music LV | 3T-U0 | .0755 | .1162 | .2036 | .0557 | .0691 | .0928 |
| music LV | 3T-U1 | .0875 | .1321 | .2258 | .0621 | .0768 | .1022 |
| music LV | no-history | .0690 | .1052 | .1846 | .0481 | .0603 | .0819 |
| music LV | no-music | .0819 | .1250 | .2162 | .0585 | .0728 | .0975 |
| music like | Pop | .0480 | .0726 | .1358 | .0342 | .0412 | .0552 |
| music like | ALS (pure) | .0433 | .0586 | .0951 | .0256 | .0303 | .0389 |
| music like | ALS (aux) | .0617 | .1054 | .2017 | .0338 | .0463 | .0677 |
| music like | 3T-U0 | .0621 | .1022 | .1793 | .0361 | .0472 | .0646 |
| music like | 3T-U1 | .0573 | .0864 | .1633 | .0325 | .0407 | .0581 |
| music like | no-history | .0560 | .0852 | .1617 | .0342 | .0425 | .0591 |
| music like | no-music | .0649 | .0940 | .1817 | .0354 | .0437 | .0632 |
