---
title: "Leakage-Controlled Multi-Action Three-Tower Retrieval on KuaiRand-Pure"
date: "2026-10-12"
---

<style>
body { font-size: 12.5px; line-height: 1.45; }
h1 { font-size: 1.55em; margin: 0.9em 0 0.35em; padding-bottom: 0.15em; }
h2 { font-size: 1.2em; margin: 0.8em 0 0.3em; }
p, ul, ol, table { margin-top: 0.3em; margin-bottom: 0.5em; }
li { margin: 0.1em 0; }
table { font-size: 0.93em; }
th, td { padding: 2px 7px !important; }
code { font-size: 0.92em; }
img { display: block; margin: 0.3em auto; }
</style>

# Leakage-Controlled Multi-Action Three-Tower Retrieval on KuaiRand-Pure

**Summary.** On KuaiRand-Pure we retrieve videos and their background-music IDs for two actions, long view (LV) and like, giving four full-catalog tasks. A three-tower model (user / video / music) is compared with Popularity and implicit ALS. All models are scored by one shared evaluator on identical test users, with 95% confidence intervals (CIs) from a 1,000-resample paired user bootstrap. Findings for the main model **U0** (user ID + strict-past history, no user metadata), on the formal **test** split:
- It beats Popularity on all four tasks.
- Against Pure ALS it is **worse on both LV tasks** (Recall@20 0.1124 vs 0.1244 on video LV, a −0.012 difference, significant) and better on both like tasks.
- On like it is **not significantly different** from Aux ALS, an ALS that adds the LV signal. This does not mean the two are equivalent.
- In a controlled cold-video holdout, content features rank new videos well *among themselves* but are **not significantly better than random on the full catalog**.
- Removing history hurts LV; removing the music input to the video tower **improves** LV.

**Post-hoc exploratory findings** (validation only, three training seeds per setting, §2.4). The study was designed after the test results were seen, and it does not change any test number:
- **Removing the music-to-video input consistently improved long-view validation Recall@20 across three seeds, without systematically harming like retrieval** (+0.0068 video LV, +0.0072 music LV). Its long-view gap to Pure ALS is no longer significant, which is not a proof of equivalence. The mechanism is not identified.
- Training-seed std (about 0.004 in the 4-task mean) is as large as most tuning effects. lr 5e-4 gains +0.0048, mostly on like. τ = 0.1 trades like for LV.

Unabridged report and audit evidence: [`REPORT_FULL.md`](REPORT_FULL.md). All metrics and differences: [`tables/`](tables/).

# 1. Design

## 1.1 Data and cleaning

**Inputs.** The loader reads only the two standard logs, `user_features_pure.csv` and `video_features_basic_pure.csv`. It refuses the random log and the monthly statistics file (§1.2).

**Data.** There are 1,436,609 raw rows from 27,077 users on 7,551 videos, and the catalog has 7,583 videos. Long view is positive on 33.2% of rows and like on only 1.85%.

**Music.** It is almost one-to-one with video: of 7,201 non-zero music IDs, only 112 are shared, by at most 15 videos. `music_id = 0` (204 videos) is treated as a "no music" sentinel. This is our assumption; the data dictionary does not confirm it. It is never a music candidate or music ground truth, and gets a learnable NO_MUSIC embedding.

**Cleaning.** Before the split, keyed on (user, video, time):
- 21,987 exact duplicates are collapsed;
- 1,486 conflict groups each keep one whole representative row;
- 2 label-conflicting events are dropped and counted;
- there is no k-core filtering.

This leaves 1,413,134 events, with all users and videos kept.

## 1.2 Temporal split and leakage control

**Split.** The split is global and chronological, at CST midnights; t < T goes to the earlier split. No timestamp spans two splits.

| split | days (2022) | events | share | LV / like positives |
|---|---|---:|---:|---:|
| train | 04-09 – 04-16 | 997,078 | 70.6% | 339,386 / 18,771 |
| val | 04-17 – 04-25 | 208,393 | 14.7% | 69,354 / 3,980 |
| test | 04-26 – 05-08 | 207,663 | 14.7% | 65,812 / 3,610 |

**No future-leaking aggregates.** `video_features_statistic_pure.csv` aggregates each video's plays, likes and shares over the whole log month. Using it at time t would encode test-period behaviour, so the loader refuses it, and none of its 51 columns appears in any processed file. The video tower reads only static metadata (type, upload type/date, music type, tags, duration, size) and IDs.

**Train-only fitting.**
- Fitted on train events only (assertion-guarded): every vocabulary, scaler, Popularity count and ALS factor.
- Validation only selects the ALS grid point and the three-tower epoch.
- Test is evaluated once, after selection, and every run records `test_used_for_selection = false`.

**Strict-past history and seen mask.**
- One function builds histories at the event time (training), at T_val (train events) and at T_test (train + val events).
- It keeps only t < t_pred, so same-timestamp siblings are also excluded (10% of rows share one), and it caps the input at the latest 100 events.
- Feedback younger than 20 min is hidden, because the dataset does not say when it becomes known (an assumption).
- The seen mask is built separately from *all* earlier interactions, uncapped.

**User metadata (U0 vs U1).** `user_features_pure.csv` is one undated snapshot. One user's registration age bounds it at no later than 2022-05-02, inside the test period; there is no lower bound. Of the 21 candidate fields:
- 2 are time-dependent by definition (`register_days_range`, cumulative `is_video_author`);
- 19 have unknown historical availability;
- none can be shown to be available at prediction time.

This is not evidence that every field leaks. We report **U0** (main: user ID + history) and **U1** (U0 + the 21 fields, same configuration and seed, with a leakage caveat).

## 1.3 Three-tower model

<img src="figures/arch_threetower.png" alt="Formal v1 three-tower architecture" width="88%">

*Figure 1. The formal three-tower architecture (U0; U1 adds the snapshot fields to the user tower).*

**Towers.**
- **Music tower:** music ID + type → MLP + LayerNorm → 128-d.
- **Video tower:** video ID, author, metadata and the music-tower output → 128-d. There is **one shared music tower**: it runs once per catalog forward, and each video reuses that output, so video losses also train it.
- **User tower:** user ID + a 2-layer Transformer with attention pooling over the last 100 history events (each token is the shared video embedding plus feedback, play ratio, recency and position) → 128-d. U1 adds 21 field embeddings.

**Heads.** Each target × action has its own `Linear(128→64) + L2` head (user 4, video 2, music 2), and the score is cosine similarity.

**Unseen IDs.** The embedding rows for padding and out-of-vocabulary (OOV, not seen in training) IDs are fixed at zero, and ID dropout 0.1 trains this path, so unseen videos are scored from content. The model has 3.77M parameters.

## 1.4 Contrastive objective: duplicates, false negatives, task balance

**Samples.** Every train event positive for either action is a sample (345,365; 18,771 likes), with four task masks (music tasks need `music_id ≠ 0`) and its own rolling strict-past history.

**Loss.** Per task, an exact full softmax over all train-vocabulary items (7,529 videos, 7,151 music IDs), so no sampling or logQ correction is needed:

$$\ell_r=-\log\frac{\exp(\cos(u_r,i_{y_r})/\tau)}{\sum_{c\notin \mathrm{FN}(r)}\exp(\cos(u_r,i_c)/\tau)},\quad \tau=0.05,\qquad \mathrm{FN}(r)=\mathrm{Pos}_{\text{train}}(u_r,\text{task})\setminus\{y_r\}$$

**Duplicate positives.** Candidate columns are unique items, so rows with the same label share one column. This covers repeats by one user, different users, and videos sharing a music ID. A row's own label is never masked.

**False negatives.**
- `Pos_train(u, task)` holds u's *train* positives for the same target and action. Those columns (except the label) are set to −inf, so they get zero weight and zero gradient.
- Validation and test positives are never used.
- Pos_train includes positives that occur **after** the sample's time (about 50% of masked columns). It is a *supervision* mask that stops the loss from pushing a user's other known positives down as negatives. It is **never a model input**: history stays strictly in the past.
- On four real batches, an independent recomputation matched every label and mask, and the loss to 8e-7.

**Task balance.**
- The loss is the equal-weight sum of the four per-task *mean* losses; like rows are not oversampled.
- A batch has about 1,006 LV rows but 56 like rows, so each like row carries about 18× the weight of an LV row: the actions are balanced at the task level.
- Like also benefits from towers and history shared with LV; only the heads are action-specific.
- The 1:1 weight was not tuned, and like gradients are noisier.

**Optimisation.** AdamW (lr 1e-3), batch 1,024, FP32, deterministic, at most 30 epochs with patience 3 on mean validation R@20, one seed. **v1 was not tuned**, although its validation LV R@20 was below ALS (video 0.1260 vs 0.1339, music 0.1308 vs 0.1361), which the design allows as a trigger for targeted adjustments. These were explored afterwards on a separate branch (§2.4).

## 1.5 Evaluation protocol

**Ground truth.** Each user is represented once at T_test. The ground truth GT(u) is the de-duplicated set of u's test positives, minus three kinds of item:
- out-of-catalog (OOC) items, i.e. items outside the ranked catalog;
- the music sentinel;
- items u interacted with before T_test (the *seen mask*, which is also applied at ranking).

**Catalog and metrics.** All 7,583 videos or 7,201 music IDs are ranked, and train-unseen items stay in the catalog and GT. Recall@K (R@K) = hits in the top K / |GT|. NDCG@K uses binary gain with IDCG over min(|GT|, K), so it also rewards ranking hits higher. Users are weighted equally, and ties break by catalog row. A *difference* is model A − model B on the same users (an absolute difference in the metric). It is *significant* if its 95% CI excludes 0; a non-significant difference does not imply equivalence.

**Population.** Evaluated users have non-empty GT and at least one history event; users with no history are reported separately.

| task | users | GT pairs | seen-excluded GT pairs | OOC positive events | sentinel-excluded events | cold users (separate) |
|---|---:|---:|---:|---:|---:|---:|
| video LV | 18,752 | 63,281 | 859 | 0 | – | 437 |
| video like | 1,924 | 3,380 | 63 | 0 | – | 68 |
| music LV | 18,595 | 61,595 | 913 | 0 | 1,670 | 434 |
| music like | 1,895 | 3,307 | 72 | 0 | 67 | 67 |

**Fair comparison.**
- Every model on a task must match one `frame_id`, a hash of users, GT and mask.
- **Popularity** uses train counts.
- **Pure ALS** uses train-only item factors for the task's own action. User vectors are folded in from pre-T_test history, and the same 8-point validation grid is used on every task. Users or items without signal score −inf, not random vectors.
- **Aux ALS** (like tasks only) adds 0.3 × the LV signal to the like signal.
- CIs come from 1,000 user resamples shared (paired) by all models on a frame.

# 2. Results

§2.1–2.3 report formal **test** results from `results/m5/`. §2.4 is a separate post-hoc **validation** study. In difference tables, a star (\*) marks a 95% CI that excludes 0 (significant). Full metrics are in `tables/`.

## 2.1 Main comparison

**Table 1. Absolute test R@20 / NDCG@20 per model.** Bold marks the best value per column; "–" means Aux ALS is defined for like only.

| model | video LV | video like | music LV | music like |
|---|---|---|---|---|
| Pop | 0.0806 / 0.0457 | 0.0716 / 0.0403 | 0.0856 / 0.0479 | 0.0726 / 0.0412 |
| Pure ALS | **0.1244** / 0.0686 | 0.0555 / 0.0282 | **0.1282** / 0.0705 | 0.0586 / 0.0303 |
| Aux ALS | – | **0.1008 / 0.0469** | – | **0.1054** / 0.0463 |
| U0 (main) | 0.1124 / 0.0676 | 0.0913 / 0.0417 | 0.1162 / 0.0691 | 0.1022 / **0.0472** |
| U1 (+ snapshot) | 0.1274 / 0.0744 | 0.0859 / 0.0397 | 0.1321 / 0.0768 | 0.0864 / 0.0407 |

**Table 2. Paired test differences in R@20, A − B [95% CI].** These are absolute differences of the values in Table 1, on the same users.

| comparison (A − B) | video LV | video like | music LV | music like |
|---|---|---|---|---|
| U0 − Pop | +0.032\* [+0.029, +0.035] | +0.020\* [+0.007, +0.032] | +0.031\* [+0.027, +0.034] | +0.030\* [+0.015, +0.043] |
| U0 − Pure ALS | **−0.012\*** [−0.016, −0.008] | +0.036\* [+0.022, +0.049] | **−0.012\*** [−0.016, −0.008] | +0.044\* [+0.029, +0.058] |
| U0 − Aux ALS | – | −0.009 [−0.025, +0.005] | – | −0.003 [−0.019, +0.014] |
| U1 − Pure ALS | +0.003 [−0.001, +0.007] | +0.030\* [+0.017, +0.043] | +0.004 [−0.000, +0.008] | +0.028\* [+0.015, +0.042] |
| U1 − U0 | +0.015\* [+0.012, +0.018] | −0.005 [−0.016, +0.005] | +0.016\* [+0.013, +0.019] | −0.016\* [−0.027, −0.004] |

**U0 vs Pure ALS.** On LV, U0's NDCG@20 difference is not significant (−0.001 / −0.001), so U0 trails ALS in recall depth rather than in ranking its hits.

**Effect of history length.** The deficit is a heavy-user effect: U0 − Pure ALS on LV is +0.028\* / +0.037\* for users with 1–5 history events, but −0.017\* / −0.016\* for users with more than 20, who are 72% of LV users.

**U1.** It reaches Pure-ALS level on LV, but this may partly be leakage (§1.2). **We do not claim that the three-tower model beats ALS.**

## 2.2 Controlled cold-video experiment

**Setup.**
- H is 750 train-seen videos (10%), drawn from train data only and stratified by train interaction count.
- Every H interaction is removed from targets, all histories (including at test time), candidates, false-negative sets, vocabularies and the cold Pop/ALS fits; 12 automated checks verify the removal.
- The cold model selects its epoch on validation without H ground truth.
- GT_H is each user's test positives on H that the user had not seen before T_test (5,936 LV / 305 like users). It is ranked against (a) the full catalog and (b) H only.

<img src="figures/fig2_cold_video.png" alt="Cold-video Recall@20, full catalog (left) and H only (right)" width="92%">

*Figure 2. Test Recall@20 on GT_H with 95% CIs. Left: candidates are the full catalog; right: candidates are the 750 H videos only. All models shown were trained without H. Cold Pure ALS is 0 by construction, because it has no factors for H.*

**Table 3. Cold-video test R@20: absolute values for the cold U0 and two cold-condition baselines, then the paired differences cold U0 − baseline [95% CI].** "Random" is the closed-form per-user expectation of a uniformly random ranking.

| task | candidates | cold U0 | cold Pop | random | cold U0 − cold Pop | cold U0 − random |
|---|---|---:|---:|---:|---|---|
| video LV | full catalog | 0.0041 | 0.0000 | 0.0027 | +0.0041\* [+0.0026, +0.0058] | +0.0014 [−0.0000, +0.0032] |
| video LV | H only | 0.1045 | 0.0207 | 0.0269 | +0.0838\* [+0.0756, +0.0923] | +0.0776\* [+0.0703, +0.0848] |
| video like | full catalog | 0.0066 | 0.0000 | 0.0027 | +0.0066 [+0.0000, +0.0164] | +0.0039 [−0.0027, +0.0137] |
| video like | H only | 0.0672 | 0.0082 | 0.0268 | +0.0590\* [+0.0328, +0.0869] | +0.0404\* [+0.0158, +0.0682] |

**H only.** Metadata, author and music carry a real ranking signal among new videos. Here cold Pop is a fixed catalog order, not a random ranking, since all its scores are 0.

**Full catalog.** The cold model is not significantly different from random, because cold videos rarely reach the top 20. Cold ALS cannot score H at all.

**Context.** The normally trained U0 reaches 0.111 / 0.433 on the same frame, but it saw H in training, so it is context only.

## 2.3 Ablations

Each ablation changes one configuration flag; everything else is as in U0. The first row gives U0's absolute test values (from Table 1). The other rows give the paired test difference *ablation − U0*; the CIs are in `tables/t2_paired_diffs.md`.

| | video LV | video like | music LV | music like |
|---|---|---|---|---|
| U0, absolute R@20 / NDCG@20 | 0.1124 / 0.0676 | 0.0913 / 0.0417 | 0.1162 / 0.0691 | 0.1022 / 0.0472 |
| no history − U0, R@20 | −0.012\* | −0.002 | −0.011\* | −0.017\* |
| no history − U0, NDCG@20 | −0.010\* | +0.001 | −0.009\* | −0.005 |
| no music input − U0, R@20 | **+0.010\*** | +0.007 | **+0.009\*** | −0.008 |
| no music input − U0, NDCG@20 | **+0.004\*** | +0.001 | **+0.004\*** | −0.004 |

**History** is the main source of LV personalisation.

**Music input.** Removing it from the video tower **improves** LV significantly, a negative result for that design choice. We report it as is and kept the specified architecture. §2.4 shows that the effect replicates across seeds on validation.

## 2.4 Post-hoc exploratory study (validation only)

**Scope.**
- After the test results above had been analysed, U0 was retrained on a separate branch under single-setting changes and seed replicates. This gives 11 runs, all with exit code 0 and bitwise export/direct parity.
- **All numbers below are validation metrics**, on the same validation users and masks as the baselines. They are read by scripts that refuse test files. They are not test results and do not replace U0.
- Two kinds of uncertainty are reported separately. *mean ± std* is over three training seeds; it describes training-seed variability and is descriptive only (n = 3). CIs (and \*) are paired user bootstraps; they cover user sampling, not training randomness.
- Same-config reruns reproduce the formal U0 (job 27) and the formal no-music run (job 35) bitwise.

<div style="break-inside: avoid; page-break-inside: avoid;">

**Table 4. Validation Recall@20 per setting.** Absolute rows: mean ± std over three training seeds. Δ rows: the mean same-seed difference vs U0, with the min–max over the three seeds in braces (a seed range, not a CI). The τ row is a single seed, compared with U0 seed 0, and its \* is the bootstrap CI.

| setting | video LV | music LV | video like | music like | 4-task mean |
|---|---|---|---|---|---|
| U0 (lr 1e-3, τ 0.05) | 0.1256 ± 0.0014 | 0.1285 ± 0.0073 | 0.1147 ± 0.0069 | 0.1190 ± 0.0037 | 0.1220 ± 0.0036 |
| no music input | **0.1324** ± 0.0018 | **0.1357** ± 0.0005 | 0.1218 ± 0.0039 | 0.1246 ± 0.0066 | **0.1286** ± 0.0018 |
| Δ no-music − U0 | +0.0068 {+0.004, +0.009} | +0.0072 {+0.001, +0.016} | +0.0071 {−0.005, +0.017} | +0.0056 {+0.003, +0.011} | +0.0066 {+0.003, +0.009} |
| lr 5e-4 | 0.1266 ± 0.0046 | 0.1320 ± 0.0068 | 0.1229 ± 0.0061 | 0.1253 ± 0.0076 | 0.1267 ± 0.0035 |
| Δ lr 5e-4 − U0 | +0.0010 {−0.005, +0.004} | +0.0035 {−0.010, +0.015} | +0.0082 {+0.003, +0.016} | +0.0063 {+0.002, +0.011} | +0.0048 {+0.003, +0.006} |
| τ 0.1 (seed 0), Δ vs U0 seed 0 | +0.0085\* | +0.0094\* | −0.0114\* | −0.0083 | −0.0005 |

</div>

<img src="../results/explore/wave3/figures/fig2_nomusic_minus_u0.png" alt="Same-seed differences no-music minus U0, validation R@20" width="74%">

*Figure 3. Validation Recall@20 of no-music minus U0 at the same seed. Dots are seeds 0/1/2; bars are the 3-seed mean. The architectures differ in layer size, so the same seed does not give the same initialisation.*

**No-music: the most consistent finding.**
- Both long-view tasks and the 4-task mean are higher on 3/3 seeds. The no-music seed ranges lie entirely above the U0 ranges.
- Like is not systematically lower: 5/6 differences are positive, and the one negative has a CI that includes 0.
- The lead holds at nearly every epoch and at per-task best epochs, so it is not an artefact of early stopping on the 4-task mean.

**Comparison with ALS on validation.** We separate "numerically above" from "CI-supported".
- Every U0 seed is significantly below Pure ALS on video LV, and 2/3 are below on music LV.
- With no music input, all six long-view R@20 CIs vs Pure ALS include 0. The gap is no longer significant, but this is not a demonstration of equivalence, and no run has a CI-supported R@20 advantage.
- On long-view NDCG@20, no-music is above Pure ALS on 6/6 seed-task pairs (+0.003 to +0.004, nominal 95% CIs exclude 0). This is a secondary metric, uncorrected for multiple comparisons and not confirmed on test.

**Tuning.**
- Across the three U0 seeds, the 4-task mean spans 0.0072. This is the observed range, not a statistical threshold, but it exceeds user-bootstrap uncertainty, so single-seed gains of that size should be read with caution.
- lr 5e-4 is higher on 3/3 seeds, but by about one seed std, mostly on like, and LV changes sign across seeds.
- τ = 0.1 (one seed) moves performance from like to LV without changing the mean.

**Limits.**
- The study is post-hoc: it was designed after the test results, it was partly motivated by the test-side no-music ablation, and choosing the best setting carries a winner's-curse bias.
- The mechanism is not identified, because removing the music input also removes the video-loss gradient into the shared music tower (E5).

Details: [`REPORT_FULL.md`](REPORT_FULL.md) §2.5 and [`docs/EXPLORATORY_RESULTS.md`](../docs/EXPLORATORY_RESULTS.md).

# 3. Error analysis

Post-hoc breakdowns of the test results (`results/final_analysis/ERROR_ANALYSIS.md`). They were not used for any choice in the formal v1, but they did motivate the later exploratory study (§2.4).

**E1. The like gain over Pure ALS is mostly coverage.**
- *Observed:* 608 of the 1,924 video-like users have no prior like, so Pure ALS cannot score them. On them, U0 − Pure ALS is +0.086\*. On users with a prior like, U0 − Pure ALS is only +0.012 [−0.005, +0.029], and U0 is *below* Aux ALS (−0.022\*).
- *Interpretation:* U0's like advantage is transfer from LV, which Aux ALS achieves explicitly.

**E2. The LV deficit grows with history length.**
- *Observed:* in the >20 bucket on video LV, U0 − Pure ALS is −0.016\* when the history fits in 100 events and −0.021\* when it was truncated. ALS fold-in uses every interaction.
- *Hypotheses (untested):* the capped, pooled representation loses long-range preference, and training histories (0–7 days) are shorter than test histories (16 days).

**E3. The empty-history path is not popularity-calibrated.**
- *Observed:* on the 437 video-LV users without history, U0 reaches R@20 0.039 versus Pop 0.142. Their shared embedding's scores correlate with popularity at only 0.48 (Spearman; no-history ablation: 0.92, R@20 0.147), and U0 sees that exact path only about 1,100 times per epoch.
- *Interpretation:* the fallback is under-trained.

**E4. Cold videos are under-scored against warm ones.**
- *Observed:* H is 9.9% of the catalog but gets 0.9% of the cold model's top-20 slots, and the best GT_H item has median rank 798. For 72% of H, all three IDs (video, author, music) are unknown, while training drops all three at once with probability 0.001.
- *Hypothesis:* ID-free embeddings are off-distribution.

**E5. Music adds little beyond video here.**
- *Observed:* per-user U0 R@20 on music LV and video LV correlates at 0.88.
- *Hypothesis (unverified):* with near one-to-one music, the music input acts as a competing second ID path (or its shared-tower gradients interfere). The no-music gain replicates on 3 validation seeds (§2.4), but these explanations are not yet separated.

# 4. Limitations

- **Main LV result is negative:** U0 is below Pure ALS on LV, and on like it is not significantly different from Aux ALS. v1 was not tuned.
- **User metadata:** U1's fields cannot be shown to be available at prediction time, so its LV gain may partly be leakage. U0 avoids the risk but uses no user metadata, a documented deviation from "ID + usable metadata + history".
- **Cold videos:** the positive result holds only for the H-only candidate set.
- **One seed per formal model:** test CIs reflect user sampling only. The exploratory three-seed study (§2.4) puts the training-seed std at about 0.004 (4-task mean validation R@20). It is post-hoc and validation-only, so no explore result is an independent test confirmation.
- **Assumptions:** a 20-minute feedback delay; the false-negative (FN) mask uses later train positives as supervision; ALS selection hit the grid edge (64 factors).
- **Numerics and scope:** re-scoring the Linux exports on macOS changes at most 2 users per frame (test means ≤ 1.1e-7). KuaiRand-Pure keeps candidate-pool videos only.

# 5. What I'd do next

1. **Music fusion:** the no-music gain replicates on 3 seeds (§2.4). Next, test a stop-gradient fusion to separate gradient interference from input competition and scale, then re-confirm on a fresh held-out period. U1 seed replicates are still to do.
2. **Empty-history fallback (E3):** a learned popularity bias or a Pop fallback for users without history.
3. **Heavy-user LV (E2):** per-epoch frozen training histories, a longer cap or a long-term mean-pooled summary, and an ensemble with ALS fold-in.
4. **Like (E1):** an LV→like auxiliary target or score mixing with Aux ALS, plus validation-tuned task weights and like oversampling.
5. **Cold calibration (E4):** whole-item ID dropout and a content-only auxiliary loss.
6. **Pretrained representations and stronger architectures (longer term):** pretrained content or sequence encoders, for example LLM or vision item embeddings for cold items, evaluated on the same frozen frames.

**Reproducibility.** U0/U1 were trained at commit `9218a76`, the cold and ablation runs at `671dde9`, and the exploratory runs on branch `explore/s9-opt` (`1051f86`, `a158e5e`, `be5ceb9`; results in `results/explore/`). Each run records its SHA, config, data fingerprint and selection log. 259 CPU tests pass, and every stage reproduces locally (`results/final_audit/REPRODUCTION.md`, `docs/REPRODUCIBILITY.md`).
