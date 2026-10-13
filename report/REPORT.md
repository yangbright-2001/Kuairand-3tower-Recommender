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

**Summary.** On KuaiRand-Pure we retrieve videos and their background-music IDs for two actions, long view (LV) and like, giving four full-catalog tasks. A three-tower model (user / video / music) is compared with Popularity and implicit ALS, all scored by one shared evaluator on identical test frames with 1,000-resample paired user-bootstrap CIs. Findings for the main model **U0** (user ID + strict-past history, no user metadata):
- It beats Popularity on all four tasks.
- Against pure ALS it is **worse on both LV tasks** (−.012 Recall@20, significant) and better on both like tasks.
- On like it is **not significantly different** from ALS with an auxiliary LV signal.
- In a controlled cold-video holdout, content features rank new videos well *among themselves* but are **not better than random on the full catalog**.
- Removing history hurts LV; removing the music input to the video tower **improves** LV.

**Post-hoc exploratory findings** (validation only, three seeds per arm, §2.4; designed after the test results were seen, and they do not change any test number):
- **Removing the music-to-video input consistently improved long-view validation Recall@20 across three seeds, without systematically harming like retrieval.** The gain is +.0068 on video LV and +.0072 on music LV, and the 4-task mean moves from .1220 ± .0036 to .1286 ± .0018. The long-view R@20 gap to pure ALS is then no longer CI-supported, which is not a proof of equivalence. The mechanism is not identified.
- Seed variation (±.004) is as large as most tuning effects. lr 5e-4 gains +.0048, mostly on like. τ = 0.1 trades like for LV.

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

**Towers.**
- **Music tower:** music ID + type → MLP + LayerNorm → 128-d.
- **Video tower:** video ID, author, metadata and the music-tower output → 128-d. There is **one shared music tower**: it runs once per catalog forward, and each video reuses that output, so video losses also train it.
- **User tower:** user ID + a 2-layer Transformer with attention pooling over the last 100 history events (each token is the shared video embedding plus feedback, play ratio, recency and position) → 128-d. U1 adds 21 field embeddings.

**Heads.** Each target × action has its own `Linear(128→64) + L2` head (user 4, video 2, music 2), and the score is cosine similarity.

**Unseen IDs.** PAD/OOV rows are fixed at zero, and ID dropout 0.1 trains this path, so unseen videos are scored from content. The model has 3.77M parameters.

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

**Optimisation.** AdamW (lr 1e-3), batch 1,024, FP32, deterministic, at most 30 epochs with patience 3 on mean validation R@20, one seed. **v1 was not tuned**, although its validation LV R@20 was below ALS (video .1260 vs .1339, music .1308 vs .1361), which the design allows as a trigger for targeted adjustments. These were explored afterwards on a separate branch (§2.4).

## 1.5 Evaluation protocol

**Ground truth.** Each user is represented once at T_test. GT(u) is the de-duplicated set of u's test positives, minus out-of-catalog items, the music sentinel, and items u interacted with before T_test (the seen mask, also applied at ranking).

**Catalog and metrics.** All 7,583 videos or 7,201 music IDs are ranked, and train-unseen items stay in the catalog and GT. Recall@K = hits/|GT|; NDCG@K uses binary gain with IDCG over min(|GT|, K). Users are weighted equally, and ties break by catalog row.

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
- **ALS** uses train-only item factors, with user vectors folded in from pre-T_test history, and the same 8-point validation grid on every task. Users or items without signal score −inf, not random vectors.
- **ALS aux** adds 0.3 × the LV signal for like.
- CIs come from 1,000 user resamples shared (paired) by all models on a frame.

# 2. Results

§2.1–2.3 report formal **test** results from `results/m5/`. §2.4 is a separate post-hoc **validation** study. A star (\*) marks a 95% CI that excludes 0; full metrics are in `tables/`.

## 2.1 Main comparison

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

**U0 vs pure ALS.** On LV, U0's NDCG@20 difference includes 0 (−.001 / −.001), so U0 trails ALS in recall depth rather than in ranking its hits.

**Effect of history length.** The deficit is a heavy-user effect: U0 − ALS on LV is +.028\* / +.037\* for users with 1–5 history events, but −.017\* / −.016\* for users with more than 20, who are 72% of LV users.

**U1.** It reaches ALS level on LV, but this may partly be leakage (§1.2). **We do not claim that the three-tower model beats ALS.**

## 2.2 Controlled cold-video experiment

**Setup.**
- H is 750 train-seen videos (10%), drawn from train data only and stratified by train interaction count.
- Every H interaction is removed from targets, all histories (including at test time), candidates, false-negative sets, vocabularies and the cold Pop/ALS fits; 12 automated checks verify the removal.
- The cold model selects its epoch on validation without H ground truth.
- GT_H is each user's unseen test positives on H (5,936 LV / 305 like users), ranked against (a) the full catalog and (b) H only.

<img src="figures/fig2_cold_video.png" alt="Cold-video Recall@20, full catalog (left) and H only (right)" width="92%">

**Table 3. Cold 3T-U0 minus cold-condition baselines, R@20 [95% CI].**

| task | candidates | cold 3T-U0 | − cold Pop | − random |
|---|---|---:|---|---|
| video LV | full catalog | 0.0041 | +.0041\* | +.0014 [−.0000, +.0032] |
| video LV | H only | 0.1045 | +.084\* | +.078\* |
| video like | full catalog | 0.0066 | +.0066 [.0000, .0164] | +.0039 [−.0027, +.0137] |
| video like | H only | 0.0672 | +.059\* | +.040\* |

**H only.** Metadata, author and music carry a real ranking signal among new videos. Here cold Pop is a fixed catalog order, since all its scores are 0, and "random" is the per-user closed-form expectation.

**Full catalog.** The cold model is not distinguishable from random, because cold videos rarely reach the top 20. Cold ALS cannot score H at all.

**Context.** The normally trained U0 reaches 0.111 / 0.433 on the same frame, but it saw H in training, so it is context only.

## 2.3 Ablations

One configuration flag changed, everything else as U0; paired difference vs U0.

| Δ vs U0 | video LV | video like | music LV | music like |
|---|---|---|---|---|
| no history, R@20 | −.012\* | −.002 | −.011\* | −.017\* |
| no history, NDCG@20 | −.010\* | +.001 | −.009\* | −.005 |
| no music input, R@20 | **+.010\*** | +.007 | **+.009\*** | −.008 |
| no music input, NDCG@20 | **+.004\*** | +.001 | **+.004\*** | −.004 |

**History** is the main source of LV personalisation.

**Music input.** Removing it from the video tower **improves** LV significantly, a negative result for that design choice. We report it as is and kept the specified architecture. §2.4 shows that the effect replicates across seeds on validation.

## 2.4 Post-hoc exploratory study (validation only)

**Scope.**
- After the test results above had been analysed, U0 was retrained on a separate branch under single-setting changes and seed replicates. This gives 11 runs, all with exit code 0 and bitwise export/direct parity.
- **All numbers below are validation metrics** (M3 val masked frames), read by scripts that refuse test files. They are not test results and do not replace U0.
- CIs are paired user bootstraps. They cover user sampling, not training randomness. Seed statistics use n = 3, so they are descriptive.
- Same-config reruns reproduce the formal U0 (job 27) and the formal no-music run (job 35) bitwise.

<div style="break-inside: avoid; page-break-inside: avoid;">

**Table 4. Validation R@20, three seeds per setting (mean ± std); brackets give the per-seed range of the same-seed difference vs U0.**

| setting | video LV | music LV | video like | music like | 4-task mean |
|---|---|---|---|---|---|
| U0 (lr 1e-3, τ 0.05) | .1256 ± .0014 | .1285 ± .0073 | .1147 ± .0069 | .1190 ± .0037 | .1220 ± .0036 |
| no music input | **.1324** ± .0018 | **.1357** ± .0005 | .1218 ± .0039 | .1246 ± .0066 | **.1286** ± .0018 |
| Δ no-music − U0 | +.0068 [+.004, +.009] | +.0072 [+.001, +.016] | +.0071 [−.005, +.017] | +.0056 [+.003, +.011] | +.0066 [+.002, +.009] |
| lr 5e-4 | .1266 ± .0046 | .1320 ± .0068 | .1229 ± .0061 | .1253 ± .0076 | .1267 ± .0035 |
| Δ lr 5e-4 − U0 | +.0010 [−.005, +.004] | +.0035 [−.010, +.015] | +.0082 [+.003, +.016] | +.0063 [+.002, +.011] | +.0048 [+.003, +.006] |
| τ 0.1 (seed 0 only), Δ vs U0 s0 | +.0085\* | +.0094\* | −.0114\* | −.0083 | −.0005 |

</div>

<img src="../results/explore/wave3/figures/fig2_nomusic_minus_u0.png" alt="Same-seed differences no-music minus U0, validation R@20" width="74%">

**No-music: the most consistent finding.**
- Both long-view tasks and the 4-task mean are higher on 3/3 seeds. The no-music seed ranges lie entirely above the U0 ranges.
- Like is not systematically lower: 5/6 differences are positive, and the one negative has a CI that includes 0.
- The lead holds at nearly every epoch and at per-task best epochs, so it is not an artefact of early stopping on the 4-task mean.

**Comparison with ALS on validation.** We separate "numerically above" from "CI-supported".
- Every U0 seed is significantly below pure ALS on video LV, and 2/3 are below on music LV.
- With no music input, all six long-view R@20 CIs vs pure ALS include 0. The gap is no longer significant, but this is not a demonstration of equivalence, and no run has a CI-supported R@20 advantage.
- On long-view NDCG@20, no-music is above pure ALS on 6/6 seed-task pairs (+.003 to +.004, nominal 95% CIs exclude 0). This is a secondary metric, uncorrected for multiple comparisons and not confirmed on test.

**Tuning.**
- Across the three U0 seeds, the 4-task mean spans .0072. This is the observed range, not a statistical threshold, but it exceeds user-bootstrap uncertainty, so single-seed gains of that size should be read with caution.
- lr 5e-4 is higher on 3/3 seeds, but by about one seed std, mostly on like, and LV changes sign across seeds.
- τ = 0.1 (one seed) moves performance from like to LV without changing the mean.

**Limits.**
- The study is post-hoc: it was designed after the test results, it was partly motivated by the test-side no-music ablation, and choosing the best setting carries a winner's-curse bias.
- The mechanism is not identified, because removing the music input also removes the video-loss gradient into the shared music tower (E5).

Details: [`REPORT_FULL.md`](REPORT_FULL.md) §2.5 and [`docs/EXPLORATORY_RESULTS.md`](../docs/EXPLORATORY_RESULTS.md).

# 3. Error analysis

Post-hoc breakdowns of the test results (`results/final_analysis/ERROR_ANALYSIS.md`). They were not used for any choice in the formal v1, but they did motivate the later exploratory study (§2.4).

**E1. The like gain over pure ALS is mostly coverage.**
- *Observed:* 608 of the 1,924 like users have no prior like, so pure ALS cannot score them; on them U0 gains +.086\*. On users with a prior like, U0 − pure ALS is only +.012 [−.005, +.029], and U0 is *below* ALS aux (−.022\*).
- *Interpretation:* U0's like advantage is transfer from LV, which ALS aux achieves explicitly.

**E2. The LV deficit grows with history length.**
- *Observed:* in the >20 bucket, U0 − ALS is −.016\* when the history fits in 100 events and −.021\* when it was truncated. ALS fold-in uses every interaction.
- *Hypotheses (untested):* the capped, pooled representation loses long-range preference, and training histories (0–7 days) are shorter than test histories (16 days).

**E3. The empty-history path is not popularity-calibrated.**
- *Observed:* on the 437 users without history, U0 reaches R@20 .039 versus Pop .142. Their shared embedding correlates with popularity at only 0.48 (no-history ablation: 0.92, R@20 .147), and U0 sees that exact path only about 1,100 times per epoch.
- *Interpretation:* the fallback is under-trained.

**E4. Cold videos are under-scored against warm ones.**
- *Observed:* H is 9.9% of the catalog but gets 0.9% of the cold model's top-20 slots, and the best GT_H item has median rank 798. For 72% of H, all three IDs (video, author, music) are unknown, while training drops all three at once with probability 0.001.
- *Hypothesis:* ID-free embeddings are off-distribution.

**E5. Music adds little beyond video here.**
- *Observed:* per-user U0 R@20 on music LV and video LV correlates at 0.88.
- *Hypothesis (unverified):* with near one-to-one music, the music input acts as a competing second ID path (or its shared-tower gradients interfere). The no-music gain replicates on 3 validation seeds (§2.4), but these explanations are not yet separated.

# 4. Limitations

- **Main LV result is negative:** U0 is below pure ALS on LV and only tied with ALS aux on like. v1 was not tuned.
- **User metadata:** U1's fields cannot be shown to be available at prediction time, so its LV gain may partly be leakage. U0 avoids the risk but uses no user metadata, a documented deviation from "ID + usable metadata + history".
- **Cold videos:** the positive result holds only for the H-only candidate set.
- **One seed per formal model:** test CIs reflect user sampling only. The exploratory three-seed study (§2.4) puts training-seed variation at about ±.004 val R@20. It is post-hoc and validation-only, so no explore result is an independent test confirmation.
- **Assumptions:** a 20-minute feedback delay; the FN mask uses later train positives as supervision; ALS selection hit the grid edge (64 factors).
- **Numerics and scope:** re-scoring the Linux exports on macOS changes at most 2 users per frame (test means ≤ 1.1e-7). KuaiRand-Pure keeps candidate-pool videos only.

# 5. What I'd do next

1. **Music fusion:** the no-music gain replicates on 3 seeds (§2.4). Next, test a stop-gradient fusion to separate gradient interference from input competition and scale, then re-confirm on a fresh held-out period. U1 seed replicates are still to do.
2. **Empty-history fallback (E3):** a learned popularity bias or a Pop fallback for users without history.
3. **Heavy-user LV (E2):** per-epoch frozen training histories, a longer cap or a long-term mean-pooled summary, and an ensemble with ALS fold-in.
4. **Like (E1):** an LV→like auxiliary target or score mixing with ALS aux, plus validation-tuned task weights and like oversampling.
5. **Cold calibration (E4):** whole-item ID dropout and a content-only auxiliary loss.
6. **Pretrained representations and stronger architectures (longer term):** pretrained content or sequence encoders, for example LLM or vision item embeddings for cold items, evaluated on the same frozen frames.

**Reproducibility.** U0/U1 were trained at commit `9218a76`, the cold and ablation runs at `671dde9`, and the exploratory runs on branch `explore/s9-opt` (`1051f86`, `a158e5e`, `be5ceb9`; results in `results/explore/`). Each run records its SHA, config, data fingerprint and selection log. 259 CPU tests pass, and every stage reproduces locally (`results/final_audit/REPRODUCTION.md`, `docs/REPRODUCIBILITY.md`).
