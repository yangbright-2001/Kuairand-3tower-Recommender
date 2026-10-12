# Project Plan

`docs/assignment.md` is the authoritative spec. This file tracks the agreed workflow, design principles, decisions, and open questions.
Items marked **OPEN** are not decided. Do not treat them as conclusions.

## 1. Workflow (agreed 2026-10-11)

```
S0 Env/skeleton -> S1 Dataset Audit (A: schema, B: plan, C: implement)
  -> P1 Protocol review [user sign-off: APPROVED 2026-10-11, see 3b]
  -> S2 Data pipeline -> S3 Shared evaluator -> S4 Pop/ALS baselines
  -> D1 Three-tower v1 design review [user sign-off: APPROVED 2026-10-11 (Revision 2), see 3d]
  -> S5 Three-tower v1 -> S6 Export + retrieve() -> S7 Experiments -> S8 Report/packaging
  -> S9 Optional improvements (only after a complete report draft)
```

Milestones:

| ID | Content | Acceptance |
|----|---------|------------|
| M0 | uv + Py3.12 env, package skeleton, pytest, .gitignore. Verify the server | `uv sync` and `pytest` pass locally. The server can SSH, has a CUDA torch build, and has a data checksum |
| M1 | Dataset audit plus P1 | A deterministic audit command produces JSON + markdown. P1 decisions are recorded here |
| M2 | Split, labels, catalogs, past-only histories, train-only popularity | Leakage, masking, and cold-holdout tests pass |
| M3 | Shared full-catalog evaluator, plus Pop and ALS (video and music) | Hand-computed metric tests pass. All 4 tasks report @10/20/50 with user, excluded, and OOC counts. Values are in [0,1]. Recall@K is non-decreasing in K (NDCG@K is not asserted monotone). Anomalies are explained. No assumption about which baseline wins |
| M4 | Three-tower v1 (after D1 sign-off) | The CPU smoke test passes. A full run goes through the same evaluator. The exported vs direct score parity test passes |
| M5 | Bootstrap CIs, history-length buckets, cold-video experiment, at least 1 ablation | Each has a config, a command, and a results file |
| M6 | 5-8 page report (Markdown to PDF), artifacts, agent logs | Uploaded |

Target timeline (PT):
- Sun: M0-M3. D1 is discussed during M2/M3.
- Mon AM: M4, then start M5 on the A100s.
- Mon PM: analysis and report.
- About 18:00 Mon: soft code freeze.

## 2. Design principles

- Correctness of data handling and evaluation comes before model performance. Well-explained negative results are fine.
- Temporal correctness:
  - Global chronological split of the two standard logs, roughly 70/15/15.
  - Histories hold only events strictly before the prediction time, capped at the most recent 100.
- `video_features_statistic_pure.csv` is never used as model features.
- Popularity, vocabularies, scalers, and any fitted statistics come from train only.
- One shared evaluator for all models. User-level bootstrap, paired across models.
- Test-first for correctness-critical logic: split/leakage, history masking, metrics, retrieval score parity.
- The target is a well-designed, competitive three-tower model, not a toy. Build a verified v1 first, then explore improvements.
- Reproducibility:
  - Configs and seeds.
  - Every run directory records its config and git SHA.
  - Raw data is read-only. No secrets and no large artifacts in git.

## 3. Agreed decisions

- Environment: uv + Python 3.12, the same locally and on the server (pyproject.toml + uv.lock). We revisit only if there are compatibility problems.
- Compute:
  - The M1 Mac is the main dev loop: audit, pipeline, tests, evaluator, baselines, smoke tests.
  - The 8xA100 node handles parallel full runs: seeds, ablations, cold-start retrains, sweeps.
  - Server workdir is `/local-scratch/liang/`. It has no backup, so important artifacts get synced back promptly.
- Data scope: only the standard logs are used for the main experiments. The random log is excluded. It may be considered later for extra analysis, without assuming it gives unbiased evaluation.
- Audit Phase A reads only the 2 standard logs, `user_features_pure.csv`, and `video_features_basic_pure.csv`.
- Report: Markdown exported to a 5-8 page PDF.
- Delivery: code and small results go to GitHub. Weights and embeddings go to Google Drive. The exact layout is decided in M6.
- Collaboration:
  - The agent implements, tests, and runs experiments.
  - Decisions that affect correctness, architecture, or result interpretation go to the user with assumptions, alternatives, and tradeoffs.

## 3b. P1 Decisions & Rationale (approved by the user 2026-10-11)

Audit numbers come from `results/audit/AUDIT.md` (provisional reference split unless stated). M2 recomputes them under the final midnight split.

### P1-1 Evaluation protocol
- **Main protocol: A, a frozen user representation evaluated against a per-user set.**
  - One user embedding per user, computed at prediction time T_test.
  - GT is the set of the user's test-period positives.
  - Rationale: this maps directly onto exported embeddings, `retrieve()`, evaluated-user counts, and user-level bootstrap.
  - Protocol B (rolling history, per-event evaluation) is an optional S9 add-on and is not committed.
- **Information boundary:**
  - At test time, history = events with t < T_test, including val-period events.
  - Every learnable parameter is fit on train only. Val is used only for early stopping and tuning.
  - ALS: item factors are trained on train only. Test user vectors are folded in from train+val interactions. An ALS row without fold-in is reported as a reference.
  - Popularity: train counts only.
  - Validation mirrors this exactly: frozen at T_val, with train-only history.
- **GT and metrics:**
  - GT(u) = the deduped set of items positive for the action in the test period.
  - Removed from GT, each counted and reported:
    - out-of-catalog items
    - `music_id=0` (music tasks only)
    - seen-masked items
    - events dropped as label conflicts
  - Evaluated user = GT is non-empty after removals, and the user has at least 1 history event before T_test.
  - Recall@K = hits / |GT|. This keeps Recall@K non-decreasing in K.
  - NDCG@K uses binary gain. IDCG is computed over min(|GT|, K) relevant items.
  - Per-user metrics are averaged with equal weight.
  - Deterministic tie-break: score descending, then item index ascending.
  - Bootstrap: 1000 user-level resamples, with the same indices shared by all models (paired). Report CIs for each model and for model differences.
- **Cold users** (no history before T_test; audit: about 2.3% of long_view and 3.3% of like test users):
  - Excluded from the main metrics, counted, and reported separately for Pop and the three-tower model.
  - The main population is then exactly covered by the 1-5 / 6-20 / >20 buckets.

### P1-2 User history
- **Main: H1.**
  - All deduped events, with an action/feedback encoding per event (long_view, is_like, is_click, and so on).
  - Strictly t < t_pred, keeping the most recent 100.
  - Rationale: it keeps negative feedback and exposure context. 1.85% like sparsity makes per-action histories (H3) nearly empty.
- **Ablation: H2** (positives only).
- **Same-timestamp events:**
  - Events at exactly t_pred are always excluded. 10% of rows sit in same-user same-timestamp multi-video groups.
  - Within one timestamp, order is deterministic by video_id.
- **One function** `history(u, t_pred)` serves train (t_pred = the target event time), val (T_val), and test/export (T_test).
- **Bucket length** = the number of events actually fed to the model (capped at 100). The same bucketing is used for Pop, ALS, and the three-tower model.
- **History cap vs seen-item mask** are separate:
  - The model input uses the most recent 100 events before t_pred.
  - The M2 mask uses **all** legal interactions before t_pred, uncapped.
  - They are produced by separate functions. The mask is never derived from the truncated sequence.

### P1-3 Temporal split
- Cut points at the nearest CST midnight:
  - T_val = 2022-04-17 00:00 CST
  - T_test = 2022-04-26 00:00 CST
- Audit split shares: 70.4 / 14.7 / 14.8. No rows fall exactly on either cut.
- Rule: t < T goes to the earlier split, and t >= T to the later one. A timestamp is never split.
- Rationale: the cut points do not depend on the cleaning rules, are easy to explain, and differ from the row-quantile candidates by at most 0.5 pp.

### P1-4 Data cleaning
- **Exact duplicates:** collapse to 1 event (21,987 redundant rows).
- **Conflict groups** (1,486; mostly `hourmin` / `date`):
  - Keep one whole representative row: the one with max `play_time_ms`, with ties broken deterministically.
  - No per-field max. This keeps labels consistent within the row.
- **Label conflicts:** events whose long_view or is_like disagree within the group are dropped entirely and counted.
- **Tabs 5/6:** kept after dedup. `tab` is not a feature. Tabs 5/6 are discussed separately in the report.
- **Not done:** no relabelling of long_view (it agrees with the official rule on 99.76% of rows), and no k-core filtering.
- **`duration_ms=0` videos:** kept, with duration treated as missing.
- **Report:** rows, events, and positives per action before and after cleaning.

### P1-5 `music_id=0`
- It is not a music retrieval candidate. GT hits on it count as "sentinel excluded" (long_view 2.5%, like 1.8% of test positive rows).
- Its video interactions are fully kept for the video tasks and for training.
- The video tower uses a dedicated "no music" embedding for it.

### P1-6 Catalog and masking
- **Catalogs:**
  - Video: all 7,583 metadata videos.
  - Music: the 7,201 non-zero metadata music ids, including 30 that never occur in the logs.
- **Masking (M2, main):**
  - Mask every item the user interacted with before t_pred. For music, mask the non-zero music ids of those videos.
  - The mask is computed only from pre-t_pred data and never from test labels.
  - Masked GT items are removed from the denominator and counted (audit: about 1.3% of long_view and 1.7% of like test positive rows).
- **M0** (no mask) is reported as a secondary view.

### P1-6b Train-unseen items and users (correctness constraints; implementation in D1)
- **Items stay in place:** train-unseen items stay in the full catalog and in GT. Their GT counts are reported per model and task.
- **No untrained parameters:** no randomly initialised, never-trained parameter may influence any score.
- **Popularity:** score = train positive count for the action (music = the sum over its videos). Unseen items score 0 and fall last under the tie-break.
- **ALS:**
  - Unseen items are explicitly marked unscorable and scored -inf.
  - The default initialisation of empty columns in `implicit` is not relied on.
  - A zero vector is not used either, because ALS scores can be negative.
- **Three-tower:**
  - Video, author, music, and user ID vocabularies are built from train only and frozen. In the cold-video condition, held-out videos are excluded from the vocabulary.
  - Out-of-vocabulary IDs map to a shared OOV index or a zeroed ID component.
  - OOV items are represented from content: video metadata, the music tower, and in-vocab author IDs.
  - OOV users are represented from their history (plus user features in U1/U2).
  - History tokens use the same mapping.
- **Train-unseen users** (about 5% of test users are not in train under the reference split):
  - They take the user-ID OOV path.
  - ALS folds them in from train+val interactions using trained item factors only.
  - Report the number of OOV-user evaluated users, and optionally metrics split by seen/unseen in train.
- **`retrieve()` consistency:**
  - One catalog index (item_id to row) is shared by the evaluator, export, and `retrieve()`.
  - The export covers every catalog row, with a scorable flag for Pop and ALS.
  - The evaluator and `retrieve()` share the same top-K function, tie-break, and M2 mask. In `retrieve()` the mask is on by default and can be turned off.
  - The score-parity test covers every catalog row, including unseen and OOV items.

### P1-7 Repeated positives (evaluation)
- GT is a deduped binary set. Music GT is deduped at the music level.
- Audit context: 66,946 test long_view rows collapse to 66,482 pairs, and only 27 train long_view pairs repeat.
- Training-side handling is D1: in-batch duplicate-column masking, same-user false negatives, and shared music.

### P1-8 Features
- **Video features:**
  - `video_features_basic` fields are usable.
  - `visible_status` is dropped: it is constant and describes the "now" state.
  - `video_features_statistic_pure.csv` is never used.
- **Evidence that the user snapshot time cannot be proven safe:**
  - `user_features_pure.csv` is a single snapshot with an undocumented date.
  - Read-only check: user 25929 has `register_days=22` but logs from 2022-04-10, so the snapshot is no later than 05-02. 12 users give upper bounds before 05-08. There is no lower bound.
  - So the snapshot may fall inside the test period, and cannot be shown to precede T_val.
  - `user_active_degree` values (`2_14_day_new`) are inconsistent with `register_days`. The fields may not come from the same moment.
- **Relation to the assignment:**
  - `docs/assignment.md` line 24: "A user tower using ID, usable metadata, and interaction history."
  - The assignment does not define "usable". Reading it as "provably time-safe" is **our own interpretation**, not the recruiter's definition.
  - Under that reading, U0 literally omits user metadata. The user accepted this risk and made U1 required to cover the requirement.
- **Main: U0.** The user tower uses user ID + interaction history only, with no user snapshot features.
- **Required: U1.**
  - U0 plus `onehot_feat*` (NaN as its own category), `is_live_streamer` (-124 as its own category), `is_video_author`, and `register_days_range`.
  - Same protocol, reported on the main evaluation with a paired bootstrap against U0.
  - This is the full "ID + usable metadata + history" user tower, with an explicit leakage-risk caveat.
- **Optional: U2.** U1 plus `user_active_degree` and the follow/fans/friend `*_range` fields. Higher risk, exploratory only.
- **Report:**
  - Gains from U1/U2 may be real signal or snapshot leakage, and offline evaluation cannot tell them apart.
  - The main table and conclusions use U0.
  - The interpretation of "usable" is argued explicitly.
- **Other:** `is_lowactive_period` is dropped as constant.

### P1-9 Controlled cold-video experiment
- **Selection:**
  - About 10% of train-seen videos, stratified by train popularity, with a fixed seed.
  - No val/test labels are used.
  - The number of selected videos with test positives is reported as an outcome, not used as a selection criterion.
  - This replaces the audit pool, which conditioned on test positives.
- **Removal:**
  - Every interaction with a held-out video is removed from training targets, all histories (including pre-T_test history at test time), in-batch negatives, Pop, ALS, and the ID vocabularies.
  - Music/author IDs absent from the cold train set follow P1-6b (OOV).
  - Report the share of H whose author or music still appears in cold train (audit: about 23% and 4%).
- **Fixed evaluation frame**, computed once from the cleaned log with H included, and shared by all models and conditions:
  - H is drawn once.
  - GT_H(u) = test positives on H, deduped, excluding videos interacted with before T_test.
  - The M2 mask is evaluation-only and uses all pre-T_test interactions.
  - Population: users with non-empty GT_H and at least 1 non-H history event before T_test.
- **Two candidate sets,** reported for every model:
  - (a) the full catalog (with the M2 mask)
  - (b) H only (with the M2 mask)
- **Comparison groups** (labels must not be mixed):
  1. **Cold-video model:** the three-tower model trained in the cold condition.
  2. **Cold-condition baselines:**
     - Pop and ALS trained on the same H-removed data. By P1-6b they score H as 0 / -inf.
     - A random-ranking reference, computed **per user from that user's actual candidate count** C_u after masking, then averaged over users:
       - E[Recall@K | u] = min(K, C_u) / C_u
       - E[NDCG@K | u] = (|GT_u| / C_u) · Σ_{r=1..min(K, C_u)} 1/log2(r+1) / IDCG_u
  3. **Seen-video Reference:** the normally trained three-tower model, Pop, and ALS.
     - This is a reference point, **not a guaranteed upper bound**, and not a same-condition cold baseline.
- **Main claims** come only from 1 vs 2, with a paired user bootstrap. 3 is shown as context.
- **Optional ablations** in the cold condition: no music input, no author input. They use the same H and the same frame.
- **Scope:** headline numbers are for video/long_view and video/like. Like may be too sparse on H; report GT and user counts, and fall back to descriptive results if needed.
- **Isolation:** the main tables use normally trained models only.

### Optional, not committed
- Protocol B as a supplementary analysis.
- A "train+val recent popularity" sensitivity row.

## 3c. M3 decisions (approved by the user 2026-10-11; P1 text above unchanged)

Full design: `.cursor/plans/m3_评估器与基线_fd18bcec.plan.md` (Revision 1).

- **Information boundary in code.**
  - `train_view()` (split == 0) is the only input to any fit.
  - `history_view(split)` (t < t_pred; val ⊆ train, test ⊆ train+val) is used only to fold in user vectors.
  - Fitted Pop counts and ALS factors are frozen. Scorers have no `fit`.
  - GT exists only in the `EvalFrame`. The evaluator applies the mask.
- **Frame identity.**
  - `population_id` = sha256 of (schema, split, target, action, t_pred, ordered catalog ids, ordered evaluated user ids).
  - `frame_id` = `population_id` + masked flag + GT CSR + mask CSR content.
  - Every model on a task must match the `frame_id`. Mask and no-mask share `population_id` (same users), with different `frame_id`s.
- **Q1 ALS signal.**
  - Like tasks have two variants:
    - `als_like_pure` (aux_w = 0) is the **headline** (O1).
    - `als_like_aux` (aux_w = 0.3, fixed; O2) is an extra comparison.
  - Long-view tasks use long_view only.
  - One `build_confidence()` defines r_ui = 1[action] + aux_w·1[long_view], p = 1[r > 0], C = 1 + α·r. It is shared by fit and fold-in.
- **Q2 strict no-signal.**
  - ALS users with no usable signal get all scores -inf and metrics 0. They stay in the population, and `n_no_signal` is reported.
  - No-signal users are:
    - fold-in: no scorable history interaction
    - no-fold-in: train-OOV, or an empty train row
  - Zero or random vectors are never used.
- **Q3 ALS scorable items.**
  - An item is scorable iff its column has non-zero train signal under the current config.
  - Unscorable items score -inf. Their GT stays in the denominator and is counted per model/task.
  - This is a tightening extension of P1-6b: an exact ALS solution for such a column is the zero vector, which P1-6b forbids. Pop still scores 0 (P1-6b).
- **Q4 no-mask.** Same population as masked (fixed main population). The GT keeps seen items, and no mask is applied at ranking.
- **ALS rows.**
  - Main row: frozen train item factors, with user vectors folded in from `history_view` (val: train; test: train+val).
  - No-fold-in reference: the raw `fit()` train user factors, no recomputation, test only.
  - Re-folding train history is a diagnostic only (`als_refold_train`), never called "no-fold-in".
- **Q5 tuning.**
  - Grid for every task: factors {64,128} × reg {0.01,0.1} × α {10,40}, 20 iterations, seed 0.
  - Selection: val masked Recall@20, tie-break NDCG@20.
  - No test tuning, and no refit on train+val.
  - A larger grid and multiple seeds are optional (time permitting).

## 3d. D1 Three-tower v1 decisions (approved by the user 2026-10-11, Revision 2; P1 3b and M3 3c unchanged)

Full design and revision history (R1, R2): `.cursor/plans/d1_三塔_v1_设计_1d3a531b.plan.md`. This section is the binding summary.

### D1-1 Architecture
- **Towers.** Three towers, each an MLP + LayerNorm (no BatchNorm) that outputs a 128-d base embedding.
  - **Music tower:** music_id embedding + `music_type`.
  - **Video tower:** video_id + author embedding + the **shared** `MusicTower` 128-d output + video metadata. Metadata is `video_type`, `upload_type`, `upload_dt`, `music_type`, mean-pooled tags (≤ 3), and z-scored `video_duration` / `server_width` / `server_height`, each with a missing flag.
  - **User tower:** user_id + history encoder (+ user features in U1).
- **Music input to the video tower.** Video losses back-propagate into the music tower. `music_id = 0` uses a learnable NO_MUSIC embedding, which is trained because 0-music videos occur in train.
- **Heads.** Separate 64-d projections per target × action, each Linear(128→64) followed by L2 normalisation:
  - user: 4 heads (video_lv, video_like, music_lv, music_like)
  - video: 2 heads
  - music: 2 heads
  - Training logits are cos/τ. Scoring and export use cos, which gives the same ranking.
- **Excluded inputs:** `tab`, the statistic file, popularity features.
- **Ablation flags** (config only, run in M5): `video.use_music`, `video.use_author`, `user.use_history`.

### D1-2 User history (P1-2 event set unchanged)
- One function, `build_history_batch(hidx, users, t_pred, cfg)`, builds history in every setting. It calls `model_history(..., 100)`: strict t < t_pred, same-timestamp events excluded, cap 100.
  - **Train:** t_pred = target event time, and `hidx` holds train events only.
  - **Val:** t_pred = T_val.
  - **Test / export:** t_pred = T_test, with `hidx` built from `history_view`.
- **Token content:**
  - `v_base` of the video from the shared video tower (Q3a). OOV and cold videos are represented by content.
  - The feedback block: 8 binary fields + a play_ratio bucket.
  - A `feedback_hidden` flag.
  - A log-bucketed recency t_pred − t_event.
  - A position embedding.
- **Encoder (Q4a):** 2-layer Transformer (d = 128, 4 heads) + learnable-query attention pooling.
  - Empty history is NaN-safe: one key is unmasked before the attention, and the output is multiplied by (len > 0).
- **Feedback availability (Q8, fixed-delay rule).**
  - **Rule:** if age = t_pred − t_event < D = 20 min, all feedback fields of that token are zeroed (play_ratio included) and `feedback_hidden = 1`. Video id, time and recency are kept.
  - The rule reads only t_pred and t_event. It never reads `play_time_ms` or labels. Train, val and test use the same function.
  - **D is a conservative assumption, not a proven upper bound on feedback completion.** Basis:
    - The official docs define `time_ms` only as "timestamp of this interaction". `play_time_ms` is the final view time. Like, follow, comment and the other actions have no timestamp of their own.
    - Pure logs are incomplete sequences: the median gap between neighbouring events is about 36 s beyond the play time. So neighbouring gaps cannot identify the semantics.
    - 312 likes occur with `play_time_ms = 0`.
    - The maximum play time is 965 s.
  - **Impact:**
    - Train: 2.6% of tokens hidden; 42% of targets have at least 1 hidden token.
    - At T_val / T_test: only a few hundred tokens.
  - **Residual risks, disclosed in the report:**
    1. The meaning of `time_ms` is unknown.
    2. Non-play actions taken more than D later cannot be excluded.
    3. Labels may be the final state of the interaction (for example, after an unlike).
  - This is an implementation refinement, not a P1-2 change. D is not tuned.

### D1-3 Training and loss
- **Samples.** Every train event that is positive for any action: 345,365 rows, of which 18,771 are likes.
  - Each row carries 4 task masks. The music tasks require music ≠ 0.
- **Loss (Q1).** Train-seen full softmax per task. Cross-entropy over cos/τ on the valid rows.
  - **Candidates:** all items in the current condition's train vocab.
    - video: 7,529 in the normal condition
    - music: 7,151 non-zero
  - **No logQ.** The softmax is exact.
  - **Evaluation is unchanged:** val/test still score the full metadata catalog (M3 frames).
  - **Duplicate positives:** candidate columns are unique, so rows with the same item share one label column. Videos with shared music share the music column.
  - **False negatives (Q7a):** for row r (user u, task (t,a), label y_r), column c ≠ y_r is set to −inf if c ∈ Pos_train(u, t, a).
    - Pos_train contains same-task train positives only. For music, it holds the non-zero music of those videos.
    - It is used only to remove loss terms. It never becomes a feature or an eval input. It may include later train positives of the same user, which is documented.
  - **Assignment fit:** the assignment asks for a contrastive retrieval objective that handles duplicate positives and false negatives in the batch. It does not mandate in-batch negative sampling.
  - **In-batch variant:** optional only. Any correction must use the exact inclusion probability of the actual batch sampler and be Monte-Carlo-tested, or the variant is reported as uncorrected.
- **Action balance (Q6).** Per-task mean, and L = Σ of the 4 task losses with equal weights. `like_oversample = 1` in v1.
  - If k > 1 is used (targeted adjustment only):
    - Each like row is duplicated k−1 times, and the copies have long_view masks off.
    - An epoch = one permutation of the expanded list.
    - Per-task valid-row counts are logged.
- **Optimisation and stability:**
  - τ = 0.05.
  - AdamW, lr 1e-3, with 300 warmup steps.
  - Weight decay 0.01 on Linear and Transformer weights; 0 on embeddings, biases and LayerNorm.
  - clip_grad_norm = 1.0.
  - Batch 1024 (to be confirmed by the benchmark).
  - ≤ 30 epochs, patience 3 on val.
  - ID dropout p = 0.1 per ID slot. Item-side dropout is drawn once per step and shared by the candidates and the history tokens.
  - FP32, TF32 off (any change must be justified by the benchmark and recorded).
  - Fail fast on a non-finite loss or grad norm, with a debug dump. Never skip silently.
  - A task with no valid rows in a batch is skipped, not computed as 0/0. The label logit is asserted finite.
- **Training history (Q2a):** rolling history(u, t_event). The mismatch with the frozen T_test plus a 13-day window is documented.

### D1-4 OOV and train-unseen (implements P1-6b)
- **Fixed-zero rows (Q5b).** PAD and OOV rows of every embedding table are fixed at zero by a constant buffer mask, so their gradient and Adam update are 0 and weight decay is 0. NO_MUSIC is learnable.
- **Content path.** OOV items are scored from trained content parameters only, so `item_scorable` is all True and `gt_unscorable = 0`.
- **Cold users** use the trained empty-history + zero-ID path and are reported separately (P1-1).

### D1-5 Cold-video condition (run in M5; P1-9 unchanged)
- The only entry point is `exclude_video_ids = H` → `fit_condition`. H is absent from training targets, all histories, the candidate columns, Pos_train and every vocab. The fixed evaluation frame is still built from the unfiltered log.
- **One condition per artifact set (implementation constraint).** The vocabs, numeric scalers, metadata vocabs, feature mappings (video/music features, encoded event indices), checkpoint and scorer of a cold run all come from the same cold condition. Mixing any of them with the normal U0 artifacts is forbidden.
  - The run directory records a `condition_id`: a hash of the condition name and the sorted H ids.
  - The checkpoint, the export manifest and the scorer each verify that their `condition_id` matches before scoring.

### D1-6 User metadata
- **U0 (main):** ID + history.
- **U1 (required: implemented, trained, evaluated):**
  - Features: U0 + `onehot_feat0..17` (NaN as its own category), `is_live_streamer` (−124 as its own category), `is_video_author`, `register_days_range`.
  - Each field gets a 16-d embedding. The vocab is fitted on users with train events. Values seen on fewer than 5 train users, and unseen values, map to a fixed zero row.
  - **Comparable protocol:** U1 reuses U0's selected config, seed, early-stopping rule and frames, with no separate tuning. If U0's config changes, U1 is rerun.
- **Risks to record:**
  - P1-8 snapshot leakage.
  - `onehot_feat3` (1,471 values) and `feat8` (454 values) may act as ID proxies for train-OOV users. Report the values kept after the threshold, and the U1 effect for OOV vs in-vocab users.
- **U2:** optional.

### D1-7 Evaluation, export, retrieve()
- **Shared evaluator.** Reuse `build_frames`, and assert each frame_id equals the one in `results/baselines/frames_summary.json`. Output per-user parquet in the M3 format. Report the masked, nomask and cold-user frames.
- **Export.** Item matrices [n_catalog, 64] per (target, action) cover every catalog row, including OOV items. Also exported:
  - user matrices at T_val and T_test
  - id lists
  - the T_test seen-mask CSR
  - `manifest.json`: git SHA, config hash, checkpoint sha256, t_pred, frame_ids, data fingerprint, condition_id
- **Reported metrics come only from `ExportedScorer`** (numpy float32 U @ I.T).
- **`DirectScorer`** reloads the checkpoint in a separate process and rebuilds features from raw ids. Both scorers use the same inference settings: eval mode, FP32, TF32 off, deterministic, the same catalog, the same M2 mask and the same `rank_topk`.
- **`retrieve(user_id, target, action, k, mask_seen=True)`** shares the scoring and `apply_mask` + `rank_topk` code with `ExportedScorer`. An unknown user raises an error.
- **Parity checks (strict).**
  1. Direct vs export scores on ≥ 2,048 users over every catalog row, for all 4 tasks: max |Δ| ≤ 1e-5. Users must cover train-OOV, the 1-5 / 6-20 / >20 buckets and cold users. Report max and p99 |Δ|.
  2. Direct vs export top-50 are identical. **No near-tie exemption.** Any difference fails and dumps the score gap and the boundary; the cause is diagnosed and the user is consulted.
  3. `retrieve()` vs `ExportedScorer`: bitwise-equal scores, the same top-K, and a mask equal to the frame CSR. `mask_seen=False` includes seen items.
  4. Direct vs export per-user metrics are identical on the val and test masked frames.

### D1-8 Model selection (implementation constraint)
- Tuning, early stopping, checkpoint selection and config selection use **validation only**: the val masked frame at T_val with train-only history.
- Q9a: one model, selected on the mean val masked R@20 over the 4 tasks, with NDCG@20 as the tie-break.
- Test is evaluated once per selected checkpoint, after selection is frozen. Test numbers never feed back into any choice: config, epoch, τ, oversampling, U0/U1 config.
- The runner computes test metrics only after the best checkpoint is fixed and records the selection log in `run_meta.json`.

### D1-9 Tests (hand-built fixtures, CPU < 60 s)
- **Leakage:**
  - L1: strict t < t_pred, same-timestamp exclusion and the cap, at all three t_pred.
  - L2: hide when age < D, visible at age == D.
  - L3: changing `play_time_ms` or labels of hidden tokens leaves the inputs bitwise unchanged.
  - L4: changing any event with t ≥ t_pred leaves the inputs unchanged.
  - L5: val events with age < D at T_test are hidden.
- **Loss and masks:**
  - F1: duplicate positives share a column.
  - F2: same-user positives are masked; the label is not.
  - F3: same-task only.
  - F4: shared-music behaviour.
  - F5: the music-0 row and the label-only row.
  - F6: hand-computed CE; under the cold condition, H is absent from the candidates and from Pos_train.
- **Gradients:** each task produces finite, non-zero grads on its own heads, and zero grads on the other heads. Video-only loss reaches the music tower; it does not with `use_music = False`.
- **Stability:** empty history, a batch with no likes, a label-only row, and an OOV user with empty history are all finite. An injected NaN fails fast.
- **OOV:** PAD/OOV rows stay exactly 0 after optimizer steps, and perturbing them leaves scores unchanged.
- **Oversampling:** row and mask counts.
- **Export:** parity checks 1-4 on the fixture.
- **Cold condition:** an artifact from another `condition_id` is rejected.
- **Smoke:** 2 epochs on the fixture run end to end. A real-data CPU run uses `--max_steps 5`.

### D1-10 Execution tiers
- **M4 required (full v1, not reduced):** everything above.
- **M4 minimum complete deliverable:**
  1. U0 (seed 0), trained on real data with val early stopping.
  2. Export, then the shared evaluator via `ExportedScorer` on the masked, nomask and cold-user frames, with frame_ids matching M3.
  3. Parity checks 1-4 pass on real data, with a working `retrieve()` example.
  4. U1 trained and evaluated under the same protocol.
  5. `metrics.csv`, per-user parquet, `run_meta.json` and `benchmark.json`.
- **Targeted adjustments.** These are used only if U0 val masked R@20 is below Pop on any task, or clearly below ALS on long_view.
  - Diagnose first:
    - per-task loss curves and train convergence
    - per-tower grad norms
    - the FN-mask share
    - the train label rank
    - overfitting a small subset
    - embedding collapse
  - Then change one thing at a time, chosen on val only, with ≤ 7 extra configs in total:
    1. τ ∈ {0.03, 0.1}
    2. lr
    3. `like_oversample = 4` (like tasks only)
    4. a mean-pooling diagnostic. This only tests whether the Transformer is at fault. It does not replace the main model.
  - Every adjustment is logged in `results/threetower/tuning.csv`.
- **Optional, after the M5 required items:**
  - seeds {1, 2}
  - U2
  - an in-batch variant
  - Q2c delayed training
  - cold-condition no-music / no-author ablations

### D1-11 Server, reproducibility, artifacts (implementation constraints)
- **Code source: the private GitHub repo** `Kuairand-3tower-Recommender`.
  - **Authentication:**
    - The server uses a **read-only Deploy Key**: a fresh `ed25519` key generated on the server. The user adds the public key without write access.
    - A per-repo `~/.ssh/config` Host alias sets `IdentitiesOnly yes`.
    - No personal SSH keys are read, copied or forwarded, and there is no agent forwarding.
  - **Fallback if authentication is blocked:** report it first, then `git archive <SHA>` + rsync, with a sha256 check and a `GIT_SHA` file.
- **Fixed SHA.**
  - The server checks out `git checkout --detach <SHA>`.
  - Every GPU run records `git rev-parse HEAD` in `run_meta.json`.
  - A run refuses to start if `git status --porcelain` is non-empty.
- **No pollution of the source tree.**
  - On the server, `RUN_ROOT=/local-scratch/liang/runs/` sits outside the clone. All checkpoints, exports, logs and debug dumps go there, under `<RUN_ROOT>/threetower/<exp>/<run_id>/`. The trainer refuses an output dir inside the git worktree.
  - Small result files (`results/threetower/<exp>/`) are pulled back to the Mac and committed there, never committed from the server.
- **Raw data** is synced separately: Zenodo `KuaiRand-Pure.tar.gz` with md5 `0820331067a3784d9691136f772b35a7`, then per-CSV md5 against the Mac. If that is not possible, rsync from the Mac. Raw data stays read-only.
- **Data consistency.** After the server reruns M2, `scripts/data_fingerprint.py` must agree with the Mac on:
  - schema
  - row counts
  - canonical content hashes (key-sorted; floats rounded to 1e-9)
  - parquet md5, as a reference only
  - frame_ids, which must equal `frames_summary.json`
- **Backup.** After each wave (benchmark, U0/U1, M5), the run's `config`, `run_meta`, selected checkpoint, export, logs and results are rsynced to the Mac. `/local-scratch` has no backup. The final layout on Google Drive is decided in M6.
- **Scheduling.**
  - Slurm is preferred. If it is unavailable, use `CUDA_VISIBLE_DEVICES` + tmux/nohup with per-job logs. No DDP: every experiment is an independent process.
  - A single-GPU smoke test and benchmark come first. Parallelism follows the measured epoch time and peak memory: 2 jobs per GPU only if peak memory < 35 GB and utilisation < 50%. Do not assume any run time in advance.
  - **Waves:**
    1. U0 ∥ U1 (U1 uses the same v1 config).
    2. Cold-video U0, no-history ablation, no-music ablation.
    3. Optional runs on idle GPUs.

## 4. Open questions

- **RESOLVED (P1, see 3b)** Evaluation protocol, information boundary, GT/denominators, cold users.
- **RESOLVED (P1, see 3b)** Split cut timestamps and tie handling.
- **RESOLVED (P1, see 3b)** History contents, same-timestamp handling, cap vs mask.
- **RESOLVED (P1, see 3b)** Catalog definition, OOC handling, seen-item masking, train-unseen item/user constraints.
- **RESOLVED (P1 3b + D1 3d)** Repeated positives: evaluation is covered by P1, and training by D1-3 (shared label column + same-task false-negative mask).
- **RESOLVED (P1, see 3b)** Raw-log cleaning: duplicates, conflicts, tabs 5/6, same-timestamp groups, `duration_ms=0`.
- **RESOLVED (P1, see 3b)** Sentinels, feature usability (U0 main, U1 required, U2 optional), `tab` not a feature.
- **RESOLVED (P1, see 3b)** Cold-video holdout selection, removal, evaluation frame, comparison groups.
- **RESOLVED (D1, see 3d)** OOV implementation (D1-4), train/test history mismatch (D1-3, Q2a), and three-tower v1 internals (D1-1 to D1-3).
- **OPEN (after the benchmark / first runs)** Batch size and max_epochs, set from the measured epoch time (D1-11). Targeted adjustments apply only under the D1-10 trigger. Seeds and extra ablations are optional.
- **RESOLVED (M5, approved by the user 2026-10-12 after independent review; P1/M3/D1 text unchanged)**
  - **Cold-model checkpoint selection = b1.**
    - In the cold condition only, the Validator scores a dedicated *cold validation selection frame* per task:
      - GT = val positives on non-H videos. Music GT is aggregated from non-H video interactions, so a shared music keeps its legitimate non-H GT.
      - Mask = the unfiltered strict-past seen interactions before T_val.
      - Candidates = the full catalog, H rows included and not extra-masked.
      - Population = non-empty GT after the mask and ≥ 1 non-H history event before T_val.
    - The D1-8 objective is unchanged: mean val R@20 over 4 tasks, N@20 tie-break, patience 3, ≤ 30 epochs. The normal condition is unchanged.
    - Rationale: H's val labels never influence the cold model's epoch choice, which makes the "controlled cold" condition closed.
    - Limitations:
      - The cold model's selection population differs from the normal U0 val population. The reduction has two parts:

        | task | normal M3 val users | − only-H val GT | − only-H history | cold selection users | total reduction |
        |---|---:|---:|---:|---:|---:|
        | video_lv | 18,664 | −668 | −20 | 17,976 | −688 |
        | video_like | 2,081 | −140 | −3 | 1,938 | −143 |
        | music_lv | 18,475 | −660 | −20 | 17,795 | −680 |
        | music_like | 2,052 | −138 | −3 | 1,911 | −141 |

        - "Only-H val GT": users whose every val GT item (after the mask) is on H, or for music reached only via H videos.
        - "Only-H history": users with non-H val GT but no non-H event before T_val. Their only pre-T_val events are on H, so the cold model would see an empty history.
      - H rows remain negatives in the selection candidates, so selection is if anything biased against H.
      - The cold run's *reported* val/test metrics still use the standard M3 frames; the selection frame is never reported as a result.
    - Selection-vs-export consistency is checked on the same (cold selection) frames, and frame-id mismatches are refused.
  - **H selection kept as implemented:** label-free train event count, 10 equal-size rank strata, round(10%) per stratum, seed 0. 750 ids stored in `results/m5/cold_video/`; not redrawn.
  - **Cold frame metadata.**
    - History length / bucket = the cold model's actual non-H input (cap 100).
    - Normal-train OOV and cold-train OOV are reported separately.
    - The label-conflict GT_H upper bound is computed from the saved `label_conflict_events` (read-only).
  - **Cold-frame H-only candidates.** Implemented as an extra mask on non-H rows. Cold Pop there is a fixed tie-break ranking (all H scores 0, catalog-row order), not a random ranking. It is reported separately from the analytical random reference.
- **To verify** (server deployment, D1-11):
  - SSH from the Mac
  - Deploy Key clone of the private repo
  - Zenodo download + md5 on the server
  - torch with CUDA
  - `implicit` on the server
  - Slurm vs direct GPU use
