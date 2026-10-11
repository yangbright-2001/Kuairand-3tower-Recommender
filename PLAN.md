# Project Plan

`docs/assignment.md` is the authoritative spec. This file tracks the agreed workflow, design principles, decisions, and open questions.
Items marked **OPEN** are not decided. Do not treat them as conclusions.

## 1. Workflow (agreed 2026-10-11)

```
S0 Env/skeleton -> S1 Dataset Audit (A: schema, B: plan, C: implement)
  -> P1 Protocol review [user sign-off: APPROVED 2026-10-11, see 3b]
  -> S2 Data pipeline -> S3 Shared evaluator -> S4 Pop/ALS baselines
  -> D1 Three-tower v1 design review [user sign-off, before first implementation]
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

## 4. Open questions

- **RESOLVED (P1, see 3b)** Evaluation protocol, information boundary, GT/denominators, cold users.
- **RESOLVED (P1, see 3b)** Split cut timestamps and tie handling.
- **RESOLVED (P1, see 3b)** History contents, same-timestamp handling, cap vs mask.
- **RESOLVED (P1, see 3b)** Catalog definition, OOC handling, seen-item masking, train-unseen item/user constraints.
- **RESOLVED (P1, see 3b) / OPEN (D1)** Repeated positives: evaluation is resolved. Training treatment is D1.
- **RESOLVED (P1, see 3b)** Raw-log cleaning: duplicates, conflicts, tabs 5/6, same-timestamp groups, `duration_ms=0`.
- **RESOLVED (P1, see 3b)** Sentinels, feature usability (U0 main, U1 required, U2 optional), `tab` not a feature.
- **RESOLVED (P1, see 3b)** Cold-video holdout selection, removal, evaluation frame, comparison groups.
- **OPEN (D1, before M4)** OOV implementation: a shared OOV index vs a zeroed ID component, and how OOV rows are trained (for example ID dropout). The P1-6b constraints themselves are fixed.
- **OPEN (D1, before M4)** Train/test mismatch: fresh rolling history during training vs history frozen at T_test plus a 13-day test window.
- **OPEN (D1, before M4)** Three-tower v1:
  - tower internals and the history encoder
  - how music and author feed the video tower
  - the contrastive loss
  - handling duplicate positives and false negatives
  - balancing the two actions
- **OPEN (after first runs)** Hyperparameters, number of seeds, extra ablations, and post-v1 exploration.
- **To verify**:
  - SSH from the Mac
  - Zenodo download on the server
  - torch with CUDA 13
  - `implicit` on Py3.12, on both arm64 and the server
  - Slurm vs direct GPU use
