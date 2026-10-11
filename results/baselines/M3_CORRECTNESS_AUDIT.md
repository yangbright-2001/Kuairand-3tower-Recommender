# M3 Correctness Audit (independent, 2026-10-11)

**Verdict: no correctness bugs found. No official artifact, metric or protocol was changed.**
The official M3 results (`metrics.csv`, `per_user/`, `frames_summary.json`, `run_meta.json`)
are reproduced exactly by independent code. 25/25 audit checks pass. `uv run pytest`: 90 passed (not the evidence for this audit).

- Script: `scripts/audit_m3_correctness.py`. Run: `uv run python scripts/audit_m3_correctness.py`. It takes 163 s on the M1 and exits 1 if any check fails.
- Machine-readable evidence: `results/baselines/m3_audit/audit_m3.json` (all checks with their numbers), plus the `audit_run.log` log.
- Environment: implicit 0.7.3 (asserted), numpy 2.5.4, pandas 3.0.6, `OPENBLAS_NUM_THREADS=1`.

## 0. How independent is this audit

| Side | What it uses |
|---|---|
| **Oracle** | `pd.read_parquet(events.parquet)` and the raw `video_features_basic_pure.csv`, then pandas/numpy/scipy only. It does **not** import `kuairand_rec.eval.*`, `data.history`, `data.catalog`, or `baselines.*` for its own computation. |
| Oracle details | - Catalog rows are re-derived from the raw metadata. Split is recomputed from the literal cut times (`2022-04-17` / `2022-04-26 00:00 Asia/Shanghai`).<br>- GT and seen sets are per-user Python `frozenset`s.<br>- Pop is a `bincount`.<br>- ALS confidence, the `implicit` fit and the closed-form fold-in are rewritten from scratch.<br>- **Ranks are computed by counting, not sorting**: `rank(g) = #{s > s_g} + #{s == s_g, row < g}`.<br>- IDCG comes from its own table. |
| **Official** | `kuairand_rec` is imported only to load the audited artifacts and to get the official score matrices and model objects for numerical comparison. |

Tolerances:
- Metrics: `1e-12` absolute. Both sides compute float64 from identical integer ranks, so any rank difference shows up as at least `1e-3`.
- Fold-in, ours vs official: `1e-5`.
- Fold-in vs `implicit.recalculate_user`: `1e-4`. The library solves in float32 Cholesky.

## 1. Independent Evaluation Oracle → `frames_summary.json`

The audit compared all **24 frames**: {val, test} × 4 tasks × {masked, nomask, cold}, covering 167,005 frame-users in total. For each frame it checked:

- All 17 count fields: positive events, sentinel, OOC, GT pairs (raw and effective), dedup-collapsed events, masked-seen pairs, all-GT-masked users, cold users and GT, evaluated users, train-OOV users, label-conflict upper bound, and train-unseen GT.
- `population_id` and `frame_id`, re-hashed from the PLAN §3c spec on oracle data.
- The per-user frame tables: the user list, then per user `n_gt`, `gt_rows`, `n_masked_candidates`, `hist_len_full`/`_capped`, `bucket`, and `is_train_oov_user`.

**Differences: 0** across all 24 frames, every field. Cross-checks on the inputs:
- Event `video_row`, `music_id` and `music_row` equal the raw metadata re-index for all events.
- The `split` column equals the literal cuts.
- No event falls exactly on a cut.
- Catalogs have 7,583 videos and 7,201 music ids (`music_id = 0` excluded).

Test / masked, as rebuilt by the oracle (identical to the official values):

| task | eval users | train-OOV | GT eff. | GT masked (seen) | all-GT-masked users | cold users (GT) | sentinel pos. |
|---|---|---|---|---|---|---|---|
| video/long_view | 18,752 | 499 | 63,281 | 859 | 102 | 437 (1,664) | 0 |
| video/is_like | 1,924 | 65 | 3,380 | 63 | 29 | 68 (158) | 0 |
| music/long_view | 18,595 | 496 | 61,595 | 913 | 111 | 434 (1,623) | 1,670 |
| music/is_like | 1,895 | 63 | 3,307 | 72 | 30 | 67 (155) | 67 |

## 2. End-to-end Score-to-Metric Parity → `per_user/*.parquet`

**Coverage is the full population, not a sample.** The oracle recomputed every user of all **46 per-user files**: 16 Pop and 30 ALS, including val, nomask, cold and no-fold-in.

Its scores came from:
- **Pop**: its own train counts.
- **ALS**: its own refit of the selected config, plus its own fold-in with train+val history for test and train history for val.

That gives **349,955 user rows and 1,110,918 GT pairs**. On each row the oracle compared `user_id` order, `n_gt`, `n_gt_unscorable`, `has_signal`, `hits@{10,20,50}`, Recall and NDCG@{10,20,50}, and `frame_id`. It also compared each `metrics.csv` row (means, `n_users`, `n_no_signal`, `gt_pairs`, `gt_unscorable`).

| Result | Value |
|---|---|
| files with any difference | **0 / 46** (file set equals the covered set; none missing) |
| max \|Δ\| per-user Recall/NDCG | 2.2e-16 |
| max \|Δ\| `metrics.csv` mean | 9.7e-17 |
| hits@K mismatches | 0 |
| population covered (summed over files) | 8,256 train-OOV user rows, 14,793 no-signal rows, 223,652 multi-GT rows |
| music: users with GT on a shared music id (≥2 videos) | 13,971 rows |
| music: users with GT collapsed from ≥2 videos | 51 rows |

The focus points were checked as follows:

- **Catalog row alignment.** Oracle scores are indexed by raw-metadata rows. On the sampled users, oracle and official score matrices are compared element-wise: the `-inf` pattern is identical and max finite \|Δscore\| is 0.0 for all 16 model × task combinations.
- **User alignment.** Users are compared in identical order in every file. The ALS fold-in user set equals the official signal-user set.
- **Mask.** It is applied from the oracle's own seen sets, and GT ∩ seen is removed from the denominator (§1).
- **`-inf`.** For every file, #GT pairs scored `-inf` equals n_gt(no-signal users) + n_gt_unscorable(signal users), so every `-inf` GT is accounted for. In total there are 35,064 such pairs, all in ALS.
- **Tie-break.** This is implemented by counting, independently of sorting. 1,450 GT pairs tied within the top 50, almost all in Pop is_like, and they match exactly.
- **GT denominator.** Recall uses \|GT_eff\|, and the IDCG uses min(\|GT\|, K).

### Explicit sort-based cross-check on representative users

On top of the full count-based parity, a third path was run on 383 sampled users (16 model × task combinations). It sorts the oracle scores with `np.lexsort` and compares them with the official `rank_topk` on official scores. The users were drawn per category: normal, multi-GT (≥3), train-OOV, no-signal (ALS), the user with the largest mask, the user with the largest GT, and for music, shared-music and collapsed-video users.

Result: **0 top-50 list mismatches, 0 GT-rank mismatches, and an identical `-inf` pattern.** Examples (test/masked, from `audit_m3.json.examples`):

| model / task | category | user | n_gt | masked | signal | hits@20 | NDCG@20 |
|---|---|---|---|---|---|---|---|
| pop video/long_view | max_mask | 22287 | 1 | 746 | ✓ | 0 | 0 |
| pop video/long_view | max_gt | 10256 | 49 | 115 | ✓ | 2 | .0766 |
| als video/long_view | train_oov | 4449 | 18 | 28 | ✓ | 1 | .0352 |
| als video/long_view | no_signal | 8184 | 1 | 5 | ✗ | 0 | 0 |
| als music/is_like | train_oov, no like before T_test | 4185 | 2 | 0 | ✗ | 0 | 0 |
| als_like_aux music/is_like | max_gt | 26147 | 17 | 28 | ✓ | 3 | .1840 |
| als_like_aux music/is_like | shared_music | 567 | 1 | 20 | ✓ | 0 | 0 |

### Negative controls: does the parity check have power?

Bugs were injected into the oracle one at a time. Each cell is the number of per-user metric cells that differ from the official values:

| injected bug | pop video/long_view | als video/is_like | pop music/is_like |
|---|---|---|---|
| none (baseline) | 0 | 0 | 0 |
| tie-break row descending | 45 | 0¹ | 293 |
| mask not applied | 13,283 | 1,047 | 801 |
| user rows shifted by 1 | 0¹ | 2,313 | 0¹ |
| catalog columns shifted by 1 | 39,045 | 1,912 | 2,319 |
| `-inf` → min finite | 0¹ | 30 | 0¹ |
| GT denominator drops 1 item | 25,124 | 942 | 953 |

¹ These are expected no-ops:
- Pop scores do not depend on the user, so a user-row shift changes nothing.
- Pop has no `-inf` outside the mask, and -1e30 still ranks below every count.
- ALS scores are continuous and have essentially no ties, so the tie-break order never matters.

Every bug that applies to a model is detected.

## 3. ALS Numerical Verification (implicit 0.7.3)

| Item | Method | Evidence |
|---|---|---|
| **Confidence semantics** | `recalculate_user` was run on a real-data fit (32 factors, 300 users) and compared against 3 hypotheses for the stored CSR value d. The final item step was also checked by item-side stationarity on a 3k-user sub-fit. | - Max error by hypothesis: **C = d, p = 1: 6.8e-6**. C = 1 + d: 0.11. p = d: 43.<br>- Stationarity (Cholesky), relative residual under C = d: median 1.0e-6, max 2.5e-6, vs 0.066 under C = 1 + d.<br>- So storing `1 + α·r` with library `alpha = 1` gives exactly C = 1 + α·r. |
| **Fold-in vs `recalculate_user`** | All val and test signal users of all 6 selected configs. The oracle's float64 closed form is compared with the official fold-in and with the library. | - Oracle vs official: **max \|Δ\| = 0.0, bitwise identical** for every row of all 12 (config × split) sets.<br>- Oracle vs library: max abs 9.1e-5 (aux) and ≤ 4.3e-5 (pure); max relative 2.0e-5. This is within 1e-4 and is float32 noise in the library. |
| **Compact user/item mapping** | Oracle refit from its own confidence frame. | - `item_rows` and `train_users` are equal, and Y and X_train are **bitwise equal** to the official ones.<br>- The confidence hash, `n_scorable_items` and the fingerprint all match `run_meta.json`.<br>- Semantic check on 500 sampled train users: cos(X_train[k], refold of k's own row) has median ≥ 0.9998, vs ≤ 0.048 for other users. Train positives rank in the top 0.5–5.7% (mean percentile).<br>- Exact check: X_train equals one implicit CG user step on the (T-1)-iteration model, max \|Δ\| = 0.0. This confirms both the mapping and the documented "no-fold-in X is half an iteration older" note. |
| **Train-only Y; test fold-in uses train+val** | - Y is refit on `t < T_val` only and is bitwise equal.<br>- Fold-in pair counts are compared with `run_meta.als_foldin_stats`.<br>- Leakage power check: fold in again with test-period events added. | - Val history pairs equal train pairs, with 0 dropped. Test history adds the val pairs and drops only items with no train signal (181–366 pairs), equal on all 3 sources.<br>- With test events leaked, 18,184–18,443 users' vectors would change (long_view/aux) and 1,218–1,229 (pure like). The official vectors are bitwise equal to the no-leak oracle. |
| **Pure vs Aux signal** | The oracle's `r = 1[any action] + aux_w·1[any long_view]` on (user, item row), with `c = 1 + α·r`. It was compared on train, val-history and test-history for all 6 configs (18 sets). | - **18/18 identical** in (user, item, c).<br>- Pure: c ∈ {11} (α = 10).<br>- Aux (α = 40, aux_w = 0.3): c ∈ {13 lv-only, 41 like-only, 53 both}. Train counts for video: 326,565 / 5,861 / 12,794.<br>- Music aggregates "any" across videos of the same music id: 13 train (and 18 test-history) pairs get c = 53 only because like and long_view came from different videos. Ours and official agree. |
| **Selection** | The argmax of `(val_recall@20, val_ndcg@20)` over `als_tuning.csv` was re-derived, keeping the first grid point on ties, and compared with the val rows in `metrics.csv`. | 6/6 selected configs equal `run_meta.als_selected`. |

Pop: the train counts by `bincount` equal the M2 `popularity_*.parquet` and `PopularityScorer.counts` for all 4 tasks. Nonzero items: 6,653 / 3,344 / 6,331 / 3,151.

## 4. Classification

### Independently Verified
- All 24 frames: GT, seen mask, evaluated / cold / OOV populations, every exclusion count, `population_id` / `frame_id`, and the per-user frame tables.
- Catalog row indexing from the raw metadata, and the split from the literal cut times.
- All 46 per-user metric files and all `metrics.csv` rows (val, test masked / nomask / cold, no-fold-in), on the full populations.
- Catalog and user alignment, mask application, `-inf` accounting, tie-break (rank counting plus an explicit sort), and the GT denominator. Every applicable injected bug is detected.
- Pop train-only counts.
- ALS:
  - the confidence semantics under implicit 0.7.3
  - fold-in, which is bitwise equal to the oracle and matches `recalculate_user` within 1e-4
  - the compact mapping and the train-only Y, both bitwise
  - test fold-in uses only `t < T_test`, with a power check
  - pure / aux signal construction
  - selection from `als_tuning.csv`

### Previously Tested Only (not independently re-derived here)
- **M2 cleaning, which feeds both sides.** The audit starts from the cleaned `events.parquet` and `label_conflict_events.parquet`. Not re-derived: the raw→clean filtering, the duplicate and label-conflict classification, and the `long_view` / `is_like` labels. These are covered by M2 tests and `DATA_REPORT.md`.
- **Validation metrics of the non-selected grid points.** Only the selection logic was re-derived; the 42 non-selected fits were not refit.
- **Runner determinism.** The byte-identical second run comes from the M3 run log and was not repeated.
- **Runner-internal sanity asserts**, the `history_view` boundary asserts, and the hand-computed metric unit tests.
- **`hist_len_capped`/`bucket`.** They were checked only as frame-table columns. The capped model history (≤100) is not used by Pop or ALS.

### Unresolved Risks / caveats (no bug, but worth knowing)
1. **Common-mode inputs and spec.** Oracle and official code read the same cleaned events and the same PLAN §3c spec. An M2 label error, or a shared misreading of the spec, would not be caught. The negative controls show power against implementation bugs, not spec bugs.
2. **The ALS fit uses the same library.** The oracle refits with `implicit`, so it is not independent of an `implicit` fit bug. This is mitigated by the stationarity / closed-form checks, which verify what the library solves.
3. **Pop tie-break sensitivity (descriptive).** The row-ascending tie-break is the protocol and is applied correctly, but on like tasks it is not neutral. With 5 random tie-break permutations, music/is_like Pop R@20 ranges over **0.0728–0.0734 vs the official 0.0726**, so the official value is slightly *below* the random range. video/is_like R@50 ranges over 0.1294–0.1314 (official 0.1304). Long-view tasks move by at most 1e-4. Pop like-task numbers carry about ±0.001 tie-break uncertainty, which is small next to every reported ALS-vs-Pop gap.
4. **No-fold-in X is half an iteration older than Y.** This is a known implicit user-then-item update order, now confirmed exactly. It affects only the `*_nofoldin` diagnostic rows.
5. **Unscorable GT stays in the denominator.** This is P1-6b by design. Pure-like ALS has many `-inf` GT pairs from no-signal users (608 / 606 users), which explains why pure-like ALS is below Pop. It is not an error.
6. `label_conflict_possible_gt_pairs` remains an upper bound (1 pair for long_view test, 0 for like). It is reproduced exactly, but it is not resolved.
