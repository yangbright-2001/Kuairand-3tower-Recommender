# S9 Wave 2: 3 seeds x 2 learning rates (post-hoc exploratory, validation only)

Post-hoc exploratory: designed after the formal v1 test results had been analysed. Only validation
files were read (`scripts/explore/analyze_wave2.py` refuses `metrics.csv` and `per_user/test_*`).
Nothing here replaces the formal U0 or changes the report.

## 1. Run validity

Jobs 44 (`x_u0_repro`), 45 (`x_lr5e4_s1`), 46 (`x_lr5e4_s2`) at explore SHA `a158e5e`: Slurm rc 0
(`.rc` files; `sacct` disabled), empty stderr, 48 files per run (best/last checkpoints, config,
selection, run_meta, train log, 21 export files, 16 per-user files, metrics, parity), best.pt sha =
selection = export manifest, `test_used_for_selection = false`, condition `133765eb12b75700`.
Parity (separate process, 2,048 stratified users): max |Δ| 0.0, 0 top-50 mismatches, retrieve and
per-user metrics identical. All 7 runs (6 grid runs + formal job 27): one data fingerprint (excl.
`data_git_sha`), one protocol config apart from lr / seed, val selection frame ids and all 24 export
frame ids equal to the M3 `frames_summary.json`.

Frame identity for the baseline comparison: the M3 val masked frames were rebuilt from
`data/processed` (`build_eval_frames`, frame_id asserted) and compared with the saved M3 frame
parquet files: same users, same GT rows, same seen-mask size per user, same capped history length;
catalog 7,583 videos / 7,201 music. Every per-user file (models and M3 baselines) carries that frame
id, the same user order and the same `n_gt`.

## 2. Reproduction control vs formal U0 job 27

`x_u0_repro` (a158e5e) reproduces job 27 (9218a76) **bitwise**: all 68 train-log records (loss,
grad norm, per-task losses), all 10 val epochs (score, tie, 8 per-task metrics), best epoch 6,
best.pt state dict (tensor by tensor), the 8 val-user / item embedding files (sha256) and every
val per-user metric. Only the best.pt *file* sha differs (pickled metadata). So the formal U0 is a
valid same-code seed-0 control and the code-version caveat of Wave 1 is resolved; seeds 1/2 (Wave 1,
1051f86) and Wave 2 (a158e5e) share the same training code.

## 3. 3 seeds x 2 learning rates (best checkpoint per run, D1-8 selection)

| lr | seed | run (job) | best ep | v_lv R@20 | v_like R@20 | m_lv R@20 | m_like R@20 | mean R@20 | mean N@20 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1e-3 | 0 | x_u0_repro (44) = job 27 | 6 | .1260 | .1223 | .1308 | .1218 | .1252 | .0693 |
| 1e-3 | 1 | x_u0_s1 (39) | 7 | .1267 | .1089 | .1344 | .1204 | .1226 | .0669 |
| 1e-3 | 2 | x_u0_s2 (40) | 5 | .1241 | .1129 | .1203 | .1149 | .1180 | .0653 |
| 5e-4 | 0 | x_lr5e4 (41) | 6 | .1304 | .1277 | .1370 | .1277 | .1307 | .0724 |
| 5e-4 | 1 | x_lr5e4_s1 (45) | 4 | .1214 | .1250 | .1242 | .1314 | .1255 | .0682 |
| 5e-4 | 2 | x_lr5e4_s2 (46) | 9 | .1281 | .1160 | .1349 | .1168 | .1239 | .0659 |

Per-task val NDCG@20 (v_lv / v_like / m_lv / m_like): 1e-3 s0 .0781/.0589/.0803/.0599, s1 .0770/.0515/.0805/.0586,
s2 .0747/.0564/.0728/.0575; 5e-4 s0 .0803/.0635/.0831/.0629, s1 .0727/.0624/.0747/.0631, s2 .0771/.0536/.0793/.0537.

Across seeds (mean ± std, n = 3; min-max):

| metric | lr 1e-3 | lr 5e-4 |
|---|---|---|
| mean R@20 | .1220 ± .0036 (.1180-.1252) | .1267 ± .0035 (.1239-.1307) |
| mean N@20 | .0672 ± .0020 (.0653-.0693) | .0689 ± .0033 (.0659-.0724) |
| v_lv R@20 | .1256 ± .0014 | .1266 ± .0046 |
| v_like R@20 | .1147 ± .0069 | .1229 ± .0061 |
| m_lv R@20 | .1285 ± .0073 | .1320 ± .0068 |
| m_like R@20 | .1190 ± .0037 | .1253 ± .0076 |

Same-seed paired differences (lr 5e-4 minus lr 1e-3); in brackets the paired val user bootstrap CI
(1000 resamples; user sampling only, no training randomness):

| seed | mean R@20 | mean N@20 | v_lv | v_like | m_lv | m_like |
|---:|---|---|---:|---:|---:|---:|
| 0 | +.0055 [+.0026, +.0085] | +.0031 [+.0020, +.0043] | +.0044 | +.0054 | +.0062 | +.0059 |
| 1 | +.0029 [-.0014, +.0073] | +.0014 [-.0006, +.0034] | -.0053 | +.0161 | -.0101 | +.0110 |
| 2 | +.0059 [+.0023, +.0099] | +.0006 [-.0013, +.0024] | +.0040 | +.0031 | +.0145 | +.0020 |
| mean | +.0048 | +.0017 | +.0010 | +.0082 | +.0035 | +.0063 |

Reading:
- Primary metric: lr 5e-4 is numerically higher on all 3 seeds (+.0029 to +.0059; mean +.0048). The direction is
  consistent; the size is comparable to the seed std (~.0035) and smaller than the seed range (~.007).
  Seed 1's own CI includes 0. The two lr groups overlap (lr 5e-4 min .1239 < lr 1e-3 max .1252).
- Tie-break NDCG@20: numerically higher on all 3 seeds, but small (+.0006 to +.0031; two of three CIs include 0).
- Trade-off: not uniform. Like tasks are numerically higher on all 3 seeds (paired CIs exclude 0 only for seed 1 video_like) (v_like +.003 to +.016, m_like +.002 to
  +.011). Long-view tasks are mixed: up on seeds 0 and 2, clearly down on seed 1 (v_lv -.0053,
  m_lv -.0101; both CIs exclude 0). Seed 1 selected epoch 4, so the long-view loss there coincides
  with an earlier stop. The mean long-view gain (+.0010 / +.0035) is small; most of the primary-metric
  gain comes from the like tasks.
- Epoch curves: lr 5e-4 does not train longer systematically (best epochs 6 / 4 / 9 vs 6 / 7 / 5);
  val curves oscillate by about ±.005 between adjacent epochs in both groups, so the selected epoch
  carries noise.

## 4. Validation comparison with the M3 baselines (same frames; val only)

Val R@20 (val N@20 in `baselines_val_comparison.csv`):

| task | Pop | ALS (pure) | ALS aux (like) | U0 lr 1e-3 s0 | lr 5e-4 s0 / s1 / s2 | lr 5e-4 mean |
|---|---:|---:|---:|---:|---|---:|
| video_lv | .0848 | **.1339** | - | .1260 | .1304 / .1214 / .1281 | .1266 |
| video_like | .0889 | .0620 | .1077 | .1223 | .1277 / .1250 / .1160 | .1229 |
| music_lv | .0891 | **.1361** | - | .1308 | .1370 / .1242 / .1349 | .1320 |
| music_like | .0904 | .0634 | .1166 | .1218 | .1277 / .1314 / .1168 | .1253 |

Paired val bootstrap, lr 5e-4 runs vs baselines (R@20):
1. Pop: every three-tower run beats Pop on every task (CIs exclude 0).
2. Pure ALS (corrected 2026-10-12 21:00 PT; the earlier "no run beats ALS" was wrong as stated).
   Like tasks: every run is numerically above pure ALS by +.05 to +.07 and every CI excludes 0 (pure
   ALS gives no signal to users without a train like) -> a supported val advantage.
   Long-view: one lr 5e-4 run is *numerically* above ALS: s0 on music_lv, .1370 vs .1361, +.0009
   [-.0035, +.0052]; the CI includes 0, so this is a tie, not an advantage. s0 on video_lv: -.0035
   [-.0076, +.0005] (tie). s1 is below with CIs excluding 0 (-.0125 / -.0118); s2 is below on
   video_lv (-.0058, CI excludes 0) and tied on music_lv (-.0012 [-.0053, +.0028]).
   Within the full explore set (incl. Wave 1, `results/explore/wave3_review/longview_vs_als.csv`):
   tau 0.1 s0 is numerically above ALS on both long-view tasks (+.0006 [-.0035, +.0048], +.0041
   [-.0001, +.0081]); no run has a long-view CI excluding 0 in its favour. The formal ablation
   no-music is a tie on both (-.0005, -.0008).
   These are VAL statements. They say nothing about the formal TEST comparison in the report.
3. ALS aux (like tasks): video_like: s0 +.0200 and s1 +.0173 (CIs exclude 0), s2 +.0084 (CI includes 0).
   music_like: s0 +.0111, s1 +.0147, s2 +.0002: all CIs include 0. U0 lr 1e-3 s0 vs aux: +.0147 /
   +.0052, both include 0.
4. Gap to ALS on long-view (ALS minus model, R@20): U0 lr 1e-3 seed mean .0083 (video) / .0076
   (music); lr 5e-4 seed mean .0073 / .0041. The gap shrinks by .0010 (video) and .0035 (music) on
   average, with a large seed spread (video gap .0035-.0125 for lr 5e-4).
5. Trade-offs: lr 5e-4 does not sacrifice a task on average; per seed, seed 1 trades long-view for
   like. On long-view, pure ALS is matched (tie) by the best three-tower runs and is never beaten with
   a CI excluding 0; the lr 1e-3 U0 seeds are below ALS on video_lv with CIs excluding 0 (all 3) and
   on music_lv for 2 of 3 seeds (seed 1 ties: -.0017 [-.0061, +.0024]).

## 5. Conclusions (exploratory)

- The lr 5e-4 effect on the primary metric has a consistent sign over 3 seeds (+.0048 mean), but its
  magnitude is about one seed std, the groups overlap, and it is driven mostly by the like tasks.
  This is suggestive, not reliable evidence; it is not a demonstrated generalisation gain and does not
  justify replacing the formal U0. A statement such as "lr 5e-4 tends to improve like retrieval on
  validation in 3/3 seeds; long-view is seed-dependent" is supported.
- The long-view deficit vs ALS is reduced but not reliably closed by lr / tau tuning (Wave 1 + 2): single
  runs reach a val tie or a small numerical lead, none with a supporting CI, and seed-to-seed variation
  is larger than these differences. That gap, plus the no-
  music ablation's long-view advantage, points at representation, not optimisation hyperparameters.

## Errata (2026-10-12 21:00 PT)

- §4.2 originally said "Long-view: no run beats ALS". Numerically, lr 5e-4 s0 is above pure ALS on
  music_lv (+.0009, CI includes 0), and in Wave 1 tau 0.1 s0 is above on both long-view tasks (CIs
  include 0). Corrected wording: no run has a supported (CI-excluding-0) long-view advantage over ALS;
  the best runs tie it. "Higher" in this document means numerically higher unless a CI is quoted.
- Wave 1 §4 (`results/explore/wave1/validation_analysis.md`) says lr 5e-4 "exceeds the U0 seed max";
  that is numerical (single seed) and is superseded by the 3-seed analysis here.

Terminology (2026-10-12 22:15 PT): in this document "tie" / "within user noise" / "持平" vs a baseline
means "the paired val bootstrap CI includes 0, i.e. no significant difference detected". It does NOT
mean demonstrated equivalence: no equivalence margin was specified or tested.
