# S9 Wave 3: no-music ablation, 3-seed replication (post-hoc exploratory, validation only)

Post-hoc exploratory: designed after the formal v1 test results (incl. the no-music ablation on
test) had been analysed. Only validation files were read (`scripts/explore/analyze_wave3.py`
refuses `metrics.csv` / `per_user/test_*`). Nothing here changes the formal v1 or the report.
Figures: `figures/fig1_u0_vs_nomusic_r20.png`, `fig2_nomusic_minus_u0.png`,
`fig3_longview_epochs.png` (`scripts/explore/plot_wave3.py`).

## 1. Run validity and reproduction

Jobs 47 / 48 / 49 at `be5ceb96b5e10bf9754983de168d8c3542e0de85`:
- Slurm `.rc` = 0 (the runner's EXIT-trap files; `sacct` is disabled) and empty stderr.
- 48 files per run, including:
  - best.pt and last.pt;
  - config, selection, run_meta and train log;
  - 21 export files (embeddings, id mappings, manifest);
  - 16 per-user files, metrics and parity.
- Consistency: best.pt sha256 = selection = export manifest; `test_used_for_selection = false`.
- Protocol:
  - condition `133765eb12b75700`;
  - one data fingerprint over all runs (excl. `data_git_sha`);
  - val selection frame ids and all 24 export frame ids equal to the M3 ones.
- Independent-process parity (2,048 stratified users): max |Δ| 0.0, 0 top-50 mismatches, retrieve and
  per-user metrics identical.
- Effective configs vs formal job 35: only `exp`, `run_root` and (s1/s2) `train.seed` differ;
  `use_music = false`, lr 1e-3, tau 0.05.

Server-side, the Wave 1 (240 files) and Wave 2 (144 files) run directories re-verify against their
sha256 manifests. The formal clone is unchanged (`671dde9`, clean) and the formal `runs/` listing
hash is unchanged.

**Job 47 reproduces the formal no-music job 35 bitwise:**
- all 68 train-log records;
- all 10 val epochs (score, tie, 8 per-task metrics); best epoch 6, same best score;
- best.pt state dict, tensor by tensor;
- the 8 item / val-user embedding files (sha256);
- every val per-user metric.

Together with `x_u0_repro` = job 27 (Wave 2), both seed-0 controls are formal-run reproductions on
the current explore code.

## 2. 3 seeds x 2 architectures (best checkpoint per run, D1-8 selection)

| arch | seed | job | sha | best ep | v_lv R / N | v_like R / N | m_lv R / N | m_like R / N | mean R@20 | mean N@20 |
|---|---:|---:|---|---:|---|---|---|---|---:|---:|
| U0 | 0 | 44 (= 27) | a158e5e | 6 | .1260 / .0781 | .1223 / .0589 | .1308 / .0803 | .1218 / .0599 | .1252 | .0693 |
| U0 | 1 | 39 | 1051f86 | 7 | .1267 / .0770 | .1089 / .0515 | .1344 / .0805 | .1204 / .0586 | .1226 | .0669 |
| U0 | 2 | 40 | 1051f86 | 5 | .1241 / .0747 | .1129 / .0564 | .1203 / .0728 | .1149 / .0575 | .1180 | .0653 |
| no-music | 0 | 47 (= 35) | be5ceb9 | 6 | .1334 / .0802 | .1177 / .0558 | .1352 / .0812 | .1248 / .0595 | .1278 | .0692 |
| no-music | 1 | 48 | be5ceb9 | 4 | .1303 / .0792 | .1254 / .0608 | .1358 / .0807 | .1311 / .0629 | .1306 | .0709 |
| no-music | 2 | 49 | be5ceb9 | 5 | .1334 / .0801 | .1221 / .0569 | .1362 / .0810 | .1179 / .0583 | .1274 | .0691 |

All 6 runs: rc 0, parity PASS.

Across seeds (mean ± std, n = 3; min-max), R@20:

| metric | U0 | no-music |
|---|---|---|
| mean R@20 | .1220 ± .0036 (.1180-.1252) | .1286 ± .0018 (.1274-.1306) |
| mean N@20 | .0672 ± .0020 | .0697 ± .0010 |
| v_lv | .1256 ± .0014 (.1241-.1267) | .1324 ± .0018 (.1303-.1334) |
| m_lv | .1285 ± .0073 (.1203-.1344) | .1357 ± .0005 (.1352-.1362) |
| v_like | .1147 ± .0069 | .1218 ± .0039 |
| m_like | .1190 ± .0037 | .1246 ± .0066 |

Same-seed differences (no-music minus U0; the same seed number is NOT a paired initialisation, since
the first-layer width differs). Brackets: paired val user bootstrap CI, 1000 resamples; user
sampling only, not training randomness.

| seed | mean R@20 | mean N@20 | v_lv R@20 | m_lv R@20 | v_like R@20 | m_like R@20 |
|---:|---:|---:|---|---|---|---|
| 0 | +.0025 | -.0001 | +.0073 [+.0050, +.0097] | +.0045 [+.0022, +.0070] | -.0047 [-.0131, +.0050] | +.0030 [-.0065, +.0118] |
| 1 | +.0080 | +.0040 | +.0036 [+.0009, +.0062] | +.0014 [-.0013, +.0039] | +.0166 [+.0070, +.0273] | +.0106 [+.0004, +.0226] |
| 2 | +.0094 | +.0037 | +.0094 [+.0069, +.0119] | +.0158 [+.0133, +.0187] | +.0093 [-.0008, +.0197] | +.0031 [-.0060, +.0128] |
| mean (std) | +.0066 (.0036) | +.0025 (.0023) | +.0068 (.0029) | +.0072 (.0076) | +.0071 (.0108) | +.0056 (.0044) |

## 3. Answers

A. Per-seed differences: see the table above.

B. Variability:
- Mean R@20 improvement: +.0066 ± .0036 (range +.0025 to +.0094).
- The no-music seed spread is smaller than U0's: std .0018 vs .0036; on m_lv .0005 vs .0073.

C. Video long-view: positive on 3/3 seeds (+.0036 to +.0094); each paired CI excludes 0. The whole
no-music range (.1303-.1334) lies above the whole U0 range (.1241-.1267).

D. Music long-view: positive on 3/3 seeds (+.0014 to +.0158).
- CIs exclude 0 on seeds 0 and 2; seed 1 is within user noise (+.0014 [-.0013, +.0039]).
- The no-music range (.1352-.1362) is above the U0 range (.1203-.1344).

E. Likes: no systematic sacrifice.
- 5/6 seed-task differences are positive.
- One is negative: v_like seed 0, -.0047, CI includes 0.
- 2/6 have CIs excluding 0 (seed 1, both likes), in favour of no-music.

F. The 4-task mean R@20 is higher on 3/3 seeds and the ranges do not overlap: U0 max .1252 < no-music
min .1274. Mean N@20 is higher on 2/3 seeds; seed 0 is a tie (-.0001).

G. Long-view vs M3 pure ALS (val):
- U0 is below ALS with CIs excluding 0 on video_lv (3/3 seeds) and on music_lv (2/3 seeds).
- No-music is within user noise of ALS on all 6 seed-task pairs: video_lv -.0005 / -.0036 / -.0005;
  music_lv -.0008 / -.0003 / +.0001. Every CI includes 0.
- Only no-music s2 on music_lv is numerically above ALS, by +.0001 [-.0040, +.0045].
- So, with the music input removed, no significant val long-view difference from pure ALS is detected
  (all 6 CIs include 0). This is absence of evidence of a difference at this sample size. It is not a
  demonstration of equivalence: no equivalence margin was pre-specified or tested. It does not
  establish an R@20 advantage over ALS either.
- On the tie-break metric, val long-view NDCG@20, no-music is above pure ALS on 6/6 seed-task pairs
  with CIs excluding 0 (video_lv +.0040 / +.0030 / +.0039; music_lv +.0043 / +.0038 / +.0042).
  U0 is mixed there:
  - video_lv: all 3 CIs include 0;
  - music_lv: s0 and s1 above ALS, s2 below (-.0040, CI excludes 0).

  Reading: no-music ranks its hits higher than ALS at a similar hit rate. This is a val-only,
  secondary-metric observation (`validation_bootstrap.csv`, metric `ndcg@20`). The primary metric is
  R@20, so it does not change the selection or the main statement.

## 4. Interpretation

1. **Replicated across seeds?** Yes, descriptively, for the long-view effect (same sign on 3/3 seeds).
   - Both long-view tasks improve on 3/3 seeds, and the no-music seed ranges sit above the U0 ranges.
   - The likes improve on average but less consistently.
2. **Larger than U0's own seed variation?** Descriptively yes, for video long-view: mean +.0068 vs a
   U0 seed range of .0027, with the architecture ranges disjoint. Music long-view and the 4-task mean: also disjoint
   ranges, but U0's spread is large (m_lv .0141). With n = 3 per arm this is a consistent descriptive
   pattern across seeds, not a cross-seed significance result. No test over training seeds was run;
   the per-user bootstrap CIs cover user sampling only.
3. **Persistent over epochs, not just at the best epoch?** Yes.
   - On common epochs, no-music is higher on video_lv at 10/10, 8/8 and 9/9 epochs (seeds 0/1/2),
     and on music_lv at 9/10, 8/8 and 9/9.
   - With each task evaluated at its own best epoch (`oracle_epoch_per_task.csv`; an optimistic,
     post-hoc view), no-music is higher on every seed for both long-view tasks: U0 max .1304 /
     .1372 vs no-music min .1343 / .1393.
4. **Is 4-task early stopping the main cause?** The data argue against it as the main explanation.
   - The selection trade-off exists in both architectures: long-view peaks after the selected epoch.
   - But the gap persists at every epoch and at per-task best epochs.
5. **How strongly is a harmful music-to-video input supported?** As a descriptive, architecture-
   level statement: "In the current concatenation fusion, removing the music-tower input from the
   video tower improves validation long-view retrieval consistently across 3 seeds, without a
   systematic cost on the like tasks." It is post-hoc, it is on val only, and the earlier test-side
   ablation motivated it.
6. **Mechanism?** Not identified. Candidates still not separated:
   - gradient interference through the shared music tower (video losses update it; consistent with
     the lower music-task train loss without the input, see the Wave 3 gate review);
   - input-scale dominance (`music_base` norm about 70x the ID embeddings at init);
   - near-ID redundancy (112 shared music ids).

   The no-music arm removes the input and the video-loss gradient into the music tower at once.
   The stop-gradient control (E-2) is the minimal experiment that separates the first candidate from
   the other two.
