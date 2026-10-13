# S9 exploratory optimisation (branch `explore/s9-opt`; NOT part of the approved v1)

Plan: `.cursor/plans/三塔优化研究方案_b178e03c.plan.md` (Revision 1, reviewer-approved direction).
Base: `671dde9b4d029f468bc405f05b624a7076fcd0fe` (the formal M5 training SHA). The formal v1
(U0/U1/ablations/cold, report) lives on `main` and is never modified from here.

## Status

Wave 1 (jobs 39-43) completed 2026-10-13 01:39 UTC: all rc 0, parity PASS; validation-only analysis in
`results/explore/wave1/` (post-hoc exploratory). Wave 2 approved 2026-10-12 19:55 PT (see below).

### Preparation (done before Wave 1)

| item | state |
|---|---|
| Local worktree `../kuairand-explore` on `explore/s9-opt` @ 671dde9 | done |
| Separate `.venv` (`uv sync --frozen`, editable install -> worktree `src/`) | done |
| `data/processed` rebuilt in the worktree from the read-only raw CSVs | done; 11/12 files md5-identical to main, `meta.json` differs only in `git_sha` |
| 24 M3 frame_ids (`build_eval_frames`) | equal to `results/baselines/frames_summary.json` |
| Condition id / fingerprint vs formal U0 (job27) and no-music (job35) | `133765eb12b75700`; condition fingerprint identical; data fingerprint differs only in `data_git_sha` |
| `uv run pytest` | 258 passed (239 formal + 19 new/parametrised explore checks incl. `test_explore_configs.py`) |
| Diagnostics D1-D12 + P1-P4 | `results/explore/diagnostics/*.json` (deterministic rerun checked on D4 / D9) |
| Server dirs `explore/`, `runs_explore/{slurm/status,threetower,checks}` | created (empty); formal clone HEAD 671dde9, clean, unchanged |
| Server explore clone | pending: needs `explore/s9-opt` pushed, then `scripts/explore/server_setup.sh <SHA>` |

## Gates (all required before Wave 1 is submitted)

1. User review of this preparation.
2. Track A: no blocking issue from the remaining User-Metadata (U1 / P1-8) compliance check that affects
   shared training data or evaluation. U0 (all explore runs) uses no user metadata, so a U1-only finding
   does not block; anything touching `data/processed`, frames, the evaluator or the trainer pauses all jobs.
3. Branch pushed; server clone set up by `server_setup.sh` (md5 / frame_id / config tests / clean tree).

## Wave 1 (config only; code path identical to 671dde9)

| exp | change vs `configs/threetower_u0.yaml` | role | D1-10 count |
|---|---|---|---|
| `x_u0_s1` | `train.seed: 1` | seed calibration | no (optional seeds) |
| `x_u0_s2` | `train.seed: 2` | seed calibration | no (optional seeds) |
| `x_lr5e4` | `train.lr: 5.0e-4` | E1, D1-10 #2 | 1 / 7 |

Every explore config also changes `exp` (prefix `x_`) and `run_root` (`/local-scratch/liang/runs_explore`).

Prepared but NOT in Wave 1 (single-variable temperature candidates, D1-10 #1; separate approval):
`x_tau003` (`train.tau: 0.03`), `x_tau010` (`train.tau: 0.1`). If both run: 3 / 7 of the D1-10 budget.
The formal v1 used 0 / 7. Diagnostic evidence for tau is weak (plan §1.7), so they are not proposed for Wave 1.

## Wave 2: same-SHA control + seed replicates of x_lr5e4 (post-hoc exploratory)

| exp | kind | diff | purpose |
|---|---|---|---|
| `x_u0_repro` | reproduction_control | vs U0: none (only exp / run_root) | same-SHA seed-0 control; tier A/B regression vs job 27 |
| `x_lr5e4_s1` | seed_replicate of `x_lr5e4` | vs x_lr5e4: `train.seed: 1` | replicate lr 5e-4 |
| `x_lr5e4_s2` | seed_replicate of `x_lr5e4` | vs x_lr5e4: `train.seed: 2` | replicate lr 5e-4 |

D1-10 count unchanged at 3 / 7 (replicates and the control are not new configurations).
Planned comparison (val only): lr 1e-3 {x_u0_repro, x_u0_s1 (39), x_u0_s2 (40)} vs lr 5e-4 {x_lr5e4 (41),
x_lr5e4_s1, x_lr5e4_s2}; same-seed paired differences, means, std, per-task trade-offs; 3 seeds are
descriptive, not a significance test. Wave 3 (e.g. E2 music fusion) is decided after Wave 2.

## Selection protocol (pre-registered)

- D1-8 unchanged for every run: normal val masked frames at T_val, mean val R@20 over the 4 tasks,
  N@20 tie-break, earliest wins exact ties, patience 3, <= 30 epochs. Each run selects its own epoch.
- Two kinds of uncertainty, reported separately:
  - user-level paired bootstrap on val per-user files (1000 resamples, seed 0, `experiments/bootstrap.py`):
    sampling uncertainty of a fixed pair of trained models;
  - training-seed variability: U0 seeds 0/1/2 (min / mean / max). Three seeds are descriptive only,
    not a significance threshold.
- Round 1 screen (U0 s0/s1/s2 vs candidate s0): (a) candidate val mean > max of U0 seeds, (b) paired
  val bootstrap CI of the 4-task mean vs U0 s0 excludes 0, (c) per-task deltas + CIs listed, tasks below
  the U0 seed minimum flagged. Otherwise reported as a negative result.
- Round 2 (time permitting): s1/s2 of a passing candidate; "stable" = wins >= 2/3 same-seed pairs and the
  3-seed mean exceeds U0's (descriptive).
- Test is sealed: the unchanged runner computes it, but explore summaries read only val per-user files,
  `selection.json` and `train_log.jsonl` until the user decides. Results already seen on test (e.g. the
  no-music ablation) are hypothesis sources only; no explore test number is an independent new test.

## Wave 2 (spec only; no code in this preparation)

E2 `x_musicgate`: `model.music_fusion: gated`, `model.music_gate_init: 1.0` (per-dimension *unconstrained*
learnable scaling of the shared music-tower output inside the video tower; may become < 0 or > 1; not a
0-1 gate). Default `music_fusion: concat` creates no parameter and reproduces U0 exactly. Tests, the
tier A/B regression (`x_u0_repro`) and the CPU smoke + Direct/Export parity are specified in plan §3.2/§3.4
and must pass before any E2 GPU run. Gate = 0 is NOT structurally equivalent to the no-music ablation.

## Wave 3 (spec only)

E3 delayed history: CPU distribution diagnostic only (plan §3.3); no trainer change, no GPU run.

## Diagnostics (`scripts/explore/diagnose_u0.py`)

Scope: train events + normal val masked frames at T_val (frame_ids asserted); formal checkpoints,
`selection.json`, `train_log.jsonl`, `per_user/val_*.parquet` read-only; no test frame or test file opened.
CPU FP32, 4 torch threads, fixed seeds. Command:

```bash
uv run python scripts/explore/diagnose_u0.py --formal_root /Users/yangliang/Desktop/kuairand-threetower-recsys
```

Inputs, checkpoint sha256s, frame_ids and timings: `results/explore/diagnostics/DIAGNOSTICS_RUN.json`.
Each JSON carries a `method` string with its exact scope and caveats. Highlights (Observed only):

| id | observation |
|---|---|
| D1 | val R@20 video_lv: U0 .1260, ALS .1339, no-music .1334; U0 - ALS by full history length: +.057 (<=5) ... -.026 (101-200) |
| D2 | epoch-wise (shared data order only, not paired RNG): no-music > U0 on video_lv 10/10, music_lv 9/10, likes 5/10, 6/10 |
| D3 | pre-clip grad norm median 6.4 after warmup, > clip 1.0 on 100% of logged steps; FN cols / row: lv 26.2, like 10.9 |
| D4/P2 | first-layer pre-activation variance share: music_base 0.575, video_id 0.064, author 0.060, cross terms 0.237 (pre-activation only) |
| D5 | train-mode, 4 batches, best.pt: like grads 2-3.5x long_view per tower; min pairwise cosine -0.315 (video tower) |
| P4 | eval-mode, 16 batches: best.pt negative pair-batch share <= 7%; last.pt up to 24% (video tower), min cos -0.38 |
| D6 | final pooling entropy 4.599 vs uniform 4.605; self-attention entropy layer0 2.64, layer1 1.19 (not uniform) |
| P3 | uniform pooling (pool_query = 0): val mean R@20 .12528 vs original .12525 |
| D7 | recency at inference: all-max-bucket .12547, shift -2 .12268, shift -4 .11498 (original .12525) |
| P1 | trained U0 depends on the music path: zeroing music input -> video_lv .0998, permuting -> .0539 (original .1260) |
| D9 | recency bucket total variation, train rolling vs val T_val: 0.582 |
| D10 | item-head effective rank (of 64): U0 6.1, no-music 7.5, no-history 3.9 (descriptive) |

Conclusions stay as in plan §1 (Observed / Hypothesis / Conclusion). New facts from this run that refine
it: (i) gradient conflict is not absent: negative cosines appear, more at last.pt (P4) -> "magnitude
imbalance observed; direction conflict occasional and checkpoint-dependent", still no evidence for
PCGrad-type methods specifically; (ii) the self-attention layers are far from uniform (D6), so the
history encoder is not mean pooling; only the final pooling step is (P3: replacing it changes nothing);
(iii) U0 relies heavily on its music input (P1), consistent with music_base acting as a near-ID path,
but dependence is not negative transfer.

## Server layout

- Explore clone: `/local-scratch/liang/explore/kuairand-explore` (read-only deploy key, detached at the
  explore SHA). Formal clone `/local-scratch/liang/kuairand-threetower-recsys` and `runs/` untouched.
- Runner: `scripts/explore/train_explore.sbatch` (M5 runner + explore guards). Run dirs:
  `/local-scratch/liang/runs_explore/threetower/<exp>/<sha7>_seed<seed>_job<id>/`; Slurm logs
  `/local-scratch/liang/runs_explore/slurm/<exp>_<id>.{out,err}`; exit codes `.../slurm/status/*.rc`.
- Status: `bash scripts/explore/status_explore.sh` (also prints the formal HEAD, which must not change).
