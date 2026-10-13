# KuaiRand-Pure: multi-action three-tower retrieval for videos and background music

## 1. Overview

### 1.1 What it does

A three-tower model (user / video / music) trained on [KuaiRand-Pure](https://kuairand.com/) retrieves, from the full catalog, both **videos** and their **background-music IDs** for two actions, **long view** and **like**. That gives four tasks:

- video / long view
- video / like
- music / long view
- music / like

### 1.2 Baselines and protocol

The model is compared with Popularity and implicit ALS, under one leakage-controlled temporal protocol, scored by one shared evaluator with paired user-bootstrap confidence intervals.

### 1.3 Quick links

| Resource | Location |
|---|---|
| Research report | **PDF: [`report/REPORT.pdf`](report/REPORT.pdf)** (8 pages; submitted version). Source: [`report/REPORT.md`](report/REPORT.md) (design, results, error analysis, limitations, next steps); extended version with all evidence and the exploratory study: [`report/REPORT_FULL.md`](report/REPORT_FULL.md) |
| Task specification | [`docs/assignment.md`](docs/assignment.md) |
| Full reproduction guide | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |
| Post-hoc exploratory results | [`docs/EXPLORATORY_RESULTS.md`](docs/EXPLORATORY_RESULTS.md) (validation only) |
| Trained runs, embeddings, logs | [Google Drive bundle](https://drive.google.com/drive/folders/1VRXe8liHr7IiZfjhHi-K9HIZbtqeaT8U?usp=sharing) (start with `00_START_HERE.md`) |
| Design decisions / progress log | [`PLAN.md`](PLAN.md) · [`PROGRESS.md`](PROGRESS.md) |

## 2. Key findings

**Model variants.** Two versions of the three-tower model differ only in the user-tower inputs:
- **U0** is the main model. It uses the user ID plus the user's strict-past interaction history, and no user metadata.
- **U1** is U0 plus 21 user-profile fields. These come from a single, undated snapshot, so their availability at prediction time cannot be verified. U1 is reported with that caveat.

### 2.1 Formal findings (held-out test)

**How to read these numbers.** Each number in this subsection is a difference in **test** Recall@20 between two models, evaluated on the same users:
- The metric is computed over the full item catalog, with items the user had already interacted with excluded from the ranking.
- Brackets give the 95% confidence interval from a paired bootstrap over 1,000 user resamples.
- A star (\*) marks a statistically significant difference, i.e. its 95% interval does not include 0.

**Findings.**

- **U0 beats Popularity on all four tasks.** The gains are +.020\* to +.032\*.
- **U0 vs pure implicit ALS is mixed:**
  - *better* on both like tasks (+.036\* video, +.044\* music);
  - *worse* on both long-view tasks (−.012\* each).
- **Against a stronger like baseline there is no significant edge.** That baseline is ALS with an auxiliary long-view signal: −.009 [−.025, +.005] video, −.003 [−.019, +.014] music.
- **Controlled cold-video holdout** (750 videos removed from training):
  - **among held-out videos only**, the content path clearly beats the same-condition baselines (+.078\* over random on long view);
  - **on the full catalog** it is not distinguishable from random (+.0014 [−.0000, +.0032]).
- **Ablations:**
  - removing history hurts long view (−.012\* / −.011\*);
  - removing the music input to the video tower **improves** long view (+.010\* / +.009\*). This is a negative result for that design choice.
- **Correctness was prioritised over winning:**
  - chronological split;
  - strict-past histories;
  - train-only statistics;
  - no future-leaking statistics file;
  - bitwise export/direct score parity;
  - a requirement-by-requirement audit.

### 2.2 Additional exploratory findings (validation only; post-hoc)

**How to read these numbers.** After the formal results were analysed, U0 was retrained under single-setting changes on a separate branch. All numbers here are **validation** Recall@20 values (the 4-task mean unless a task is named). They do not change the formal test results above, and no explore model replaces U0.

**Findings.**

- **Removing the music-to-video input consistently improved long-view validation Recall@20 across three seeds, without systematically harming like retrieval.**
  - The ablation was repeated at seeds 0/1/2 (seed 0 reproduces the formal ablation bitwise).
  - Mean val R@20 over 4 tasks: .1286 ± .0018 vs U0 .1220 ± .0036.
  - Average same-seed gain: +.0068 on video long view, +.0072 on music long view, higher on 3/3 seeds for both.
  - With the music input removed, the long-view R@20 difference from pure ALS is no longer CI-supported: all six CIs include 0. This is **not** a proof of equivalence, and not a win over ALS.
  - On the secondary metric, long-view NDCG@20, no-music is above pure ALS on 6/6 seed-task pairs. The nominal 95% CIs exclude 0, uncorrected for multiple comparisons and not confirmed on test.
  - The mechanism is not identified; a stop-gradient control is future work.
- **Seed variation is large.** Across three U0 seeds, the 4-task mean val R@20 spans .0072 (.1180–.1252). Single-seed optimisation gains of that size should be read with caution.
- **Learning rate.** lr 5e-4 raises the 4-task mean val R@20 on all three seeds, by +.0048 on average (about one seed std). The gain comes mainly from the like tasks; long view is seed-dependent.
- **Temperature.** On seed 0, τ = 0.1 raises long view and lowers like; the 4-task mean does not improve.

Details: §4.4 and [`docs/EXPLORATORY_RESULTS.md`](docs/EXPLORATORY_RESULTS.md).

## 3. Three-tower architecture (formal v1)

![Formal v1 three-tower architecture](report/figures/arch_threetower.png)

### 3.1 Towers

- **One shared music tower.** It runs once per step. The video tower gathers its music input from that same output, and the video losses back-propagate into the music tower.
- **The video tower** combines video ID, author, the music representation and metadata.
- **The user tower** combines user ID with a 2-layer Transformer over the last 100 events strictly before the prediction time. History tokens are the shared video embeddings plus feedback, play-ratio, recency and position.

### 3.2 Representations and heads

- Each tower outputs a 128-d base embedding.
- Separate `Linear(128→64) + L2` heads are used per target × action: 4 on the user side, 2 for video and 2 for music.
- The score is cosine similarity.

### 3.3 Training objective

- Per task: an exact full softmax over all train-vocabulary items (τ = 0.05). There is no sampling, so no logQ correction is needed.
- Duplicate positives share one candidate column.
- Other train positives of the same user and task are masked as false negatives. The row's own label is never masked.
- The four task losses are summed with equal weights.

### 3.4 U0 vs U1

- **U0** (main) uses no user metadata.
- **U1** adds 21 user-snapshot fields whose historical availability could not be verified, so it is reported with a leakage caveat and is not the headline.

The code is in [`models/towers.py`](src/kuairand_rec/models/towers.py) and [`models/loss.py`](src/kuairand_rec/models/loss.py); the diagram is drawn by [`scripts/draw_architecture.py`](scripts/draw_architecture.py).

## 4. Experimental results

The figures in §4.1–4.3 are generated from the formal **test** result CSVs by [`scripts/make_report_tables.py`](scripts/make_report_tables.py). Their error bars are 95% paired-bootstrap CIs, and their bar-chart y-axes start at 0. The exploratory figure in §4.4 shows **validation** metrics, is generated by [`scripts/explore/plot_wave3.py`](scripts/explore/plot_wave3.py) from `results/explore/wave3/`, and uses a zoomed y-axis.

### 4.1 Main comparison

Each task is evaluated on the same users for every model.

![Test Recall@20, four tasks](report/figures/fig1_baselines.png)

- U0 is above Popularity everywhere.
- U0 is below pure ALS on long view and above it on like.
- On like, U0 is statistically tied with ALS aux.
- U1 reaches ALS level on long view, with the snapshot caveat.
- Source: `results/m5/main_formal/bootstrap_models.csv`. Full R/NDCG@10/20/50 are in [`report/tables/t1_main_metrics.md`](report/tables/t1_main_metrics.md); all paired differences are in [`t2_paired_diffs.md`](report/tables/t2_paired_diffs.md).

### 4.2 Controlled cold-video experiment

Only same-condition baselines are shown; the seen-video reference is context, not a baseline.

![Cold-video Recall@20](report/figures/fig2_cold_video.png)

- **Left, full catalog (7,583 videos):** cold videos rarely reach the top 20, and the result is not distinguishable from random.
- **Right, held-out videos only (750):** the cold model ranks clearly better than cold Pop, cold ALS (0 by construction) and the analytical random ranking.
- Source: `results/m5/cold_video/eval_671dde9_job33/`. Full table: [`report/tables/t5_cold_video.md`](report/tables/t5_cold_video.md).

### 4.3 Ablations

Each is a single configuration flag, with the same data, seed and selection rule.

![Ablation differences vs U0](report/figures/fig4_ablations.png)

- No history: lower long view.
- No music input: higher long view.
- Like-task differences are mostly not significant.
- History-length buckets and the full error analysis are in the [extended report](report/REPORT_FULL.md) §2.3 and §3.

### 4.4 Post-hoc exploratory findings (validation only)

After the formal results were analysed, a separate exploratory study (branch `explore/s9-opt`) retrained U0 under single-setting changes. **These are validation metrics, not test results, and they do not replace the formal model.**

![No-music minus U0 at the same seed, validation R@20 (post-hoc exploratory)](results/explore/wave3/figures/fig2_nomusic_minus_u0.png)

- **No-music, three seeds.** Removing the music-to-video input raises both long-view tasks on 3/3 seeds (mean +.0068 / +.0072 val R@20).
  - The no-music seed range lies above the U0 seed range.
  - Like is not systematically lower.
  - The lead holds at nearly every epoch, so it is not an early-stopping artefact.
  - Same seed numbers are not paired initialisations.
- **Seeds matter.** Across three U0 seeds, the 4-task mean val R@20 is .1220 ± .0036. The observed range of .0072 is larger than the user-bootstrap uncertainty; it is not a statistical threshold. Single-seed differences of that size should be read with caution.
- **Learning rate.** lr 5e-4 is numerically higher than the formal lr 1e-3 on all three seeds (+.0048 on average, about one seed std). The gain comes mainly from like; long view is seed-dependent.
- **Temperature.** τ = 0.1 lifts long view and lowers like, leaving the mean unchanged.
- **Against ALS on validation.**
  - No-music removes U0's significant long-view R@20 deficit: all six CIs vs pure ALS include 0.
  - No run has a CI-supported R@20 advantage, so this is not equivalence and not a win.
  - Long-view NDCG@20 of no-music is above pure ALS (nominal CIs, secondary metric).
- **Reproduction.** Same-config reruns reproduce the formal U0 (job 27) and the formal no-music ablation (job 35) bitwise.
- **Details.** Full tables are in [`docs/EXPLORATORY_RESULTS.md`](docs/EXPLORATORY_RESULTS.md) and [`report/REPORT_FULL.md`](report/REPORT_FULL.md) §2.5.

## 5. Quick start

### 5.1 Environment and tests

`uv run` executes a command inside the project's uv-managed Python environment (Python 3.12, pinned by `uv.lock`).

```bash
uv sync                    # install the environment
uv run pytest              # 259 CPU tests (~2 min); real-data tests skip without data
```

### 5.2 Data pipeline and CPU smoke run

Download KuaiRand-Pure from [Zenodo](https://zenodo.org/records/10439422) into `data/raw/KuaiRand-Pure/data/`, then run:

```bash
uv run python -m kuairand_rec.data.pipeline --config configs/data.yaml        # M2, ~12 s
RUN_ROOT=/tmp/kuairand_runs uv run python -m kuairand_rec.models.train \
    --config configs/threetower_smoke.yaml --max_steps 5                      # CPU smoke, ~7 min
```

The smoke run trains 5 steps, then performs selection, export, evaluation and strict export/direct parity end to end.

### 5.3 Inspect results without training

Reading the results needs no training. See `results/` and the report, or download a trained run from Drive and query it:

```bash
uv run python -m kuairand_rec.models.retrieve --export_dir artifacts/u0/9218a76_seed0_job27/export \
    --user_id 1 --target video --action long_view --k 10
```

## 6. Reproducibility

### 6.1 Local CPU

- Setup, tests and the CPU smoke run.
- M1 dataset audit, M2 data pipeline, M3 Popularity/ALS baselines and the independent M3 oracle.
- Cold-video H selection and evaluation, bootstrap/bucket analysis, `retrieve()` and parity re-checks.
- Report tables and figures.

### 6.2 Remote GPU (Slurm)

- The five formal three-tower trainings: U0, U1, cold U0, no-history, no-music. Each run also exports, evaluates and checks parity in the same job.
- The formal runs were submitted with `sbatch` after SSH-ing to the GPU server. The job runs on a compute node and does not depend on the SSH session.
- The batch scripts (with our cluster paths, which must be replaced) are in the Drive bundle.

The stage-by-stage commands, their order and dependencies, which outputs overwrite committed files, and how to re-derive every formal analysis from the Drive runs on a CPU are all in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

### 6.3 Report helper scripts

Both are read-only on the formal results, and neither trains or re-scores a model.

- [`scripts/make_report_tables.py`](scripts/make_report_tables.py) regenerates the Markdown tables in `report/tables/` and figures 1–4 in `report/figures/` from the formal M5 and cold-video CSVs.
- [`scripts/check_report_claims.py`](scripts/check_report_claims.py) parses the report Markdown (written for the unabridged formal report) and checks 321 numbers against the formal CSV/JSON files, including significance stars:
  - every cell of the key result tables;
  - a fixed list of anchored prose numbers.

  Qualitative text, figure captions and dataset/model descriptions are listed as manual-review scope in `results/final_audit/report_claims.json`.

### 6.4 Formal runs and provenance


| run | training commit | condition |
|---|---|---|
| U0, U1 | `9218a76` | normal |
| cold U0, no-history, no-music | `671dde9` | normal / cold (H = 750 videos) |

- Each run records its git SHA, config, data and condition fingerprints, the selection log (`test_used_for_selection: false`) and per-file sha256 hashes.
- Post-processing ran at `671dde9`.
- Every stage was re-run locally and matches the formal outputs ([`results/final_audit/REPRODUCTION.md`](results/final_audit/REPRODUCTION.md)).

## 7. Repository structure

```
src/kuairand_rec/
  data/         loaders (allowed files only), cleaning, split, catalogs + train-only vocabs, history, pipeline
  audit/        dataset audit
  eval/         train/history views, evaluation frames (GT, seen mask, population), metrics, evaluator
  baselines/    popularity, implicit ALS, baseline runner
  models/       features, towers, loss, training, validation, export, parity, retrieve
  experiments/  bootstrap, history buckets, cold-video experiment, main analysis
configs/        one YAML per stage (data, baselines, 6 three-tower configs, cold video, analysis)
tests/          259 tests for leakage, masking, metrics, loss, parity, cold-video isolation
scripts/        report tables/figures, claim check, error analysis, audits, architecture diagram
results/        committed small outputs: audit, data report, baselines, M5 analysis, final audit, explore (val copies)
report/         REPORT.md (concise), REPORT_FULL.md (extended), figures/, tables/
docs/           task specification, reproducibility guide, exploratory results
```

## 8. Artifacts and report

| where | what |
|---|---|
| This repository | Source, configs, tests, small result files (metrics, bootstrap CSVs, frames summary, audits) and the report |
| [Google Drive](https://drive.google.com/drive/folders/1VRXe8liHr7IiZfjhHi-K9HIZbtqeaT8U?usp=sharing) | The 5 complete formal runs (checkpoints, exported embeddings, ID mappings, per-user metrics, parity, logs); processed data and baseline per-user tables; the results and audit bundle; Cursor coding-agent session logs; and `05_Exploratory_Experiments.zip`, the post-hoc exploratory Waves 1–3 (11 runs, validation only, kept separate from the formal bundle). `00_START_HERE.md` is the navigation page, and `MANIFEST_SHA256.txt` lists every file's hash |

The research report is [`report/REPORT.pdf`](report/REPORT.pdf) (source [`report/REPORT.md`](report/REPORT.md)); the extended version is [`report/REPORT_FULL.md`](report/REPORT_FULL.md).

## 9. Limitations

- **User metadata.** None of the 21 snapshot fields used by U1 can be shown to be available at the historical prediction time: 2 are time-dependent by definition, and 19 have unknown availability. U0 therefore uses no user metadata. This is a documented deviation from "ID + usable metadata + history".
- **Cold videos on the full catalog.** The content path ranks new videos sensibly among themselves, but rarely surfaces them against warm videos.
- **One seed per formal model.** CIs reflect user sampling, not training variance. The exploratory three-seed study shows seed variation of about ±.004 in mean val R@20, so differences of about 0.01 between three-tower variants may not hold across seeds.
- **Cross-platform numerics.** The formal metrics were scored on Linux. Re-scoring the same exports on macOS changes at most 2 users per frame (test means ≤ 1.1e-7) through exact score ties at the cut-off.
- **Exploratory study.** It is post-hoc: designed after the test results were seen, with three seeds and settings that are subject to selection bias. It is reported on validation only.
- **Further caveats.** The 20-minute feedback delay is an assumption, and KuaiRand-Pure keeps candidate-pool videos only. See report §4.
