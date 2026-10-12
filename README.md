# KuaiRand-Pure multi-action three-tower retrieval

This is a research take-home. On [KuaiRand-Pure](https://kuairand.com/), we retrieve **videos** and their **background-music IDs** for two engagement actions, **long view** and **like**. That gives four full-catalog retrieval tasks: video/long view, video/like, music/long view and music/like.

The three-tower model (user / video / music) is compared with Popularity and implicit ALS. All models run under one leakage-controlled temporal protocol and are scored by one shared evaluator with paired bootstrap CIs.

- **Report:** [`report/REPORT.md`](report/REPORT.md). It covers design, results, error analysis, limitations and next steps, with figures in `report/figures/` and full tables in `report/tables/`.
- **Specification:** [`docs/assignment.md`](docs/assignment.md).
- **Design decisions:** [`PLAN.md`](PLAN.md) §3b (protocol), §3c (evaluator and baselines) and §3d (model).
- **Progress log:** [`PROGRESS.md`](PROGRESS.md).
- **Requirement-by-requirement audit:** `artifacts/final_compliance_review/ASSIGNMENT_COMPLIANCE_AUDIT.md` (the artifacts bundle).

## Headline results

Test split, full catalog, seen-masked ranking; Recall@20 with 95% paired user-bootstrap CIs.

| model | video LV | video like | music LV | music like |
|---|---:|---:|---:|---:|
| Popularity | .0806 | .0716 | .0856 | .0726 |
| ALS (pure) | **.1244** | .0555 | **.1282** | .0586 |
| ALS (like + 0.3·LV) | – | **.1008** | – | **.1054** |
| 3T-U0 (main: ID + history) | .1124 | .0913 | .1162 | .1022 |
| 3T-U1 (+ user snapshot; leakage risk) | .1274 | .0859 | .1321 | .0864 |

- U0 beats Popularity on all four tasks.
- It loses to pure ALS on long view (−0.012, significant) and beats it on like.
- It is statistically tied with the auxiliary-signal ALS on like.
- In the controlled cold-video experiment, the content path clearly beats random *among held-out videos*, but is not significantly better than random on the full catalog.

These negative results are reported as is; see the report for the full picture.

## Repository layout

```
src/kuairand_rec/
  data/        io (allowed-file loaders), clean, split, catalog (catalogs + train-only vocabs),
               history (strict-past history + uncapped seen mask), train_stats, pipeline (M2 CLI)
  audit/       dataset audit (M1 CLI)
  eval/        views (train_view / history_view), frame (GT / mask / population), metrics, evaluator
  baselines/   popularity, als (implicit), run (M3 CLI)
  models/      features, towers, loss, data (samples / Pos_train / batches), train (M4 CLI),
               evaluation (validation / selection), export, inference, parity (CLI), retrieve (CLI),
               artifacts (fingerprints), benchmark (CLI)
  experiments/ bootstrap, buckets, sources, analysis (M5 CLI), cold_video (M5 CLI)
configs/       one YAML per step (data, baselines, threetower_{smoke,u0,u1,cold_u0,abl_nohist,abl_nomusic},
               cold_video, m5_analysis)
scripts/       audit_m3_correctness.py, audit_final_compliance.py, error_analysis.py,
               make_report_tables.py, check_report_claims.py (+ superseded audit_phase_a*.py)
tests/         259 CPU tests (hand-built fixtures + real-data checks that skip if data is absent)
results/       small committed outputs (audit, data report, baselines, M5, final audit / analysis)
report/        REPORT.md, figures/, tables/
artifacts/     (git-ignored, shipped separately) checkpoints, exports, run logs, review documents
```

## Setup

**Environment.**
- Python 3.12 with [uv](https://docs.astral.sh/uv/). The pinned dependencies are in `uv.lock`: torch 2.14.1, implicit 0.7.3, pandas 3.0.6 and numpy 2.5.4.
- Install with `uv sync`.
- Optionally, `export MPLCONFIGDIR=/tmp/mpl` avoids a matplotlib cache rebuild when the home directory is read-only.

**Data.**
- Download `KuaiRand-Pure.tar.gz` from [Zenodo](https://zenodo.org/records/10439422) (md5 `0820331067a3784d9691136f772b35a7`) and extract it to `data/raw/KuaiRand-Pure/data/`.
- Alternatively, set `KUAIRAND_RAW_DIR` to the extracted `data/` directory.
- Raw files are read-only.
- Only the two standard logs, `user_features_pure.csv` and `video_features_basic_pure.csv` are read.
- The loader refuses `video_features_statistic_pure.csv` (future-leaking monthly aggregates) and the random log.

## Reproduction workflow

Every step is a CLI with a YAML config. The commands below were checked against `--help`. Indicative runtimes are for an M1 Mac (CPU) or one A100.

```bash
# M1  Dataset audit (no filtering; ~30 s, ~2 min in a sandbox) -> results/audit/{AUDIT.md, audit.json, figures/}
uv run python -m kuairand_rec.audit.run --out results/audit

# M2  Clean, split, catalogs, train-only vocabs/stats (~12 s, deterministic)
#     -> data/processed/ (git-ignored), results/data_pipeline/{DATA_REPORT.md, data_report.json}
uv run python -m kuairand_rec.data.pipeline --config configs/data.yaml

# M3  Shared evaluator frames + Popularity + ALS (val grid, then test once; ~10 min)
#     -> results/baselines/{BASELINES.md, metrics.csv, als_tuning.csv, frames_summary.json, run_meta.json}
uv run python -m kuairand_rec.baselines.run --config configs/baselines.yaml
uv run python scripts/audit_m3_correctness.py          # independent oracle (~3 min; exit 1 on failure)

# M4  Three-tower: CPU smoke on real data (5 steps -> val selection -> export -> metrics -> parity; ~7 min)
RUN_ROOT=/tmp/kuairand_runs uv run python -m kuairand_rec.models.train --config configs/threetower_smoke.yaml --max_steps 5

# M4  Formal runs (GPU; ~13 min train + ~5 min export/eval + ~5 min parity each). Output goes to
#     $RUN_ROOT/threetower/<exp>/<timestamp>/ (or --out_dir); the trainer refuses dirs inside the worktree,
#     and require_clean_git: true refuses a dirty tree.
uv run python -m kuairand_rec.models.train --config configs/threetower_u0.yaml
uv run python -m kuairand_rec.models.train --config configs/threetower_u1.yaml
uv run python -m kuairand_rec.models.parity --run_dir <run_dir>          # re-run strict parity (also run by train)
uv run python -m kuairand_rec.models.benchmark --config configs/threetower_u0.yaml --steps 300 --val_once

# M5  Controlled cold-video experiment
uv run python -m kuairand_rec.experiments.cold_video select --config configs/cold_video.yaml   # draws H once (train only)
uv run python -m kuairand_rec.models.train --config configs/threetower_cold_u0.yaml             # GPU, H removed
uv run python -m kuairand_rec.experiments.cold_video evaluate --config configs/cold_video.yaml \
    --cold_run <cold_u0 run dir> --seen_run <u0 run dir> --out_dir results/m5/cold_video/<name>

# M5  Ablations (GPU; single-flag configs) + main analysis (bootstrap CIs, history buckets; CPU, seconds)
uv run python -m kuairand_rec.models.train --config configs/threetower_abl_nohist.yaml
uv run python -m kuairand_rec.models.train --config configs/threetower_abl_nomusic.yaml
uv run python -m kuairand_rec.experiments.analysis --config configs/m5_analysis.yaml \
    --run u0=<u0 run> --run u1=<u1 run> --run abl_nohist=<run> --run abl_nomusic=<run>

# Report tables / figures, error analysis, claim check (read-only on formal results)
uv run python scripts/make_report_tables.py
uv run python scripts/error_analysis.py
uv run python scripts/check_report_claims.py
```

**GPU runs.** The formal runs used Slurm with `artifacts/m5_gpu_jobs/scripts_server/train_threetower_m5.sbatch`. That script checks the expected git SHA and a clean worktree, sets `OPENBLAS_NUM_THREADS=1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and then calls `models.train` with `--out_dir`.

**Reproducing the formal analysis from the shipped artifacts.**
- `configs/m5_analysis.yaml` points U0 and U1 at `artifacts/`. Add `--run abl_nohist=artifacts/abl_nohist/671dde9_seed0_job34 --run abl_nomusic=artifacts/abl_nomusic/671dde9_seed0_job35` for the ablations.
- For the cold-video evaluation, use `--cold_run artifacts/cold_u0/671dde9_seed0_job33 --seen_run artifacts/u0/9218a76_seed0_job27`.
- Both reproduce `results/m5/main_formal/` and `results/m5/cold_video/eval_671dde9_job33/` locally (see `results/final_audit/REPRODUCTION.md`).

## Evaluation protocol in one paragraph

**Split and representation.**
- The logs are split globally by time at CST midnights: train up to 04-16, val 04-17–04-25, test 04-26–05-08.
- Users are frozen at T_test. Each user is represented by their history strictly before T_test (train + val events), capped at the 100 most recent events.

**Ground truth and ranking.**
- GT is the de-duplicated set of a user's test positives. Out-of-catalog items, `music_id = 0` and items the user interacted with before T_test are removed from it.
- Seen items are masked using all pre-T_test interactions, uncapped.
- The catalog is all 7,583 metadata videos and all 7,201 non-zero music IDs.
- Ties are broken by catalog row.

**Metrics and population.**
- Recall@K = hits / |GT|. NDCG@K uses binary gain.
- Users with a non-empty GT and at least one history event are evaluated. Cold users (no history) are reported separately.

**Fitting and statistics.**
- Every fitted quantity (vocabularies, scalers, Popularity, ALS, U1 vocabulary) uses train only. Validation selects ALS configs and three-tower epochs. Test is evaluated once.
- CIs: 1,000 paired user-level bootstrap resamples with percentile intervals.

## Architecture

- **Music tower:** music ID + music type, through an MLP + LayerNorm, gives a 128-d embedding.
- **Video tower:** video ID, author, the shared music-tower output (computed once per step) and metadata (type, upload type/date, tags, log duration/width/height with missing flags) give a 128-d embedding.
- **User tower:** user ID plus a 2-layer Transformer history encoder over the shared video embeddings, with feedback bits, play ratio, recency and position. U1 adds 21 snapshot fields. The output is 128-d.
- **Heads:** separate Linear(128 → 64) + L2 projections per target × action (4 user, 2 video, 2 music). The score is cosine similarity.

**Training.**
- The loss is an exact full softmax over the train-vocab items per task (τ = 0.05). Duplicate positives share one candidate column.
- Same-user, same-task train positives are masked as false negatives; the label is never masked.
- The four task losses are summed with equal weights.
- PAD and OOV rows are fixed at zero, and ID dropout of 0.1 trains the zero-ID content path for unseen items and users.

## Export and `retrieve()`

Each run exports, to `<run_dir>/export/`:
- item matrices for every catalog row;
- user matrices at T_val and T_test;
- id lists and the T_test seen-mask CSR;
- a `manifest.json` with every file's sha256, the git SHA, the condition ID and the data fingerprint.

**Reported metrics.** These come only from `ExportedScorer` (numpy float32). A separate process reloads the checkpoint and recomputes the scores (`models.parity`). Across 2,048 stratified users, every catalog row, the 4 tasks and both val and test, the scores are bitwise equal, and the top-50 lists and per-user metrics are identical.

**`retrieve()` in Python.**

```python
from kuairand_rec.models.retrieve import load_retriever
retrieve = load_retriever("artifacts/u0/9218a76_seed0_job27/export")
retrieve(user_id=1, target="video", action="long_view", k=10)   # [(video_id, cosine score), ...]
```

**`retrieve()` from the command line.**

```bash
uv run python -m kuairand_rec.models.retrieve --export_dir artifacts/u0/9218a76_seed0_job27/export \
    --user_id 1 --target video --action long_view --k 10      # --no_mask_seen to include seen items
```

**Behaviour.**
- Users are represented at T_test.
- Items the user interacted with before T_test are excluded by default.
- An unknown user raises `KeyError`.
- The scores and ranking are exactly those of the evaluator. `tests/test_retrieve_interface.py` recomputes the formal U0 R@20 and NDCG@20 from `retrieve()` output.

## Tests

```bash
uv run pytest            # 259 tests, ~2 min on an M1; real-data tests skip if data/processed is absent
```

The tests are focused on correctness-critical logic:
- **Leakage:** strict t < t_pred and same-timestamp exclusion, the cap of 100, the feedback delay, invariance to any event at or after t_pred, and cold-video H removal everywhere.
- **Masking:** the uncapped seen mask, GT de-duplication, sentinel and OOC handling.
- **Metrics:** hand-computed Recall/NDCG, tie-break, and −inf handling.
- **Loss:** duplicate positives, the FN mask against a brute-force definition, shared music, and hand-computed CE.
- **Train-only fitting guards.**
- **ALS semantics:** fold-in against `implicit`.
- **Bootstrap:** correctness and determinism.
- **Export/direct parity, and `retrieve()`.**

## Official results and provenance

| run | git SHA | condition | best epoch | artifacts |
|---|---|---|---:|---|
| U0 | `9218a76` | normal `133765eb12b75700` | 6 | `artifacts/u0/9218a76_seed0_job27` |
| U1 | `9218a76` | normal | 8 | `artifacts/u1/9218a76_seed0_job28` |
| cold U0 | `671dde9` | cold `1eee3839b6cc5c90` (H = 750 videos) | 7 | `artifacts/cold_u0/671dde9_seed0_job33` |
| no-history | `671dde9` | normal | 6 | `artifacts/abl_nohist/671dde9_seed0_job34` |
| no-music | `671dde9` | normal | 6 | `artifacts/abl_nomusic/671dde9_seed0_job35` |

**Run artifacts.** Each run directory holds:
- `config.yaml`, `run_meta.json` (git SHA, data and condition fingerprints, sample counts, timing) and `selection.json` (validation history, with `test_used_for_selection: false`);
- `best.pt` and `last.pt`;
- `export/` with its manifest;
- `metrics.csv` and per-user parquet;
- `parity.json`, `train_log.jsonl` and the Slurm logs.

**Checksums.** They are in `artifacts/SHA256SUMS.server` (M4) and `artifacts/m5_gpu_jobs/SHA256SUMS_m5_*.txt` (M5), and all verify.

**Result files.**
- Formal M5 analysis: `results/m5/main_formal/` (server job 37, git `671dde9`).
- Formal cold-video evaluation: `results/m5/cold_video/eval_671dde9_job33/`.
- `results/m5/main/` is an earlier U0/U1-only run of the same analysis. Its rows are identical to the corresponding rows of `main_formal`.

## Known limitations

These are discussed in report §4.
- One seed per model.
- U1 uses a user snapshot of unknown date that is no later than 2022-05-02 (inside the test period), so U1 may leak. U0, the main model, uses no user metadata.
- The 20-minute feedback-delay rule is an assumption.
- Cold-video retrieval on the full catalog is at random level.
- The music input to the video tower hurts long view.
- The formal metrics were scored on Linux. Re-scoring on macOS changes at most 2 users per frame through exact ties at the cut-off (test means ≤ 1.1e-7).
- Pure contains candidate-pool videos only, so absolute metric levels are not production-comparable.
