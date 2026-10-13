# Reproducibility guide

This file holds every stage command of the project. The README keeps only a summary.

**Conventions.**
- All commands run from the repository root.
- `uv run` executes the command inside the project's uv-managed Python 3.12 environment, pinned by `uv.lock`.
- Every command below was checked against the CLI's `--help` at commit `8d9d187`.

**You do not need to retrain anything to check the results.**
- The formal results are committed under `results/` and summarised in `report/REPORT.md`.
- The trained runs (checkpoints, embeddings, per-user metrics, logs) are in the [Google Drive bundle](https://drive.google.com/drive/folders/1VRXe8liHr7IiZfjhHi-K9HIZbtqeaT8U?usp=sharing).
- Section 5 shows how to re-derive the analysis from those runs on a CPU.

## 0. Setup

```bash
uv sync                                   # Python 3.12 + pinned deps (torch 2.14.1, implicit 0.7.3, pandas 3.0.6)
export MPLCONFIGDIR=/tmp/mpl              # optional: avoids a matplotlib cache rebuild in read-only homes
```

**Data.**
- Download `KuaiRand-Pure.tar.gz` from [Zenodo](https://zenodo.org/records/10439422) (md5 `0820331067a3784d9691136f772b35a7`) and extract it to `data/raw/KuaiRand-Pure/data/`, or point `KUAIRAND_RAW_DIR` at the extracted `data/` directory.
- Only the two standard logs, `user_features_pure.csv` and `video_features_basic_pure.csv` are read.
- The loader refuses `video_features_statistic_pure.csv` and the random log.
- The dataset is CC BY-SA 4.0.

## 1. Where each stage runs

| stage | runs on | depends on | needed to read the results? |
|---|---|---|---|
| Tests, CPU smoke | local CPU | data for the real-data tests (they skip otherwise) | no |
| M1 dataset audit | local CPU (~30 s; ~2 min in a sandbox) | raw data | no |
| M2 data pipeline | local CPU (~12 s, deterministic) | raw data | yes, if you re-run anything below |
| M3 baselines + oracle | local CPU (~10 min + ~3 min) | M2 | no (committed) |
| M4 / M5 three-tower training | **remote GPU via Slurm** (~13 min train + ~10 min export/eval/parity per run on one A100) | M2, M3 frames | no (Drive) |
| M5 cold-video selection / evaluation, main analysis | local CPU (seconds to ~2 min) | M2, M3, finished runs | no (committed) |
| Parity re-check, benchmark | CPU (parity) / GPU (benchmark) | a finished run | no |
| Report tables / figures, error analysis, claim check | local CPU (seconds) | committed results (+ Drive runs for error analysis) | no |

## 2. Tests and CPU smoke (local)

```bash
uv run pytest                                                   # 259 tests, ~2 min on an M1
RUN_ROOT=/tmp/kuairand_runs uv run python -m kuairand_rec.models.train \
    --config configs/threetower_smoke.yaml --max_steps 5        # needs M2; ~7 min on CPU
```

**What the smoke run does.** It runs on real data and goes through 5 training steps, validation-based selection, the canonical CPU export, metrics through `ExportedScorer`, and strict Direct-vs-Export parity in a separate process.

**Outputs.** Run outputs go to `$RUN_ROOT/threetower/<exp>/<timestamp>/`. The trainer refuses an output directory inside the git worktree.

## 3. Data, audit and baselines (local CPU, in order)

```bash
# M1  dataset audit (no filtering)          -> results/audit/{AUDIT.md, audit.json, figures/}
uv run python -m kuairand_rec.audit.run --out results/audit

# M2  clean, split, catalogs, train-only vocabularies / statistics
#     -> data/processed/ (git-ignored), results/data_pipeline/{DATA_REPORT.md, data_report.json}
uv run python -m kuairand_rec.data.pipeline --config configs/data.yaml

# M3  shared evaluation frames + Popularity + implicit ALS (validation grid, then test once)
#     -> results/baselines/{BASELINES.md, metrics.csv, als_tuning.csv, frames_summary.json, run_meta.json}
uv run python -m kuairand_rec.baselines.run --config configs/baselines.yaml
uv run python scripts/audit_m3_correctness.py      # independent oracle; exits 1 on any failure
```

**Overwrite warning.** These commands write to the committed `results/` paths. To compare against the committed files without overwriting them, copy the config and point `output_dir` / `report_dir` to a scratch directory. The audit takes `--out <dir>`.

## 4. Three-tower training (remote GPU, Slurm)

**Formal run configs.** Each run is independent; every config differs from `threetower_u0.yaml` only where stated, and this is test-enforced.

| config | run | needs first |
|---|---|---|
| `configs/threetower_u0.yaml` | U0, the main model (user ID + history) | M2, M3 |
| `configs/threetower_u1.yaml` | U1 = U0 + 21 user-snapshot fields | M2, M3 |
| `configs/threetower_cold_u0.yaml` | cold-video U0 (H removed) | M2, M3, H selection (§5) |
| `configs/threetower_abl_nohist.yaml` | ablation: `model.use_history: false` | M2, M3 |
| `configs/threetower_abl_nomusic.yaml` | ablation: `model.use_music: false` | M2, M3 |

**The runner command.** It is the same on any machine with a CUDA build of torch:

```bash
uv run python -m kuairand_rec.models.train --config configs/threetower_u0.yaml [--out_dir <dir outside the repo>]
```

**What one run does, in one process.**
1. Training with validation selection only.
2. `best.pt` and `selection.json` are frozen.
3. Canonical CPU export.
4. Validation and test metrics through `ExportedScorer`.
5. Strict parity in a subprocess.

**Formal-config guards.**
- The formal configs set `device: cuda` and `require_clean_git: true`: a run refuses a dirty worktree.
- Their `run_root` is a cluster path (`/local-scratch/liang/runs`). Override it with `RUN_ROOT=<dir>` or `--out_dir <dir>`.

**How the formal runs were launched.**
- We SSH'd to the GPU server, checked out a pinned commit and submitted with `sbatch`.
- Slurm runs the job on a compute node, so the job does not depend on the SSH session staying open.
- The scripts are shipped in the Drive bundle (`03_Results_and_Audits.zip` → `m5_gpu_jobs/scripts_server/train_threetower_m5.sbatch`). Each one checks the expected git SHA and a clean worktree, sets `OPENBLAS_NUM_THREADS=1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and then calls the runner above.
- Usage pattern (illustrative; the cluster paths and partition inside the script are ours and must be replaced):

```bash
sbatch -J u0 train_threetower_m5.sbatch configs/threetower_u0.yaml <expected_git_sha>
```

**Optional tools.** Neither is needed for the results.

```bash
uv run python -m kuairand_rec.models.parity --run_dir <run_dir>        # re-run strict parity (CPU; also run by train)
uv run python -m kuairand_rec.models.benchmark --config configs/threetower_u0.yaml --steps 300 --val_once   # GPU throughput
```

## 5. Controlled cold-video experiment and main analysis (local CPU)

```bash
# Draw the held-out set H once (train data only). The committed H is reproduced; refuses to overwrite a different H.
uv run python -m kuairand_rec.experiments.cold_video select --config configs/cold_video.yaml

# (GPU) train the cold model: configs/threetower_cold_u0.yaml (section 4)

# Evaluate every comparison group on the fixed cold frames
uv run python -m kuairand_rec.experiments.cold_video evaluate --config configs/cold_video.yaml \
    --cold_run <cold_u0 run dir> --seen_run <u0 run dir> --out_dir <output dir>

# Paired bootstrap CIs + history buckets over finished runs (seconds)
uv run python -m kuairand_rec.experiments.analysis --config configs/m5_analysis.yaml \
    --run u0=<u0 run> --run u1=<u1 run> --run abl_nohist=<run> --run abl_nomusic=<run>
```

**Reproducing the formal analysis from the Drive runs** (unzip the `01_*.zip` files into `artifacts/`):
- **Cold video.** Use `--cold_run artifacts/cold_u0/671dde9_seed0_job33 --seen_run artifacts/u0/9218a76_seed0_job27`. This reproduces `results/m5/cold_video/eval_671dde9_job33/` byte-identically.
- **Main analysis.** `configs/m5_analysis.yaml` already points U0/U1 at `artifacts/`. Add `--run abl_nohist=artifacts/abl_nohist/671dde9_seed0_job34 --run abl_nomusic=artifacts/abl_nomusic/671dde9_seed0_job35`. This reproduces `results/m5/main_formal/` (CI bounds within 3e-16).
- **Output paths.** The analysis config writes to `output_dir` = `results/m5/main`; it has no `--out_dir` flag, so copy the YAML and edit `output_dir` to write elsewhere. `cold_video evaluate` accepts `--out_dir`, and its default is `results/m5/cold_video`.
- **Evidence.** See `results/final_audit/REPRODUCTION.md`.

## 6. Export and `retrieve()` (local CPU, needs a finished run)

```bash
uv run python -m kuairand_rec.models.retrieve --export_dir artifacts/u0/9218a76_seed0_job27/export \
    --user_id 1 --target video --action long_view --k 10       # --no_mask_seen to include seen items
```

```python
from kuairand_rec.models.retrieve import load_retriever
retrieve = load_retriever("artifacts/u0/9218a76_seed0_job27/export")
retrieve(user_id=1, target="video", action="long_view", k=10)   # [(video_id, cosine score), ...]
```

## 7. Report tables, figures and checks (local CPU, read-only on results)

```bash
uv run python scripts/make_report_tables.py      # report/tables/*.md + report/figures/fig{1..4}_*.png from the formal CSVs
uv run python scripts/check_report_claims.py     # parses report/REPORT.md and checks its numbers against the formal CSVs
uv run python scripts/error_analysis.py          # post-hoc breakdowns -> results/final_analysis/ (needs the Drive runs)
uv run python scripts/audit_final_compliance.py checks    # independent real-data checks -> results/final_audit/checks.json
uv run python scripts/draw_architecture.py       # README architecture diagram -> report/figures/arch_threetower.png
```

**What `check_report_claims.py` checks.** It reads `REPORT.md` and checks two things against the formal CSVs/JSON:
- every cell of the key result tables: Table 1/2/3, the bucket and cold-user tables, the §1.5 population table and Table B1, including significance stars;
- a fixed list of anchored prose numbers. An edited anchor fails the check; it never silently passes.

It currently checks 321 numbers. Qualitative text, figure captions and the §1.1–1.4 dataset/model numbers are manual-review scope; they are listed in `results/final_audit/report_claims.json`.

**What these scripts write.** None of them trains or re-scores a model.
- `make_report_tables.py` regenerates `report/`.
- `check_report_claims.py` writes only `report_claims.json`.
