#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/local-scratch/liang/runs_explore/slurm/%x_%j.out
#SBATCH --error=/local-scratch/liang/runs_explore/slurm/%x_%j.err
# S9 explore runner. Usage (from the explore clone):
#   sbatch -J <exp> scripts/explore/train_explore.sbatch configs/explore/<exp>.yaml <expected_sha>
# Same checks, env and runner command as the M5 formal runner (train_threetower_m5.sbatch), except:
#   * REPO     = the separate explore clone (never the formal clone)
#   * RUN_ROOT = /local-scratch/liang/runs_explore (never the formal runs/), incl. Slurm logs/status
#   * run dir  = <RUN_ROOT>/threetower/<exp>/<sha7>_seed<seed from config>_job<id>
#   * refuses configs whose exp does not start with "x_" or that live outside configs/explore/
# Slurm accounting is disabled on this cluster (no sacct): the exit code goes to status/<name>_<id>.rc.
set -euo pipefail
CFG=$1; WANT_SHA=$2
REPO=/local-scratch/liang/explore/kuairand-explore
FORMAL_REPO=/local-scratch/liang/kuairand-threetower-recsys
RUN_ROOT=/local-scratch/liang/runs_explore
ST=$RUN_ROOT/slurm/status
mkdir -p "$ST"
OUT=""
finish() {
  rc=$?
  echo "end=$(date -Is) rc=$rc"
  echo "rc=$rc end=$(date -Is) job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME} out=${OUT}" > "$ST/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.rc"
  if [ -n "$OUT" ] && [ -d "$OUT" ]; then
    cp "$RUN_ROOT/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUT/stdout.log" || true
    cp "$RUN_ROOT/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$OUT/stderr.log" || true
  fi
}
trap finish EXIT
cd "$REPO"
[ "$(pwd -P)" != "$(cd "$FORMAL_REPO" && pwd -P)" ] || { echo "refusing to run in the formal clone"; exit 2; }
case "$CFG" in configs/explore/x_*.yaml) ;; *) echo "not an explore config: $CFG"; exit 2;; esac
export KUAIRAND_RAW_DIR=/local-scratch/liang/data/KuaiRand-Pure/data
export RUN_ROOT
export UV_CACHE_DIR=/local-scratch/liang/.cache/uv
export OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
SHA=$(git rev-parse HEAD)
[ "$SHA" = "$WANT_SHA" ] || { echo "SHA mismatch: $SHA != $WANT_SHA"; exit 2; }
[ -z "$(git status --porcelain)" ] || { echo "dirty worktree"; exit 2; }
read -r EXP SEED < <(.venv/bin/python -c "import yaml,sys;c=yaml.safe_load(open(sys.argv[1]));print(c['exp'],c['train']['seed'])" "$CFG")
case "$EXP" in x_*) ;; *) echo "explore exp must start with x_: $EXP"; exit 2;; esac
OUT_TARGET=$RUN_ROOT/threetower/$EXP/${SHA:0:7}_seed${SEED}_job${SLURM_JOB_ID}
[ ! -e "$OUT_TARGET" ] || { echo "run dir exists: $OUT_TARGET"; exit 2; }
OUT=$OUT_TARGET
echo "job=$SLURM_JOB_ID host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES sha=$SHA cfg=$CFG exp=$EXP seed=$SEED out=$OUT start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
.venv/bin/python -m kuairand_rec.models.train --config "$CFG" --out_dir "$OUT"
