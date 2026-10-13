#!/bin/bash
# One-time server setup for the S9 explore clone (run on gpu228-liang AFTER explore/s9-opt is pushed).
# Never touches the formal clone (/local-scratch/liang/kuairand-threetower-recsys) or runs/.
# Usage: bash server_setup.sh <EXPLORE_SHA>
set -euo pipefail
SHA=$1
BASE=/local-scratch/liang/explore
REPO=$BASE/kuairand-explore
FORMAL=/local-scratch/liang/kuairand-threetower-recsys
RUN_ROOT=/local-scratch/liang/runs_explore
KEY="ssh -i ~/.ssh/kuairand_3tower_deploy_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes"
FORMAL_HEAD_BEFORE=$(git -C "$FORMAL" rev-parse HEAD)
mkdir -p "$BASE" "$RUN_ROOT/slurm/status" "$RUN_ROOT/threetower" "$RUN_ROOT/checks"
[ ! -e "$REPO" ] || { echo "explore clone exists: $REPO"; exit 2; }
# Read-only deploy key (same key and options as the formal clone); clone only the explore branch.
GIT_SSH_COMMAND="$KEY" git clone --branch explore/s9-opt --single-branch \
  git@github.com:yangbright-2001/Kuairand-3tower-Recommender.git "$REPO"
git -C "$REPO" config core.sshCommand "$KEY"
git -C "$REPO" checkout --detach "$SHA"
cd "$REPO"
export UV_CACHE_DIR=/local-scratch/liang/.cache/uv KUAIRAND_RAW_DIR=/local-scratch/liang/data/KuaiRand-Pure/data
uv sync --frozen
# Processed data rebuilt inside the explore clone (git-ignored), then compared with the formal clone.
.venv/bin/python -m kuairand_rec.data.pipeline --config configs/data.yaml
git checkout -- results/data_pipeline          # the report only re-stamps the git SHA; keep the tree clean
for f in "$FORMAL"/data/processed/*.parquet "$FORMAL"/data/processed/vocab_meta.json; do
  n=$(basename "$f"); [ "$(md5sum < "$f")" = "$(md5sum < data/processed/$n)" ] || { echo "DIFF $n"; exit 3; }
done
.venv/bin/python -c "from kuairand_rec.data.pipeline import load_processed; from kuairand_rec.models.evaluation import build_eval_frames; print('frames ok', len(build_eval_frames(load_processed('data/processed'))))"
.venv/bin/python -m pytest -q tests/test_explore_configs.py
[ -z "$(git status --porcelain)" ] || { echo "explore clone dirty after setup"; exit 4; }
[ "$(git -C "$FORMAL" rev-parse HEAD)" = "$FORMAL_HEAD_BEFORE" ] || { echo "formal HEAD changed!"; exit 5; }
echo "explore clone ready at $(git rev-parse HEAD); formal clone untouched at $FORMAL_HEAD_BEFORE"
