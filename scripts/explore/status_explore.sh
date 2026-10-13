#!/bin/bash
# S9 explore job status (server, read-only). Usage: bash scripts/explore/status_explore.sh
R=/local-scratch/liang/runs_explore; D=$R/slurm
echo "## $(date -u +%FT%TZ)  explore HEAD: $(git -C /local-scratch/liang/explore/kuairand-explore rev-parse HEAD)"
echo "## formal HEAD (must be unchanged): $(git -C /local-scratch/liang/kuairand-threetower-recsys rev-parse HEAD)"
echo "## squeue"; squeue -u "$USER" -o "%.6i %.18j %.9T %.10M %.10l %.30R"
echo "## exit codes"; cat $D/status/*.rc 2>/dev/null || echo "(none yet)"
for f in $D/x_*.out; do
  [ -f "$f" ] || continue
  echo "## $(basename "$f" .out): last val epoch / final / errors"
  grep '^{"epoch": [0-9]*, "step"' "$f" | tail -1
  grep -E '^\{"out_dir"|^\{"ok"' "$f" | cut -c1-300
  grep -iE "Traceback|Error|NonFinite" "$f" "${f%.out}.err" 2>/dev/null | tail -3
done
ls -d $R/threetower/x_*/* 2>/dev/null
