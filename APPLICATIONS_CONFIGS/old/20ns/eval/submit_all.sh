#!/bin/bash
set -euo pipefail

# Submit all MSD jobs in 323K/ and 353K/
#
# Usage:
#   bash submit_all.sh            # submit
#   bash submit_all.sh --dry-run  # print sbatch commands only

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

ROOT_DIR="/u/yjian1/project/MLFF-distill/APPLICATIONS_CONFIG/20ns/eval"
DIRS=(
  "$ROOT_DIR/323K"
  "$ROOT_DIR/353K"
)

for d in "${DIRS[@]}"; do
  if [[ ! -d "$d" ]]; then
    echo "Missing directory: $d" >&2
    exit 1
  fi

  shopt -s nullglob
  files=("$d"/*.sh)
  shopt -u nullglob

  if (( ${#files[@]} == 0 )); then
    echo "No .sh files found in $d" >&2
    continue
  fi

  for f in "${files[@]}"; do
    if (( DRY_RUN )); then
      echo sbatch "$f"
    else
      sbatch "$f"
    fi
  done
done
