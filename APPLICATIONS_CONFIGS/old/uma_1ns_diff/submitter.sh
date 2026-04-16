#!/bin/bash
set -euo pipefail

cd /u/yjian1/project/MLFF-distill


DIR="/u/yjian1/project/MLFF-distill/APPLICATIONS_CONFIG/uma_1ns_diff"
DRY_RUN="${DRY_RUN:-0}"

shopt -s nullglob
for f in "$DIR"/*.sh; do
  if [[ "$(basename "$f")" == "submitter.sh" ]]; then
    continue
  fi
  echo "sbatch $f"
  if [[ "$DRY_RUN" != "1" ]]; then
    sbatch "$f"
  fi
done
shopt -u nullglob
