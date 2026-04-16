#!/bin/bash
set -euo pipefail

# this is running on login node 2

# Activate environment and submit all ablation jobs in this directory.

cd /u/yjian1/project/MLFF-distill
source /u/yjian1/project/Generative_3d_turbulence_flow/yjian1/env/miniconda3/bin/activate fairchemV2

# How long to wait between cycles (hours). Override by exporting SLEEP_HOURS=...
SLEEP_HOURS="${SLEEP_HOURS:-48}"
SLEEP_SECONDS=$((SLEEP_HOURS * 3600))
# How many cycles to run (must be >=1). Hardcoded.
MAX_RUNS=6

submit_jobs() {
  dir="APPLICATIONS_CONFIG/ablation_on_factors_after_temperature"
  echo "[$(date)] Submitting jobs in $dir"
  shopt -s nullglob
  for f in "$dir"/*.sh; do
    # Safety: never submit anything under eval/, even if the glob changes later.
    if [[ "$f" == */eval/* ]]; then
      echo "Skipping eval script: $f"
      continue
    fi
    base="$(basename "$f")"
    if [[ "$base" == submitter*.sh ]]; then
      continue
    fi
    echo "sbatch $f"
    sbatch "$f"
  done
  shopt -u nullglob
}

# INITIAL_SLEEP_HOURS=48
# INITIAL_SLEEP_SECONDS=$((INITIAL_SLEEP_HOURS * 3600))
# echo "[$(date)] Initial sleep for ${INITIAL_SLEEP_HOURS}h (${INITIAL_SLEEP_SECONDS}s) before first submission cycle"
# sleep "${INITIAL_SLEEP_SECONDS}"

for ((run=1; run<=MAX_RUNS; run++)); do
  echo "[$(date)] Cycle ${run}/${MAX_RUNS}: cancelling existing jobs for $USER"
  scancel -u "$USER" || true

  submit_jobs

  if (( run < MAX_RUNS )); then
    echo "[$(date)] Cycle ${run}/${MAX_RUNS}: sleeping for ${SLEEP_HOURS}h (${SLEEP_SECONDS}s) before next cycle"
    sleep "${SLEEP_SECONDS}"
  else
    echo "[$(date)] Cycle ${run}/${MAX_RUNS}: max cycles reached, exiting"
  fi
done
