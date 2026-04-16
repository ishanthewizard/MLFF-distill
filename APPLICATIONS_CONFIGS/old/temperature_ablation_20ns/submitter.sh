#!/bin/bash
set -euo pipefail

# Activate environment and submit all ablation jobs for 323K and 353K ckpts.

cd /u/yjian1/project/MLFF-distill
source /u/yjian1/project/Generative_3d_turbulence_flow/yjian1/env/miniconda3/bin/activate fairchemV2

# How long to wait between cycles (hours). Override by exporting SLEEP_HOURS=...
SLEEP_HOURS="${SLEEP_HOURS:-48}"
SLEEP_SECONDS=$((SLEEP_HOURS * 3600))
# How many cycles to run (must be >=1). Hardcoded.
MAX_RUNS=6


submit_jobs() {
  for dir in APPLICATIONS_CONFIG/temperature_ablation_20ns/323K-ckpt APPLICATIONS_CONFIG/temperature_ablation_20ns/353K-ckpt; do
    echo "[$(date)] Submitting jobs in $dir"
    shopt -s nullglob
    for f in "$dir"/*.sh; do
      echo "sbatch $f"
      sbatch "$f"
    done
    shopt -u nullglob
  done
}

# Optional initial sleep before starting cycles (hardcoded to 40h)
# INITIAL_SLEEP_HOURS=40
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
