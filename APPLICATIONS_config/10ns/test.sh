#!/usr/bin/env bash

# Launch md_flex_ablation_20ns_g6.sh every 24 hours, making sure
# no overlapping runs of the same job are active to avoid concurrent writes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/md_flex_ablation_20ns_g6.sh"
JOB_NAME="g6_outliers"
SLEEP_SECONDS=$((24 * 60 * 60))

echo "Starting daily submit loop for ${JOB_SCRIPT}"
echo "Canceling any existing jobs named ${JOB_NAME} before each submission"

while true; do
  echo "[$(date)] Cancelling existing jobs named ${JOB_NAME} for user ${USER}..."
  # Cancel any running or pending jobs with the target name for this user.
  scancel -u "${USER}" -n "${JOB_NAME}" || true

  echo "[$(date)] Submitting ${JOB_SCRIPT}..."
  sbatch "${JOB_SCRIPT}"

  echo "[$(date)] Sleeping for ${SLEEP_SECONDS} seconds (~24h)..."
  sleep "${SLEEP_SECONDS}"
done

