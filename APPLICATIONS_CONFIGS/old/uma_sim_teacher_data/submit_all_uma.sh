#!/bin/bash
# Submit all UMA SLURM jobs in this directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting all UMA jobs under: ${SCRIPT_DIR}"

for job in "${SCRIPT_DIR}"/*.sh; do
    # Skip this submit helper itself
    if [[ "$(basename "$job")" == "submit_all_uma.sh" ]]; then
        continue
    fi
    echo "sbatch $job"
    sbatch "$job"
done

echo "Done submitting all UMA jobs."

