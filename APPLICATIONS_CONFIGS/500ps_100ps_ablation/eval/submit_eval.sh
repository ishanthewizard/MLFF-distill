#!/bin/bash
# Submit all MSD eval jobs for 500ps and 100ps ablation trajectories
#
# Usage:
#   ./submit_eval.sh           # submit all
#   ./submit_eval.sh 500ps     # submit only 500ps evals
#   ./submit_eval.sh 100ps     # submit only 100ps evals

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILTER="${1:-}"

for script in "$SCRIPT_DIR"/msd_*.sh; do
    if [[ ! -f "$script" ]]; then
        continue
    fi
    fname=$(basename "$script")
    # Skip example.sh if present
    if [[ "$fname" == "example.sh" ]]; then
        continue
    fi
    if [[ -n "$FILTER" ]] && [[ "$fname" != *"${FILTER}"* ]]; then
        continue
    fi
    echo "Submitting $fname..."
    sbatch "$script"
done
