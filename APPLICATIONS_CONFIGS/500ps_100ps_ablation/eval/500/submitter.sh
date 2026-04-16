#!/bin/bash
# Submit all MSD eval jobs for 500ps ablation trajectories

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in "$SCRIPT_DIR"/msd_500ps_*.sh; do
    if [[ -f "$script" ]]; then
        echo "Submitting $(basename "$script")..."
        sbatch "$script"
    fi
done
