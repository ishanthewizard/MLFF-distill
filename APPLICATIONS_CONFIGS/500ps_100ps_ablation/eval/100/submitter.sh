#!/bin/bash
# Submit all MSD eval jobs for 100ps ablation trajectories

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in "$SCRIPT_DIR"/msd_100ps_*.sh; do
    if [[ -f "$script" ]]; then
        echo "Submitting $(basename "$script")..."
        sbatch "$script"
    fi
done
