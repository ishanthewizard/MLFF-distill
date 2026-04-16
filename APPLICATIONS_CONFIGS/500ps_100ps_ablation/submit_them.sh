#!/bin/bash
# Submit all ablation jobs from 500ps and 100ps directories

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in "$SCRIPT_DIR"/500ps/*.sh "$SCRIPT_DIR"/100ps/*.sh; do
    if [[ -f "$script" ]]; then
        echo "Submitting $(basename "$script")..."
        sbatch "$script"
    fi
done
