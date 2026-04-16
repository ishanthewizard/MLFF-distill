#!/bin/bash
# Submit all micro_very_aggressive inference jobs, skipping example files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in "$SCRIPT_DIR"/*.sh; do
    if [[ -f "$script" && "$(basename "$script")" != example_* ]]; then
        echo "Submitting $(basename "$script")..."
        sbatch "$script"
    fi
done
