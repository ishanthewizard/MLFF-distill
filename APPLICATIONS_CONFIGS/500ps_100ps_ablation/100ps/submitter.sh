#!/bin/bash
# Submit all 100ps ablation jobs in this directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in "$SCRIPT_DIR"/*.sh; do
    # Skip this submitter script itself
    if [[ -f "$script" && "$(basename "$script")" != "submitter.sh" ]]; then
        echo "Submitting $(basename "$script")..."
        sbatch "$script"
    fi
done
