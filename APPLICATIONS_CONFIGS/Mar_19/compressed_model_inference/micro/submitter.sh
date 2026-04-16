#!/bin/bash
# Submit all micro inference jobs in this directory, skipping example files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in "$SCRIPT_DIR"/*.sh; do
    if [[ -f "$script" && "$(basename "$script")" != "submitter.sh" && "$(basename "$script")" != example_* ]]; then
        echo "Submitting $(basename "$script")..."
        sbatch "$script"
    fi
done
