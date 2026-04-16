#!/bin/bash
# Submit all eval jobs in this directory, skipping examples

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in "$SCRIPT_DIR"/*.sh; do
    base="$(basename "$script")"
    if [[ -f "$script" && "$base" != "submitter.sh" && "$base" != example_* ]]; then
        echo "Submitting $base..."
        sbatch "$script"
    fi
done
