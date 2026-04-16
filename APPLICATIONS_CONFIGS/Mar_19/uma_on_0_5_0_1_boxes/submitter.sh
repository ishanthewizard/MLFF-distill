#!/bin/bash
# Submit all UMA 100ps jobs for uma_on_0_5_0_1_boxes systems

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in "$SCRIPT_DIR"/uma_100ps_*.sh; do
    if [[ -f "$script" ]]; then
        echo "Submitting $(basename "$script")..."
        sbatch "$script"
    fi
done
