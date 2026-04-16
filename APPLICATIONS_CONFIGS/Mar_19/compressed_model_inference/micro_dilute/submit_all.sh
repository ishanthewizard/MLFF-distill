#!/bin/bash

# Submit all micro_dilute inference jobs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "infer_micro_naotf_dme_0_1M_298K.sh"
    "infer_micro_naotf_diglyme_0_1M_298K.sh"
    "infer_micro_naotf_pc_0_1M_298K.sh"
    "infer_micro_naotf_tgdme_0_1M_298K.sh"
    "infer_micro_napf6_diglyme_0_1M_298K.sh"
    "infer_micro_napf6_dme_0_1M_298K.sh"
    "infer_micro_napf6_pc_0_1M_298K.sh"
    "infer_micro_lipf6_0_5M_273_2K.sh"
    "infer_micro_napf6_dme_0_5M_273_2K.sh"
    "infer_micro_lipf6_0_5M_298_2K.sh"
    "infer_micro_napf6_dme_0_5M_298_2K.sh"
    "infer_micro_lipf6_0_5M_323_2K.sh"
    "infer_micro_napf6_dme_0_5M_323_2K.sh"
)

echo "Submitting ${#SCRIPTS[@]} jobs..."
for script in "${SCRIPTS[@]}"; do
    path="$SCRIPT_DIR/$script"
    if [ ! -f "$path" ]; then
        echo "WARNING: $script not found, skipping."
        continue
    fi
    job_id=$(sbatch "$path" | awk '{print $NF}')
    echo "Submitted $script -> job $job_id"
done
echo "Done."
