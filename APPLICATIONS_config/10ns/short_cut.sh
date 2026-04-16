#!/bin/bash

set -euo pipefail

# 0. Activate micromamba environment
if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook --shell=bash)"
    micromamba activate fairchemV2
    echo "fairchemV2 activated"
else
    echo "micromamba not found in PATH; skipping activation" >&2
fi

# 1. Cancel all jobs belonging to your user
echo "Cancelling all jobs for $USER..."
scancel -u "$USER"
echo "Waiting 120s to ensure cancellations propagate..."
sleep 120

# 2. Submit new jobs sequentially
echo "Submitting g6..."
sbatch /global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS_config/10ns/md_10ns_nvt.sh

echo "Submitting g7..."
sbatch /global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS_config/10ns/md_10ns_nvt_naotf.sh


echo "Done."
