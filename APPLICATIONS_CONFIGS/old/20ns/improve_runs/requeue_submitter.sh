#!/bin/bash

set -euo pipefail

# Run 4 cycles, every 48.1 hours:
#  1) cancel all of this user's jobs
#  2) submit every .sh in the improve_runs outlier dir
INTERVAL_SECONDS=$((48*3600 + 6*60)) # 48.1 hours = 48h + 6m
MAX_RUNS=2

OUTLIER_DIR="/u/yjian1/project/MLFF-distill/APPLICATIONS_CONFIG/20ns/improve_runs/outliers_high_T_separate"
NVT_DIR="/u/yjian1/project/MLFF-distill/APPLICATIONS_CONFIG/20ns/improve_runs/nvt_separate"
activate_env() {
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate fairchemV2
        export LD_LIBRARY_PATH="/u/yjian1/project/MLFF-distill/yjian1/env/miniconda3/envs/fairchemV2/lib:${LD_LIBRARY_PATH:-}"
        echo "fairchemV2 activated (conda)"
    else
        echo "conda not found in PATH; skipping activation" >&2
    fi
}

cancel_jobs() {
    echo "$(date -Iseconds) cancelling all jobs for ${USER}..."
    scancel -u "${USER}" || echo "warning: scancel returned non-zero" >&2
    echo "$(date -Iseconds) waiting 120s to let cancellations propagate"
    sleep 120
}

submit_dir() {
    local dir="$1"
    if [ ! -d "${dir}" ]; then
        echo "skip: directory not found: ${dir}" >&2
        return
    fi

    # Sort for deterministic submission order.
    mapfile -t scripts < <(find "${dir}" -maxdepth 1 -type f -name '*.sh' | sort)
    if [ ${#scripts[@]} -eq 0 ]; then
        echo "no scripts to submit in ${dir}"
        return
    fi

    for script in "${scripts[@]}"; do
        echo "$(date -Iseconds) submitting ${script}"
        sbatch "${script}"
    done
}

activate_env
cd /u/yjian1/project/MLFF-distill
echo "$(date -Iseconds) scheduler started; ${MAX_RUNS} cycles, interval ${INTERVAL_SECONDS}s"

for ((run=1; run<=MAX_RUNS; run++)); do
    echo "========== cycle ${run}/${MAX_RUNS} =========="
    cancel_jobs
    # submit_dir "${NVT_DIR}"
    submit_dir "${OUTLIER_DIR}"
    if [ "${run}" -lt "${MAX_RUNS}" ]; then
        echo "$(date -Iseconds) cycle ${run} complete; sleeping ${INTERVAL_SECONDS}s"
        sleep "${INTERVAL_SECONDS}"
    else
        echo "$(date -Iseconds) cycle ${run} complete; max cycles reached, exiting"
    fi
done

