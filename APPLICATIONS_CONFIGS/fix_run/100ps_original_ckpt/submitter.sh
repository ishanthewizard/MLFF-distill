#!/bin/bash
# Submit all SLURM jobs in this directory (run_*.sh only; not this file).
# Checkpoint (all run_*.sh here): .../outlier_ablation_Feb_2026/ckpt/202602-1315-1942-821a-first100ps_all_salts_293K/final/inference_ckpt.pt
# Trajectories: .../fix_run/origin_100ps_student/... (distinct from 100ps_micro_ckpt; separate .traj/.log).
# Usage: bash submitter.sh   or   ./submitter.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
RUNS=( "$SCRIPT_DIR"/run_*.sh )

if ((${#RUNS[@]} == 0)); then
    echo "ERROR: no run_*.sh scripts in $SCRIPT_DIR"
    exit 1
fi

readarray -t RUNS < <(printf '%s\n' "${RUNS[@]}" | sort)

echo "Submitting ${#RUNS[@]} jobs (no SLURM dependencies; all enter the queue now)..."
JIDS=()
for i in "${!RUNS[@]}"; do
    r="${RUNS[$i]}"
    jid="$(sbatch --parsable "$r")"
    JIDS+=("$jid")
    echo "[$((i + 1))/${#RUNS[@]}] $(basename "$r") -> job $jid"
done

echo "Done. Job IDs: ${JIDS[*]}"
echo "Monitor: squeue -u \"\$USER\""
