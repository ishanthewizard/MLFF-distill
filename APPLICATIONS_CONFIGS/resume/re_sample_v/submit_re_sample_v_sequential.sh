#!/bin/bash
# Submit all resume jobs at once (no SLURM dependencies; scheduler may run them in parallel if resources allow).
# Usage: bash submit_re_sample_v_sequential.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNS=(
    "${SCRIPT_DIR}/run_01_naotf_diglyme_1m_s1p1.sh"
    "${SCRIPT_DIR}/run_02_naotf_tgdme_1m_s1p1.sh"
    "${SCRIPT_DIR}/run_03_napf6_diglyme_pfactor_0.1_1fs.sh"
    "${SCRIPT_DIR}/run_04_napf6_dme_re1_0p1M.sh"
    "${SCRIPT_DIR}/run_05_napf6_dme_re1_273_2K.sh"
)

for f in "${RUNS[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing $f"
        exit 1
    fi
done

echo "Submitting ${#RUNS[@]} jobs (no afterok chain; all enter the queue now)..."
JIDS=()
for i in "${!RUNS[@]}"; do
    r="${RUNS[$i]}"
    jid="$(sbatch --parsable "$r")"
    JIDS+=("$jid")
    echo "[$((i + 1))/${#RUNS[@]}] $(basename "$r") -> job $jid"
done

echo "Done. Job IDs: ${JIDS[*]}"
echo "Monitor: squeue -u \"\$USER\""
