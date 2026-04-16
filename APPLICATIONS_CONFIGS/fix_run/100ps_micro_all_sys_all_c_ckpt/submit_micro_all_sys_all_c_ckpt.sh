#!/bin/bash
# Submit all micro_100ps_all_sys_all_c_student fix jobs (no SLURM dependencies).
# Usage: bash submit_micro_all_sys_all_c_ckpt.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNS=(
    "${SCRIPT_DIR}/run_01_s1m_napf6_dgly_pf.sh"
    "${SCRIPT_DIR}/run_02_s1m_napf6_pc_mask.sh"
    "${SCRIPT_DIR}/run_03_s1m_naotf_dgly.sh"
    "${SCRIPT_DIR}/run_04_s1m_naotf_dme.sh"
    "${SCRIPT_DIR}/run_05_s1m_napf6_dme.sh"
    "${SCRIPT_DIR}/run_06_01m_naotf_dgly_1m.sh"
    "${SCRIPT_DIR}/run_07_01m_naotf_dme_omol.sh"
    "${SCRIPT_DIR}/run_08_01m_naotf_pc_1m.sh"
    "${SCRIPT_DIR}/run_09_01m_naotf_tgdme.sh"
    "${SCRIPT_DIR}/run_10_01m_napf6_dgly_pf.sh"
    "${SCRIPT_DIR}/run_11_01m_napf6_dme_re1.sh"
    "${SCRIPT_DIR}/run_12_01m_napf6_pc_mask.sh"
    "${SCRIPT_DIR}/run_13_05m_273_lipf6.sh"
    "${SCRIPT_DIR}/run_14_05m_273_napf6_dme.sh"
    "${SCRIPT_DIR}/run_15_05m_298_lipf6.sh"
    "${SCRIPT_DIR}/run_16_05m_298_napf6_dme.sh"
    "${SCRIPT_DIR}/run_17_05m_323_lipf6.sh"
    "${SCRIPT_DIR}/run_18_05m_323_napf6_dme.sh"
)

for f in "${RUNS[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing $f"
        exit 1
    fi
done

echo "Submitting ${#RUNS[@]} jobs (parallel queue)..."
JIDS=()
for i in "${!RUNS[@]}"; do
    r="${RUNS[$i]}"
    jid="$(sbatch --parsable "$r")"
    JIDS+=("$jid")
    echo "[$((i + 1))/${#RUNS[@]}] $(basename "$r") -> job $jid"
done

echo "Done. Job IDs: ${JIDS[*]}"
echo "Monitor: squeue -u ${USER}"
