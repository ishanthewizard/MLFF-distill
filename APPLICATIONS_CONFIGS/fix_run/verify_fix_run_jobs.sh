#!/bin/bash
# Sanity-check 100ps_micro_ckpt vs 100ps_original_ckpt: ckpt separation, job counts,
# unique trajectory dirs per suite, and paired run script names.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICRO="${ROOT}/100ps_micro_ckpt"
ORIG="${ROOT}/100ps_original_ckpt"

CKPT_MICRO="/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/Mar_19/ckpt/202603-1401-0647-f49b-micro/final/inference_ckpt.pt"
CKPT_ORIG="/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/outlier_ablation_Feb_2026/ckpt/202602-1315-1942-821a-first100ps_all_salts_293K/final/inference_ckpt.pt"

fail() { echo "FAIL: $*" >&2; exit 1; }

check_dir() {
    local label=$1 dir=$2 ckpt=$3 student_substr=$4 forbidden=$5
    local n runs
    runs=( "$dir"/run_*.sh )
    n=${#runs[@]}
    (( n > 0 )) || fail "$label: no run_*.sh"
    local uniq_traj
    uniq_traj=$(grep -F '"/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/fix_run/' "$dir"/run_*.sh \
        | grep -v MODEL_CHECKPOINTS | sort -u | wc -l)
    (( uniq_traj == n )) || fail "$label: expected $n unique TRAJECTORY_DIRS, got $uniq_traj"
    local dups
    dups=$(grep -oh 'fix_run/[^"]*student/[^"]*"' "$dir"/run_*.sh | tr -d '"' | sort | uniq -d | wc -l)
    (( dups == 0 )) || fail "$label: duplicate trajectory directories in run scripts"
    for f in "$dir"/run_*.sh; do
        grep -qF "$ckpt" "$f" || fail "$label: $(basename "$f") missing expected MODEL_CHECKPOINTS"
        grep -qF "$student_substr" "$f" || fail "$label: $(basename "$f") TRAJECTORY_DIRS missing $student_substr"
        if [[ -n "$forbidden" ]]; then
            ! grep -qF "$forbidden" "$f" || fail "$label: $(basename "$f") must not reference $forbidden"
        fi
    done
}

echo "Checking $MICRO ..."
check_dir "micro_ckpt" "$MICRO" "$CKPT_MICRO" "micro_100ps_student" "origin_100ps_student"
! grep -qE 'outlier_ablation_Feb_2026|first100ps_all_salts' "$MICRO"/run_*.sh || fail "micro_ckpt: must not reference original ckpt path"

echo "Checking $ORIG ..."
check_dir "original_ckpt" "$ORIG" "$CKPT_ORIG" "origin_100ps_student" "micro_100ps_student"
! grep -qE 'Mar_19/ckpt/202603-1401-0647-f49b-micro|f49b-micro' "$ORIG"/run_*.sh || fail "original_ckpt: must not reference micro ckpt path"

nmicro=$(ls -1 "$MICRO"/run_*.sh | wc -l)
norig=$(ls -1 "$ORIG"/run_*.sh | wc -l)
(( nmicro == norig )) || fail "job count mismatch: micro=$nmicro original=$norig"

diff -q \
    <(ls -1 "$MICRO"/run_*.sh | xargs -n1 basename | sort) \
    <(ls -1 "$ORIG"/run_*.sh | xargs -n1 basename | sort) \
    || fail "run_*.sh basenames differ between micro and original dirs"

echo "OK: $nmicro jobs per dir; checkpoints and student trees are separated; traj dirs unique per suite."
echo "Note: MD writes to TRAJECTORY_DIRS/<leaf>.traj and .log — micro vs origin use different parent trees, so no cross-writes."
