#!/bin/bash

# === SLURM Job Parameters (Delta) ===
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=48:00:00
#SBATCH --job-name=fix_masac_r09
#SBATCH --account=bfnb-dtai-gh
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

#SBATCH --output=/u/yjian1/project/MLFF-distill/yjian1/log/md_temperature_ablation_%x_%j_%Y%m%d_%H%M%S.out
#SBATCH --error=/u/yjian1/project/MLFF-distill/yjian1/log/md_temperature_ablation_%x_%j_%Y%m%d_%H%M%S.err

#SBATCH --mail-user=yuejian@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Job Name: ${SLURM_JOB_NAME:-}"
echo "Node: ${SLURM_NODELIST:-}"
echo "Start Time: $(date)"
echo "=========================================="

mkdir -p /u/yjian1/project/MLFF-distill/yjian1/log
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd /u/yjian1/project/MLFF-distill

MODEL_CHECKPOINTS=(
    "/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/Mar_19/ckpt/202604-0917-3915-a13d-micro-50ps-all-systems-all-concentration/inference_ckpt.pt"
)
TRAJECTORY_DIRS=(
    "/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/fix_run/micro_100ps_all_sys_all_c_student/20ns_solvent_0_1M/298K/md_omol_naotf_tgdme_1m_s1p1"
)
TEMPERATURES=(298)
INITIAL_TEMPERATURES=(298)
TARGET_STEPS=20000000
INTERVAL=100
dt=1

if [ ${#TRAJECTORY_DIRS[@]} -ne ${#MODEL_CHECKPOINTS[@]} ]; then
    echo "ERROR: trajectories/models count mismatch"
    exit 1
fi
if [ ${#TEMPERATURES[@]} -ne ${#TRAJECTORY_DIRS[@]} ]; then
    echo "ERROR: temperatures count mismatch"
    exit 1
fi
if [ ${#INITIAL_TEMPERATURES[@]} -ne ${#TRAJECTORY_DIRS[@]} ]; then
    echo "ERROR: initial temperatures count mismatch"
    exit 1
fi

echo "Starting MD (single trajectory)..."
echo "Trajectories: ${TRAJECTORY_DIRS[@]}"
echo "Target steps: $TARGET_STEPS | Interval: $INTERVAL | dt: ${dt} fs"

CMD=(python -u APPLICATIONS/electrolytes/solv_uma_npt_flex_ablation_resume.py)
CMD+=("${TRAJECTORY_DIRS[@]}")
CMD+=(--models "${MODEL_CHECKPOINTS[@]}")
CMD+=(--steps "$TARGET_STEPS")
CMD+=(--interval "$INTERVAL")
CMD+=(--timestep "$dt")
CMD+=(--temperature "${TEMPERATURES[@]}")
CMD+=(--initial_temperature "${INITIAL_TEMPERATURES[@]}")

echo "Running: ${CMD[*]}"
"${CMD[@]}"

exit_code=$?
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "MD simulation completed successfully!"
else
    echo "MD simulation exited with code: $exit_code"
fi
echo "End Time: $(date)"
echo "=========================================="
exit "$exit_code"
