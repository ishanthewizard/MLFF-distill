#!/bin/bash

# === SLURM Job Parameters (Delta) ===
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=48:00:00
#SBATCH --job-name=202602-1315-2137-4d14-first50ps_all_salts_353K__20ns_solvent_0_1M__md_omol_naotf_dme_s1p1_omol
#SBATCH --account=bfnb-dtai-gh
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

#SBATCH --output=/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/outlier_ablation_Feb_2026/log/md_temperature_ablation_%x_%j_%Y%m%d_%H%M%S.out
#SBATCH --error=/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/outlier_ablation_Feb_2026/log/md_temperature_ablation_%x_%j_%Y%m%d_%H%M%S.err

#SBATCH --mail-user=yuejian@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# === Environment Setup and Logging ===
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Job Name: ${SLURM_JOB_NAME:-}"
echo "Node: ${SLURM_NODELIST:-}"
echo "Start Time: $(date)"
echo "=========================================="

mkdir -p /u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/outlier_ablation_Feb_2026/log
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd /u/yjian1/project/MLFF-distill

# === Configuration ===
MODEL_CHECKPOINT="/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/outlier_ablation_Feb_2026/ckpt/202602-1315-2137-4d14-first50ps_all_salts_353K/final/inference_ckpt.pt"
TRAJECTORY_DIRS=(
    "/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/outlier_ablation_Feb_2026/md_simulation/first_50ps_all_salts_teacher_data_353K/20ns_solvent_0_1M/md_omol_naotf_dme_s1p1_omol"
)

TEMPERATURE=298.0
INITIAL_TEMPERATURE=298.0
TARGET_STEPS=20000000
INTERVAL=100

echo "Trajectory directories (hard-coded):"
printf '  - %s\n' "${TRAJECTORY_DIRS[@]}"

# === Run simulations (one trajectory per run; required for 1 GPU) ===
for traj_dir in "${TRAJECTORY_DIRS[@]}"; do
    echo "------------------------------------------"
    echo "Trajectory: ${traj_dir}"
    echo "Checkpoint: ${MODEL_CHECKPOINT}"
    echo "Temperature: ${TEMPERATURE} K"
    echo "Initial temperature: ${INITIAL_TEMPERATURE} K"
    echo "Target steps: ${TARGET_STEPS}"
    echo "Interval: ${INTERVAL}"

    CMD=(python -u APPLICATIONS/electrolytes/solv_uma_npt_flex_ablation.py)
    CMD+=("${traj_dir}")
    CMD+=(--models "${MODEL_CHECKPOINT}")
    CMD+=(--steps "${TARGET_STEPS}")
    CMD+=(--interval "${INTERVAL}")
    CMD+=(--temperature "${TEMPERATURE}")
    CMD+=(--initial_temperature "${INITIAL_TEMPERATURE}")

    echo "Running command: ${CMD[*]}"
    "${CMD[@]}"
done

echo "=========================================="
echo "All trajectories completed successfully."
echo "End Time: $(date)"
echo "=========================================="
