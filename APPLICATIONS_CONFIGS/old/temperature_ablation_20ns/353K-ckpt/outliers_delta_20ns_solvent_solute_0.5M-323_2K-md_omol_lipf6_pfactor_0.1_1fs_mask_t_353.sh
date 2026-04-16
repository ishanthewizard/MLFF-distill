#!/bin/bash

# === SLURM Job Parameters (Delta) ===
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=48:00:00
#SBATCH --job-name=temp_ablation_20ns_solvent_solute_0.5M-323_2K-md_omol_lipf6_pfactor_0.1_1fs_mask_t
#SBATCH --account=bfnb-dtai-gh
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

#SBATCH --output=/u/yjian1/project/MLFF-distill/yjian1/log/md_temperature_ablation_%x_%j_%Y%m%d_%H%M%S.out
#SBATCH --error=/u/yjian1/project/MLFF-distill/yjian1/log/md_temperature_ablation_%x_%j_%Y%m%d_%H%M%S.err

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

# Make sure log dir exists
mkdir -p /u/yjian1/project/MLFF-distill/yjian1/log

# Avoid CPU oversubscription
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# === Change to Working Directory ===
cd /u/yjian1/project/MLFF-distill

# === Configuration ===
MODEL_CHECKPOINTS=(
    "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/models/trained_on_all_systems/202601-1919-1500-55dc-353k/checkpoints/final/inference_ckpt.pt"
)
TRAJECTORY_DIRS=(
    "/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solvent_solute_0.5M/323_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t"
)
TEMPERATURES=(323.2)
INITIAL_TEMPERATURES=(323.2)
TARGET_STEPS=20000000
INTERVAL=100

# === Validation ===
if [ ${#TRAJECTORY_DIRS[@]} -ne ${#MODEL_CHECKPOINTS[@]} ]; then
    echo "ERROR: Number of trajectories (${#TRAJECTORY_DIRS[@]}) must equal number of models (${#MODEL_CHECKPOINTS[@]})"
    exit 1
fi
if [ ${#TEMPERATURES[@]} -ne ${#TRAJECTORY_DIRS[@]} ]; then
    echo "ERROR: Number of temperatures (${#TEMPERATURES[@]}) must equal number of trajectories (${#TRAJECTORY_DIRS[@]})"
    exit 1
fi
if [ ${#INITIAL_TEMPERATURES[@]} -ne ${#TRAJECTORY_DIRS[@]} ]; then
    echo "ERROR: Number of initial temperatures (${#INITIAL_TEMPERATURES[@]}) must equal number of trajectories (${#TRAJECTORY_DIRS[@]})"
    exit 1
fi
if [ ${#TRAJECTORY_DIRS[@]} -gt 4 ]; then
    echo "ERROR: Number of trajectory-model pairs (${#TRAJECTORY_DIRS[@]}) cannot exceed 4"
    exit 1
fi

# === Launch MD Simulation ===
echo "Starting MD simulation with trajectory-model pairs..."
echo "Number of pairs: ${#TRAJECTORY_DIRS[@]}"
echo "Models: ${MODEL_CHECKPOINTS[@]}"
echo "Trajectories: ${TRAJECTORY_DIRS[@]}"
echo "Target steps: $TARGET_STEPS"
echo "Interval: $INTERVAL"
echo "Temperatures: ${TEMPERATURES[@]} K"
echo "Initial temperatures: ${INITIAL_TEMPERATURES[@]} K"

CMD=(python -u APPLICATIONS/electrolytes/solv_uma_npt_flex_ablation.py)
CMD+=("${TRAJECTORY_DIRS[@]}")
CMD+=(--models "${MODEL_CHECKPOINTS[@]}")
CMD+=(--steps "$TARGET_STEPS")
CMD+=(--interval "$INTERVAL")
CMD+=(--temperature "${TEMPERATURES[@]}")
CMD+=(--initial_temperature "${INITIAL_TEMPERATURES[@]}")

echo "Running command: ${CMD[*]}"
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
