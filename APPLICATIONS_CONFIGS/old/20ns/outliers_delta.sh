#!/bin/bash

# === SLURM Job Parameters (Delta) ===
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00
#SBATCH --job-name=mlff_outliers_20ns
#SBATCH --account=bfoy-dtai-gh
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=verbose,closest

#SBATCH --output=/u/yjian1/project/MLFF-distill/yjian1/log/md_flex_%x_%j_%Y%m%d_%H%M%S.out
#SBATCH --error=/u/yjian1/project/MLFF-distill/yjian1/log/md_flex_%x_%j_%Y%m%d_%H%M%S.err

#SBATCH --mail-user=yuejian@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# === Interactive Debug (Delta) ===
# If you want to debug in an interactive allocation (no sbatch), copy/paste something like:
#
# srun --partition=ghx4 --account=bfoy-dtai-gh \
#      --nodes=1 --ntasks=1 --cpus-per-task=32 \
#      --gpus-per-node=4 --gpu-bind=verbose,closest \
#      --time=01:00:00 --mem=256g \
#      --pty bash -lc 'cd /u/yjian1/project/MLFF-distill/yjian1/MLFF-distill && \
#      python -u APPLICATIONS/electrolytes/solv_uma_npt_flex_ablation.py \
#        "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/ablation_diffusivity/outliers/20ns_solvent_0.1M/md_omol_naotf_dme_s1p1_omol" \
#        "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/ablation_diffusivity/outliers/20ns_solvent_solute_0.5M/298_2K/md_omol_napf6_dme_re1" \
#        "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/ablation_diffusivity/outliers/20ns_solvent_solute_0.5M/323_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t" \
#        "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/ablation_diffusivity/outliers/20ns_solvent_solute_0.5M/323_2K/md_omol_napf6_dme_re1" \
#        --models \
#        "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/models/small_batch/202510-1614-0349-0f8b-naotf-wo/checkpoints/final/inference_ckpt.pt" \
#        "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/models/small_batch/202510-2322-2657-3d7e-napf6-wo/checkpoints/final/inference_ckpt.pt" \
#        "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/models/small_batch/202510-1400-4222-c26e-lipf6-wo/checkpoints/final/inference_ckpt.pt" \
#        "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/models/small_batch/202510-2322-2657-3d7e-napf6-wo/checkpoints/final/inference_ckpt.pt" \
#        --steps 20000000 --interval 10 \
#        --temperature 298.2 298.2 323.2 323.2 \
#        --initial_temperature 298.2 298.2 323.2 323.2'
#
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
cd /u/yjian1/project/MLFF-distill/yjian1/MLFF-distill

# === Configuration - EDIT THESE PATHS FOR YOUR SIMULATIONS ===
# Model checkpoint paths (space-separated list)
MODEL_CHECKPOINTS=(
    "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/models/small_batch/202510-1614-0349-0f8b-naotf-wo/checkpoints/final/inference_ckpt.pt"
    "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/models/small_batch/202510-2322-2657-3d7e-napf6-wo/checkpoints/final/inference_ckpt.pt"
    "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/models/small_batch/202510-1400-4222-c26e-lipf6-wo/checkpoints/final/inference_ckpt.pt"
    "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/models/small_batch/202510-2322-2657-3d7e-napf6-wo/checkpoints/final/inference_ckpt.pt"
)

# Trajectory directories (space-separated list, same length as models)
TRAJECTORY_DIRS=(
    "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/ablation_diffusivity/outliers/20ns_solvent_0.1M/md_omol_naotf_dme_s1p1_omol"
    "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/ablation_diffusivity/outliers/20ns_solvent_solute_0.5M/298_2K/md_omol_napf6_dme_re1"
    "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/ablation_diffusivity/outliers/20ns_solvent_solute_0.5M/323_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t"
    "/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/ablation_diffusivity/outliers/20ns_solvent_solute_0.5M/323_2K/md_omol_napf6_dme_re1"
)

# Temperature arrays (same length as models and trajectories)
TEMPERATURES=(298.2 298.2 323.2 323.2)

# Initial temperature arrays (same length as models and trajectories)
INITIAL_TEMPERATURES=(298.2 298.2 323.2 323.2)

# Simulation parameters
TARGET_STEPS=20000000
INTERVAL=10

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

# IMPORTANT (Delta): this script uses multi-GPU internally via torch.multiprocessing,
# so we run a SINGLE python process and just request multiple GPUs from Slurm.
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


