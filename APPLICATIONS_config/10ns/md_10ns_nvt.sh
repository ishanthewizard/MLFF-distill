#!/bin/bash

# === SLURM Job Parameters ===
#SBATCH --account=m5024_g              # Billing account for compute time
#SBATCH --constraint=gpu               # Request GPU nodes only
#SBATCH --cpus-per-task=32             # 32 CPU cores per task (for each GPU)
#SBATCH --gpus-per-node=4              # Request 4 GPUs per node
#SBATCH --job-name=g6_nvt_10ns         # Job name
#SBATCH --mem=200GB                    # Total memory per node
#SBATCH --nodes=1                      # Use only 1 compute node
#SBATCH --ntasks-per-node=4            # 4 tasks per node (1 per GPU)
#SBATCH --qos=premium                  # Quality of service
#SBATCH --time=24:00:00                # Maximum runtime
#SBATCH --open-mode=append             # Append to output files (important for requeued jobs)
#SBATCH --output=/global/homes/y/yuejian/project/MLFF-distill/yuejian/log/md_flex_%x_%j_%Y%m%d_%H%M%S.out  # Standard output file
#SBATCH --error=/global/homes/y/yuejian/project/MLFF-distill/yuejian/log/md_flex_%x_%j_%Y%m%d_%H%M%S.err   # Standard error file

# === Environment Setup and Logging ===
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Requeue Count: ${SLURM_RESTART_COUNT:-0}"
echo "=========================================="

# === Change to Working Directory ===
cd /global/homes/y/yuejian/project/MLFF-distill

# === Configuration - EDIT THESE PATHS FOR YOUR SIMULATIONS ===
# Model checkpoint paths (space-separated list)
MODEL_CHECKPOINTS=(
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablation_ckpt/Ishan_all_salt_wo_hessian/final/inference_ckpt.pt"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablation_ckpt/Ishan_all_salt_wo_hessian/final/inference_ckpt.pt"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablation_ckpt/Ishan_all_salt_wo_hessian/final/inference_ckpt.pt"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablation_ckpt/Ishan_all_salt_wo_hessian/final/inference_ckpt.pt"
)

# Trajectory directories (space-separated list, same length as models)
TRAJECTORY_DIRS=(
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/nvt_unstable/20ns_solvent_0_1M/md_omol_napf6_dme_re1"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/nvt_unstable/20ns_solvent_solute_0.5M/298_2K/md_omol_napf6_dme_re1"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/nvt_unstable/20ns_solvent_solute_0.5M/323_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/nvt_unstable/20ns_solvent_solute_0.5M/323_2K/md_omol_napf6_dme_re1"
)

# Temperature arrays (space-separated list, same length as models and trajectories)
TEMPERATURES=(
    298.0
    298.2
    323.2
    323.2
)

# Initial temperature arrays (space-separated list, same length as models and trajectories)
INITIAL_TEMPERATURES=(
    298.0
    298.2
    323.2
    323.2
)

# Simulation parameters
TARGET_STEPS=10000000    # Total MD steps (10 ns at 1 fs timestep)
INTERVAL=100              # Output interval for trajectory and status, save every 100 steps

# === Validation ===
# Check that trajectory and model lists have the same length
if [ ${#TRAJECTORY_DIRS[@]} -ne ${#MODEL_CHECKPOINTS[@]} ]; then
    echo "ERROR: Number of trajectories (${#TRAJECTORY_DIRS[@]}) must equal number of models (${#MODEL_CHECKPOINTS[@]})"
    exit 1
fi

# Check that temperature arrays have the same length
if [ ${#TEMPERATURES[@]} -ne ${#TRAJECTORY_DIRS[@]} ]; then
    echo "ERROR: Number of temperatures (${#TEMPERATURES[@]}) must equal number of trajectories (${#TRAJECTORY_DIRS[@]})"
    exit 1
fi

if [ ${#INITIAL_TEMPERATURES[@]} -ne ${#TRAJECTORY_DIRS[@]} ]; then
    echo "ERROR: Number of initial temperatures (${#INITIAL_TEMPERATURES[@]}) must equal number of trajectories (${#TRAJECTORY_DIRS[@]})"
    exit 1
fi

# Check that both lists don't exceed 4 items
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

# Build command with all arguments
CMD="python APPLICATIONS/electrolytes/solv_uma_nvt_flex_ablation.py"
CMD="$CMD ${TRAJECTORY_DIRS[@]}"
CMD="$CMD --models ${MODEL_CHECKPOINTS[@]}"
CMD="$CMD --steps $TARGET_STEPS"
CMD="$CMD --interval $INTERVAL"
CMD="$CMD --temperature ${TEMPERATURES[@]}"
CMD="$CMD --initial_temperature ${INITIAL_TEMPERATURES[@]}"

echo "Running command: $CMD"

# Run Python script in background (&) so we can capture its process ID
$CMD &
python_pid=$!                           # Store process ID of background Python job

echo "Python process started with PID: $python_pid"

# === Wait for Completion ===
wait $python_pid                        # Wait for Python process to finish
exit_code=$?                            # Capture exit code of Python process

# === Final Status Report ===
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "MD simulation completed successfully!"
else
    echo "MD simulation exited with code: $exit_code"
fi
echo "End Time: $(date)"
echo "=========================================="

