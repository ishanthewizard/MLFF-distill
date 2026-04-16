#!/bin/bash

# === SLURM Job Parameters ===
#SBATCH --account=m4558_g              # Billing account for compute time
#SBATCH --constraint=gpu               # Request GPU nodes only
#SBATCH --cpus-per-task=32              # 5 CPU cores per task (for each GPU)
#SBATCH --gpus-per-node=4              # Request 4 GPUs per node
#SBATCH --job-name=nvt_outliers    # Name of the job (shows in queue)
#SBATCH --mem=200GB                     # Total memory per node (80GB)
#SBATCH --nodes=1                      # Use only 1 compute node
#SBATCH --ntasks-per-node=4            # 4 tasks per node (1 per GPU)
#SBATCH --qos=regular                  # Quality of service
#SBATCH --time=15:00:00                 # Maximum runtime: 2 hours
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
    "/global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/models/uma-s-1p1.pt"
    "/global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/models/uma-s-1p1.pt"
    "/global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/models/uma-s-1p1.pt"
    "/global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/models/uma-s-1p1.pt"
)

# Trajectory directories (space-separated list, same length as models)
TRAJECTORY_DIRS=(
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/speed_test/UMA/lipf6_dme"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/speed_test/UMA/naotf_dme"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/speed_test/UMA/napf6_dme"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/speed_test/UMA/napf6_pc"
)

# Temperature arrays (space-separated list, same length as models and trajectories)
TEMPERATURES=(
    298.0
    298.0
    298.0
    298.0
)

# Initial temperature arrays (space-separated list, same length as models and trajectories)
INITIAL_TEMPERATURES=(
    298.0
    298.0
    298.0
    298.0
)

# Simulation parameters
TARGET_STEPS=10000000    # Total MD steps (10 ns at 1 fs timestep)
INTERVAL=100            # Output interval for trajectory and status



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
CMD="python APPLICATIONS/electrolytes/solv_uma_npt_flex_ablation.py"
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
exit_code=$?                           # Capture exit code of Python process

# === Final Status Report ===
echo "=========================================="
if [ $exit_code -eq 0 ]; then          
    echo "MD simulation completed successfully!"
else
    echo "MD simulation exited with code: $exit_code"  
fi
echo "End Time: $(date)"                
echo "=========================================="
