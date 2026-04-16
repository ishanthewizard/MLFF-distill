#!/bin/bash

# === SLURM Job Parameters ===
#SBATCH --account=m5024_g              # Billing account for compute time
#SBATCH --constraint=gpu               # Request GPU nodes only
#SBATCH --cpus-per-task=5              # 5 CPU cores per task (for each GPU)
#SBATCH --gpus-per-node=4              # Request 4 GPUs per node
#SBATCH --job-name=fl2    # Name of the job (shows in queue)
#SBATCH --mem=128GB                     # Total memory per node (80GB)
#SBATCH --nodes=1                      # Use only 1 compute node
#SBATCH --ntasks-per-node=4            # 4 tasks per node (1 per GPU)
#SBATCH --qos=premium                  # Quality of service
#SBATCH --time=24:00:00                 # Maximum runtime: 2 hours
#SBATCH --signal=USR1@30               # Send USR1 signal 30 seconds before timeout
#SBATCH --requeue                      # Automatically requeue job when preempted
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
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_ckpt/small_batch/202510-1714-3336-8b81-naotf-80/cleaned_inference_ckpt.pt"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_ckpt/small_batch/202510-1714-3356-164a-naotf-500/cleaned_inference_ckpt.pt"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_ckpt/small_batch/202510-1614-0433-0ca3-naotf-10/cleaned_inference_ckpt.pt"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_ckpt/small_batch/202510-1614-0433-0ca3-naotf-10/cleaned_inference_ckpt.pt"
)

# Trajectory directories (space-separated list, same length as models)
TRAJECTORY_DIRS=(
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablate_distillation/ablation_section_3/ablate_cols/md_omol_naotf_pc_1m_s1p1_80"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablate_distillation/ablation_section_3/ablate_cols/md_omol_naotf_pc_1m_s1p1_500"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablate_distillation/ablation_section_3/ablation_md_simulation_1ns_wwo_hessian_pert_10/md_omol_naotf_diglyme_1m_s1p1"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablate_distillation/ablation_section_3/ablation_md_simulation_1ns_wwo_hessian_pert_10/md_omol_naotf_dme_s1p1_omol"
)

# Simulation parameters
TARGET_STEPS=1000000    # Total MD steps (1 ns at 1 fs timestep)
INTERVAL=10            # Output interval for trajectory and status
TEMPERATURE=323        # Simulation temperature (K)
INITIAL_TEMPERATURE=300 # Initial temperature (K)

# === Validation ===
# Check that trajectory and model lists have the same length
if [ ${#TRAJECTORY_DIRS[@]} -ne ${#MODEL_CHECKPOINTS[@]} ]; then
    echo "ERROR: Number of trajectories (${#TRAJECTORY_DIRS[@]}) must equal number of models (${#MODEL_CHECKPOINTS[@]})"
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
echo "Temperature: $TEMPERATURE K"
echo "Initial temperature: $INITIAL_TEMPERATURE K"

# Build command with all arguments
CMD="python APPLICATIONS/electrolytes/solv_uma_npt_flex_ablation.py"
CMD="$CMD --models ${MODEL_CHECKPOINTS[@]}"
CMD="$CMD --steps $TARGET_STEPS"
CMD="$CMD --interval $INTERVAL"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --initial_temperature $INITIAL_TEMPERATURE"
CMD="$CMD ${TRAJECTORY_DIRS[@]}"

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
