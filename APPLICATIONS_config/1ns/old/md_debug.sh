#!/bin/bash
# SLURM script for requeuing MD simulations (handles both preemption and timeout)
#
# =============================================================================
# USAGE INSTRUCTIONS:
# =============================================================================
# 
# 1. CONFIGURE SIMULATION PARAMETERS (lines 67-78):
#    - MODEL_CHECKPOINT: Path to your UMA model file
#    - TRAJECTORY_DIRS: Array of trajectory directory paths
#    - TARGET_STEPS: Total MD steps to run (default: 1,000,000 = 1 ns)
#    - INTERVAL: Output frequency for trajectory and status
#
# 2. SUBMIT THE JOB:
#    sbatch APPLICATIONS/electrolytes/md.sh
#
# 3. MONITOR THE JOB:
#    squeue -u $USER                    # Check job status
#    tail -f yuejian/log/md_*.out       # Watch live output
#
# 4. CANCEL IF NEEDED:
#    scancel JOBID
#
# REQUEUE BEHAVIOR:
# - Job automatically requeues on preemption or timeout
# - Each requeue continues from the last saved checkpoint
# - Simulation stops when all trajectories reach TARGET_STEPS
#
# =============================================================================

# === SLURM Job Parameters ===
#SBATCH --account=m5024_g              # Billing account for compute time
#SBATCH --constraint=gpu               # Request GPU nodes only
#SBATCH --cpus-per-task=20             # 20 CPU cores for the single task
#SBATCH --gpus-per-node=4              # Request 4 GPUs per node
#SBATCH --job-name=md_debug          # Name of the job (shows in queue)
#SBATCH --mem=40GB                     # Total memory per node (80GB)
#SBATCH --nodes=1                      # Use only 1 compute node
#SBATCH --ntasks-per-node=1            # 1 task per node (the Python script will handle parallelism)
#SBATCH --qos=debug                  # Quality of service (debug queue for testing)
#SBATCH --time=00:01:00                 # Maximum runtime: 5 minutes for testing
#SBATCH --signal=USR1@30               # Send USR1 signal 30 seconds before timeout
#SBATCH --requeue                      # Automatically requeue job when preempted (NOT for timeout)
#SBATCH --open-mode=append             # Append to output files (important for requeued jobs)
#SBATCH --output=/global/homes/y/yuejian/project/MLFF-distill/yuejian/log/md_%x_%j_%Y%m%d_%H%M%S.out  # Standard output file
#SBATCH --error=/global/homes/y/yuejian/project/MLFF-distill/yuejian/log/md_%x_%j_%Y%m%d_%H%M%S.err   # Standard error file

# The preempt_handler is no longer needed in the batch script,
# because srun will handle propagating the SIGTERM signal directly to the Python process.
# SLURM will automatically requeue the job on preemption because of the #SBATCH --requeue flag.

timeout_handler() {
    # This function handles timeout (USR1 from SLURM, 30 seconds before time limit)
    echo "Received USR1 signal - job will reach time limit in 30 seconds"
    echo "Initiating graceful shutdown for timeout..."
    
    # Forward SIGTERM signal to the srun process, which will forward it to Python
    kill -TERM ${1}
    echo "Sent SIGTERM to srun process (PID: ${1})"
    
    # Wait for graceful shutdown
    wait ${1}
    echo "Python process terminated - manually requeuing job"
    
    # Manually requeue because SLURM will NOT do it automatically for timeout
    scontrol requeue ${SLURM_JOB_ID}
    echo "Job requeued successfully"
}

# === Environment Setup and Logging ===
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"              
echo "Job Name: $SLURM_JOB_NAME"          
echo "Node: $SLURM_NODELIST"              
echo "Start Time: $(date)"                
echo "Requeue Count: ${SLURM_RESTART_COUNT:-0}"  
echo "=========================================="

# Manually set SLURM_JOB_ID for local testing
export SLURM_JOB_ID="$$"

# Print the PID of this script so we can send it signals
echo "Batch script running with PID: $$"
echo "=========================================="

# === Change to Working Directory ===
cd /global/homes/y/yuejian/project/MLFF-distill

# === Configuration - EDIT THESE PATHS FOR YOUR SIMULATIONS ===
# Model checkpoint path
MODEL_CHECKPOINT="/global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/models/uma-s-1p1.pt"

# Trajectory directories (space-separated list)
TRAJECTORY_DIRS=(
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_debug/md_omol_cspf6_pfactor_0.1_1fs_mask_t"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_debug/md_omol_lipf6_pfactor_0.1_1fs_mask_t"
)

# Simulation parameters
TARGET_STEPS=1000000    # Total MD steps (1 ns at 1 fs timestep)
INTERVAL=10            # Output interval for trajectory and status

# === Launch MD Simulation ===
echo "Starting MD simulation..."
echo "Model: $MODEL_CHECKPOINT"
echo "Trajectories: ${TRAJECTORY_DIRS[@]}"
echo "Target steps: $TARGET_STEPS"
echo "Interval: $INTERVAL"

# Build command with all arguments
CMD="python APPLICATIONS/electrolytes/solv_uma_npt_pr_requeue_2.py"
CMD="$CMD --model $MODEL_CHECKPOINT"
CMD="$CMD --steps $TARGET_STEPS"
CMD="$CMD --interval $INTERVAL"
CMD="$CMD ${TRAJECTORY_DIRS[@]}"

# --- Temporarily use the fake simulation for local debugging ---
# The real command is commented out
# echo "Running command: $CMD"
CMD="python APPLICATIONS/electrolytes/fake_simulation.py"
echo "Running FAKE command for debugging: $CMD"
# ----------------------------------------------------------------

# For local testing, we run Python directly. In the real job, srun is used.
echo "NOTE: Bypassing srun for local test."
$CMD &
python_pid=$!                           # Store process ID of the python job

echo "Python process started with PID: $python_pid"

# === Set Up Signal Handlers ===
# The trap will pass the python_pid to the handler for the local test.
trap "timeout_handler '$python_pid'" USR1       # Catches timeout USR1 from SLURM

# === Wait for Completion ===
wait $python_pid                        # Wait for python process to finish
exit_code=$?                           # Capture exit code of the python process

# Clean up the fake simulation file after the test
echo "Cleaning up fake simulation script."
# rm APPLICATIONS/electrolytes/fake_simulation.py

# === Final Status Report ===
echo "=========================================="
if [ $exit_code -eq 0 ]; then          
    echo "MD simulation completed successfully!"
else
    echo "MD simulation exited with code: $exit_code"  
fi
echo "End Time: $(date)"                
echo "=========================================="

# === Keep Process Alive for SLURM Signal Handling ===
# This sleep ensures there's still a process running so SLURM can correctly
# record the job state as PREEMPTED.
sleep 120