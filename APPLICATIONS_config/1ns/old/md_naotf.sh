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
#SBATCH --cpus-per-task=5              # 5 CPU cores per task (for each GPU)
#SBATCH --gpus-per-node=4              # Request 4 GPUs per node
#SBATCH --job-name=md_naotf          # Name of the job (shows in queue)
#SBATCH --mem=80GB                     # Total memory per node (80GB)
#SBATCH --nodes=1                      # Use only 1 compute node
#SBATCH --ntasks-per-node=4            # 4 tasks per node (1 per GPU)
#SBATCH --qos=regular                  # Quality of service (debug queue for testing)
#SBATCH --time=24:00:00                 # Maximum runtime: 5 minutes for testing
#SBATCH --signal=USR1@30               # Send USR1 signal 30 seconds before timeout
#SBATCH --requeue                      # Automatically requeue job when preempted (NOT for timeout)
#SBATCH --open-mode=append             # Append to output files (important for requeued jobs)
#SBATCH --output=/global/homes/y/yuejian/project/MLFF-distill/yuejian/log/md_%x_%j_%Y%m%d_%H%M%S.out  # Standard output file
#SBATCH --error=/global/homes/y/yuejian/project/MLFF-distill/yuejian/log/md_%x_%j_%Y%m%d_%H%M%S.err   # Standard error file

# === Signal Handlers ===
preempt_handler() {
    # This function handles preemption (SIGTERM from SLURM)
    echo "Received SIGTERM signal - job is being preempted"
    echo "Initiating graceful shutdown for preemption..."
    
    # Forward SIGTERM signal to the user application
    kill -TERM ${1}
    echo "Sent SIGTERM to Python process (PID: ${1})"
    
    # Wait for graceful shutdown (SLURM will automatically requeue due to --requeue flag)
    wait ${1}
    echo "Python process terminated - job will be automatically requeued by SLURM"
}

timeout_handler() {
    # This function handles timeout (USR1 from SLURM, 60 seconds before time limit)
    echo "Received USR1 signal - job will reach time limit in 60 seconds"
    echo "Initiating graceful shutdown for timeout..."
    
    # Forward SIGTERM signal to the user application
    kill -TERM ${1}
    echo "Sent SIGTERM to Python process (PID: ${1})"
    
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

# === Change to Working Directory ===
cd /global/homes/y/yuejian/project/MLFF-distill

# === Configuration - EDIT THESE PATHS FOR YOUR SIMULATIONS ===
# Model checkpoint path
MODEL_CHECKPOINT="/global/homes/y/yuejian/project/MLFF-distill/logs/202509-1821-2449-6971/checkpoints/step_215000/inference_ckpt.pt"

# Trajectory directories (space-separated list)
TRAJECTORY_DIRS=(
    # # group 1
    # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_rest/md_omol_cspf6_pfactor_0.1_1fs_mask_t",
    # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_rest/md_omol_lipf6_pfactor_0.1_1fs_mask_t",
    # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_rest/md_omol_naotf_diglyme_1m_s1p1",
    # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_rest/md_omol_naotf_dme_s1p1_omol",
    # # group 2
    # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_rest/md_omol_naotf_pc_1m_s1p1",
    # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_rest/md_omol_naotf_tgdme_1m_s1p1",
    # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_rest/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t",
    # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_rest/md_omol_napf6_dimethylcarbonate_pfactor_0.1_1fs_mask_t_re2",
    # # group 3
    # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_rest/md_omol_napf6_ethylene_carbonate_pfactor_0.1_1fs_mask_t_re2",
    # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_rest/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_re1_s1p1",
    # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_rest/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_re3_s1p1",
    # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_data_rest/md_omol_napf6_tetrahydrofuran_pfactor_0.1_1fs_mask_t_re2",
    # group 4
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/md_omol_naotf_diglyme_1m_s1p1"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/md_omol_naotf_dme_s1p1_omol"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/md_omol_naotf_pc_1m_s1p1"
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/md_omol_naotf_tgdme_1m_s1p1"
)

# Simulation parameters
TARGET_STEPS=10000000    # Total MD steps (10 ns at 1 fs timestep)
INTERVAL=50            # Output interval for trajectory and status

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

echo "Running command: $CMD"

# Run Python script in background (&) so we can capture its process ID
$CMD &
python_pid=$!                           # Store process ID of background Python job

echo "Python process started with PID: $python_pid"

# # === Set Up Signal Handlers ===
trap "preempt_handler '$python_pid'" SIGTERM    # Catches preemption SIGTERM from SLURM
trap "timeout_handler '$python_pid'" USR1       # Catches timeout USR1 from SLURM

# # === Wait for Completion ===
wait $python_pid                        # Wait for Python process to finish
exit_code=$?                           # Capture exit code of Python process

# # === Final Status Report ===
echo "=========================================="
if [ $exit_code -eq 0 ]; then          
    echo "MD simulation completed successfully!"
else
    echo "MD simulation exited with code: $exit_code"  
fi
echo "End Time: $(date)"                
echo "=========================================="

# # === Keep Process Alive for SLURM Signal Handling ===
# # This sleep ensures there's still a process running so SLURM can send signals
# # Critical for proper preemption detection
# sleep 120