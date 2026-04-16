#!/bin/bash

# === SLURM Job Parameters ===
#SBATCH --account=m4319              # Billing account for compute time
#SBATCH --constraint=cpu               # Request CPU nodes only
#SBATCH --cpus-per-task=128              # 5 CPU cores per task
#SBATCH --job-name=msd    # Name of the job (shows in queue)
#SBATCH --mem=450GB                     # Total memory per node (128GB)
#SBATCH --nodes=1                      # Use only 1 compute node
#SBATCH --ntasks-per-node=1            # 1 task per node
#SBATCH --qos=premium                  # Quality of service
#SBATCH --time=14:00:00                 # Maximum runtime: 24 hours
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

# Set matplotlib backend to non-interactive for batch jobs
export MPLBACKEND=Agg

# === MSD Calculation Parameters ===
# Define solvents (one per line, easier to read and manage)
SOLVENTS=(
    "dme"
    # Add more solvents here, one per line:
    # "solvent2"
    # "solvent3"
)

# Define trajectory paths (one per line, easier to read and manage)
TRAJECTORIES=(
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablate_distillation/10ns_solvent_0_1M/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj"
    # Add more trajectory paths here, one per line:
    # "/path/to/trajectory3.traj"
    # "/path/to/trajectory4.traj"
)

# Number of frames to analyze
ANALYZE_FIRST_N_FRAMES=1000000

# Output root directory
ROOT_DIR="/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablate_distillation/msd/0_1M/"

# Convert arrays to comma-separated strings for Python script
# Using IFS (Internal Field Separator) to join array elements with commas
IFS=','
SOLVENTS_STR="${SOLVENTS[*]}"
TRAJECTORIES_STR="${TRAJECTORIES[*]}"
unset IFS

# Run MSD calculation script
python APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msd_calculation.py \
    --solvents "$SOLVENTS_STR" \
    --trajectories "$TRAJECTORIES_STR" \
    --root-dir "$ROOT_DIR" \
    --analyze_first_n_frames "$ANALYZE_FIRST_N_FRAMES"


# give me a direct command to run the script, no $
python APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msd_calculation.py \
    --solvents "dme" \
    --trajectories "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/temp/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t_wo_hessian/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t_wo_hessian.traj" \
    --root-dir "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablate_distillation/msd/0_1M/" \
    --analyze_first_n_frames 1000000
