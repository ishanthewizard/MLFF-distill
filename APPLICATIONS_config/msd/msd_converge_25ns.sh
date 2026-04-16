#!/bin/bash
#
# MSD Convergence Calculation Script
# 
# Usage:
#   sbatch msd_converge.sh
#
# Note: Output directory is configured in the script (see OUT_DIR variable below)
#
# The TARGETS configuration is defined in the MSD_TARGETS_JSON variable below.
# Modify it to change which trajectories are analyzed.

# === SLURM Job Parameters ===
#SBATCH --account=m4319              # Billing account for compute time
#SBATCH --constraint=cpu               # Request CPU nodes only
#SBATCH --cpus-per-task=128              # 5 CPU cores per task
#SBATCH --job-name=msd_convergence    # Name of the job (shows in queue)
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

# === Configuration ===
# TARGETS format: (traj_path, system_name, cat_symbol, anion_symbol, solvent_symbol, concentration_M, temperature_K)
# Note: cat_symbol, anion_symbol, and solvent_symbol must match keys in cation_dict, anion_dict, and solvent_dict respectively.
# concentration_M is a string like "1M", "0_5M", etc.
# temperature_K is a string like "298K", "300K", etc.
# user input .traj path, LLM will infer the rest of the information
MSD_TARGETS_JSON='[
    ["/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solute_solvent_1M/naotf_dme/naotf_dme.traj", "NaOTf — DME", "Na", "OTf", "DME", "1M", "298K"],
    ["/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solute_solvent_1M/napf6_dme/napf6_dme.traj", "NaPF6 — DME", "Na", "PF6", "DME", "1M", "298K"]
]'

# Export TARGETS as environment variable
export MSD_TARGETS_JSON

# Output directory (configure manually here)
OUT_DIR="/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/observables/msd/converge_12_6/31ns"

# Set tau_max_fit_ps to 20000 (20 ns) - fixed value
TAU_MAX_FIT_PS=31000
# ====end of user config====


# Run the MSD calculation batch script
python /global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch.py \
    --out-dir "$OUT_DIR" \
    --tau-max-fit-ps "$TAU_MAX_FIT_PS"