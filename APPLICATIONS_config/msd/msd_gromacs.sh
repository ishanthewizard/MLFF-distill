#!/bin/bash
#
# MSD Calculation for GROMACS trajectories
#
# Usage:
#   sbatch msd_gromacs.sh
#
# Note: Output directory and tau_max_fit_ps are configured below.
#

# === SLURM Job Parameters ===
#SBATCH --account=m4319              # Billing account for compute time
#SBATCH --constraint=cpu             # Request CPU nodes only
#SBATCH --cpus-per-task=128          # CPU cores per task
#SBATCH --job-name=msd_gromacs       # Job name (appears in queue)
#SBATCH --mem=450GB                  # Total memory per node
#SBATCH --nodes=1                    # Number of compute nodes
#SBATCH --ntasks-per-node=1          # Tasks per node
#SBATCH --qos=premium                # Quality of service
#SBATCH --time=20:00:00              # Max runtime
#SBATCH --output=/global/homes/y/yuejian/project/MLFF-distill/yuejian/log/md_flex_%x_%j_%Y%m%d_%H%M%S.out  # Stdout file
#SBATCH --error=/global/homes/y/yuejian/project/MLFF-distill/yuejian/log/md_flex_%x_%j_%Y%m%d_%H%M%S.err   # Stderr file

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

# Non-interactive backend for matplotlib
export MPLBACKEND=Agg

# === Configuration ===
OUT_DIR="/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/OPLS_diffusivities/OPLS_init_try_3_systems"
TAU_MAX_FIT_PS=20000

# === Run ===
python /global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch_gromacs.py \
    -o "$OUT_DIR" \
    -t "$TAU_MAX_FIT_PS"

