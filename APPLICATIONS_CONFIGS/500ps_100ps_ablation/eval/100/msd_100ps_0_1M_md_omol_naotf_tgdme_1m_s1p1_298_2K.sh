#!/bin/bash
# MSD Convergence Calculation Script
#
# Usage:
#   sbatch msd_100ps_0_1M_md_omol_naotf_tgdme_1m_s1p1_298_2K.sh
#
# Note: Output directory is configured in the script (see OUT_DIR variable below)
#
# The TARGETS configuration is defined in the MSD_TARGETS_JSON variable below.
# Modify it to change which trajectories are analyzed.

# === SLURM Job Parameters (Delta) ===
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=1:00:00
#SBATCH --job-name=eval_msd
#SBATCH --account=bfoy-dtai-gh
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

#SBATCH --output=/u/yjian1/project/MLFF-distill/yjian1/log/md_flex_%x_%j_%Y%m%d_%H%M%S.out
#SBATCH --error=/u/yjian1/project/MLFF-distill/yjian1/log/md_flex_%x_%j_%Y%m%d_%H%M%S.err

set -euo pipefail

# Make sure log dir exists (for Slurm --output/--error paths)
mkdir -p /u/yjian1/project/MLFF-distill/yjian1/log

# === Change to Working Directory ===
cd /u/yjian1/project/MLFF-distill

# Set matplotlib backend to non-interactive for batch jobs
export MPLBACKEND=Agg

# === Single target ===
MSD_TARGETS_JSON='[
    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/outlier_and_all_sys_ablation_Mar_13_2026/100ps_double_data/initial_box/20ns_solvent_0_1M/md_omol_naotf_tgdme_1m_s1p1/md_omol_naotf_tgdme_1m_s1p1.traj", "md_omol_naotf_tgdme_1m_s1p1_298_2K", "Na", "OTf", "TGDME", "0_1M", "298_2K"]
]'

export MSD_TARGETS_JSON

# Output directory (isolated per system to allow parallel runs)
OUT_DIR="/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/outlier_and_all_sys_ablation_Mar_13_2026/eval"

# Set tau_max_fit_ps to 20000 ps (20 ns)
TAU_MAX_FIT_PS=8000
# Trajectory base timestep in picoseconds (100 fs)
KNOWN_DT_PS=0.1

# Avoid CPU oversubscription if SLURM sets cpus-per-task
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Run the MSD calculation batch script
python /u/yjian1/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch.py \
    --out-dir "$OUT_DIR" \
    --tau-max-fit-ps "$TAU_MAX_FIT_PS" \
    --known-dt-ps "$KNOWN_DT_PS"
