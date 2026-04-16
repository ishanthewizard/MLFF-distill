#!/bin/bash
# MSD Convergence Calculation Script
#
# Usage:
#   sbatch msd_converge.sh
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
#SBATCH --time=24:00:00
#SBATCH --job-name=eval_msd
#SBATCH --account=bfoy-dtai-gh
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

#SBATCH --output=/u/yjian1/project/MLFF-distill/yjian1/log/md_flex_%x_%j_%Y%m%d_%H%M%S.out
#SBATCH --error=/u/yjian1/project/MLFF-distill/yjian1/log/md_flex_%x_%j_%Y%m%d_%H%M%S.err

set -euo pipefail

# === Environment Setup and Logging ===
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"              
echo "Job Name: $SLURM_JOB_NAME"          
echo "Node: $SLURM_NODELIST"              
echo "Start Time: $(date)"                
echo "Requeue Count: ${SLURM_RESTART_COUNT:-0}"  
echo "=========================================="

# Make sure log dir exists
mkdir -p /u/yjian1/project/MLFF-distill/yjian1/log

# === Change to Working Directory ===
cd /u/yjian1/project/MLFF-distill

# Set matplotlib backend to non-interactive for batch jobs
export MPLBACKEND=Agg




# === Configuration ===
# TARGETS format: (traj_path, system_name, cat_symbol, anion_symbol, solvent_symbol, concentration_M, temperature_K)
# Note: cat_symbol, anion_symbol, and solvent_symbol must match keys in cation_dict, anion_dict, and solvent_dict respectively.
# concentration_M is a string like "1M", "0_5M", etc.
# temperature_K is a string like "298K", "300K", etc.
# user input .traj path, LLM will infer the rest of the information
MSD_TARGETS_JSON='[
    ["/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/ablation_diffusivity/nvt_unstable/20ns_solvent_0_1M/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj", "NaOTf - DME", "Na", "OTf", "DME", "0_1M", "298K"]
]'

# Export TARGETS as environment variable
export MSD_TARGETS_JSON

# Output directory (configure manually here)
OUT_DIR="/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_distillation/observables/msd/nvt_unstable/7ns"

# Set tau_max_fit_ps to 20000 (20 ns) - fixed value
TAU_MAX_FIT_PS=7000
# Trajectory base timestep in picoseconds (100 fs)
KNOWN_DT_PS=0.1
# ====end of user config====


# Avoid CPU oversubscription if SLURM sets cpus-per-task
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Run the MSD calculation batch script
python /u/yjian1/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch.py \
    --out-dir "$OUT_DIR" \
    --tau-max-fit-ps "$TAU_MAX_FIT_PS" \
    --known-dt-ps "$KNOWN_DT_PS"

# === Debug Command (inline Python) ===
# For debugging, use this single-line command:
# cd /u/yjian1/project/MLFF-distill && MPLBACKEND=Agg MSD_TARGETS_JSON='[["/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/ablation_diffusivity/nvt_unstable/20ns_solvent_0_1M/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj", "NaOTf - DME", "Na", "OTf", "DME", "0_1M", "298K"]]' python /u/yjian1/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch.py --out-dir "/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_distillation/observables/msd/test" --tau-max-fit-ps 3000 --known-dt-ps 0.1
#
# Or use Python directly with inline JSON (more readable):
# python -c "import os, json, subprocess; os.chdir('/u/yjian1/project/MLFF-distill'); os.environ['MPLBACKEND']='Agg'; os.environ['MSD_TARGETS_JSON']=json.dumps([['/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/ablation_diffusivity/nvt_unstable/20ns_solvent_0_1M/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj', 'NaOTf - DME', 'Na', 'OTf', 'DME', '0_1M', '298K']]); subprocess.run(['python', '/u/yjian1/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch.py', '--out-dir', '/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_distillation/observables/msd/test', '--tau-max-fit-ps', '3000', '--known-dt-ps', '0.1'])"