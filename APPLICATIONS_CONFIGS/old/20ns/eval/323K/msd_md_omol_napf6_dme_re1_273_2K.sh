#!/bin/bash
# MSD Convergence Calculation Script

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

cd /u/yjian1/project/MLFF-distill
export MPLBACKEND=Agg

# === Single target ===
MSD_TARGETS_JSON='[
    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_323K/20ns_solvent_solute_0.5M/273_2K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj", "md_omol_napf6_dme_re1_273_2K", "Na", "PF6", "DME", "0_5M", "273_2K"]
]'
export MSD_TARGETS_JSON

# Output directory (isolated per system to allow parallel runs)
OUT_DIR="/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/analysis/2_3/323/md_omol_napf6_dme_re1_273_2K"
TAU_MAX_FIT_PS=20000
KNOWN_DT_PS=0.1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python /u/yjian1/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch.py \
    --out-dir "$OUT_DIR" \
    --tau-max-fit-ps "$TAU_MAX_FIT_PS" \
    --known-dt-ps "$KNOWN_DT_PS"
