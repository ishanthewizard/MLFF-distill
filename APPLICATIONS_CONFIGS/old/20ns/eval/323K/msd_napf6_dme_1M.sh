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

mkdir -p /u/yjian1/project/MLFF-distill/yjian1/log

cd /u/yjian1/project/MLFF-distill
export MPLBACKEND=Agg

MSD_TARGETS_JSON='[
    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_323K/20ns_solute_solvent_1M/napf6_dme/napf6_dme.traj", "napf6_dme_1M", "Na", "PF6", "DME", "1M", "298K"]
]'
export MSD_TARGETS_JSON

OUT_DIR="/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/analysis/2_3/323/napf6_dme_1M"
TAU_MAX_FIT_PS=20000
KNOWN_DT_PS=0.1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python /u/yjian1/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch.py \
    --out-dir "$OUT_DIR" \
    --tau-max-fit-ps "$TAU_MAX_FIT_PS" \
    --known-dt-ps "$KNOWN_DT_PS"
