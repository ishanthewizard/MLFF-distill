#!/bin/bash
#
# Example Slurm submission script to compute MSD/diffusivity for all OPLS 1M
# trajectories in:
#   m5024/distillation_project/opls_baseline/tpr_files_1M
#
# Adjust `-A` (account), time, and CPUs as needed.
#
# Usage:
#   sbatch submit_opls_1M_msd.slurm
#

#SBATCH -J opls_msd_1M
#SBATCH -o opls_msd_1M.%j.out
#SBATCH -e opls_msd_1M.%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -A m5024
#SBATCH -q regular

set -euo pipefail

# If you use conda/env modules, activate here.
# Example:
#   source ~/.bashrc
#   conda activate fairchemV2

INPUT_DIR="/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/opls_baseline/tpr_files_1M"
OUT_DIR="/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/OPLS_diffusivities/opls_1M_from_tpr_files_1M"

TAU_MAX_FIT_PS=20000
TAU_MIN_FIT_PS=1000
EQ_TIME_PS=100
TEMP_K=298

python3 "/global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_gromacs_CLI.py" \
  --input-dir "${INPUT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --tau-max-fit-ps "${TAU_MAX_FIT_PS}" \
  --tau-min-fit-ps "${TAU_MIN_FIT_PS}" \
  --eq-time-ps "${EQ_TIME_PS}" \
  --temperature-k "${TEMP_K}" \
  --n-workers 8 \
  --parallel-msd

