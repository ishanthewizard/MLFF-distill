#!/bin/bash

# SLURM Parameters for DFT Calculations
#SBATCH --account=m5024_g
#SBATCH --cpus-per-task=128
#SBATCH --error=/global/homes/y/yuejian/project/MLFF-distill/yuejian/DFT_output/%j_%t_log.err
#SBATCH --job-name=dft_single_point_calculations
#SBATCH --mem=256GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/global/homes/y/yuejian/project/MLFF-distill/yuejian/DFT_output/%j_%t_log.out
#SBATCH --qos=regular
#SBATCH --signal=USR2@90
#SBATCH --time=24:00:00
#SBATCH --wckey=submitit

# Note: Activate your conda environment before running this script
# Example: conda activate your_env_name

# Create log directory if it doesn't exist
mkdir -p /global/homes/y/yuejian/project/MLFF-distill/yuejian/DFT_output

# Set the root directory containing ligand subdirectories
# Path to the ligand data directory
ROOT_DIR="/global/homes/y/yuejian/project/MLFF-distill/yuejian/ligandboundconf3/xtb_local_min"

# Set the ORCA executable path
# Path to ORCA installation on NERSC
ORCA_PATH="/global/homes/y/yuejian/project/MLFF-distill/yuejian/orca"

# Change to the project directory
cd /global/homes/y/yuejian/project/MLFF-distill

# Run the DFT calculation script (batch version)
python src_v2/DFT_calculation/calculate_single_point_energy_batch.py "$ROOT_DIR" --orca-path "$ORCA_PATH" --no-confirm

# Optional: Add any post-processing or cleanup commands here
echo "DFT calculations completed at $(date)" 