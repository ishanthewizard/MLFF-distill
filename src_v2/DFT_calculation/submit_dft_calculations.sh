#!/bin/bash

# SLURM Parameters for DFT Calculations
#SBATCH --account=m5024_g
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=8
#SBATCH --error=/global/homes/y/yuejian/project/MLFF-distill/yuejian/runs/dft_calculations/logs/%j_%t_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=dft_single_point_calculations
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/global/homes/y/yuejian/project/MLFF-distill/yuejian/runs/dft_calculations/logs/%j_%t_log.out
#SBATCH --qos=regular
#SBATCH --signal=USR2@90
#SBATCH --time=48:00:00
#SBATCH --wckey=submitit

# Load necessary modules (adjust as needed for your NERSC setup)
module load python/3.9

# Set up environment variables
export PYTHONPATH="/global/homes/y/yuejian/project/MLFF-distill:$PYTHONPATH"

# Create log directory if it doesn't exist
mkdir -p /global/homes/y/yuejian/project/MLFF-distill/yuejian/runs/dft_calculations/logs

# Set the root directory containing ligand subdirectories
# Modify this path to point to your actual data directory
ROOT_DIR="/global/homes/y/yuejian/project/MLFF-distill/data/ligands"

# Set the ORCA executable path
# Modify this path to point to your ORCA installation
ORCA_PATH="/global/homes/y/yuejian/project/MLFF-distill/yuejian/orca/orca_6_0_0_shared_openmpi416/orca"

# Change to the project directory
cd /global/homes/y/yuejian/project/MLFF-distill

# Run the DFT calculation script (batch version)
python src_v2/DFT_calculation/calculate_single_point_energy_batch.py "$ROOT_DIR" --orca-path "$ORCA_PATH" --no-confirm

# Optional: Add any post-processing or cleanup commands here
echo "DFT calculations completed at $(date)" 