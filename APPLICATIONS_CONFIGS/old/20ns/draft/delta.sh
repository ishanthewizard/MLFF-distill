#!/bin/bash


#SBATCH --mem=256g #<=120GB per gpu service unit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32 # 72 cores per gpu service unit
#SBATCH --partition=ghx4 # more info see "sinfo -s"
#SBATCH --time=00:15:00
#SBATCH --job-name=meanflow_delta
#SBATCH --account=bfoy-dtai-gh
### GPU options ###
#SBATCH --gpus-per-node=4     # max 4 gh200 GPUs per node
#SBATCH --gpu-bind=verbose,closest

#SBATCH --output=/u/yjian1/project/Generative_3d_turbulence_flow/yjian1/3d_turbulence/log/meanflow_turbulence_%x_%j_%Y%m%d_%H%M%S.out
#SBATCH --error=/u/yjian1/project/Generative_3d_turbulence_flow/yjian1/3d_turbulence/log/meanflow_turbulence_%x_%j_%Y%m%d_%H%M%S.err

#SBATCH --mail-user=yuejian@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29500 + $RANDOM % 1000))  # Random port to avoid conflicts
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID

cd /u/yjian1/project/Generative_3d_turbulence_flow

# Use torchrun for distributed training



# # overfit
# torchrun --standalone --nproc_per_node=4 --master_port=12345 scripts/train_meanflow.py --config-path config/meanflow_exp/config_meanflow_overfit --config-name main

# # toy
# torchrun --standalone --nproc_per_node=4 --master_port=12345 scripts/train_meanflow.py --config-path config/meanflow_exp/config_meanflow_toy --config-name main

# # normal
# torchrun --standalone --nproc_per_node=4 --master_port=12345 scripts/train_meanflow.py --config-path config/meanflow_exp/config_meanflow --config-name main


# fourier noise
torchrun --standalone --nproc_per_node=4 --master_port=12345 scripts/train_meanflow.py --config-path config/meanflow_exp/config_meanflow_fourier_noise --config-name main