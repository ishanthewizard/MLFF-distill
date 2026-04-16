#!/bin/bash

# === SLURM Job Parameters ===
#SBATCH --account=m5024              # Billing account for compute time
#SBATCH --constraint=cpu               # Request GPU nodes only
#SBATCH --cpus-per-task=32             # 32 CPU cores per task
#SBATCH --job-name=uma_compare_eval         # Job name
#SBATCH --ntasks-per-node=1            # Use only 1 task per node
#SBATCH --mem=200GB                    # Total memory per node
#SBATCH --nodes=1                      # Use only 1 compute node
#SBATCH --qos=debug                  # Quality of service
#SBATCH --time=00:30:00                # Maximum runtime
#SBATCH --open-mode=append             # Append to output files (important for requeued jobs)
#SBATCH --output=/global/homes/y/yuejian/project/MLFF-distill/yuejian/log/md_flex_%x_%j_%Y%m%d_%H%M%S.out  # Standard output file
#SBATCH --error=/global/homes/y/yuejian/project/MLFF-distill/yuejian/log/md_flex_%x_%j_%Y%m%d_%H%M%S.err   # Standard error file


# Set matplotlib backend to non-interactive for batch jobs
export MPLBACKEND=Agg

# === Single target ===
MSD_TARGETS_JSON='[
    ["/global/homes/y/yuejian/project/MLFF-distill/m5250/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solvent_0_1M/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj", "uma_1ns_naotf_dme_0_1M_298K_md_omol", "Na", "OTf", "DME", "0_1M", "298K"]
]'

export MSD_TARGETS_JSON

# Output directory (isolated per system to allow parallel runs)
OUT_DIR="/global/homes/y/yuejian/project/MLFF-distill/m5250/electrolyte_application/analysis/1ns/msd_uma_1ns_naotf_dme_0_1M_298K_md_omol"

# Set tau_max_fit_ps to 1000 ps (1 ns)
TAU_MAX_FIT_PS=1000
# Trajectory base timestep in picoseconds (100 fs)
KNOWN_DT_PS=0.01

# Run the MSD calculation batch script
python /global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch_new.py \
    --out-dir "$OUT_DIR" \
    --tau-max-fit-ps "$TAU_MAX_FIT_PS" \
    --known-dt-ps "$KNOWN_DT_PS"
