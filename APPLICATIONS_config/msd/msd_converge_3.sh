#!/bin/bash
#
# MSD Convergence Calculation Script
# 
# Usage:
#   sbatch msd_converge.sh
#
# Note: Output directory is configured in the script (see OUT_DIR variable below)
#
# The TARGETS configuration is defined in the MSD_TARGETS_JSON variable below.
# Modify it to change which trajectories are analyzed.

# === SLURM Job Parameters ===
#SBATCH --account=m4319              # Billing account for compute time
#SBATCH --constraint=cpu               # Request CPU nodes only
#SBATCH --cpus-per-task=128              # 5 CPU cores per task
#SBATCH --job-name=msd_convergence_3    # Name of the job (shows in queue)
#SBATCH --mem=450GB                     # Total memory per node (128GB)
#SBATCH --nodes=1                      # Use only 1 compute node
#SBATCH --ntasks-per-node=1            # 1 task per node
#SBATCH --qos=premium                  # Quality of service
#SBATCH --time=20:00:00                 # Maximum runtime: 24 hours
#SBATCH --output=/global/homes/y/yuejian/project/MLFF-distill/yuejian/log/md_flex_%x_%j_%Y%m%d_%H%M%S.out  # Standard output file
#SBATCH --error=/global/homes/y/yuejian/project/MLFF-distill/yuejian/log/md_flex_%x_%j_%Y%m%d_%H%M%S.err   # Standard error file

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

# Set matplotlib backend to non-interactive for batch jobs
export MPLBACKEND=Agg

# === Configuration ===
# TARGETS format: (traj_path, system_name, cat_symbol, anion_symbol, solvent_symbol, concentration_M, temperature_K)
# Note: cat_symbol, anion_symbol, and solvent_symbol must match keys in cation_dict, anion_dict, and solvent_dict respectively.
# concentration_M is a string like "1M", "0_5M", etc.
# temperature_K is a string like "298K", "300K", etc.
# user input .traj path, LLM will infer the rest of the information
MSD_TARGETS_JSON='[
    ["/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solute_solvent_1M/md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj", "NaPF6 — Diglyme", "Na", "PF6", "Diglyme", "1M", "298K"],
    ["/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solute_solvent_1M/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj", "NaPF6 — PC", "Na", "PF6", "PC", "1M", "298K"]
]'

# Export TARGETS as environment variable
export MSD_TARGETS_JSON

# Output directory (configure manually here)
OUT_DIR="/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/observables/msd/converge_12_6/20ns_batch_3"

# Set tau_max_fit_ps to 20000 (20 ns) - fixed value
TAU_MAX_FIT_PS=20000
# ====end of user config====


# Run the MSD calculation batch script
python /global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch.py \
    --out-dir "$OUT_DIR" \
    --tau-max-fit-ps "$TAU_MAX_FIT_PS"

# === Debug Command (inline Python) ===
# For debugging, use this single-line command:
# cd /global/homes/y/yuejian/project/MLFF-distill && MPLBACKEND=Agg MSD_TARGETS_JSON='[["/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solute_solvent_1M/naotf_dme/naotf_dme.traj", "NaOTf — DME", "Na", "OTf", "DME", "1M", "298K"], ["/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solute_solvent_1M/napf6_dme/napf6_dme.traj", "NaPF6 — DME", "Na", "PF6", "DME", "1M", "298K"], ["/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solvent_0_1M/md_omol_naotf_diglyme_1m_s1p1/md_omol_naotf_diglyme_1m_s1p1.traj", "NaOTf — Diglyme", "Na", "OTf", "Diglyme", "0_1M", "298K"], ["/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solvent_0_1M/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj", "NaOTf — DME", "Na", "OTf", "DME", "0_1M", "298K"], ["/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solvent_0_1M/md_omol_naotf_pc_1m_s1p1/md_omol_naotf_pc_1m_s1p1.traj", "NaOTf — PC", "Na", "OTf", "PC", "0_1M", "298K"], ["/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solvent_0_1M/md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj", "NaPF6 — Diglyme", "Na", "PF6", "Diglyme", "0_1M", "298K"], ["/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solvent_0_1M/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj", "NaPF6 — PC", "Na", "PF6", "PC", "0_1M", "298K"]]' python /global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch.py --out-dir "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/observables/msd/test" --tau-max-fit-ps 3000
#
# Or use Python directly with inline JSON (more readable):
# python -c "import os, json, subprocess; os.chdir('/global/homes/y/yuejian/project/MLFF-distill'); os.environ['MPLBACKEND']='Agg'; os.environ['MSD_TARGETS_JSON']=json.dumps([['/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solute_solvent_1M/naotf_dme/naotf_dme.traj', 'NaOTf — DME', 'Na', 'OTf', 'DME', '1M', '298K'], ['/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solute_solvent_1M/napf6_dme/napf6_dme.traj', 'NaPF6 — DME', 'Na', 'PF6', 'DME', '1M', '298K'], ['/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solvent_0_1M/md_omol_naotf_diglyme_1m_s1p1/md_omol_naotf_diglyme_1m_s1p1.traj', 'NaOTf — Diglyme', 'Na', 'OTf', 'Diglyme', '0_1M', '298K'], ['/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solvent_0_1M/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj', 'NaOTf — DME', 'Na', 'OTf', 'DME', '0_1M', '298K'], ['/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solvent_0_1M/md_omol_naotf_pc_1m_s1p1/md_omol_naotf_pc_1m_s1p1.traj', 'NaOTf — PC', 'Na', 'OTf', 'PC', '0_1M', '298K'], ['/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solvent_0_1M/md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj', 'NaPF6 — Diglyme', 'Na', 'PF6', 'Diglyme', '0_1M', '298K'], ['/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity/20ns_solvent_0_1M/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj', 'NaPF6 — PC', 'Na', 'PF6', 'PC', '0_1M', '298K']]); subprocess.run(['python', '/global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch.py', '--out-dir', '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/observables/msd/test', '--tau-max-fit-ps', '3000'])"