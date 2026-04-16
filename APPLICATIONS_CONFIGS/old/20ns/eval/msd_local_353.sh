#!/bin/bash
# MSD Convergence Calculation Script (local execution)
#
# Collects all .traj files under the outliers directory and runs the MSD batch
# calculator locally. Target metadata is listed explicitly below (no inference).

set -euo pipefail

cd /u/yjian1/project/MLFF-distill
export MPLBACKEND=Agg

# === Configuration ===
# TARGETS format: (traj_path, system_name, cat_symbol, anion_symbol, solvent_symbol, concentration_M, temperature_K)
# Note: cat_symbol, anion_symbol, and solvent_symbol must match keys in cation_dict, anion_dict, and solvent_dict respectively.
# concentration_M is a string like "1M", "0_5M", etc.
# temperature_K is a string like "298K", "300K", etc.
MSD_TARGETS_JSON='[
    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solvent_solute_0.5M/273_2K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj", "md_omol_napf6_dme_re1_273_2K", "Na", "PF6", "DME", "0_5M", "273_2K"],
    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solvent_solute_0.5M/273_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t/md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj", "md_omol_lipf6_pfactor_0.1_1fs_mask_t_273_2K", "Li", "PF6", "DME", "0_5M", "273_2K"],

    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solvent_solute_0.5M/298_2K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj", "md_omol_napf6_dme_re1_298_2K", "Na", "PF6", "DME", "0_5M", "298_2K"],

    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solvent_solute_0.5M/323_2K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj", "md_omol_napf6_dme_re1_323_2K", "Na", "PF6", "DME", "0_5M", "323_2K"],

    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solvent_0_1M/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj", "md_omol_naotf_dme_s1p1_omol_0_1M", "Na", "OTf", "DME", "0_1M", "298K"],
    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solvent_0_1M/md_omol_naotf_pc_1m_s1p1/md_omol_naotf_pc_1m_s1p1.traj", "md_omol_naotf_pc_1m_s1p1_0_1M", "Na", "OTf", "PC", "0_1M", "298K"],
    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solvent_0_1M/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj", "md_omol_napf6_dme_re1_0_1M", "Na", "PF6", "DME", "0_1M", "298K"],
    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solvent_0_1M/md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj", "md_omol_napf6_diglyme_pfactor_0.1_1fs_0_1M", "Na", "PF6", "Diglyme", "0_1M", "298K"],
    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solvent_0_1M/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj", "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1_0_1M", "Na", "PF6", "PC", "0_1M", "298K"],

    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solute_solvent_1M/napf6_dme/napf6_dme.traj", "napf6_dme_1M", "Na", "PF6", "DME", "1M", "298K"],
    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solute_solvent_1M/naotf_dme/naotf_dme.traj", "naotf_dme_1M", "Na", "OTf", "DME", "1M", "298K"],
    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solute_solvent_1M/naotf_diglyme/naotf_diglyme.traj", "naotf_diglyme_1M", "Na", "OTf", "Diglyme", "1M", "298K"],
    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solute_solvent_1M/md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj", "md_omol_napf6_diglyme_pfactor_0.1_1fs_1M", "Na", "PF6", "Diglyme", "1M", "298K"],
    ["/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/temp_ablation/initial_box_inference_md_353K/20ns_solute_solvent_1M/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj", "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1_1M", "Na", "PF6", "PC", "1M", "298K"]
]'

# Export TARGETS as environment variable
export MSD_TARGETS_JSON

# Output directory (configure manually here)
OUT_DIR="/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/analysis_results/353K"

# Set tau_max_fit_ps to 7 ns (7000 ps)
TAU_MAX_FIT_PS=20000
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