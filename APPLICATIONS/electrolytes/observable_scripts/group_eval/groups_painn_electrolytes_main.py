"""Group config for the PAINN electrolytes_data simulation set.

Plots density parity (MD vs experiment) for this PAINN production set.

Prereq (not run by this config — build the parity csv first):
  python build_parity_csv.py --source painn \
      --root /global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data/simulation \
      --exp-csv /global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/experiment_data/cleaned_version/conductivity.csv \
      --output /global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data/analysis/group_parity/conductivity_parity_painn.csv

Then:
  python run_groups.py groups_painn_electrolytes_main

NOTE: "msd" parity is NOT included here — build_parity_csv.py currently only
computes conductivity and density per system, with no diffusivity/MSD columns
or experimental D_* matching. Adding an "msd" parity property would require
extending build_parity_csv.py to run the MSD analysis per system and adding
matching exp/sim diffusivity columns (properties.py already defines
diffusivity_cation/anion/solvent entries, but nothing currently populates
their sim_/exp_ columns for this source).
"""

# Separate from per-trajectory analysis output (config_painn_electrolytes_main.py's
# OUTPUT_DIR = .../analysis/per_traj) -- group parity csv + plots live here.
OUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data/analysis/group_parity"
)

GROUPS = {
    "painn_electrolytes_main": {
        "label": "PAINN electrolytes_data",
        "parity_csv": f"{OUT_DIR}/conductivity_parity_painn.csv",
        "properties": ["density", "conductivity", "conductivity_NE", "conductivity_linear", "conductivity_NE_linear"],
    },
}
