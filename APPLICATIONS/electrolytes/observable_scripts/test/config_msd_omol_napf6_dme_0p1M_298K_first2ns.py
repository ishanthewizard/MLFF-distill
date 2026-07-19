"""Config: MSD analysis for omol NaPF6/DME 0.1M 298K — first 2 ns.

Trajectory: micro_acas_50ps student model run
  path: diffusivity_main_results_20ns_final/micro_acas_50ps/20ns_solvent_0_1M/298K/
        md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj
dt_fs = 100 fs/frame; analysis capped at 2 ns.
"""
from pathlib import Path

_TRAJ = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5024/distillation_project/results/diffusivity_main_results_20ns_final"
    "/micro_acas_50ps/20ns_solvent_0_1M/298K"
    "/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj"
)

SYSTEMS = [
    {
        "name":           "NaPF6/DME 0.1M 298K (omol, first 2ns)",
        "traj_paths":     {"omol": _TRAJ},
        "model_colors":   {"omol": "#1f77b4"},
        "dt_fs":          100.0,
        "max_traj_ns":    2.0,
        "n_frames":       2000,
        # MSD parameters
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
        "eq_cut_ns":      0.0,
        "fit_pct":        0.8,
        "tau_min_fit_ns": 0.1,
        "slide_window_ns":0.5,
        "slide_step_ns":  0.05,
        "n_conv_points":  100,
    },
]

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/analysis/msd_omol_napf6_dme_0p1M_298K_first2ns"
)
ANALYSES = ["msd"]
WORKERS  = 1
