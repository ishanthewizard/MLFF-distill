"""Config: MSD analysis — 3 systems, full 20 ns.

  LiPF6/DME  0.5M 323K  (micro_acas_50ps)
  NaPF6/DME  0.5M 323K  (micro_acas_50ps)
  NaOTf/DME  0.1M 298K  (micro_acas_50ps)

dt_fs = 100 fs/frame; analysis capped at 20 ns.
"""
from pathlib import Path

_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5024/distillation_project/results/diffusivity_main_results_20ns_final"
    "/micro_acas_50ps"
)

_SHARED = dict(
    dt_fs           = 100.0,
    max_traj_ns     = 20.0,
    n_frames        = 2000,
    model_colors    = {"omol": "#1f77b4"},
    eq_cut_ns       = 0.0,
    fit_pct         = 0.8,
    tau_min_fit_ns  = 1.0,
    slide_window_ns = 10.0,
    slide_step_ns   = 0.5,
    n_conv_points   = 200,
)

def _traj(rel):
    p = _BASE / rel
    return str(p / f"{p.name}.traj")

SYSTEMS = [
    {
        **_SHARED,
        "name":           "LiPF6/DME 0.5M 323K (omol, 20ns)",
        "traj_paths":     {"omol": _traj("20ns_solvent_solute_0.5M/323_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t")},
        "cat_symbol":     "Li",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaPF6/DME 0.5M 323K (omol, 20ns)",
        "traj_paths":     {"omol": _traj("20ns_solvent_solute_0.5M/323_2K/md_omol_napf6_dme_re1")},
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaOTf/DME 0.1M 298K (omol, 20ns)",
        "traj_paths":     {"omol": _traj("20ns_solvent_0_1M/298K/md_omol_naotf_dme_s1p1_omol")},
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "DME",
    },
]

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/analysis/msd_omol_3systems_20ns"
)
ANALYSES = ["msd"]
WORKERS  = 3
