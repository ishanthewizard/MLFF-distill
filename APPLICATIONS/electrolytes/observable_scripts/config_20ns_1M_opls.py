"""Config: 1M / 298K systems — OPLS only (GROMACS XTC). Run on login node."""
from pathlib import Path

_OPLS = Path("/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/results/opls_baseline/tpr_files_1M")

_SHARED = {
    "dt_fs":          100.0,
    "max_traj_ns":    20.0,
    "skip_ns":        0.1,
    "window_ns":      10.0,
    "n_frames":       2000,
    "sliding_window": True,
    "rdf_slide_ns":   1.0,
    "density_roll_window_ns": 0.5,
    "eq_cut_ns":      0.0,
    "fit_pct":        0.8,
    "tau_min_fit_ns": 1.0,
    "slide_window_ns":10.0,
    "slide_step_ns":  0.5,
    "n_conv_points":  200,
    "model_colors": {
        "OPLS": "#2ca02c",
    },
}

SYSTEMS = [
    {
        **_SHARED,
        "name":       "NaOTf/DME 1M 298K",
        "traj_paths": {
            "OPLS": {"xtc": str(_OPLS / "npt_1M_naotf_dme.xtc"), "tpr": str(_OPLS / "npt_1M_naotf_dme.tpr")},
        },
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":       "NaPF6/DME 1M 298K",
        "traj_paths": {
            "OPLS": {"xtc": str(_OPLS / "npt_1M_napf6_dme.xtc"), "tpr": str(_OPLS / "npt_1M_napf6_dme.tpr")},
        },
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":       "NaOTf/Diglyme 1M 298K",
        "traj_paths": {
            "OPLS": {"xtc": str(_OPLS / "npt_1M_naotf_diglyme.xtc"), "tpr": str(_OPLS / "npt_1M_naotf_diglyme.tpr")},
        },
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "Diglyme",
    },
    {
        **_SHARED,
        "name":       "NaPF6/Diglyme 1M 298K",
        "traj_paths": {
            "OPLS": {"xtc": str(_OPLS / "npt_1M_napf6_diglyme.xtc"), "tpr": str(_OPLS / "npt_1M_napf6_diglyme.tpr")},
        },
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "Diglyme",
    },
    {
        **_SHARED,
        "name":       "NaPF6/PC 1M 298K",
        "traj_paths": {
            "OPLS": {"xtc": str(_OPLS / "npt_1M_napf6_pc.xtc"), "tpr": str(_OPLS / "npt_1M_napf6_pc.tpr")},
        },
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "PC",
    },
]

OUTPUT_DIR = "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/analysis/20ns"
ANALYSES   = ["rdf", "density", "energy", "msd"]
WORKERS    = 5
