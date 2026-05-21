"""Config: 1M / 298K systems — ASE only (original + micro). Run on compute node."""
from pathlib import Path

_ORIG  = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/other_fix_run/original_100ps/20ns_solute_solvent_1M/298K")
_MICRO = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/other_fix_run/micro_acas_rest/20ns_solute_solvent_1M/298K")

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
        "original": "#ff7f0e",
        "micro":    "#1f77b4",
    },
}

SYSTEMS = [
    {
        **_SHARED,
        "name":       "NaOTf/DME 1M 298K",
        "traj_paths": {
            "original": str(_ORIG  / "naotf_dme/naotf_dme.traj"),
            "micro":    str(_MICRO / "naotf_dme/naotf_dme.traj"),
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
            "original": str(_ORIG  / "napf6_dme/napf6_dme.traj"),
            "micro":    str(_MICRO / "napf6_dme/napf6_dme.traj"),
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
            "original": str(_ORIG  / "naotf_diglyme/naotf_diglyme.traj"),
            "micro":    str(_MICRO / "naotf_diglyme/naotf_diglyme.traj"),
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
            "original": str(_ORIG  / "md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj"),
            "micro":    str(_MICRO / "md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj"),
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
            "original": str(_ORIG  / "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj"),
            "micro":    str(_MICRO / "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj"),
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
