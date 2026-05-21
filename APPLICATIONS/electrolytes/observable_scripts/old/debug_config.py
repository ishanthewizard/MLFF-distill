"""Debug config: 5 systems from original_100ps/20ns_solute_solvent_1M/298K.

All trajectories: 200k frames at 100 fs/frame = 20 ns, student model (original).
n_frames reduced for fast debug run.
"""
from pathlib import Path

_ROOT = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application"
             "/other_fix_run/original_100ps/20ns_solute_solvent_1M/298K")

SYSTEMS = [
    {
        "name": "NaOTf/DME 1M",
        "traj_paths": {
            "original": str(_ROOT / "naotf_dme/naotf_dme.traj"),
        },
        "dt_fs": 100.0,
        "rdf_pairs": [("Na", "S"), ("Na", "O")],
        "skip_ns": 0.1,
        "window_ns": 15.0,
        "n_frames": 300,
        "sliding_window": True,
        "density_roll_window_ns": 0.5,
        "cat_symbol":      "Na",
        "anion_symbol":    "OTf",
        "solvent_symbol":  "DME",
        "eq_cut_ns":       0.0,
        "fit_pct":         0.8,
        "tau_min_fit_ns":  1.0,
        "slide_window_ns": 10.0,
        "slide_step_ns":   0.5,
        "n_conv_points":   50,
        "max_traj_ns":     20.0,
    },
    {
        "name": "NaPF6/DME 1M",
        "traj_paths": {
            "original": str(_ROOT / "napf6_dme/napf6_dme.traj"),
        },
        "dt_fs": 100.0,
        "rdf_pairs": [("Na", "O"), ("Na", "F")],
        "skip_ns": 0.1,
        "window_ns": 15.0,
        "n_frames": 300,
        "sliding_window": True,
        "density_roll_window_ns": 0.5,
        "cat_symbol":      "Na",
        "anion_symbol":    "PF6",
        "solvent_symbol":  "DME",
        "eq_cut_ns":       0.0,
        "fit_pct":         0.8,
        "tau_min_fit_ns":  1.0,
        "slide_window_ns": 10.0,
        "slide_step_ns":   0.5,
        "n_conv_points":   50,
        "max_traj_ns":     20.0,
    },
    {
        "name": "NaOTf/Diglyme 1M",
        "traj_paths": {
            "original": str(_ROOT / "naotf_diglyme/naotf_diglyme.traj"),
        },
        "dt_fs": 100.0,
        "rdf_pairs": [("Na", "S"), ("Na", "O")],
        "skip_ns": 0.1,
        "window_ns": 15.0,
        "n_frames": 300,
        "sliding_window": True,
        "density_roll_window_ns": 0.5,
        "cat_symbol":      "Na",
        "anion_symbol":    "OTf",
        "solvent_symbol":  "Diglyme",
        "eq_cut_ns":       0.0,
        "fit_pct":         0.8,
        "tau_min_fit_ns":  1.0,
        "slide_window_ns": 10.0,
        "slide_step_ns":   0.5,
        "n_conv_points":   50,
        "max_traj_ns":     20.0,
    },
    {
        "name": "NaPF6/Diglyme 1M",
        "traj_paths": {
            "original": str(_ROOT / "md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj"),
        },
        "dt_fs": 100.0,
        "rdf_pairs": [("Na", "O"), ("Na", "F")],
        "skip_ns": 0.1,
        "window_ns": 15.0,
        "n_frames": 300,
        "sliding_window": True,
        "density_roll_window_ns": 0.5,
        "cat_symbol":      "Na",
        "anion_symbol":    "PF6",
        "solvent_symbol":  "Diglyme",
        "eq_cut_ns":       0.0,
        "fit_pct":         0.8,
        "tau_min_fit_ns":  1.0,
        "slide_window_ns": 10.0,
        "slide_step_ns":   0.5,
        "n_conv_points":   50,
        "max_traj_ns":     20.0,
    },
    {
        "name": "NaPF6/PC 1M",
        "traj_paths": {
            "original": str(_ROOT / "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj"),
        },
        "dt_fs": 100.0,
        "rdf_pairs": [("Na", "O"), ("Na", "F")],
        "skip_ns": 0.1,
        "window_ns": 15.0,
        "n_frames": 300,
        "sliding_window": True,
        "density_roll_window_ns": 0.5,
        "cat_symbol":      "Na",
        "anion_symbol":    "PF6",
        "solvent_symbol":  "PC",
        "eq_cut_ns":       0.0,
        "fit_pct":         0.8,
        "tau_min_fit_ns":  1.0,
        "slide_window_ns": 10.0,
        "slide_step_ns":   0.5,
        "n_conv_points":   50,
        "max_traj_ns":     20.0,
    },
]

OUTPUT_DIR = "/pscratch/sd/y/yuejian/observable_debug"
ANALYSES   = ["rdf", "density", "energy", "msd"]
WORKERS    = 5  # one per system — tests full parallelism
