"""Config: ablation 50ps original student — all systems.

Same systems as config_20ns_origin.py but trajectories are saved every 10 fs/frame
(vs 100 fs/frame previously), so dt_fs = 10.0.  Total analysis window: 20 ns.
"""
from pathlib import Path

_BASE  = Path("/global/homes/y/yuejian/project/MLFF-distill/m5250/electrolyte_application/ablate_distillation/ablation_diffusivity")
_SS1M  = _BASE / "20ns_solute_solvent_1M"
_S01M  = _BASE / "20ns_solvent_0_1M"
_SM05  = _BASE / "20ns_solvent_solute_0.5M"

_SHARED = {
    "model_colors":            {"ablation_50ps": "#2ca02c"},
    "dt_fs":                   10.0,        # 10 fs per frame (was 100 fs)
    "max_traj_ns":             20.0,
    "fallback_to_traj_length": True,
    "skip_ns":                 0.1,
    "window_ns":               10.0,
    "n_frames":                2000,
    "sliding_window":          True,
    "rdf_slide_ns":            1.0,
    "density_roll_window_ns":  0.5,
    "eq_cut_ns":               0.0,
    "fit_pct":                 0.8,
    "tau_min_fit_ns":          1.0,
    "slide_window_ns":         10.0,
    "slide_step_ns":           0.5,
    "n_conv_points":           200,
}

def _p(base, name):
    return str(base / name / f"{name}.traj")

SYSTEMS = [
    # ── solute+solvent 1M 298K ──────────────────────────────────────────────
    {
        **_SHARED,
        "name":           "NaOTf/DME 1M 298K",
        "traj_paths":     {"ablation_50ps": _p(_SS1M, "naotf_dme")},
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaPF6/DME 1M 298K",
        "traj_paths":     {"ablation_50ps": _p(_SS1M, "napf6_dme")},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaOTf/Diglyme 1M 298K",
        "traj_paths":     {"ablation_50ps": _p(_SS1M, "naotf_diglyme")},
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "Diglyme",
    },
    {
        **_SHARED,
        "name":           "NaPF6/Diglyme 1M 298K",
        "traj_paths":     {"ablation_50ps": _p(_SS1M, "md_omol_napf6_diglyme_pfactor_0.1_1fs")},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "Diglyme",
    },
    {
        **_SHARED,
        "name":           "NaPF6/PC 1M 298K",
        "traj_paths":     {"ablation_50ps": _p(_SS1M, "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1")},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "PC",
    },

    # ── 0.1M 298K ───────────────────────────────────────────────────────────
    {
        **_SHARED,
        "name":           "NaOTf/Diglyme 0.1M 298K",
        "traj_paths":     {"ablation_50ps": _p(_S01M, "md_omol_naotf_diglyme_1m_s1p1")},
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "Diglyme",
    },
    {
        **_SHARED,
        "name":           "NaOTf/DME 0.1M 298K",
        "traj_paths":     {"ablation_50ps": _p(_S01M, "md_omol_naotf_dme_s1p1_omol")},
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaOTf/PC 0.1M 298K",
        "traj_paths":     {"ablation_50ps": _p(_S01M, "md_omol_naotf_pc_1m_s1p1")},
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "PC",
    },
    {
        **_SHARED,
        "name":           "NaOTf/TGDME 0.1M 298K",
        "traj_paths":     {"ablation_50ps": _p(_S01M, "md_omol_naotf_tgdme_1m_s1p1")},
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "TGDME",
    },
    {
        **_SHARED,
        "name":           "NaPF6/Diglyme 0.1M 298K",
        "traj_paths":     {"ablation_50ps": _p(_S01M, "md_omol_napf6_diglyme_pfactor_0.1_1fs")},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "Diglyme",
    },
    {
        **_SHARED,
        "name":           "NaPF6/DME 0.1M 298K",
        "traj_paths":     {"ablation_50ps": _p(_S01M, "md_omol_napf6_dme_re1")},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaPF6/PC 0.1M 298K",
        "traj_paths":     {"ablation_50ps": _p(_S01M, "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1")},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "PC",
    },

    # ── 0.5M — 273.2 K ──────────────────────────────────────────────────────
    {
        **_SHARED,
        "name":           "LiPF6 0.5M 273K",
        "traj_paths":     {"ablation_50ps": _p(_SM05 / "273_2K", "md_omol_lipf6_pfactor_0.1_1fs_mask_t")},
        "rdf_pairs":      [("Li", "O"), ("Li", "F")],
        "cat_symbol":     "Li",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaPF6/DME 0.5M 273K",
        "traj_paths":     {"ablation_50ps": _p(_SM05 / "273_2K", "md_omol_napf6_dme_re1")},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },

    # ── 0.5M — 298.2 K ──────────────────────────────────────────────────────
    {
        **_SHARED,
        "name":           "LiPF6 0.5M 298K",
        "traj_paths":     {"ablation_50ps": _p(_SM05 / "298_2K", "md_omol_lipf6_pfactor_0.1_1fs_mask_t")},
        "rdf_pairs":      [("Li", "O"), ("Li", "F")],
        "cat_symbol":     "Li",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaPF6/DME 0.5M 298K",
        "traj_paths":     {"ablation_50ps": _p(_SM05 / "298_2K", "md_omol_napf6_dme_re1")},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },

    # ── 0.5M — 323.2 K ──────────────────────────────────────────────────────
    {
        **_SHARED,
        "name":           "LiPF6 0.5M 323K",
        "traj_paths":     {"ablation_50ps": _p(_SM05 / "323_2K", "md_omol_lipf6_pfactor_0.1_1fs_mask_t")},
        "rdf_pairs":      [("Li", "O"), ("Li", "F")],
        "cat_symbol":     "Li",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaPF6/DME 0.5M 323K",
        "traj_paths":     {"ablation_50ps": _p(_SM05 / "323_2K", "md_omol_napf6_dme_re1")},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
]

OUTPUT_DIR = "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/results/analysis/ablation_50ps"
ANALYSES   = ["rdf", "density", "energy", "msd"]
WORKERS    = 36
