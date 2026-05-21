"""Full comparison config: OPLS vs original student vs micro student, all concentrations.

Groups
------
  OPLS     : GROMACS .xtc — 1M/298K only
  original : ASE .traj    — 0.1M/298K, 0.5M/273K+298K+323K, 1M/298K
  micro    : ASE .traj    — 0.5M/273K+298K+323K, 1M/298K  (no 0.1M)

WARNING: GROMACS (.xtc) format is recognised but the compute modules use
ASE internally — OPLS analyses will log an error and be skipped; student
model results within the same system will still be produced.
"""
from pathlib import Path

_OPLS  = Path("/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/results/opls_baseline/tpr_files_1M")
_ORIG  = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/other_fix_run/original_100ps")
_MICRO = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/other_fix_run/micro_acas_rest")

_ORIG_1M  = _ORIG  / "20ns_solute_solvent_1M/298K"
_ORIG_01M = _ORIG  / "20ns_solvent_0_1M/298K"
_ORIG_05M = _ORIG  / "20ns_solvent_solute_0.5M"
_MICRO_1M = _MICRO / "20ns_solute_solvent_1M/298K"
_MICRO_05M= _MICRO / "20ns_solvent_solute_0.5M"

_COLORS = {
    "OPLS":     "#2ca02c",   # green
    "original": "#ff7f0e",   # orange
    "micro":    "#1f77b4",   # blue
}

_SHARED = {
    # ── trajectory ────────────────────────────────────────────────────────────
    "dt_fs":          100.0,   # frame spacing in femtoseconds (all models)
    "max_traj_ns":    20.0,    # cap analysis at 20 ns; falls back to actual if shorter

    # ── RDF ───────────────────────────────────────────────────────────────────
    "skip_ns":        0.1,     # ns to skip from traj start before RDF/density
    "window_ns":      10.0,    # RDF window width (ns); also sliding window width
    "n_frames":       2000,    # frames sampled per window
    "sliding_window": True,    # enable sliding-window g(r) and n(r) plots
    "rdf_slide_ns":   1.0,     # step between sliding windows (ns)
                               # → 10 windows: [0.1-10.1],[1.1-11.1],...,[10.1-20.0]

    # ── density ───────────────────────────────────────────────────────────────
    "density_roll_window_ns": 0.5,  # rolling avg width for density timeseries (ns)

    # ── MSD / diffusivity ─────────────────────────────────────────────────────
    "eq_cut_ns":      0.0,     # ns to skip from traj start before MSD time origins
    "fit_pct":        0.8,     # fit up to this fraction of max lag → ~16 ns for 20 ns traj
    "tau_min_fit_ns": 1.0,     # lower bound of linear fit; skips ballistic + cage regime
    "slide_window_ns":10.0,    # Panel 4: fixed window width for sliding D diagnostic (ns)
    "slide_step_ns":  0.5,     # Panel 4: step size (ns)
    "n_conv_points":  200,     # resolution of D vs tau_max sweep (Panels 2/3)

    "model_colors":   _COLORS,
}

# ── 1M / 298K ─────────────────────────────────────────────────────────────────

_1M = [
    {
        **_SHARED,
        "name":       "NaOTf/DME 1M 298K",
        "traj_paths": {
            "OPLS":     {"xtc": str(_OPLS / "npt_1M_naotf_dme.xtc"),
                         "tpr": str(_OPLS / "npt_1M_naotf_dme.tpr")},
            "original": str(_ORIG_1M  / "naotf_dme/naotf_dme.traj"),
            "micro":    str(_MICRO_1M / "naotf_dme/naotf_dme.traj"),
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
            "OPLS":     {"xtc": str(_OPLS / "npt_1M_napf6_dme.xtc"),
                         "tpr": str(_OPLS / "npt_1M_napf6_dme.tpr")},
            "original": str(_ORIG_1M  / "napf6_dme/napf6_dme.traj"),
            "micro":    str(_MICRO_1M / "napf6_dme/napf6_dme.traj"),
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
            "OPLS":     {"xtc": str(_OPLS / "npt_1M_naotf_diglyme.xtc"),
                         "tpr": str(_OPLS / "npt_1M_naotf_diglyme.tpr")},
            "original": str(_ORIG_1M  / "naotf_diglyme/naotf_diglyme.traj"),
            "micro":    str(_MICRO_1M / "naotf_diglyme/naotf_diglyme.traj"),
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
            "OPLS":     {"xtc": str(_OPLS / "npt_1M_napf6_diglyme.xtc"),
                         "tpr": str(_OPLS / "npt_1M_napf6_diglyme.tpr")},
            "original": str(_ORIG_1M  / "md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj"),
            "micro":    str(_MICRO_1M / "md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj"),
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
            "OPLS":     {"xtc": str(_OPLS / "npt_1M_napf6_pc.xtc"),
                         "tpr": str(_OPLS / "npt_1M_napf6_pc.tpr")},
            "original": str(_ORIG_1M  / "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj"),
            "micro":    str(_MICRO_1M / "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj"),
        },
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "PC",
    },
]

# ── 0.1M / 298K  (original only — no micro, no OPLS) ─────────────────────────

_01M = [
    {
        **_SHARED,
        "name":       "NaOTf/DME 0.1M 298K",
        "traj_paths": {
            "original": str(_ORIG_01M / "md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj"),
        },
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":       "NaOTf/Diglyme 0.1M 298K",
        "traj_paths": {
            "original": str(_ORIG_01M / "md_omol_naotf_diglyme_1m_s1p1/md_omol_naotf_diglyme_1m_s1p1.traj"),
        },
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "Diglyme",
    },
    {
        **_SHARED,
        "name":       "NaOTf/PC 0.1M 298K",
        "traj_paths": {
            "original": str(_ORIG_01M / "md_omol_naotf_pc_1m_s1p1/md_omol_naotf_pc_1m_s1p1.traj"),
        },
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "PC",
    },
    {
        **_SHARED,
        "name":       "NaOTf/TGDME 0.1M 298K",
        "traj_paths": {
            "original": str(_ORIG_01M / "md_omol_naotf_tgdme_1m_s1p1/md_omol_naotf_tgdme_1m_s1p1.traj"),
        },
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "TGDME",
    },
    {
        **_SHARED,
        "name":       "NaPF6/DME 0.1M 298K",
        "traj_paths": {
            "original": str(_ORIG_01M / "md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj"),
        },
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":       "NaPF6/Diglyme 0.1M 298K",
        "traj_paths": {
            "original": str(_ORIG_01M / "md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj"),
        },
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "Diglyme",
    },
    {
        **_SHARED,
        "name":       "NaPF6/PC 0.1M 298K",
        "traj_paths": {
            "original": str(_ORIG_01M / "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj"),
        },
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "PC",
    },
]

# ── 0.5M / multi-temperature  (original + micro) ──────────────────────────────

def _05m_system(name, temp, traj_stem, rdf_pairs, cat, anion, solvent):
    return {
        **_SHARED,
        "name":       f"{name} 0.5M {temp}",
        "traj_paths": {
            "original": str(_ORIG_05M  / temp / traj_stem / f"{traj_stem.split('/')[-1]}.traj"),
            "micro":    str(_MICRO_05M / temp / traj_stem / f"{traj_stem.split('/')[-1]}.traj"),
        },
        "rdf_pairs":      rdf_pairs,
        "cat_symbol":     cat,
        "anion_symbol":   anion,
        "solvent_symbol": solvent,
    }

_05M = []
for _temp in ["273_2K", "298_2K", "323_2K"]:
    _05M += [
        _05m_system("LiPF6/DME", _temp,
                    "md_omol_lipf6_pfactor_0.1_1fs_mask_t",
                    [("Li", "O"), ("Li", "F")], "Li", "PF6", "DME"),
        _05m_system("NaPF6/DME", _temp,
                    "md_omol_napf6_dme_re1",
                    [("Na", "O"), ("Na", "F")], "Na", "PF6", "DME"),
    ]

# ── assemble all systems ───────────────────────────────────────────────────────

SYSTEMS = _1M + _01M + _05M   # 5 + 7 + 6 = 18 systems total

OUTPUT_DIR = "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/analysis/20ns"
ANALYSES   = ["rdf", "density", "energy", "msd"]
WORKERS    = 18   # one process per system → all 18 run simultaneously
