#!/usr/bin/env python3
"""Config for conductivity analysis of 18 electrolyte systems (20 ns trajs).

Two model groups:
  micro    -> micro_acas_50ps  (dt = 50 fs, but stored at 100 fs intervals)
  original -> original_100ps   (dt = 100 fs)

Both groups share identical relative path structure under their respective
base directories.
"""

_MICRO_BASE    = "/global/cfs/cdirs/m5024/distillation_project/results/diffusivity_main_results_20ns_final/micro_acas_50ps"
_ORIG_BASE     = "/global/cfs/cdirs/m5024/distillation_project/results/diffusivity_main_results_20ns_final/original_100ps"

OUTPUT_DIR = "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/conductivity_results/eval_runs"
ANALYSES   = ["conductivity"]
WORKERS    = 4


def _traj(base, *parts):
    """Build traj path.
    Usage: _traj(base, subdir1, subdir2, dir_name, stem)
    Returns: base/subdir1/subdir2/dir_name/stem.traj
    Typically dir_name == stem for these trajectories.
    """
    import os
    return os.path.join(base, *parts[:-1], f"{parts[-1]}.traj")


# Experimental conductivities (all converted from µS/cm to mS/cm, i.e. / 1000)
# None means no experimental data available
_EXP = {
    # 1M systems
    "napf6_dme_1M_298K":        None,          # NaPF6 1M DME: no data
    "naotf_dme_1M_298K":        1234.5 / 1000, # 1.2345 mS/cm
    "naotf_diglyme_1M_298K":    2793.0 / 1000,
    "napf6_diglyme_1M_298K":    6701.6 / 1000,
    "napf6_pc_1M_298K":         6495.9 / 1000,
    # 0.1M systems
    "napf6_dme_0.1M_298K":      613.1  / 1000,
    "naotf_dme_0.1M_298K":      46.1   / 1000,
    "naotf_diglyme_0.1M_298K":  114.5  / 1000,
    "napf6_diglyme_0.1M_298K":  518.2  / 1000,
    "napf6_pc_0.1M_298K":       1952.6 / 1000,
    "naotf_pc_0.1M_298K":       1516.9 / 1000,
    "naotf_tgdme_0.1M_298K":    91.2   / 1000,
    # 0.5M temperature-dependent
    "lipf6_dme_0.5M_273K":      None,          # no experimental data at 0°C
    "napf6_dme_0.5M_273K":      None,          # no experimental data at 0°C
    "lipf6_dme_0.5M_298K":      5676.0 / 1000,
    "napf6_dme_0.5M_298K":      6377.5 / 1000,
    "lipf6_dme_0.5M_323K":      4407.1 / 1000,
    "napf6_dme_0.5M_323K":      4052.2 / 1000,
}


SYSTEMS = [

    # ── 1 M systems ────────────────────────────────────────────────────────────

    {
        "name": "NaPF6 DME 1M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solute_solvent_1M", "298K", "napf6_dme", "napf6_dme"),
            "original": _traj(_ORIG_BASE,  "20ns_solute_solvent_1M", "298K", "napf6_dme", "napf6_dme"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "PF6",
        "solvent_symbol":   "DME",
        "conductivity_T_K": 298.0,
        "conductivity_exp_mS_cm": _EXP["napf6_dme_1M_298K"],
    },

    {
        "name": "NaOTf DME 1M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solute_solvent_1M", "298K", "naotf_dme", "naotf_dme"),
            "original": _traj(_ORIG_BASE,  "20ns_solute_solvent_1M", "298K", "naotf_dme", "naotf_dme"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "OTf",
        "solvent_symbol":   "DME",
        "conductivity_T_K": 298.0,
        "conductivity_exp_mS_cm": _EXP["naotf_dme_1M_298K"],
    },

    {
        "name": "NaOTf Diglyme 1M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solute_solvent_1M", "298K", "naotf_diglyme", "naotf_diglyme"),
            "original": _traj(_ORIG_BASE,  "20ns_solute_solvent_1M", "298K", "naotf_diglyme", "naotf_diglyme"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "OTf",
        "solvent_symbol":   "Diglyme",
        "conductivity_T_K": 298.0,
        "conductivity_exp_mS_cm": _EXP["naotf_diglyme_1M_298K"],
    },

    {
        "name": "NaPF6 Diglyme 1M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solute_solvent_1M", "298K",
                              "md_omol_napf6_diglyme_pfactor_0.1_1fs",
                              "md_omol_napf6_diglyme_pfactor_0.1_1fs"),
            "original": _traj(_ORIG_BASE,  "20ns_solute_solvent_1M", "298K",
                              "md_omol_napf6_diglyme_pfactor_0.1_1fs",
                              "md_omol_napf6_diglyme_pfactor_0.1_1fs"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "PF6",
        "solvent_symbol":   "Diglyme",
        "conductivity_T_K": 298.0,
        "conductivity_exp_mS_cm": _EXP["napf6_diglyme_1M_298K"],
    },

    {
        "name": "NaPF6 PC 1M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solute_solvent_1M", "298K",
                              "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1",
                              "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1"),
            "original": _traj(_ORIG_BASE,  "20ns_solute_solvent_1M", "298K",
                              "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1",
                              "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "PF6",
        "solvent_symbol":   "PC",
        "conductivity_T_K": 298.0,
        "conductivity_exp_mS_cm": _EXP["napf6_pc_1M_298K"],
    },

    # ── 0.1 M systems ──────────────────────────────────────────────────────────

    {
        "name": "NaPF6 DME 0.1M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solvent_0_1M", "298K",
                              "md_omol_napf6_dme_re1", "md_omol_napf6_dme_re1"),
            "original": _traj(_ORIG_BASE,  "20ns_solvent_0_1M", "298K",
                              "md_omol_napf6_dme_re1", "md_omol_napf6_dme_re1"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "PF6",
        "solvent_symbol":   "DME",
        "conductivity_T_K": 298.0,
        "conductivity_exp_mS_cm": _EXP["napf6_dme_0.1M_298K"],
    },

    {
        "name": "NaOTf DME 0.1M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solvent_0_1M", "298K",
                              "md_omol_naotf_dme_s1p1_omol", "md_omol_naotf_dme_s1p1_omol"),
            "original": _traj(_ORIG_BASE,  "20ns_solvent_0_1M", "298K",
                              "md_omol_naotf_dme_s1p1_omol", "md_omol_naotf_dme_s1p1_omol"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "OTf",
        "solvent_symbol":   "DME",
        "conductivity_T_K": 298.0,
        "conductivity_exp_mS_cm": _EXP["naotf_dme_0.1M_298K"],
    },

    {
        "name": "NaOTf Diglyme 0.1M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solvent_0_1M", "298K",
                              "md_omol_naotf_diglyme_1m_s1p1", "md_omol_naotf_diglyme_1m_s1p1"),
            "original": _traj(_ORIG_BASE,  "20ns_solvent_0_1M", "298K",
                              "md_omol_naotf_diglyme_1m_s1p1", "md_omol_naotf_diglyme_1m_s1p1"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "OTf",
        "solvent_symbol":   "Diglyme",
        "conductivity_T_K": 298.0,
        "conductivity_exp_mS_cm": _EXP["naotf_diglyme_0.1M_298K"],
    },

    {
        "name": "NaPF6 Diglyme 0.1M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solvent_0_1M", "298K",
                              "md_omol_napf6_diglyme_pfactor_0.1_1fs",
                              "md_omol_napf6_diglyme_pfactor_0.1_1fs"),
            "original": _traj(_ORIG_BASE,  "20ns_solvent_0_1M", "298K",
                              "md_omol_napf6_diglyme_pfactor_0.1_1fs",
                              "md_omol_napf6_diglyme_pfactor_0.1_1fs"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "PF6",
        "solvent_symbol":   "Diglyme",
        "conductivity_T_K": 298.0,
        "conductivity_exp_mS_cm": _EXP["napf6_diglyme_0.1M_298K"],
    },

    {
        "name": "NaPF6 PC 0.1M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solvent_0_1M", "298K",
                              "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1",
                              "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1"),
            "original": _traj(_ORIG_BASE,  "20ns_solvent_0_1M", "298K",
                              "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1",
                              "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "PF6",
        "solvent_symbol":   "PC",
        "conductivity_T_K": 298.0,
        "conductivity_exp_mS_cm": _EXP["napf6_pc_0.1M_298K"],
    },

    {
        "name": "NaOTf PC 0.1M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solvent_0_1M", "298K",
                              "md_omol_naotf_pc_1m_s1p1", "md_omol_naotf_pc_1m_s1p1"),
            "original": _traj(_ORIG_BASE,  "20ns_solvent_0_1M", "298K",
                              "md_omol_naotf_pc_1m_s1p1", "md_omol_naotf_pc_1m_s1p1"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "OTf",
        "solvent_symbol":   "PC",
        "conductivity_T_K": 298.0,
        "conductivity_exp_mS_cm": _EXP["naotf_pc_0.1M_298K"],
    },

    {
        "name": "NaOTf TGDME 0.1M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solvent_0_1M", "298K",
                              "md_omol_naotf_tgdme_1m_s1p1", "md_omol_naotf_tgdme_1m_s1p1"),
            "original": _traj(_ORIG_BASE,  "20ns_solvent_0_1M", "298K",
                              "md_omol_naotf_tgdme_1m_s1p1", "md_omol_naotf_tgdme_1m_s1p1"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "OTf",
        "solvent_symbol":   "TGDME",
        "conductivity_T_K": 298.0,
        "conductivity_exp_mS_cm": _EXP["naotf_tgdme_0.1M_298K"],
    },

    # ── 0.5 M temperature-dependent ────────────────────────────────────────────

    {
        "name": "LiPF6 DME 0.5M 273K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solvent_solute_0.5M", "273_2K",
                              "md_omol_lipf6_pfactor_0.1_1fs_mask_t",
                              "md_omol_lipf6_pfactor_0.1_1fs_mask_t"),
            "original": _traj(_ORIG_BASE,  "20ns_solvent_solute_0.5M", "273_2K",
                              "md_omol_lipf6_pfactor_0.1_1fs_mask_t",
                              "md_omol_lipf6_pfactor_0.1_1fs_mask_t"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Li",
        "anion_symbol":     "PF6",
        "solvent_symbol":   "DME",
        "conductivity_T_K": 273.2,
        "conductivity_exp_mS_cm": _EXP["lipf6_dme_0.5M_273K"],
    },

    {
        "name": "NaPF6 DME 0.5M 273K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solvent_solute_0.5M", "273_2K",
                              "md_omol_napf6_dme_re1", "md_omol_napf6_dme_re1"),
            "original": _traj(_ORIG_BASE,  "20ns_solvent_solute_0.5M", "273_2K",
                              "md_omol_napf6_dme_re1", "md_omol_napf6_dme_re1"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "PF6",
        "solvent_symbol":   "DME",
        "conductivity_T_K": 273.2,
        "conductivity_exp_mS_cm": _EXP["napf6_dme_0.5M_273K"],
    },

    {
        "name": "LiPF6 DME 0.5M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solvent_solute_0.5M", "298_2K",
                              "md_omol_lipf6_pfactor_0.1_1fs_mask_t",
                              "md_omol_lipf6_pfactor_0.1_1fs_mask_t"),
            "original": _traj(_ORIG_BASE,  "20ns_solvent_solute_0.5M", "298_2K",
                              "md_omol_lipf6_pfactor_0.1_1fs_mask_t",
                              "md_omol_lipf6_pfactor_0.1_1fs_mask_t"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Li",
        "anion_symbol":     "PF6",
        "solvent_symbol":   "DME",
        "conductivity_T_K": 298.2,
        "conductivity_exp_mS_cm": _EXP["lipf6_dme_0.5M_298K"],
    },

    {
        "name": "NaPF6 DME 0.5M 298K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solvent_solute_0.5M", "298_2K",
                              "md_omol_napf6_dme_re1", "md_omol_napf6_dme_re1"),
            "original": _traj(_ORIG_BASE,  "20ns_solvent_solute_0.5M", "298_2K",
                              "md_omol_napf6_dme_re1", "md_omol_napf6_dme_re1"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "PF6",
        "solvent_symbol":   "DME",
        "conductivity_T_K": 298.2,
        "conductivity_exp_mS_cm": _EXP["napf6_dme_0.5M_298K"],
    },

    {
        "name": "LiPF6 DME 0.5M 323K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solvent_solute_0.5M", "323_2K",
                              "md_omol_lipf6_pfactor_0.1_1fs_mask_t",
                              "md_omol_lipf6_pfactor_0.1_1fs_mask_t"),
            "original": _traj(_ORIG_BASE,  "20ns_solvent_solute_0.5M", "323_2K",
                              "md_omol_lipf6_pfactor_0.1_1fs_mask_t",
                              "md_omol_lipf6_pfactor_0.1_1fs_mask_t"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Li",
        "anion_symbol":     "PF6",
        "solvent_symbol":   "DME",
        "conductivity_T_K": 323.2,
        "conductivity_exp_mS_cm": _EXP["lipf6_dme_0.5M_323K"],
    },

    {
        "name": "NaPF6 DME 0.5M 323K",
        "traj_paths": {
            "micro":    _traj(_MICRO_BASE, "20ns_solvent_solute_0.5M", "323_2K",
                              "md_omol_napf6_dme_re1", "md_omol_napf6_dme_re1"),
            "original": _traj(_ORIG_BASE,  "20ns_solvent_solute_0.5M", "323_2K",
                              "md_omol_napf6_dme_re1", "md_omol_napf6_dme_re1"),
        },
        "dt_fs":            100.0,
        "cat_symbol":       "Na",
        "anion_symbol":     "PF6",
        "solvent_symbol":   "DME",
        "conductivity_T_K": 323.2,
        "conductivity_exp_mS_cm": _EXP["napf6_dme_0.5M_323K"],
    },
]
