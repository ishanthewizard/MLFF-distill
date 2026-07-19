#!/usr/bin/env python3
"""Conductivity (Onsager / Nernst-Einstein) for one PAINN electrolyte system.

System: NaOTf / DME, 1 M, 298 K, NPT (PAINN/FENNIX electrolytes_data run).

Trajectory is the full 20 ns despite the "2ns" label in the dirname
(200000 frames @ dt_fs=100 -> 20.0 ns; the trailing <duration>_<dtlabel> in
PAINN electrolytes_data dir names is not reliable, see analyze-md notes).
"""

_TRAJ = (
    "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
    "simulation_results/PAINN/electrolytes_data/simulation/naotf_dme/"
    "npt_1M_298K_2ns_100fs/npt_1M_298K_2ns_100fs.traj"
)

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/"
    "conductivity_results/painn_naotf_dme_1M_298K"
)
ANALYSES = ["conductivity"]
WORKERS  = 1

SYSTEMS = [
    {
        "name": "NaOTf DME 1M 298K (PAINN)",
        "traj_paths": {
            "PAINN": _TRAJ,
        },
        "dt_fs":            100.0,   # tdump = 0.1 ps -> 100 fs/frame
        "max_traj_ns":      20.0,    # eval.py caps to actual traj length
        # ── species (naotf -> Na/OTf, dme -> DME) ──
        "cat_symbol":       "Na",
        "anion_symbol":     "OTf",
        "solvent_symbol":   "DME",
        # ── conductivity ──
        "conductivity_T_K":        298.0,
        "conductivity_z_cat":      1.0,
        "conductivity_z_anion":   -1.0,
        "conductivity_eq_cut_ns":  2.0,   # skip first 2 ns for equilibration (analyze 2-20 ns)
        "conductivity_exp_mS_cm":  1234.5 / 1000,  # NaOTf DME 1M 298K experiment
    },
]
