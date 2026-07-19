#!/usr/bin/env python3
"""Conductivity (Onsager / Nernst-Einstein) for ONE PAINN electrolyte system — TEST.

System: LiPF6 / DME, 0.5 M, 273 K, NPT Langevin
        (PAINN FP32 In-distribution simulation, replica_1).

Trajectory is the full 20 ns despite the "1ns_1ps" label in the dirname
(200000 frames @ dt_fs=100 -> 20.0 ns; the trailing <duration>_<dtlabel> in
PAINN electrolytes_data dir names is not reliable. The run's .fennol.yaml has
dt[fs]=1.0, tdump[ps]=0.1 -> 100 fs/frame, nsteps=20000000 -> 20 ns target).
"""

_TRAJ = (
    "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
    "simulation_results/PAINN/electrolytes_data/simulation/FP32_simulation/"
    "In_distribution_exp/NPT_langevin/npt_replica_1/lipf6_dme/"
    "npt_0_5M_273K_1ns_1ps/npt_0_5M_273K_1ns_1ps.traj"
)

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/"
    "conductivity_results/painn_lipf6_dme_0p5M_273K_test"
)
ANALYSES = ["conductivity"]
WORKERS  = 1

SYSTEMS = [
    {
        "name": "LiPF6 DME 0.5M 273K (PAINN FP32 In-dist r1)",
        "traj_paths": {
            "PAINN": _TRAJ,
        },
        "dt_fs":            100.0,   # tdump = 0.1 ps -> 100 fs/frame
        "max_traj_ns":      20.0,    # eval.py caps to actual traj length (200k frames -> 20 ns)
        # ── species (lipf6 -> Li/PF6, dme -> DME) ──
        "cat_symbol":       "Li",
        "anion_symbol":     "PF6",
        "solvent_symbol":   "DME",
        # ── conductivity ──
        "conductivity_T_K":         273.0,
        "conductivity_z_cat":        1.0,
        "conductivity_z_anion":     -1.0,
        "conductivity_eq_cut_ns":    0.5,   # skip first 0.5 ns for equilibration
        "conductivity_load_dt_ps":   5.0,   # effective subsample / MSD-lag dt (ps)
        # no experimental value supplied for this test
    },
]
