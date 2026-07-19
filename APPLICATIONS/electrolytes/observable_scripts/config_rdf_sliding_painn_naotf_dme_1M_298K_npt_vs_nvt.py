#!/usr/bin/env python3
"""Sliding-window RDF for PAINN NaOTf/DME 1 M 298 K: NPT vs NVT.

Two trajectories of the same system (different ensembles) are loaded as two
models under one system, so eval.py produces:
  - per-model sliding-window RDF panels (time evolution of each g(r))
  - a static npt-vs-nvt overlay comparison g(r)

RDF element pairs (NaOTf in DME -> elements Na, O, F, S, C, H):
  Na-O : total cation solvation shell (DME oxygens + triflate oxygens)
  Na-S : cation-anion contact (S is unique to triflate)
  Na-F : cation-anion contact (F is unique to triflate, CF3 end)

PAINN electrolytes_data convention: dirname duration/dt label is unreliable;
both runs are dt_fs=100 (tdump 0.1 ps) and ~20 ns. eval.py caps max_traj_ns
to the real trajectory length.
"""

_BASE = ("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
         "simulation_results/PAINN/electrolytes_data/simulation/naotf_dme")

OUTPUT_DIR = ("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
              "simulation_results/PAINN/electrolytes_data/analysis/"
              "rdf_sliding/naotf_dme_1M_298K_npt_vs_nvt")
ANALYSES = ["rdf"]
WORKERS  = 1

SYSTEMS = [
    {
        "name": "NaOTf DME 1M 298K",
        "traj_paths": {
            "npt": f"{_BASE}/npt_1M_298K_2ns_100fs/npt_1M_298K_2ns_100fs.traj",
            "nvt": f"{_BASE}/nvt_1M_298K_20ns_100fs/nvt_1M_298K_20ns_100fs.traj",
        },
        "model_colors": {"npt": "#1f77b4", "nvt": "#d62728"},
        "dt_fs":        100.0,
        "max_traj_ns":  20.0,

        # ── RDF ──
        "rdf_pairs":      [("Na", "O"), ("Na", "S"), ("Na", "F")],
        "skip_ns":        0.5,    # equilibration skip / sliding-window start offset
        "n_frames":       1000,    # frames sampled per window
        # ── sliding window ──
        "sliding_window": True,
        "window_ns":      1.0,    # PER-WINDOW width (not the total span!)
        "rdf_slide_ns":   1.0,    # step between window starts -> ~19 windows over 0.5-20 ns
    },
]
