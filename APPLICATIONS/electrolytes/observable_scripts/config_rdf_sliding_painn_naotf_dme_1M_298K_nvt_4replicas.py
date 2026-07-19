#!/usr/bin/env python3
"""Sliding-window RDF for PAINN NaOTf/DME 1 M 298 K (NVT Langevin), 4 replicas.

Window width 3 ns, sliding in 1 ns steps from 3 ns to 20 ns:
windows [3-6], [4-7], ..., [17-20]  ->  15 windows per replica.
(get_windows emits t..t+window while t+window <= total_ns, stepping by slide_ns.)

Each replica is a separate "model" under one system, so eval.py produces:
  - a per-replica sliding-window RDF panel (time evolution of each g(r))
  - a static 4-replica overlay g(r) per pair

RDF element pairs (NaOTf in DME -> elements Na, O, F, S, C, H):
  Na-O : cation solvation shell (DME oxygens + triflate oxygens)
  Na-S : cation-anion contact (S is unique to triflate)
  Na-F : cation-anion contact (F is unique to triflate CF3 end)

PAINN electrolytes_data convention: dirname duration/dt label is unreliable;
runs are dt_fs=100 (tdump 0.1 ps) and ~20 ns. eval.py caps max_traj_ns to the
real trajectory length.
"""

_R = ("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
      "simulation_results/PAINN/electrolytes_data/simulation/FP32_simulation/"
      "In_distribution_exp/NVT_langevin")
_SYS = "naotf_dme/nvt_1M_298K_20ns_100fs/nvt_1M_298K_20ns_100fs.traj"

OUTPUT_DIR = ("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
              "simulation_results/PAINN/electrolytes_data/analysis/"
              "rdf_sliding/naotf_dme_1M_298K_nvt_4replicas")
ANALYSES = ["rdf"]
WORKERS  = 1

SYSTEMS = [
    {
        "name": "NaOTf DME 1M 298K NVT (4 replicas)",
        "traj_paths": {
            "replica_0": f"{_R}/nvt_replica_0/{_SYS}",
            "replica_1": f"{_R}/nvt_replica_1/{_SYS}",
            "replica_2": f"{_R}/nvt_replica_2/{_SYS}",
            "replica_3": f"{_R}/nvt_replica_3/{_SYS}",
        },
        "model_colors": {
            "replica_0": "#1f77b4", "replica_1": "#d62728",
            "replica_2": "#2ca02c", "replica_3": "#9467bd",
        },
        "dt_fs":        100.0,
        "max_traj_ns":  20.0,

        # ── RDF ──
        "rdf_pairs":      [("Na", "O"), ("Na", "S"), ("Na", "F")],
        "skip_ns":        3.0,    # sliding-window start (3 ns)
        "n_frames":       1000,   # frames sampled per window (static overlay)
        # ── sliding window ──
        "sliding_window": True,
        "window_ns":      3.0,    # PER-WINDOW width (NOT the total span)
        "rdf_slide_ns":   1.0,    # step between window starts -> 15 windows over 3-20 ns
    },
]
