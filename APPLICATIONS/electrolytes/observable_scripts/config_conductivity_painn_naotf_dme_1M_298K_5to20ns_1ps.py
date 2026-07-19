#!/usr/bin/env python3
"""Conductivity (byteff2 Onsager/NE + mdcraft collective) for NaOTf/DME 1 M 298 K,
PAINN FP32 In-distribution NVT-langevin, all 4 replicas.

Per user request (2026-07):
  - displacement (MSD) window : 5 ns -> 20 ns   -> conductivity_eq_cut_ns = 5.0
                                (discard the first 5 ns of every replica)
  - byteff2 Onsager/NE fit    : 50-200 ps       -> conductivity_load_dt_ps = 1.0
      byteff2's onsager_calc() hardcodes its MSD-slope fit to lags [50,200) frames
      (independent of nt_start/nt_end; see conductivity/compute.py:271 where those
      are NOT forwarded). At 1 ps/frame those 50-200 frames == 50-200 ps, applied
      to the MSD averaged over the 5-20 ns window. conductivity_tau_min_ns=0.05 is
      set ONLY so the reported tau_*_fit_ns labels read 0.05-0.2 ns honestly; it
      does not change the computed sigma.
  - mdcraft collective-Onsager: kept at its DEFAULT 0.3-2.0 ns lag fit (tau_max
      left unset). Forcing the collective backend to 50-200 ps would fit it in the
      sub-diffusive regime; the 0.3-2.0 ns window is the reliable CSD number.

PARALLELISM: eval.py parallelizes across SYSTEMS only (replicas/models run serially
within a system). So each replica is its own system here -> WORKERS=4 runs all 4
replicas concurrently. Every system name starts with `naotf_dme__` and carries the
`1M`/`298K` tokens, so the group merge (which groups on system.split("__")[0] +
parsed concentration + T) still collapses the 4 into ONE mean +/- s.d. row.

dt_fs=100 (tdump=0.1 ps) and max_traj_ns=20 come from the .fennol.yaml, not the
dirname label `20ns_100fs`. eval.py caps max_traj_ns to each traj's real length.
"""
from pathlib import Path

_REPLICA_ROOT = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data"
    "/simulation/FP32_simulation/In_distribution_exp/NVT_langevin"
)
_RUN = "naotf_dme/nvt_1M_298K_20ns_100fs"
_TRAJ_NAME = "nvt_1M_298K_20ns_100fs.traj"

REPLICA_COLORS = {
    "replica_0": "#1f77b4",
    "replica_1": "#ff7f0e",
    "replica_2": "#2ca02c",
    "replica_3": "#d62728",
}


def _traj(r: int) -> str:
    return str(_REPLICA_ROOT / f"nvt_replica_{r}" / _RUN / _TRAJ_NAME)


# One system per replica so eval.py runs them in parallel (WORKERS=4). The merge
# script re-groups them into a single naotf_dme / 1 M / 298 K row.
SYSTEMS = []
for _r in range(4):
    _lbl = f"replica_{_r}"
    SYSTEMS.append({
        "name": f"naotf_dme__nvt_1M_298K_20ns_100fs__rep{_r}",
        "traj_paths": {_lbl: _traj(_r)},
        "model_colors": {_lbl: REPLICA_COLORS[_lbl]},
        "dt_fs": 100.0,           # tdump = 0.1 ps -> 100 fs/saved-frame
        "max_traj_ns": 20.0,      # eval.py caps to actual traj length
        "temperature_K": 298.0,
        "concentration_M": 1.0,
        # ── species (naotf -> Na/OTf, dme -> DME) ──
        "cat_symbol": "Na",
        "anion_symbol": "OTf",
        "solvent_symbol": "DME",
        # ── conductivity ──
        "conductivity_T_K": 298.0,
        "conductivity_z_cat": 1.0,
        "conductivity_z_anion": -1.0,
        "conductivity_eq_cut_ns": 5.0,     # displacement window: 5 -> 20 ns
        "conductivity_load_dt_ps": 1.0,    # 1 ps/frame -> byteff2 fit = 50-200 ps
        "conductivity_tau_min_ns": 0.05,   # honest label only (byteff2 window hardcoded)
        # conductivity_tau_max_ns intentionally UNSET -> mdcraft keeps 0.3-2.0 ns fit
    })

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data"
    "/analysis/fp32_simulation/In_distribution/multi_replicas"
    "/naotf_dme_1M_298K_cond_5to20ns_1ps"
)
ANALYSES = ["conductivity"]
WORKERS = 4   # 4 systems (one per replica) run concurrently
