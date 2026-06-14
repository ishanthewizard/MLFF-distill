"""Config: MSD analysis for berendsen teacher trajectories (~1.8 ns).

Systems
-------
  LiPF6/DME 323K 0.5M
  NaOTf/DME 298K 0.1M
  NaPF6/DME 298K 0.1M
  NaPF6/DME 323K 0.5M

dt_fs = 100 fs; traj length ~1.8 ns.
"""
from pathlib import Path

_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/distort_temp_md_nvt/simulations/nvt/berendsen"
)

_DT_FS  = 100.0   # fs between consecutive saved frames
_MAX_NS = 2.0     # cap at 2 ns

_SHARED = dict(
    dt_fs           = _DT_FS,
    max_traj_ns     = _MAX_NS,
    n_frames        = 2000,
    eq_cut_ns       = 0.0,
    fit_pct         = 0.8,
    tau_min_fit_ns  = 0.1,
    slide_window_ns = 0.5,
    slide_step_ns   = 0.05,
    n_conv_points   = 100,
)


def _sys(label: str, cat: str, anion: str, name_base: str) -> dict:
    traj = str(_BASE / f"{name_base}_teacher" / f"{name_base}_teacher.traj")
    return {
        **_SHARED,
        "name":           label,
        "cat_symbol":     cat,
        "anion_symbol":   anion,
        "solvent_symbol": "DME",
        "traj_paths":     {"teacher": traj},
        "model_colors":   {"teacher": "#1f77b4"},
    }


SYSTEMS = [
    _sys("LiPF6/DME 323K 0.5M", "Li", "PF6", "li_pf6_dme_323K_0.5M"),
    _sys("NaOTf/DME 298K 0.1M", "Na", "OTf", "na_otf_dme_298K_0.1M"),
    _sys("NaPF6/DME 298K 0.1M", "Na", "PF6", "na_pf6_dme_298K_0.1M"),
    _sys("NaPF6/DME 323K 0.5M", "Na", "PF6", "na_pf6_dme_323K_0.5M"),
]

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/distort_temp_md_nvt/analysis/msd_berendsen_teacher"
)
ANALYSES = ["msd"]
WORKERS  = 4
