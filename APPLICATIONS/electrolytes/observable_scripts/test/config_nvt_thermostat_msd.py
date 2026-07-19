"""Config: NVT thermostat comparison — MSD analysis only.

Trajectories are actively running (do NOT write to them).
All traj paths are opened read-only via ASE Trajectory default mode.

Systems: LiPF6/DME 323K 0.5M, NaOTf/DME 298K 0.1M,
         NaPF6/DME 298K 0.1M, NaPF6/DME 323K 0.5M.
Each system compares berendsen_teacher / nh_teacher / nh_student.

dt_fs  = 100 fs  (1 fs MD timestep, write interval = 100 steps)
Current traj length: ~11k frames ≈ 1.1 ns (simulations targeting 20 ns).
"""
from pathlib import Path

_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/distort_temp_md_nvt/simulations/nvt"
)

_DT_FS   = 100.0   # fs between consecutive saved frames
_MAX_NS  = 20.0    # cap — simulations target 20 ns total

_SHARED_MSD = dict(
    dt_fs              = _DT_FS,
    max_traj_ns        = _MAX_NS,
    n_frames           = 2000,
    # MSD — tuned for ~1 ns trajectories (growing toward 20 ns)
    eq_cut_ns          = 0.0,
    fit_pct            = 0.8,
    tau_min_fit_ns     = 0.1,
    slide_window_ns    = 0.2,
    slide_step_ns      = 0.02,
    n_conv_points      = 50,
)

_COLORS = {
    "berendsen_teacher": "#1f77b4",
    "nh_teacher":        "#ff7f0e",
    "nh_student":        "#2ca02c",
}


def _p(thermostat: str, name: str) -> str:
    return str(_BASE / thermostat / name / f"{name}.traj")


def _sys(label: str, cat: str, anion: str, name_base: str) -> dict:
    """Build one system entry with three models."""
    b_name = f"{name_base}_teacher"
    nh_t   = f"{name_base}_teacher"
    nh_s   = f"{name_base}_student"
    return {
        **_SHARED_MSD,
        "name": label,
        "cat_symbol":    cat,
        "anion_symbol":  anion,
        "solvent_symbol": "DME",
        "traj_paths": {
            "berendsen_teacher": _p("berendsen",  b_name),
            "nh_teacher":        _p("nose_hoover", nh_t),
            "nh_student":        _p("nose_hoover", nh_s),
        },
        "model_colors": _COLORS,
    }


SYSTEMS = [
    _sys("LiPF6/DME 323K 0.5M", "Li",  "PF6", "li_pf6_dme_323K_0.5M"),
    _sys("NaOTf/DME 298K 0.1M", "Na",  "OTf", "na_otf_dme_298K_0.1M"),
    _sys("NaPF6/DME 298K 0.1M", "Na",  "PF6", "na_pf6_dme_298K_0.1M"),
    _sys("NaPF6/DME 323K 0.5M", "Na",  "PF6", "na_pf6_dme_323K_0.5M"),
]

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/distort_temp_md_nvt/analysis/msd"
)
ANALYSES = ["msd"]
WORKERS  = 4
