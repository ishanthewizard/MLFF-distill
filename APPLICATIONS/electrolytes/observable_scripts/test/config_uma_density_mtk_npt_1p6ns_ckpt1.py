"""Config: MSD + cell_size + density for UMA isotropic MTK NPT (~1.6 ns).

Systems
-------
  LiPF6/DME 323K 0.5M   : teacher, ckpt1
  NaOTf/DME 298K 0.1M   : teacher, ckpt1
  NaPF6/DME 298K 0.1M   : teacher, ckpt1
  NaPF6/DME 323K 0.5M   : teacher, ckpt1

Trajectory dt: 100 fs/frame (~1.6 ns, ~16 000 frames).
"""
from pathlib import Path

_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/uma_density_isotropic_mtk_npt"
)

_DT_FS  = 100.0
_MAX_NS = 1.6

_COLORS = {
    "teacher": "#1f77b4",
    "ckpt1":   "#ff7f0e",
}

_SHARED = dict(
    dt_fs                  = _DT_FS,
    max_traj_ns            = _MAX_NS,
    model_colors           = _COLORS,
    # density
    density_roll_window_ns = 0.2,
    # cell_size
    cell_size_analyze_dt_ps = 100.0,
    # msd
    n_frames               = 1000,
    eq_cut_ns              = 0.0,
    fit_pct                = 0.8,
    tau_min_fit_ns         = 0.1,
    slide_window_ns        = 0.4,
    slide_step_ns          = 0.05,
    n_conv_points          = 100,
)


def _paths(name_base: str) -> dict:
    return {
        "teacher": str(_BASE / f"{name_base}_teacher" / f"{name_base}_teacher.traj"),
        "ckpt1":   str(_BASE / f"{name_base}_student" / f"{name_base}_student.traj"),
    }


SYSTEMS = [
    {
        **_SHARED,
        "name":           "LiPF6/DME 323K 0.5M",
        "traj_paths":     _paths("li_pf6_dme_323K_0.5M"),
        "cat_symbol":     "Li",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaOTf/DME 298K 0.1M",
        "traj_paths":     _paths("na_otf_dme_298K_0.1M"),
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaPF6/DME 298K 0.1M",
        "traj_paths":     _paths("na_pf6_dme_298K_0.1M"),
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaPF6/DME 323K 0.5M",
        "traj_paths":     _paths("na_pf6_dme_323K_0.5M"),
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
]

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/uma_density_isotropic_mtk_npt/analysis"
)
ANALYSES = ["msd", "cell_size", "density"]
WORKERS  = 4
