"""Config: NVT nose-Hoover teacher trajectories — 4 systems, first 1.7 ns.

Systems
-------
  LiPF6/DME 323K 0.5M  (teacher)
  NaOTf/DME 298K 0.1M  (teacher)
  NaPF6/DME 298K 0.1M  (teacher)
  NaPF6/DME 323K 0.5M  (teacher)

Trajectory dt: 100 fs/frame  (~1.75 ns available; capped at 1.7 ns).

Analyses: energy, energy_mae, force_mae, msd, rdf, pressure
  - force/energy MAE: teacher = UMA-S-1p1, student = stored in traj (None)
  - analyze_dt_ps = 100 → stride = 1000 → ~17 eval points over 1.7 ns
"""
from pathlib import Path

_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/distort_temp_md_nvt/simulations/nvt/nose_hoover"
)

_TEACHER_CKPT = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m4558/distillation_project/models/uma-s-1p1.pt"
)

_DT_FS      = 100.0   # fs between consecutive saved frames
_MAX_NS     = 1.7     # analyse only the first 1.7 ns
_ANALYZE_DT = 100.0   # ps between evaluation frames → ~17 pts over 1.7 ns


def _p(name: str) -> str:
    return str(_BASE / name / f"{name}.traj")


_SHARED = dict(
    dt_fs                    = _DT_FS,
    max_traj_ns              = _MAX_NS,
    n_frames                 = 2000,
    # RDF / density windowing
    skip_ns                  = 0.1,
    window_ns                = 1.5,
    density_roll_window_ns   = 0.3,
    # energy (stored in traj, no inference needed)
    # force_mae
    force_mae_teacher_ckpt   = _TEACHER_CKPT,
    force_mae_student_ckpt   = None,
    force_mae_analyze_dt_ps  = _ANALYZE_DT,
    # energy_mae
    energy_mae_teacher_ckpt  = _TEACHER_CKPT,
    energy_mae_student_ckpt  = None,
    energy_mae_analyze_dt_ps = _ANALYZE_DT,
    # pressure
    pressure_analyze_dt_ps   = _ANALYZE_DT,
    # msd — scaled for ~1.7 ns trajectories
    eq_cut_ns                = 0.0,
    fit_pct                  = 0.8,
    tau_min_fit_ns           = 0.2,
    slide_window_ns          = 0.5,
    slide_step_ns            = 0.05,
    n_conv_points            = 50,
)

_COLOR = "#1f77b4"   # single model per system — use one colour

SYSTEMS = [
    {
        **_SHARED,
        "name":           "LiPF6/DME 323K 0.5M teacher",
        "traj_paths":     {"teacher": _p("li_pf6_dme_323K_0.5M_teacher")},
        "model_colors":   {"teacher": _COLOR},
        "rdf_pairs":      [("Li", "O"), ("Li", "F")],
        "cat_symbol":     "Li",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaOTf/DME 298K 0.1M teacher",
        "traj_paths":     {"teacher": _p("na_otf_dme_298K_0.1M_teacher")},
        "model_colors":   {"teacher": _COLOR},
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaPF6/DME 298K 0.1M teacher",
        "traj_paths":     {"teacher": _p("na_pf6_dme_298K_0.1M_teacher")},
        "model_colors":   {"teacher": _COLOR},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaPF6/DME 323K 0.5M teacher",
        "traj_paths":     {"teacher": _p("na_pf6_dme_323K_0.5M_teacher")},
        "model_colors":   {"teacher": _COLOR},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
]

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/distort_temp_md_nvt/analysis/nvt_ckpt_1_5"
)
ANALYSES = ["energy", "energy_mae", "force_mae", "msd", "rdf", "pressure"]
WORKERS  = 1   # GPU-bound (force_mae / energy_mae); serialize to avoid OOM
