"""Config: cell size + energy MAE + stress MAE for selected NaPF6 systems.

Student ckpt : trained on all systems, first 100 ps, 293 K
Teacher ckpt : UMA-S-1p1
Traj dt      : 100 fs/frame  →  analyze_dt_ps=100 gives stride=1000 (~200 pts/20ns)

Systems
-------
  NaPF6/DME  0.5M 298K
  NaPF6/DME  0.1M 298K
  NaPF6/DME  1M   298K
  NaPF6/Diglyme 0.1M 298K
"""
from pathlib import Path

_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5024/distillation_project/results"
    "/diffusivity_main_results_20ns_final/original_100ps"
)

_TEACHER_CKPT = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m4558/distillation_project/models/uma-s-1p1.pt"
)
_STUDENT_CKPT = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5024/distillation_project/model_checkpoints/ablation_ckpt"
    "/trained_on_all_systems"
    "/202602-1315-1942-821a-first100ps_all_salts_293K"
    "/final/inference_ckpt.pt"
)

_DT_FS = 100.0   # fs between consecutive saved frames


def _p(subdir, name):
    return str(_BASE / subdir / name / f"{name}.traj")


_SHARED = dict(
    dt_fs                    = _DT_FS,
    max_traj_ns              = 20.0,
    model_colors             = {"student": "#ff7f0e"},
    # energy_mae
    energy_mae_teacher_ckpt  = _TEACHER_CKPT,
    energy_mae_student_ckpt  = None,     # use energy stored in traj
    energy_mae_analyze_dt_ps = 100.0,    # 100 ps → stride 1000 → ~200 eval pts
    # stress_mae
    stress_mae_teacher_ckpt  = _TEACHER_CKPT,
    stress_mae_student_ckpt  = None,     # use stress stored in traj
    stress_mae_analyze_dt_ps = 100.0,
    # cell_size
    cell_size_analyze_dt_ps  = 100.0,
)

_01M = "20ns_solvent_0_1M/298K"
_05M = "20ns_solvent_solute_0.5M/298_2K"
_1M  = "20ns_solute_solvent_1M/298K"

SYSTEMS = [
    {**_SHARED, "name": "NaPF6/DME 0.5M 298K",
     "traj_paths": {"student": _p(_05M, "md_omol_napf6_dme_re1")}},

    {**_SHARED, "name": "NaPF6/DME 0.1M 298K",
     "traj_paths": {"student": _p(_01M, "md_omol_napf6_dme_re1")}},

    {**_SHARED, "name": "NaPF6/DME 1M 298K",
     "traj_paths": {"student": _p(_1M, "napf6_dme")}},

    {**_SHARED, "name": "NaPF6/Diglyme 0.1M 298K",
     "traj_paths": {"student": _p(_01M, "md_omol_napf6_diglyme_pfactor_0.1_1fs")}},
]

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/analysis/cell_energy_stress_mae/original_100ps_napf6"
)
ANALYSES = ["cell_size", "energy_mae", "stress_mae"]
WORKERS  = 1   # GPU-bound; process one system at a time
