"""Config: Energy (PE, KE, total, temperature) for UMA isotropic MTK NPT trajectories.

Systems (4 chemistries, teacher + student):
  LiPF6/DME  323K 0.5M
  NaOTf/DME  298K 0.1M
  NaPF6/DME  298K 0.1M
  NaPF6/DME  323K 0.5M

Trajectory dt: 100 fs/frame (~17.4 ns available).
"""
from pathlib import Path

_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/uma_density_isotropic_mtk_npt"
)

_DT_FS  = 100.0
_MAX_NS = 20.0  # cap higher than available to use all frames

_COLORS = {
    "teacher": "#1f77b4",
    "student": "#ff7f0e",
}

_SHARED = dict(
    dt_fs        = _DT_FS,
    max_traj_ns  = _MAX_NS,
    n_frames     = 1000,
    model_colors = _COLORS,
)


def _paths(name_base: str) -> dict:
    return {
        "teacher": str(_BASE / f"{name_base}_teacher" / f"{name_base}_teacher.traj"),
        "student": str(_BASE / f"{name_base}_student" / f"{name_base}_student.traj"),
    }


SYSTEMS = [
    {**_SHARED, "name": "LiPF6/DME 323K 0.5M", "traj_paths": _paths("li_pf6_dme_323K_0.5M")},
    {**_SHARED, "name": "NaOTf/DME 298K 0.1M", "traj_paths": _paths("na_otf_dme_298K_0.1M")},
    {**_SHARED, "name": "NaPF6/DME 298K 0.1M", "traj_paths": _paths("na_pf6_dme_298K_0.1M")},
    {**_SHARED, "name": "NaPF6/DME 323K 0.5M", "traj_paths": _paths("na_pf6_dme_323K_0.5M")},
]

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/uma_density_isotropic_mtk_npt/analysis"
)
ANALYSES = ["energy"]
WORKERS  = 4
