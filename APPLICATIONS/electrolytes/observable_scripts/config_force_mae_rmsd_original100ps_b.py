"""original_100ps — GPU 1 — systems [9:18] (0.5M 323K + 1M)."""
from config_force_mae_rmsd_original100ps import SYSTEMS as _ALL, _TEACHER_CKPT, _STUDENT_CKPT  # noqa: F401

SYSTEMS    = _ALL[9:18]
OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/analysis/force_mae_rmsd/original_100ps"
)
ANALYSES = ["force_mae", "rmsd"]
WORKERS  = 1
