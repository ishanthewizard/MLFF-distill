"""original_100ps — GPU 0 — systems [0:9] (0.1M + 0.5M 273K/298K)."""
from config_force_mae_rmsd_original100ps import SYSTEMS as _ALL, _TEACHER_CKPT, _STUDENT_CKPT  # noqa: F401

SYSTEMS    = _ALL[0:9]
OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/analysis/force_mae_rmsd/original_100ps"
)
ANALYSES = ["force_mae", "rmsd"]
WORKERS  = 1
