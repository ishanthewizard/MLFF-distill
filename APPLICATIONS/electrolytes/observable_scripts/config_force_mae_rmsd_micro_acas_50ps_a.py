"""micro_acas_50ps — GPU 2 — systems [0:9] (1M + 0.1M NaOTf)."""
from config_force_mae_rmsd_micro_acas_50ps import SYSTEMS as _ALL, _TEACHER_CKPT, _STUDENT_CKPT  # noqa: F401

SYSTEMS    = _ALL[0:9]
OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/analysis/force_mae_rmsd/micro_acas_50ps"
)
ANALYSES = ["force_mae", "rmsd"]
WORKERS  = 1
