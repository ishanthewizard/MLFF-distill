"""Config: force MAE + RMSD for the original_100ps student model — all systems.

Student ckpt : trained on all systems, first 100 ps, 293 K
Teacher ckpt : UMA-S-1p1
Traj dt      : 100 fs/frame  →  analyze_dt_ps=100 gives stride=1000 (~200 pts/20ns)
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
    dt_fs                  = _DT_FS,
    max_traj_ns            = 20.0,
    model_colors           = {"student": "#ff7f0e"},
    force_mae_teacher_ckpt = _TEACHER_CKPT,
    force_mae_student_ckpt = None,      # use forces stored in traj
    force_mae_analyze_dt_ps= 100.0,     # 100 ps → stride 1000 → ~200 eval pts
    force_mae_small_model  = False,
    rmsd_ref_frame_idx     = 0,
    rmsd_analyze_dt_ps     = 100.0,
)

_01M = "20ns_solvent_0_1M/298K"
_05M = "20ns_solvent_solute_0.5M"
_1M  = "20ns_solute_solvent_1M/298K"

SYSTEMS = [
    # ── 0.1 M 298 K ────────────────────────────────────────────────────────
    {**_SHARED, "name": "NaOTf/Diglyme 0.1M 298K",
     "traj_paths": {"student": _p(_01M, "md_omol_naotf_diglyme_1m_s1p1")}},
    {**_SHARED, "name": "NaOTf/PC 0.1M 298K",
     "traj_paths": {"student": _p(_01M, "md_omol_naotf_pc_1m_s1p1")}},
    {**_SHARED, "name": "NaOTf/DME 0.1M 298K",
     "traj_paths": {"student": _p(_01M, "md_omol_naotf_dme_s1p1_omol")}},
    {**_SHARED, "name": "NaOTf/TGDME 0.1M 298K",
     "traj_paths": {"student": _p(_01M, "md_omol_naotf_tgdme_1m_s1p1")}},
    {**_SHARED, "name": "NaPF6/DME 0.1M 298K",
     "traj_paths": {"student": _p(_01M, "md_omol_napf6_dme_re1")}},
    {**_SHARED, "name": "NaPF6/Diglyme 0.1M 298K",
     "traj_paths": {"student": _p(_01M, "md_omol_napf6_diglyme_pfactor_0.1_1fs")}},
    {**_SHARED, "name": "NaPF6/PC 0.1M 298K",
     "traj_paths": {"student": _p(_01M, "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1")}},

    # ── 0.5 M 273 K ────────────────────────────────────────────────────────
    {**_SHARED, "name": "LiPF6/DME 0.5M 273K",
     "traj_paths": {"student": _p(_05M + "/273_2K", "md_omol_lipf6_pfactor_0.1_1fs_mask_t")}},
    {**_SHARED, "name": "NaPF6/DME 0.5M 273K",
     "traj_paths": {"student": _p(_05M + "/273_2K", "md_omol_napf6_dme_re1")}},

    # ── 0.5 M 298 K ────────────────────────────────────────────────────────
    {**_SHARED, "name": "LiPF6/DME 0.5M 298K",
     "traj_paths": {"student": _p(_05M + "/298_2K", "md_omol_lipf6_pfactor_0.1_1fs_mask_t")}},
    {**_SHARED, "name": "NaPF6/DME 0.5M 298K",
     "traj_paths": {"student": _p(_05M + "/298_2K", "md_omol_napf6_dme_re1")}},

    # ── 0.5 M 323 K ────────────────────────────────────────────────────────
    {**_SHARED, "name": "LiPF6/DME 0.5M 323K",
     "traj_paths": {"student": _p(_05M + "/323_2K", "md_omol_lipf6_pfactor_0.1_1fs_mask_t")}},
    {**_SHARED, "name": "NaPF6/DME 0.5M 323K",
     "traj_paths": {"student": _p(_05M + "/323_2K", "md_omol_napf6_dme_re1")}},

    # ── 1 M 298 K ──────────────────────────────────────────────────────────
    {**_SHARED, "name": "NaPF6/Diglyme 1M 298K",
     "traj_paths": {"student": _p(_1M, "md_omol_napf6_diglyme_pfactor_0.1_1fs")}},
    {**_SHARED, "name": "NaPF6/PC 1M 298K",
     "traj_paths": {"student": _p(_1M, "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1")}},
    {**_SHARED, "name": "NaOTf/Diglyme 1M 298K",
     "traj_paths": {"student": _p(_1M, "naotf_diglyme")}},
    {**_SHARED, "name": "NaOTf/DME 1M 298K",
     "traj_paths": {"student": _p(_1M, "naotf_dme")}},
    {**_SHARED, "name": "NaPF6/DME 1M 298K",
     "traj_paths": {"student": _p(_1M, "napf6_dme")}},
]

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/analysis/force_mae_rmsd/original_100ps"
)
ANALYSES = ["force_mae", "rmsd"]
WORKERS  = 4   # GPU-bound; process one system at a time
