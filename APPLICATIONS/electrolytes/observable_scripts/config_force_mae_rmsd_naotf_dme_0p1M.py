"""Config: force MAE + RMSD — single system: NaOTf/DME 0.1M 298K (original_100ps student).

Trajectory : original_100ps student model, 20 ns, dt = 100 fs/frame
Teacher    : UMA-S-1p1
Student    : ablation ckpt — trained on all systems, first 100 ps, 293 K
"""
from pathlib import Path

_TRAJ_DIR = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5024/distillation_project/results"
    "/diffusivity_main_results_20ns_final/original_100ps"
    "/20ns_solvent_0_1M/298K"
    "/md_omol_naotf_dme_s1p1_omol"
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

SYSTEMS = [
    {
        "name": "NaOTf/DME 0.1M 298K student",
        "traj_paths": {
            "student": str(_TRAJ_DIR / "md_omol_naotf_dme_s1p1_omol.traj"),
        },
        "dt_fs":       _DT_FS,
        "max_traj_ns": 20.0,
        "model_colors": {"student": "#ff7f0e"},

        # ── force_mae ──────────────────────────────────────────────────────
        "force_mae_teacher_ckpt":  _TEACHER_CKPT,
        "force_mae_student_ckpt":  None,      # use forces stored in traj
        "force_mae_analyze_dt_ps": 100.0,     # stride=1000 → ~200 eval pts
        "force_mae_small_model":   False,

        # ── rmsd ──────────────────────────────────────────────────────────
        "rmsd_ref_frame_idx":  0,
        "rmsd_analyze_dt_ps":  100.0,
    },
]

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/analysis/force_mae_rmsd/original_100ps"
)
ANALYSES = ["force_mae", "rmsd"]
WORKERS  = 1
