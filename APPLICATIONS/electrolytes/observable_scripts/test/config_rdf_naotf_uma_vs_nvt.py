"""Config: RDF comparison — NaOTf/DME 298K 0.1M, UMA (NPT) vs NVT teacher.

UMA traj  : 10  fs/frame, 1.0 ns total
NVT traj  : 100 fs/frame, 1.81 ns (capped at 1.0 ns to match UMA)

Both are analysed over the same physical time window: skip 0.1 ns, use 0.85 ns.
RDF pairs: Na-S, Na-O
"""
from pathlib import Path

_UMA_TRAJ = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5024/distillation_project/data/raw_data_from_UMA_simulation"
    "/other_temperature_for_comparing_with_student_model"
    "/20ns_solvent_0_1M/298K/md_omol_naotf_dme_s1p1_omol"
    "/md_omol_naotf_dme_s1p1_omol.traj"
)

_NVT_TRAJ = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/distort_temp_md_nvt/simulations/nvt/nose_hoover"
    "/na_otf_dme_298K_0.1M_teacher/na_otf_dme_298K_0.1M_teacher.traj"
)

_MAX_NS     = 1.0    # UMA caps at 1.0 ns; NVT capped to match
_SKIP_NS    = 0.1
_WINDOW_NS  = 0.85

SYSTEMS = [
    {
        "name":       "NaOTf/DME 298K 0.1M",
        "traj_paths": {
            "UMA":     _UMA_TRAJ,
            "NVT_teacher": _NVT_TRAJ,
        },
        "dt_fs": {
            "UMA":         10.0,
            "NVT_teacher": 100.0,
        },
        "model_colors": {
            "UMA":         "#1f77b4",
            "NVT_teacher": "#ff7f0e",
        },
        "max_traj_ns": _MAX_NS,
        "rdf_pairs":   [("Na", "S"), ("Na", "O")],
        "skip_ns":     _SKIP_NS,
        "window_ns":   _WINDOW_NS,
        "n_frames":    1000,
    },
]

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/distort_temp_md_nvt/analysis/nvt_ckpt_1_5"
    "/rdf_uma_vs_nvt"
)
ANALYSES = ["rdf"]
WORKERS  = 1
