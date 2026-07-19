"""Config: force/energy/stress MAE for the PAINN electrolytes_data
simulation set, comparing against the UMA-S-1p1 teacher.

Walks `_BASE` for every `<salt>_<solvent>/<npt|nvt>_<conc>M_<temp>K_<dur>ns_<dt>`
run and builds one SYSTEMS entry per `.traj` found.

Run with:
  python eval.py --config config_painn_electrolytes_mae.py --workers 1
"""
import re
from pathlib import Path

_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data/simulation"
)

_TEACHER_CKPT = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m4558/distillation_project/models/uma-s-1p1.pt"
)

ION_MAP = {
    "lipf6": ("Li", "PF6"),
    "napf6": ("Na", "PF6"),
    "naotf": ("Na", "OTf"),
}

SOLVENT_SYMBOL = {
    "dme":     "DME",
    "diglyme": "Diglyme",
    "tgdme":   "TGDME",
    "pc":      "PC",
}

# Every run's .fennol.yaml has dt[fs]=1.0, tdump[ps]=0.1 (i.e. dt_fs=100),
# regardless of dirname suffix. eval.py caps max_traj_ns to the real
# trajectory length, so 20.0 here means "up to the whole trajectory".
SYSTEM_DIR_RE = re.compile(r"^([a-z0-9]+)_([a-z]+)$")
RUN_DIR_RE = re.compile(r"^(npt|nvt)_(\d+(?:_\d+)?)M_(\d+)K_")
DT_FS = 100.0
MAX_TRAJ_NS = 20.0

# MAE eval cadence: analyze_dt_ps = 1000 -> stride = 10 frames @ dt_fs=100
_MAE_ANALYZE_DT_PS = 1000.0
_MAE_N_FRAMES = 200

SYSTEMS = []
for sys_dir in sorted(_BASE.glob("*/")):
    m = SYSTEM_DIR_RE.match(sys_dir.name)
    if not m:
        continue
    salt, solvent_tok = m.groups()
    if salt not in ION_MAP or solvent_tok not in SOLVENT_SYMBOL:
        continue
    cat_symbol, anion_symbol = ION_MAP[salt]
    solvent_symbol = SOLVENT_SYMBOL[solvent_tok]

    for run_dir in sorted(sys_dir.glob("*/")):
        mr = RUN_DIR_RE.match(run_dir.name)
        if not mr:
            continue
        ensemble, conc_tok, temp_tok = mr.groups()
        traj_path = run_dir / f"{run_dir.name}.traj"
        if not traj_path.exists():
            continue

        concentration = float(conc_tok.replace("_", "."))
        temperature_K = float(temp_tok)

        SYSTEMS.append({
            "name":           f"{salt}_{solvent_tok} {ensemble} {concentration}M {temperature_K:.0f}K ({run_dir.name})",
            "traj_paths":     {"PAINN": str(traj_path)},
            "dt_fs":          DT_FS,
            "max_traj_ns":    MAX_TRAJ_NS,
            "model_colors":   {"PAINN": "#1f77b4"},
            "temperature_K":  temperature_K,
            "cat_symbol":     cat_symbol,
            "anion_symbol":   anion_symbol,
            "solvent_symbol": solvent_symbol,
            # force_mae
            "force_mae_teacher_ckpt":   _TEACHER_CKPT,
            "force_mae_student_ckpt":   None,
            "force_mae_analyze_dt_ps":  _MAE_ANALYZE_DT_PS,
            "force_mae_n_frames":       _MAE_N_FRAMES,
            # energy_mae
            "energy_mae_teacher_ckpt":  _TEACHER_CKPT,
            "energy_mae_student_ckpt":  None,
            "energy_mae_analyze_dt_ps": _MAE_ANALYZE_DT_PS,
            "energy_mae_n_frames":      _MAE_N_FRAMES,
            # stress_mae
            "stress_mae_teacher_ckpt":  _TEACHER_CKPT,
            "stress_mae_student_ckpt":  None,
            "stress_mae_analyze_dt_ps": _MAE_ANALYZE_DT_PS,
            "stress_mae_n_frames":      _MAE_N_FRAMES,
        })

OUTPUT_DIR = "/pscratch/sd/y/yuejian/painn_electrolytes_mae_analysis"
ANALYSES = ["force_mae", "energy_mae", "stress_mae"]
WORKERS  = 1
