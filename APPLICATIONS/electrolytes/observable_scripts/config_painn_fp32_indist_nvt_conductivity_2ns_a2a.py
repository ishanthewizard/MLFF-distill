"""Config: multi-replica CONDUCTIVITY for the PAINN FP32 In-distribution NVT set,
apple-to-apple [0.1, 2.1] ns window (matches the UMA npt_2ns + foam runs).

Trajectory set / auto-discovery identical to
`config_painn_fp32_indist_multireplica_nvt_conductivity.py`; only the analysis
window differs so this can be compared 1:1 against the UMA and foam conductivity
runs (same start, same length, same effective sampling dt).

Layout: <root>/nvt_replica_<i>/<salt>_<solvent>/<run_dir>/<run_dir>.traj
Each replica with the trajectory becomes a model `replica_<i>`.

Window (apple-to-apple, user request):
  - dt_fs                    = 100.0   (tdump 0.1 ps, dt 1 fs -> 100 fs/frame; dirname unreliable)
  - conductivity_eq_cut_ns   = 0.1     (start from 0.1 ns)
  - max_traj_ns              = 2.1     (analyze [0.1, 2.1] ns; trajs are full 20 ns)
  - conductivity_load_dt_ps  = 0.5     (effective sampling / MSD-lag dt; == UMA run)

byteff2 onsager_calc() drops the first 200 loaded frames then fits MSD-slope lags
[50, 200). The 2 ns window / 0.5 ps = 4000 loaded frames >> 400, so the
Nernst-Einstein and byteff2-Onsager numbers are well defined (25-100 ps lag fit).
The mdcraft collective-Onsager backend fits its own 0.3-2.0 ns window.

Run with:
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_painn_fp32_indist_nvt_conductivity_2ns_a2a.py
"""
import re
from pathlib import Path

_REPLICA_ROOT = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data"
    "/simulation/FP32_simulation/In_distribution_exp/NVT_langevin"
)
# auto-discover nvt_replica_<i> dirs (PAINN has replica_0..3)
REPLICA_DIRS = {
    f"replica_{p.name.split('_')[-1]}": p
    for p in sorted(_REPLICA_ROOT.glob("nvt_replica_*")) if p.is_dir()
}
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
REPLICA_COLORS = {lbl: _PALETTE[i % len(_PALETTE)]
                  for i, lbl in enumerate(sorted(REPLICA_DIRS))}

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

DT_FS       = 100.0    # 100 fs/saved-frame
MAX_TRAJ_NS = 2.1      # analyze [0.1, 2.1] ns; eval.py caps to actual length
EQ_CUT_NS   = 0.1      # start from 0.1 ns
LOAD_DT_PS  = 0.5      # effective sampling / MSD-lag dt (apple-to-apple w/ UMA)
Z_CAT       =  1.0
Z_ANION     = -1.0

SYSTEM_DIR_RE = re.compile(r"^([a-z0-9]+)_([a-z]+)$")    # naotf_dme, lipf6_dme
RUN_DIR_RE    = re.compile(r"^(npt|nvt)_(\d+(?:_\d+)?)M_(\d+)K_")

_systems: dict[str, dict[str, str]] = {}
for rlabel, rroot in REPLICA_DIRS.items():
    if not rroot.is_dir():
        continue
    for sys_dir in sorted(p for p in rroot.glob("*/") if p.is_dir()):
        if not SYSTEM_DIR_RE.match(sys_dir.name):
            continue
        for run_dir in sorted(p for p in sys_dir.glob("*/") if p.is_dir()):
            traj = run_dir / f"{run_dir.name}.traj"
            if not traj.exists():
                continue
            key = f"{sys_dir.name}/{run_dir.name}"
            _systems.setdefault(key, {})[rlabel] = str(traj)

SYSTEMS = []
for key in sorted(_systems):
    sys_name, run_name = key.split("/", 1)
    ms = SYSTEM_DIR_RE.match(sys_name)
    salt, solvent_tok = ms.groups()
    if salt not in ION_MAP or solvent_tok not in SOLVENT_SYMBOL:
        continue
    cat_symbol, anion_symbol = ION_MAP[salt]
    solvent_symbol = SOLVENT_SYMBOL[solvent_tok]

    mr = RUN_DIR_RE.match(run_name)
    concentration_M = float(mr.group(2).replace("_", ".")) if mr else None
    temperature_K   = float(mr.group(3)) if mr else 298.0

    name = key.replace("/", "__")

    traj_paths = {r: _systems[key][r] for r in REPLICA_DIRS if r in _systems[key]}
    SYSTEMS.append({
        "name":            name,
        "traj_paths":      traj_paths,
        "model_colors":    dict(REPLICA_COLORS),
        "dt_fs":           DT_FS,
        "max_traj_ns":     MAX_TRAJ_NS,
        "temperature_K":   temperature_K,
        "concentration_M": concentration_M,
        "cat_symbol":      cat_symbol,
        "anion_symbol":    anion_symbol,
        "solvent_symbol":  solvent_symbol,
        "conductivity_T_K":        temperature_K,
        "conductivity_z_cat":      Z_CAT,
        "conductivity_z_anion":    Z_ANION,
        "conductivity_eq_cut_ns":  EQ_CUT_NS,
        "conductivity_load_dt_ps": LOAD_DT_PS,
    })

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/eleyte_temp/painn"
)
ANALYSES = ["conductivity"]
WORKERS  = 6
