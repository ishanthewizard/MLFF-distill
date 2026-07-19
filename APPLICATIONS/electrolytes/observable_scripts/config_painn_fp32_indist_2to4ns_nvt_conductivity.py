"""Config: [2 ns, 4 ns]-window CONDUCTIVITY for the PAINN FP32 In-distribution NVT set.

Strict 2->4 ns window (per user request, 2026-07-16). Same trajectory set /
auto-discovery as ``config_painn_fp32_indist_4ns_nvt_conductivity.py`` (which used
[2, 6] ns), but restricted to the [2 ns, 4 ns] slice.

Conductivity windowing:
  - window = [2 ns, 4 ns]             -> conductivity_eq_cut_ns = 2.0 AND
                                         max_traj_ns = 4.0  => 2 ns of usable data
  - effective sampling / lag dt 2 ps  -> conductivity_load_dt_ps = 2.0
  - dt = 100 fs/saved-frame           -> dt_fs = 100.0

Frame budget: n_loaded = (40000 - 20000)//20 = 1000 frames > 400 (byteff2 OK).
(A 5 ps lag here would give only 400 frames and byteff2 would fail; 2 ps chosen
to keep the strict [2,4] ns window well above the 400-frame minimum.)

Run with:
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_painn_fp32_indist_2to4ns_nvt_conductivity.py
"""
import re
from pathlib import Path

_REPLICA_ROOT = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data"
    "/simulation/FP32_simulation/In_distribution_exp/NVT_langevin"
)
REPLICA_DIRS = {
    "replica_0": _REPLICA_ROOT / "nvt_replica_0",
    "replica_1": _REPLICA_ROOT / "nvt_replica_1",
    "replica_2": _REPLICA_ROOT / "nvt_replica_2",
    "replica_3": _REPLICA_ROOT / "nvt_replica_3",
}

REPLICA_COLORS = {
    "replica_0": "#1f77b4",
    "replica_1": "#ff7f0e",
    "replica_2": "#2ca02c",
    "replica_3": "#d62728",
}

# salt -> (cation, anion) ; solvent token -> solvent symbol (keys into msd dicts)
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

DT_FS = 100.0          # tdump=0.1 ps, dt=1 fs -> 100 fs/saved-frame (dirname unreliable)
MAX_TRAJ_NS = 4.0      # [2,4] ns check: window UPPER bound 4 ns (with eq_cut=2 -> 2 ns usable)

# ── conductivity windowing ────────────────────────────────────────────────────
EQ_CUT_NS   = 2.0      # start from 2 ns (discard first 2 ns of every trajectory)
LOAD_DT_PS  = 2.0      # effective sampling / MSD-lag dt = 2 ps -> 1000 frames in [2,4] ns
Z_CAT       =  1.0     # all cations here are monovalent (Li+, Na+)
Z_ANION     = -1.0     # all anions here are monovalent (PF6-, OTf-)

SYSTEM_DIR_RE = re.compile(r"^([a-z0-9]+)_([a-z]+)$")    # e.g. naotf_dme, lipf6_dme
RUN_DIR_RE    = re.compile(r"^(npt|nvt)_(\d+(?:_\d+)?)M_(\d+)K_")

# ── discover the union of <salt_solvent>/<run_dir> across all replicas ──────────
# {relative "system/run": {replica_label: traj_path}}
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
    temperature_K = float(mr.group(3)) if mr else 298.0

    name = key.replace("/", "__")   # e.g. naotf_dme__nvt_1M_298K_20ns_100fs

    traj_paths = {r: _systems[key][r] for r in REPLICA_DIRS if r in _systems[key]}
    SYSTEMS.append({
        "name":            name,
        "traj_paths":      traj_paths,
        "model_colors":    dict(REPLICA_COLORS),
        "dt_fs":           DT_FS,
        "max_traj_ns":     MAX_TRAJ_NS,
        "temperature_K":   temperature_K,
        # ── species (required by conductivity) ──
        "cat_symbol":      cat_symbol,
        "anion_symbol":    anion_symbol,
        "solvent_symbol":  solvent_symbol,
        # ── conductivity ──
        "conductivity_T_K":        temperature_K,
        "conductivity_z_cat":      Z_CAT,
        "conductivity_z_anion":    Z_ANION,
        "conductivity_eq_cut_ns":  EQ_CUT_NS,
        "conductivity_load_dt_ps": LOAD_DT_PS,
    })

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data"
    "/analysis/2to4ns_conductivity_check/nvt"
)
ANALYSES = ["conductivity"]
WORKERS  = 8   # parallel across systems (conductivity = no teacher re-inference)
