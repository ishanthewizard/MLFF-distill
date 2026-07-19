"""Config: multi-replica CONDUCTIVITY for the MLP `foam_all_systems` NVT set.

MLP analogue of `config_painn_fp32_indist_multireplica_nvt_conductivity.py`.
Mirrors the foam MSD config (same trajectory set / auto-discovery), but runs the
conductivity analysis instead of MSD.

For every unique system `<salt>_<solvent>/<run_dir>`, each replica that has the
trajectory becomes a "model" labelled `replica_<i>`, so eval.py computes (and
writes one conductivity.csv row for) every replica of every system. All 18
systems have all 4 replicas (72 trajectories total).

Conductivity windowing (per user request):
  - frames start at 2 ns            -> conductivity_eq_cut_ns = 2.0
  - to the end (20 ns target)        -> max_traj_ns = 20.0  (eval.py caps each
                                        replica to its actual length)
  - effective sampling / lag dt 5 ps -> conductivity_load_dt_ps = 5.0  (user choice)
  - dt = 100 fs/saved-frame          -> dt_fs = 100.0
(dt_fs / max_traj_ns match the .fennol.yaml: dt[fs]=1.0, tdump[ps]=0.1,
nsteps=20e6; the dirname labels `20ns_100fs` are NOT reliable.)

NOTE on load_dt_ps=5.0 and the byteff2 fit window:
  byteff2's onsager_calc() hardcodes its MSD-slope fit to lags [50, 200) frames
  (after dropping the first 200 loaded frames). With load_dt_ps=5.0 ps that is a
  250-1000 ps lag window for the Nernst-Einstein / byteff2-Onsager numbers. The
  mdcraft collective-Onsager backend instead fits 0.3-2.0 ns (independent of
  load_dt).

cat/anion/solvent symbols + temperature are inferred from the dir names via the
lookup tables below (required by the conductivity analysis).

Run with (multi-hour job: 72 trajectories, ~18 ns each, two backends):
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_foam_all_systems_nvt_conductivity.py
"""
import re
from pathlib import Path

_REPLICA_ROOT = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/MLP/foam_all_systems"
)
REPLICA_DIRS = {
    "replica_1": _REPLICA_ROOT / "nvt_replica_1",
    "replica_2": _REPLICA_ROOT / "nvt_replica_2",
    "replica_3": _REPLICA_ROOT / "nvt_replica_3",
    "replica_4": _REPLICA_ROOT / "nvt_replica_4",
}

REPLICA_COLORS = {
    "replica_1": "#1f77b4",
    "replica_2": "#ff7f0e",
    "replica_3": "#2ca02c",
    "replica_4": "#d62728",
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
MAX_TRAJ_NS = 20.0     # 20 ns target; eval.py caps each replica to actual traj length

# ── conductivity windowing (user request) ─────────────────────────────────────
EQ_CUT_NS   = 2.0      # start from 2 ns (discard first 2 ns of every trajectory)
LOAD_DT_PS  = 5.0      # effective sampling / MSD-lag dt = 5 ps (user choice)
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
    temperature_K   = float(mr.group(3)) if mr else 298.0
    concentration_M = float(mr.group(2).replace("_", ".")) if mr else None  # 0_1M->0.1

    name = key.replace("/", "__")   # e.g. naotf_dme__nvt_1M_298K_20ns_100fs

    traj_paths = {r: _systems[key][r] for r in REPLICA_DIRS if r in _systems[key]}
    SYSTEMS.append({
        "name":            name,
        "traj_paths":      traj_paths,
        "model_colors":    dict(REPLICA_COLORS),
        "dt_fs":           DT_FS,
        "max_traj_ns":     MAX_TRAJ_NS,
        "temperature_K":   temperature_K,
        "concentration_M": concentration_M,   # -> exp match at the right conc
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
    "/m5250/distillation_proj/simulation_results/MLP/foam_all_systems/analysis"
)
ANALYSES = ["conductivity"]
WORKERS  = 6   # parallel across systems (conductivity = no teacher re-inference)
