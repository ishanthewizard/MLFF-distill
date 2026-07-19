"""Config: multi-replica MSD (-> diffusivity) for the MLP `foam_all_systems`
NVT set.

MLP analogue of `config_painn_fp32_indist_multireplica_nvt_msd.py`.
Computes MSD (-> diffusivity / Nernst-Einstein conductivity) for the 4 NVT
replicas (nvt_replica_1..4) of every electrolyte system, overlaying the replicas
on the same axes per system.

Each unique system = `<salt>_<solvent>/<run_dir>` (so 0.1 M vs 1 M and the three
lipf6 temperatures stay separate). Within a system, every replica that has the
trajectory becomes a "model" labelled `replica_<i>`, so the 4 replicas' MSD
curves are drawn together — appropriate here because they share
composition/box/temperature. All 18 systems have all 4 replicas.

All NVT runs in this set target the full 20 ns, so no per-system SHORT window
capping is needed (eval.py caps max_traj_ns to the actual length).

MSD windowing (per user request):
  - frames start at 2 ns          -> eq_cut_ns = 2.0
  - total trajectory is 20 ns      -> max_traj_ns = 20.0
  - dt = 100 fs/saved-frame        -> dt_fs = 100.0
  - effective MSD lag dt = 5 ps    -> TARGET_MSD_DT_PS = 5.0 (user choice)
(dt_fs / max_traj_ns match the .fennol.yaml: dt[fs]=1.0, tdump[ps]=0.1,
nsteps=20e6; the dirname labels `20ns_100fs` etc. are NOT reliable.)

cat/anion/solvent symbols + temperature/concentration are inferred from the
`<salt>_<solvent>` / `<run_dir>` names via the lookup tables below (required by
the msd analysis; concentration_M makes eval match the exp reference at the right
concentration).

Run with:
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_foam_all_systems_nvt_msd.py
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
MAX_TRAJ_NS = 20.0     # 20 ns target; eval.py caps to actual traj length

# MSD windowing (user request)
EQ_CUT_NS       = 2.0    # discard first 2 ns of every trajectory
FIT_PCT         = 0.8    # fit up to 80% of max lag
TAU_MIN_FIT_NS  = 1.0    # lower bound of the linear (diffusive) fit
SLIDE_WINDOW_NS = 10.0   # sliding-window width (convergence panel)
SLIDE_STEP_NS   = 1.0    # sliding-window step
N_CONV_POINTS   = 20     # convergence-sweep resolution

# Effective MSD lag spacing (after subsampling), pinned across ALL systems.
# eval.py subsamples each trajectory to n_frames points; the MSD lag step is
# stride*dt_ps with stride = ceil(avail/n_frames) and avail = usable frames
# after eq_cut. Setting n_frames = usable_ns * 1000 / TARGET_MSD_DT_PS pins the
# lag step to exactly TARGET_MSD_DT_PS everywhere (all runs are full 20 ns here,
# so this is a single value, but we compute it per system for robustness).
TARGET_MSD_DT_PS = 5.0    # ps (user choice)

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
    temperature_K   = float(mr.group(3)) if mr else None
    concentration_M = float(mr.group(2).replace("_", ".")) if mr else None  # 0_1M->0.1

    name = key.replace("/", "__")   # e.g. naotf_dme__nvt_1M_298K_20ns_100fs
    max_traj_ns = MAX_TRAJ_NS

    # n_frames sized so the subsampled MSD lag step == TARGET_MSD_DT_PS
    n_frames = max(1, round((max_traj_ns - EQ_CUT_NS) * 1000.0 / TARGET_MSD_DT_PS))

    traj_paths = {r: _systems[key][r] for r in REPLICA_DIRS if r in _systems[key]}
    SYSTEMS.append({
        "name":            name,
        "traj_paths":      traj_paths,
        "model_colors":    dict(REPLICA_COLORS),
        "dt_fs":           DT_FS,
        "max_traj_ns":     max_traj_ns,
        "n_frames":        n_frames,
        "temperature_K":   temperature_K,
        "concentration_M": concentration_M,   # -> eval matches exp at the right conc
        # ── msd ──
        "cat_symbol":      cat_symbol,
        "anion_symbol":    anion_symbol,
        "solvent_symbol":  solvent_symbol,
        "eq_cut_ns":       EQ_CUT_NS,
        "fit_pct":         FIT_PCT,
        "tau_min_fit_ns":  TAU_MIN_FIT_NS,
        "slide_window_ns": SLIDE_WINDOW_NS,
        "slide_step_ns":   SLIDE_STEP_NS,
        "n_conv_points":   N_CONV_POINTS,
    })

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/MLP/foam_all_systems/analysis"
)
ANALYSES = ["msd"]
WORKERS  = 6   # parallel across systems; msd (no teacher re-inference)
