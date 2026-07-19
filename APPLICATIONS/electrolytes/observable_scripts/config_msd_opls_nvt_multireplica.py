"""Config: multi-replica MSD / diffusivity for the OPLS GROMACS NVT production set.

GROMACS analogue of `config_painn_fp32_indist_multireplica_nvt_msd.py`.
Computes MSD (-> diffusivity / Nernst-Einstein conductivity) for the 4 NVT
replicas (replicas_1..4) of every OPLS electrolyte system, overlaying the
replicas on the same axes per system.

Layout walked (LOCAL pscratch copy -- MDAnalysis's offset-cache flock() hangs
indefinitely on CFS, so nvt.xtc/nvt.tpr were rsync'd here first):
    <root>/replicas_<N>/<conc>/<system>/nvt.xtc (+ nvt.tpr)
where
    <conc>   in {1M, 0.1M, 0.5M}
    <system> is  <salt>_<solvent>_<conc>     for 1M / 0.1M      (e.g. naotf_dme_1M)
             or  lipf6_dme_<temp>            for the 0.5M set   (e.g. lipf6_dme_298)

Each unique system = <conc>/<system dir> (so 0.1 M vs 0.5 M vs 1 M and the three
lipf6 temperatures stay separate). Within a system, every replica that has the
trajectory becomes a "model" labelled `replica_<i>`, so the (up to) 4 replicas'
MSD curves are drawn together -- appropriate here because they share
composition / box / temperature.

GROMACS specifics (from nvt.mdp): dt = 1 fs, nstxout-compressed = 1000
-> frames every 1 ps -> dt_fs = 1000.0; nsteps = 2e7 -> 20 ns.

MSD windowing (per user request):
  - frames start at 2 ns          -> eq_cut_ns = 2.0
  - total trajectory is 20 ns      -> max_traj_ns = 20.0
  - effective MSD lag dt = 5 ps    -> TARGET_MSD_DT_PS = 5.0 (user choice)
    => n_frames = (20 - 2) * 1000 / 5 = 3600

cat/anion/solvent symbols are inferred from the <salt>_<solvent> tokens via the
lookup tables below (required by the msd analysis; these are component_dictionary
keys, matching the previous OPLS msd configs).

Run with (fairchemV2_new has MDAnalysis 2.10):
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_msd_opls_nvt_multireplica.py
"""
import re
from pathlib import Path

# LOCAL pscratch copy of the OPLS NVT production tree (see module docstring).
_ROOT = Path("/pscratch/sd/y/yuejian/opls_nvt_msd_local")

REPLICA_DIRS = {
    "replica_1": _ROOT / "replicas_1",
    "replica_2": _ROOT / "replicas_2",
    "replica_3": _ROOT / "replicas_3",
    "replica_4": _ROOT / "replicas_4",
}
REPLICA_COLORS = {
    "replica_1": "#1f77b4",
    "replica_2": "#ff7f0e",
    "replica_3": "#2ca02c",
    "replica_4": "#d62728",
}

# salt token -> (cation, anion) ; solvent token -> solvent symbol
# (component_dictionary keys required by the msd analysis)
ION_MAP = {
    "lipf6": ("Li", "PF6"),
    "napf6": ("Na", "PF6"),
    "naotf": ("Na", "OTf"),
}
SOLVENT_SYMBOL = {
    "dme":     "DME",
    "diglyme": "Diglyme",
    "tegdme":  "TGDME",
    "tgdme":   "TGDME",
    "pc":      "PC",
}

DT_FS       = 1000.0   # 1 fs MD step, frames every 1000 steps -> 1 ps/saved-frame
MAX_TRAJ_NS = 20.0     # 20 ns target; eval.py caps to actual traj length

# MSD windowing (user request)
EQ_CUT_NS        = 2.0    # discard first 2 ns of every trajectory
FIT_PCT          = 0.8    # fit up to 80% of max lag
TAU_MIN_FIT_NS   = 1.0    # lower bound of the linear (diffusive) fit
SLIDE_WINDOW_NS  = 5.0    # sliding-window width (convergence panel)
SLIDE_STEP_NS    = 0.5    # sliding-window step
N_CONV_POINTS    = 20     # convergence-sweep resolution

# Effective MSD lag spacing (after subsampling), pinned across ALL systems.
# eval.py subsamples each trajectory to n_frames points; the MSD lag step is
# stride*dt_ps with stride = ceil(avail/n_frames), avail = usable frames after
# eq_cut. n_frames = usable_ns * 1000 / TARGET_MSD_DT_PS pins the lag step.
TARGET_MSD_DT_PS = 5.0    # ps (user choice)

# <salt>_<solvent>_<conclabel-or-temp>, e.g. naotf_dme_1M, napf6_pc_0.1M, lipf6_dme_298
_SYS_RE = re.compile(r"^([a-z0-9]+)_([a-z]+)_([0-9.]+M?|[0-9]+)$")

# ── discover the union of <conc>/<system> across all replicas ──────────────────
# {relative "conc/system": {replica_label: {"xtc":..., "tpr":...}}}
_systems: dict[str, dict[str, dict]] = {}
for rlabel, rroot in REPLICA_DIRS.items():
    if not rroot.is_dir():
        continue
    for conc_dir in sorted(p for p in rroot.glob("*/") if p.is_dir()):
        for sys_dir in sorted(p for p in conc_dir.glob("*/") if p.is_dir()):
            xtc = sys_dir / "nvt.xtc"
            tpr = sys_dir / "nvt.tpr"
            if not xtc.exists() or not tpr.exists():
                continue
            key = f"{conc_dir.name}/{sys_dir.name}"
            _systems.setdefault(key, {})[rlabel] = {"xtc": str(xtc), "tpr": str(tpr)}

SYSTEMS = []
for key in sorted(_systems):
    conc_label, sys_name = key.split("/", 1)
    m = _SYS_RE.match(sys_name)
    if not m:
        continue
    salt, solvent_tok, third = m.groups()
    if salt not in ION_MAP or solvent_tok not in SOLVENT_SYMBOL:
        continue
    cat_symbol, anion_symbol = ION_MAP[salt]
    solvent_symbol = SOLVENT_SYMBOL[solvent_tok]

    # concentration from the parent folder (1M / 0.1M / 0.5M)
    concentration_M = float(conc_label[:-1]) if conc_label.endswith("M") else float(conc_label)
    # temperature: encoded in the system dir only for the 0.5M lipf6 set
    # (lipf6_dme_273/298/323); everything else is 298 K.
    temperature_K = float(third) if third.isdigit() else 298.0

    # clean, unambiguous display name: <salt>_<solvent>_<conc>_<temp>K
    name = f"{salt}_{solvent_tok}_{conc_label}_{int(temperature_K)}K"

    # n_frames sized so the subsampled MSD lag step == TARGET_MSD_DT_PS
    n_frames = max(1, round((MAX_TRAJ_NS - EQ_CUT_NS) * 1000.0 / TARGET_MSD_DT_PS))

    traj_paths = {r: _systems[key][r] for r in REPLICA_DIRS if r in _systems[key]}
    SYSTEMS.append({
        "name":            name,
        "traj_paths":      traj_paths,
        "model_colors":    dict(REPLICA_COLORS),
        "dt_fs":           DT_FS,
        "max_traj_ns":     MAX_TRAJ_NS,
        "n_frames":        n_frames,
        "concentration_M": concentration_M,
        "temperature_K":   temperature_K,
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

# quick single-system test: MSD_TEST_SINGLE=<substr>
import os
_TEST_SINGLE = os.environ.get("MSD_TEST_SINGLE")
if _TEST_SINGLE:
    SYSTEMS = [s for s in SYSTEMS if _TEST_SINGLE in s["name"]]

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/OPLS/analysis/nvt"
)
ANALYSES = ["msd"]
WORKERS  = 1 if _TEST_SINGLE else 6   # parallel across systems (msd: no teacher re-inference)
