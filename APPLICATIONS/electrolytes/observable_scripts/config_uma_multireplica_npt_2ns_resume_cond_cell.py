"""Config: multi-replica MSD + conductivity + cell size for the UMA NPT
"concentrate" 2 ns set (UMA_multi_replicas/npt/npt_2ns).

Layout (ONE dir per system per replica, traj is `<dir>.traj` inside it):

  npt/npt_2ns/UMA_ref_concentrate_rep{1..5}/<salt>_<solvent>_npt_<conc>M_<temp>K/
        └── <salt>_<solvent>_npt_<conc>M_<temp>K.traj

The 5 replicas (rep1..rep5) become the "models" (`replica_1..replica_5`) of each
unique system, so all 5 replicas are overlaid on the same axes / land in the same
combined CSV. 11 systems × 5 replicas = 55 trajectories.

Trajectory params (user):
  - dt_fs        = 10.0     (10 fs / saved-frame)
  - max_traj_ns  = 2.1      (analyze the first 2.1 ns; eval.py caps to real length)

MSD / conductivity windowing (user choices):
  - eq_cut_ns          = 0.1     (start MSD/conductivity from 0.1 ns)
  - effective lag dt   = 0.5 ps  (TARGET_MSD_DT_PS / conductivity_load_dt_ps)
  - MSD n_frames pinned so the subsampled lag step == 0.5 ps:
        n_frames = (max_traj_ns - eq_cut_ns) * 1000 / TARGET_MSD_DT_PS
                 = (2.1 - 0.1) * 1000 / 0.5 = 4000
  - conductivity_load_dt_ps = 0.5 -> ~4000 loaded frames from the 2 ns window.
    byteff2 onsager_calc() drops the first 200 loaded frames then fits MSD-slope
    lags [50, 200); 4000 >> 400 so both the byteff2 Onsager and Nernst-Einstein
    numbers are well defined. The mdcraft collective-Onsager backend fits its own
    0.3-2.0 ns window.

cat/anion/solvent symbols + concentration + temperature are parsed from the dir
name via the lookup tables below (required by msd/conductivity, and by the group
merge exp-matcher).

NOTE (cost): MSD is a serial O(n_frames^2) python loop and these CFS trajectories
are slow random-access reads. With n_frames=4000 and 55 trajs this is a multi-hour
job — run on a SLURM CPU node, NOT a login node.

Run with:
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_uma_multireplica_npt_2ns.py
"""
import re
from pathlib import Path

_NPT_ROOT = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/UMA/UMA_multi_replicas/npt/npt_2ns"
)
REPLICA_DIRS = {
    "replica_1": _NPT_ROOT / "UMA_ref_concentrate_rep1",
    "replica_2": _NPT_ROOT / "UMA_ref_concentrate_rep2",
    "replica_3": _NPT_ROOT / "UMA_ref_concentrate_rep3",
    "replica_4": _NPT_ROOT / "UMA_ref_concentrate_rep4",
    "replica_5": _NPT_ROOT / "UMA_ref_concentrate_rep5",
}
REPLICA_COLORS = {
    "replica_1": "#1f77b4",
    "replica_2": "#ff7f0e",
    "replica_3": "#2ca02c",
    "replica_4": "#d62728",
    "replica_5": "#9467bd",
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

# ── trajectory params ──────────────────────────────────────────────────────────
DT_FS       = 10.0     # 10 fs/saved-frame (user)
MAX_TRAJ_NS = 2.1      # analyze the first 2.1 ns; eval.py caps to actual length

# ── MSD / conductivity windowing (user choices) ────────────────────────────────
EQ_CUT_NS        = 0.1     # start MSD/conductivity from 0.1 ns
TARGET_MSD_DT_PS = 0.5     # effective MSD lag / conductivity sampling dt (user)
LOAD_DT_PS       = 0.5     # conductivity subsample dt (== TARGET_MSD_DT_PS)
Z_CAT            =  1.0    # Li+ / Na+ monovalent
Z_ANION          = -1.0    # PF6- / OTf- monovalent

# mdcraft collective-Onsager diffusive fit window (user): 30-100 ps.
# These knobs drive the MSD-slope + mdcraft L_ij fit; byteff2's own Onsager
# window is hardcoded so it is unaffected (see MEMORY conductivity fit window).
COND_TAU_MIN_NS = 0.03     # 30 ps  -> mdcraft fit_start_ns
COND_TAU_MAX_NS = 0.10     # 100 ps -> mdcraft fit_stop_ns

# MSD linear-fit / convergence knobs, scaled for a ~2 ns usable window
FIT_PCT         = 0.5     # fit up to 50% of max lag (short traj -> avoid noisy tail)
TAU_MIN_FIT_NS  = 0.1     # lower bound of the diffusive fit
SLIDE_WINDOW_NS = 0.5     # sliding-window width for the convergence panel
SLIDE_STEP_NS   = 0.2     # sliding-window step
N_CONV_POINTS   = 10      # convergence-sweep resolution (keeps MSD recomputes modest)

# <salt>_<solvent>_npt_<conc>M_<temp>K   e.g. napf6_dme_npt_1M_298K, lipf6_dme_npt_0_5M_273K
SYS_RE = re.compile(r"^([a-z]+[0-9]*)_([a-z]+)_npt_(\d+(?:_\d+)?)M_(\d+)K$")

# ── discover the union of <system dir> across all replicas ──────────────────────
# {system_dir_name: {replica_label: traj_path}}
_systems: dict[str, dict[str, str]] = {}
for rlabel, rroot in REPLICA_DIRS.items():
    if not rroot.is_dir():
        continue
    for sys_dir in sorted(p for p in rroot.glob("*/") if p.is_dir()):
        if not SYS_RE.match(sys_dir.name):
            continue
        traj = sys_dir / f"{sys_dir.name}.traj"
        if not traj.exists():
            continue
        _systems.setdefault(sys_dir.name, {})[rlabel] = str(traj)

SYSTEMS = []
for name in sorted(_systems):
    m = SYS_RE.match(name)
    salt, solvent_tok, conc_tok, temp_tok = m.groups()
    if salt not in ION_MAP or solvent_tok not in SOLVENT_SYMBOL:
        continue
    cat_symbol, anion_symbol = ION_MAP[salt]
    solvent_symbol = SOLVENT_SYMBOL[solvent_tok]
    concentration_M = float(conc_tok.replace("_", "."))   # 0_5 -> 0.5, 1 -> 1.0
    temperature_K   = float(temp_tok)                     # 298, 273, 323

    # n_frames sized so the subsampled MSD lag step == TARGET_MSD_DT_PS
    n_frames = max(1, round((MAX_TRAJ_NS - EQ_CUT_NS) * 1000.0 / TARGET_MSD_DT_PS))

    traj_paths = {r: _systems[name][r] for r in REPLICA_DIRS if r in _systems[name]}
    SYSTEMS.append({
        "name":            name,
        "traj_paths":      traj_paths,
        "model_colors":    dict(REPLICA_COLORS),
        "dt_fs":           DT_FS,
        "max_traj_ns":     MAX_TRAJ_NS,
        "n_frames":        n_frames,           # MSD subsample
        "temperature_K":   temperature_K,
        "concentration_M": concentration_M,    # -> exp matched at the right conc
        # ── species (required by msd / conductivity) ──
        "cat_symbol":      cat_symbol,
        "anion_symbol":    anion_symbol,
        "solvent_symbol":  solvent_symbol,
        # ── msd ──
        "eq_cut_ns":       EQ_CUT_NS,
        "fit_pct":         FIT_PCT,
        "tau_min_fit_ns":  TAU_MIN_FIT_NS,
        "slide_window_ns": SLIDE_WINDOW_NS,
        "slide_step_ns":   SLIDE_STEP_NS,
        "n_conv_points":   N_CONV_POINTS,
        # ── conductivity ──
        "conductivity_T_K":        temperature_K,
        "conductivity_z_cat":      Z_CAT,
        "conductivity_z_anion":    Z_ANION,
        "conductivity_eq_cut_ns":  EQ_CUT_NS,
        "conductivity_load_dt_ps": LOAD_DT_PS,
        # mdcraft diffusive fit window 30-100 ps (fit_start_ns / fit_stop_ns)
        "conductivity_tau_min_ns": COND_TAU_MIN_NS,
        "conductivity_tau_max_ns": COND_TAU_MAX_NS,
        # ── cell size ──
        "cell_size_analyze_dt_ps": 5.0,        # NPT box: sample every 5 ps (~400 pts)
    })

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/eleyte_temp"
)
# resume: MSD already finished in the m5250 run; conductivity only now
ANALYSES = ["conductivity"]
WORKERS  = 11   # parallel across the 11 systems; each worker runs its 5 replicas serially
