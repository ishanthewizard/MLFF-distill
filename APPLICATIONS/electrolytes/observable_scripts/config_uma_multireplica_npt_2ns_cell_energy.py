"""Config: cell_size + potential-energy timeseries for the UMA NPT
"concentrate" 2 ns set (UMA_multi_replicas/npt/npt_2ns).

Same discovery / layout as config_uma_multireplica_npt_2ns.py, but restricted to
the two cheap per-trajectory analyses the user asked for:

  ANALYSES = ["cell_size", "energy"]

Layout (ONE dir per system per replica, traj is `<dir>.traj` inside it):

  npt/npt_2ns/UMA_ref_concentrate_rep{1..5}/<salt>_<solvent>_npt_<conc>M_<temp>K/
        └── <salt>_<solvent>_npt_<conc>M_<temp>K.traj

The 5 replicas (rep1..rep5) become the "models" (`replica_1..replica_5`) of each
unique system, so all 5 replicas are overlaid on the same axes.
11 systems x 5 replicas = 55 trajectories.

Trajectory params (user):
  - dt_fs        = 10.0     (10 fs / saved-frame)
  - max_traj_ns  = 2.1      (analyze the first 2.1 ns; eval.py caps to real length;
                             the actual trajs are ~2.6 ns / 260k frames)

Sampling (both analyses subsample to ~5 ps / frame -> ~420 pts over 2.1 ns):
  - cell_size: cell_size_analyze_dt_ps = 5.0
  - energy:    uses the global n_frames as the sample count. n_frames sized so the
               energy timeseries is also ~5 ps-spaced:
                   n_frames = (max_traj_ns * 1000) / 5.0 = 2100/5 = 420
               (energy samples over [0, max_ns], no eq_cut, so it uses max_traj_ns
               not max_traj_ns-eq_cut.)

Both analyses read the energy/cell straight off the stored ASE calc results
(UMA trajs store energy+forces+stress per frame) -- no re-inference, so this is a
fast, GPU-free job. Still, 55 slow CFS trajectories x random-access reads: run on a
SLURM CPU node or expect a while on a login node.

Run with:
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_uma_multireplica_npt_2ns_cell_energy.py
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

# salt -> (cation, anion) ; solvent token -> solvent symbol
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

# ── sampling: ~5 ps/frame for both cell_size and energy ────────────────────────
SAMPLE_DT_PS  = 5.0
CELL_N_FRAMES = None   # let cell_size_analyze_dt_ps drive the stride
# energy uses the global n_frames as its sample count -> ~5 ps spacing over 2.1 ns
ENERGY_N_FRAMES = max(1, round(MAX_TRAJ_NS * 1000.0 / SAMPLE_DT_PS))   # 420

# <salt>_<solvent>_npt_<conc>M_<temp>K   e.g. napf6_dme_npt_1M_298K, lipf6_dme_npt_0_5M_273K
SYS_RE = re.compile(r"^([a-z]+[0-9]*)_([a-z]+)_npt_(\d+(?:_\d+)?)M_(\d+)K$")

# ── discover the union of <system dir> across all replicas ──────────────────────
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
    concentration_M = float(conc_tok.replace("_", "."))
    temperature_K   = float(temp_tok)

    traj_paths = {r: _systems[name][r] for r in REPLICA_DIRS if r in _systems[name]}
    SYSTEMS.append({
        "name":            name,
        "traj_paths":      traj_paths,
        "model_colors":    dict(REPLICA_COLORS),
        "dt_fs":           DT_FS,
        "max_traj_ns":     MAX_TRAJ_NS,
        "n_frames":        ENERGY_N_FRAMES,       # energy sample count (~5 ps spacing)
        "temperature_K":   temperature_K,
        "concentration_M": concentration_M,
        "cat_symbol":      cat_symbol,
        "anion_symbol":    anion_symbol,
        "solvent_symbol":  solvent_symbol,
        # ── cell size ──
        "cell_size_analyze_dt_ps": SAMPLE_DT_PS,  # NPT box: sample every 5 ps (~420 pts)
    })

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/UMA/UMA_multi_replicas/npt/npt_2ns/analysis"
)
ANALYSES = ["cell_size", "energy"]
WORKERS  = 11   # parallel across the 11 systems; each worker runs its 5 replicas serially
