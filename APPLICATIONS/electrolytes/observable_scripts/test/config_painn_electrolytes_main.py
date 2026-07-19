"""Config: cell size, energy/temperature, pressure, and MSD for the PAINN
electrolytes_data simulation set.

Walks `_BASE` for every `<salt>_<solvent>/<npt|nvt>_<conc>M_<temp>K_<dur>ns_<dt>` run
and builds one SYSTEMS entry per `.traj` found, with cation/anion/solvent symbols,
concentration, temperature, and dt_fs inferred from the directory names.

NOTE: this config only defines SYSTEMS/ANALYSES — it is not run automatically.
Run with:
  python eval.py --config config_painn_electrolytes_main.py
"""
import re
from pathlib import Path

_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data/simulation"
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

# NOTE: directory-name suffixes (e.g. "1ns_1ps", "2ns_100fs", "20ns_100fs") do
# NOT reflect the actual sampling -- every run's .fennol.yaml has
# dt[fs]=1.0, tdump[ps]=0.1 (i.e. dt_fs=100) and nsteps=20e6 (20 ns target),
# regardless of the dirname. dt_fs/max_traj_ns below are hardcoded to match
# the actual yaml configs; eval.py caps max_traj_ns to the real trajectory
# length anyway.
SYSTEM_DIR_RE = re.compile(r"^([a-z0-9]+)_([a-z]+)$")
RUN_DIR_RE = re.compile(r"^(npt|nvt)_(\d+(?:_\d+)?)M_(\d+)K_")
# nsteps=20e6 fs @ dt[fs]=1.0 -> 20ns target; eval.py caps to the actual
# trajectory length via min(max_traj_ns, n_frames*dt_fs*1e-6) (~14.86ns for
# the ~148k-frame files), so 20.0 here means "analyze the whole trajectory".
DT_FS = 100.0
MAX_TRAJ_NS = 20.0

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
        max_traj_ns = MAX_TRAJ_NS
        dt_fs = DT_FS

        SYSTEMS.append({
            "name":           f"{salt}_{solvent_tok} {ensemble} {concentration}M {temperature_K:.0f}K ({run_dir.name})",
            "traj_paths":     {"PAINN": str(traj_path)},
            "dt_fs":          dt_fs,
            "max_traj_ns":    max_traj_ns,
            "n_frames":       2000,
            "model_colors":   {"PAINN": "#1f77b4"},
            "temperature_K":  temperature_K,
            "cat_symbol":     cat_symbol,
            "anion_symbol":   anion_symbol,
            "solvent_symbol": solvent_symbol,
            "eq_cut_ns":      min(0.1, 0.05 * max_traj_ns),
            "fit_pct":        0.8,
            "tau_min_fit_ns": 0.05 * max_traj_ns,
            "slide_window_ns": 0.5 * max_traj_ns,
            "slide_step_ns":  0.05 * max_traj_ns,
            "n_conv_points":  20,
        })

# Per-trajectory analysis output (eval.py creates one subdir per system under
# a timestamped run dir here). Group parity output goes to a separate dir --
# see group_eval/groups_painn_electrolytes_main.py's OUT_DIR.
OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data/analysis/per_traj"
)
ANALYSES = ["msd"]
WORKERS  = 4
