"""Config: temperature (energy) timeseries for the PAINN electrolytes_data
`simulation` set.

The `energy` analysis is the one that yields temperature: it produces a
4-panel timeseries per (system, run) — PE / KE / E_tot / **Temperature** — and
saves the raw arrays (incl. `temp`) to `energy_<run>.npz`.

Directory layout under `_BASE`:

    simulation/<salt>_<solvent>/<npt|nvt>_<conc>M_<temp>K_<dur>_<dtlabel>/<run>.traj

One SYSTEMS entry per `<salt>_<solvent>`; every run inside it becomes a "model"
labelled `<ensemble>_<conc>M_<temp>K`, so each run gets its own energy/temperature
PNG + npz in the system's output folder. The `lipf6_dme` system spans 273/298/323 K
(npt + nvt); the rest are 298 K at 0.1 M / 1 M.

dt_fs / max_traj_ns are hardcoded per the analyze-md notes: despite the dirname
labels (`1ps`, `100fs`, `1ns`/`2ns`/`20ns`), every run's `.fennol.yaml` has
dt[fs]=1.0, tdump[ps]=0.1 -> dt_fs=100 (0.1 ps/saved-frame). eval.py caps
max_traj_ns to the actual trajectory length, so 20.0 means "analyze the whole run".

Run with:
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_painn_electrolytes_simulation_temperature.py
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

SYSTEM_DIR_RE = re.compile(r"^([a-z0-9]+)_([a-z]+)$")
RUN_DIR_RE = re.compile(r"^(npt|nvt)_(\d+(?:_\d+)?)M_(\d+)K_")

DT_FS = 100.0          # tdump = 0.1 ps -> 100 fs/saved-frame (dirname label unreliable)
MAX_TRAJ_NS = 20.0     # eval.py caps to actual traj length

SYSTEMS = []
for sys_dir in sorted(p for p in _BASE.glob("*/") if p.is_dir()):
    m = SYSTEM_DIR_RE.match(sys_dir.name)
    if not m:
        continue
    salt, solvent_tok = m.groups()
    if salt not in ION_MAP or solvent_tok not in SOLVENT_SYMBOL:
        continue

    traj_paths = {}
    for run_dir in sorted(p for p in sys_dir.glob("*/") if p.is_dir()):
        mr = RUN_DIR_RE.match(run_dir.name)
        if not mr:
            continue
        ensemble, conc_tok, temp_tok = mr.groups()
        traj_path = run_dir / f"{run_dir.name}.traj"
        if not traj_path.exists():
            continue
        conc = conc_tok.replace("_", ".")
        label = f"{ensemble}_{conc}M_{temp_tok}K"   # unique per run within a system
        traj_paths[label] = str(traj_path)

    if not traj_paths:
        continue

    SYSTEMS.append({
        "name":        sys_dir.name,
        "traj_paths":  traj_paths,
        "dt_fs":       DT_FS,
        "max_traj_ns": MAX_TRAJ_NS,
        "n_frames":    2000,    # frames sampled per run for the timeseries
        "skip_ns":     0.5,
    })

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data/analysis"
)
# energy -> PE/KE/E_tot/**Temperature** 4-panel timeseries + energy_<run>.npz (temp inside)
ANALYSES = ["energy"]
WORKERS  = 4   # parallel across systems; energy is not MAE so no teacher re-inference
