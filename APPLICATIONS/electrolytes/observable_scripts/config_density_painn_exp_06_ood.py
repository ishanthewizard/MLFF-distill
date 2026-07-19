"""Config: DENSITY only for the PAINN electrolytes_data `exp_06_ood` set.

Only the NPT runs are analyzed (NVT has a fixed box -> constant, uninformative
density). For each system we frame-average the density over the FULL 20 ns
trajectory, sampling ~200 frames (stride ~1000 over the 200k-frame trajectories).

Directory layout under `_BASE`:
    exp_06_ood/<outer>/<salt>_<solvent>/npt_<conc>M_<temp>K_len20ns_dt100fs/<run>.traj

`<outer>` is a pairing label that may repeat the same physical `<salt>_<solvent>`
across different outer dirs, so the per-system identity is `<outer>__<salt>_<solvent>`
to stay unique. dt_fs=100, max_traj_ns=20 per the analyze-md notes.

Run with:
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_density_painn_exp_06_ood.py
"""
import re
from pathlib import Path

_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data/exp_06_ood"
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

DT_FS = 100.0          # tdump = 0.1 ps -> 100 fs/frame
MAX_TRAJ_NS = 20.0     # eval.py caps to actual traj length (200k frames -> 20 ns)
NPT_COLOR = "#1f77b4"

SYSTEMS = []
for outer_dir in sorted(p for p in _BASE.glob("*/") if p.is_dir()):
    for sys_dir in sorted(p for p in outer_dir.glob("*/") if p.is_dir()):
        m = SYSTEM_DIR_RE.match(sys_dir.name)
        if not m:
            continue
        salt, solvent_tok = m.groups()
        if salt not in ION_MAP or solvent_tok not in SOLVENT_SYMBOL:
            continue
        cat_symbol, anion_symbol = ION_MAP[salt]
        solvent_symbol = SOLVENT_SYMBOL[solvent_tok]

        # NPT only
        npt_traj = None
        concentration = temperature_K = None
        for run_dir in sorted(p for p in sys_dir.glob("*/") if p.is_dir()):
            mr = RUN_DIR_RE.match(run_dir.name)
            if not mr or mr.group(1) != "npt":
                continue
            traj_path = run_dir / f"{run_dir.name}.traj"
            if not traj_path.exists():
                continue
            npt_traj = str(traj_path)
            concentration = float(mr.group(2).replace("_", "."))
            temperature_K = float(mr.group(3))

        if npt_traj is None:
            continue

        SYSTEMS.append({
            "name":            f"{outer_dir.name}__{salt}_{solvent_tok}",
            "traj_paths":      {"npt": npt_traj},
            "dt_fs":           DT_FS,
            "max_traj_ns":     MAX_TRAJ_NS,
            # full 20 ns, ~200 frames (stride ~1000 over 200k frames)
            "skip_ns":         0.0,
            "window_ns":       20.0,
            "n_frames":        200,
            "model_colors":    {"npt": NPT_COLOR},
            "density_roll_window_ns": 1.0,
            # carried through for the downstream parity CSV
            "salt":            salt,
            "solvent_tok":     solvent_tok,
            "concentration_M": concentration,
            "temperature_K":   temperature_K,
        })

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data/analysis/ood_results"
)
ANALYSES = ["density"]
WORKERS  = 5
