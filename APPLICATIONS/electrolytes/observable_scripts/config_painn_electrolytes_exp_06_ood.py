"""Config: MSD, conductivity, pressure, energy/temperature, and cell size for the
PAINN electrolytes_data `exp_06_ood` simulation set.

Directory layout under `_BASE`:

    exp_06_ood/<outer>/<salt>_<solvent>/<npt|nvt>_<conc>M_<temp>K_len20ns_dt100fs/<run>.traj

`<outer>` is a batch/pairing label that may repeat the same physical
`<salt>_<solvent>` across different outer dirs, so the system identity (and the
per-system output folder name) is `<outer>__<salt>_<solvent>` to stay unique.
Within one (outer, salt_solvent) the `npt` and `nvt` runs are grouped as two
*models* of one SYSTEMS entry so their pressure / cell / energy / MSD /
conductivity land in overlaid comparison plots and shared CSVs.

Species / concentration / temperature are inferred from the dir names via the
lookup tables below. dt_fs / max_traj_ns are hardcoded per the analyze-md notes:
every run's `.fennol.yaml` has dt[fs]=1.0, tdump[ps]=0.1 (-> dt_fs=100) and
nsteps=20e6 (20 ns target) regardless of the dirname; eval.py caps max_traj_ns
to the actual trajectory length, so 20.0 means "analyze the whole trajectory".

Run with (env has ase + byteff2 + numpy/pandas/matplotlib):
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_painn_electrolytes_exp_06_ood.py
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
MAX_TRAJ_NS = 20.0     # eval.py caps to actual traj length
ENSEMBLE_COLOR = {"npt": "#1f77b4", "nvt": "#d62728"}

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

        traj_paths = {}
        concentration = temperature_K = None
        for run_dir in sorted(p for p in sys_dir.glob("*/") if p.is_dir()):
            mr = RUN_DIR_RE.match(run_dir.name)
            if not mr:
                continue
            ensemble, conc_tok, temp_tok = mr.groups()
            traj_path = run_dir / f"{run_dir.name}.traj"
            if not traj_path.exists():
                continue
            traj_paths[ensemble] = str(traj_path)
            concentration = float(conc_tok.replace("_", "."))
            temperature_K = float(temp_tok)

        if not traj_paths:
            continue

        # deterministic model order: npt first, then nvt
        ordered = {k: traj_paths[k] for k in ("npt", "nvt") if k in traj_paths}
        ordered.update({k: v for k, v in traj_paths.items() if k not in ordered})
        colors = {k: ENSEMBLE_COLOR.get(k, "#2ca02c") for k in ordered}

        SYSTEMS.append({
            "name":            f"{outer_dir.name}__{salt}_{solvent_tok}",
            "traj_paths":      ordered,
            "dt_fs":           DT_FS,
            "max_traj_ns":     MAX_TRAJ_NS,
            "n_frames":        2000,
            "model_colors":    colors,
            "skip_ns":         0.5,
            "concentration_M": concentration,
            "temperature_K":   temperature_K,
            # ── species (MSD + conductivity) ──
            "cat_symbol":      cat_symbol,
            "anion_symbol":    anion_symbol,
            "solvent_symbol":  solvent_symbol,
            # ── MSD / diffusivity ──
            "eq_cut_ns":       0.5,
            "fit_pct":         0.8,
            "tau_min_fit_ns":  1.0,
            "slide_window_ns": 10.0,
            "slide_step_ns":   1.0,
            "n_conv_points":   20,
            # ── conductivity (Onsager / Nernst-Einstein) ──
            "conductivity_T_K":       temperature_K or 298.0,
            "conductivity_z_cat":     1.0,
            "conductivity_z_anion":  -1.0,
            "conductivity_eq_cut_ns": 0.5,
            "conductivity_tau_min_ns": 1.0,
            # ── pressure / cell size (sample every 10 ps -> ~2000 pts over 20 ns) ──
            "pressure_analyze_dt_ps":  10.0,
            "cell_size_analyze_dt_ps": 10.0,
        })

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data/analysis"
)
# energy -> potential-energy + temperature timeseries (4-panel: PE/KE/Etot/T)
ANALYSES = ["msd", "conductivity", "pressure", "energy", "cell_size"]
WORKERS  = 4
