"""Config: MSD/diffusivity for all OPLS NPT production systems.

Walks .../simulation_results/OPLS/npt/<salt>_<solvent>_<conc>M_<temp>K/
(GROMACS npt.xtc + npt.tpr, dt=1 fs, frames every 1 ps -> dt_fs=1000) and
runs the MSD/diffusivity analysis (incl. Yeh-Hummer correction + parity vs
experimental diffusivity.csv).
"""
import re
from pathlib import Path

# NOTE: MDAnalysis's offset-cache flock() can hang indefinitely on the CFS
# filesystem -- if this happens, rsync the relevant system dirs to a local
# /pscratch copy (see config_msd_opls_nvt.py) and point _ROOT there instead.
_ROOT = Path("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/simulation_results/OPLS/npt")

# directory-name salt token -> (cat_symbol, anion_symbol)  (component_dictionary keys)
ION_MAP = {
    "lipf6": ("Li", "PF6"),
    "napf6": ("Na", "PF6"),
    "naotf": ("Na", "OTf"),
}

# directory-name solvent token -> component_dictionary solvent key
SOLVENT_DICT_KEY = {
    "dme":     "DME",
    "diglyme": "Diglyme",
    "tegdme":  "TGDME",
    "tgdme":   "TGDME",
    "pc":      "PC",
}

_DIRNAME_RE = re.compile(r"^([a-z0-9]+)_([a-z]+)_([\d.]+)M_(\d+)K$")

_SHARED = {
    "dt_fs":          1000.0,   # 1 fs MD step, frames every 1000 steps -> 1 ps/frame
    "max_traj_ns":    20.0,
    "eq_cut_ns":      2.0,
    "fit_pct":        0.8,
    "tau_min_fit_ns": 1.0,
    "n_frames":       2000,
    "n_conv_points":  200,
    "slide_window_ns": 5.0,
    "slide_step_ns":  0.5,
    "model_colors": {"OPLS": "#2ca02c"},
}

SYSTEMS = []
for sys_dir in sorted(_ROOT.glob("*/")):
    m = _DIRNAME_RE.match(sys_dir.name)
    if not m:
        continue
    salt, solvent_tok, conc, temp_label = m.groups()
    if salt not in ION_MAP or solvent_tok not in SOLVENT_DICT_KEY:
        continue
    cat_symbol, anion_symbol = ION_MAP[salt]
    solvent_symbol = SOLVENT_DICT_KEY[solvent_tok]

    xtc = sys_dir / "npt.xtc"
    tpr = sys_dir / "npt.tpr"
    if not xtc.exists() or not tpr.exists():
        continue

    SYSTEMS.append({
        **_SHARED,
        "name":             sys_dir.name,
        "traj_paths":       {"OPLS": {"xtc": str(xtc), "tpr": str(tpr)}},
        "cat_symbol":       cat_symbol,
        "anion_symbol":     anion_symbol,
        "solvent_symbol":   solvent_symbol,
        "concentration_M":  float(conc),
        "temperature_K":    float(temp_label),
    })

# for a quick single-system test: set _TEST_SINGLE to a system name substring
import os
_TEST_SINGLE = os.environ.get("MSD_TEST_SINGLE")
if _TEST_SINGLE:
    SYSTEMS = [s for s in SYSTEMS if _TEST_SINGLE in s["name"]]

OUTPUT_DIR = "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/analysis/opls_npt_msd"
ANALYSES   = ["msd"]
WORKERS    = 1 if _TEST_SINGLE else 6
