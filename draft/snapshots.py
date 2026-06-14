"""
snapshots.py — batch GIF renderer for MD trajectories.

Edit TRAJ_PATHS and GIF_PARAMS below, then run:
    python snapshots.py
"""

import sys, os, re
sys.path.insert(0, '/pscratch/sd/y/yuejian/envs/fairchemV2/lib/python3.10/site-packages')
sys.path.insert(0, '/global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/plots/MD_vis')

from ase.io import Trajectory
from vis import save_gif_3d

# ── Input trajectories ────────────────────────────────────────────────────────
_NVT_BASE = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/exp_density_nvt/simulations"
)
_NVT_SYSTEMS = [
    "li_pf6_dme_323K_0.5M",
    "na_otf_dme_298K_0.1M",
    "na_pf6_dme_298K_0.1M",
    "na_pf6_dme_323K_0.5M",
]
_NVT_THERMOSTATS = ["berendsen", "nose_hover"]
_NVT_ROLES       = ["student", "teacher"]

TRAJ_PATHS = [
    f"{_NVT_BASE}/{thermo}/{sys}_{role}/{sys}_{role}.traj"
    for thermo in _NVT_THERMOSTATS
    for sys    in _NVT_SYSTEMS
    for role   in _NVT_ROLES
]

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/exp_density_nvt/gif"
)

# ── GIF parameters (shared across all trajectories) ──────────────────────────
# dt = 100 fs/frame; 3 ns = 30 000 frames; stride 300 → ~100 frames per GIF
GIF_PARAMS = dict(
    start       = 0,
    end         = 30000,
    stride      = 300,
    hide_H      = True,
    fps         = 5,
    size        = 400,
    elev        = 20,
    azim        = 45,
    spin        = False,
    n_workers   = 4,
    atom_scale  = 0.5,
    timestep_fs = 100,
)

# ── Name parsing ──────────────────────────────────────────────────────────────
KNOWN_CATIONS  = ['na', 'li', 'k', 'mg', 'ca']
KNOWN_ANIONS   = ['pf6', 'tfsi', 'fsi', 'bf4', 'otf', 'dca', 'no3']
KNOWN_SOLVENTS = ['diglyme', 'dme', 'ec', 'dmc', 'pc', 'thf', 'acn', 'dmso', 'water']
_CONC_RE       = re.compile(r'(\d+(?:_\d+)?)M')

def parse_traj_name(traj_path):
    parts = traj_path.replace('\\', '/').split('/')

    temperature = next((p for p in parts if re.fullmatch(r'\d+K', p)), 'unknownK')

    conc_part     = next((p for p in parts if _CONC_RE.search(p)), '')
    conc_raw      = _CONC_RE.search(conc_part).group(1) if conc_part else None
    concentration = (conc_raw.replace('_', '.') + 'M') if conc_raw else 'unknownM'

    if   any('micro'  in p for p in parts): model = 'micro'
    elif any('origin' in p or 'orig' in p for p in parts): model = 'original'
    else: model = 'unknown'

    run_dir = os.path.basename(os.path.dirname(traj_path)).lower()
    cation  = next((c for c in KNOWN_CATIONS  if c in run_dir), 'unknown')
    anion   = next((a for a in KNOWN_ANIONS   if a in run_dir), 'unknown')
    solvent = next((s for s in KNOWN_SOLVENTS if s in run_dir), 'unknown')

    return temperature, concentration, model, cation, anion, solvent

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = len(TRAJ_PATHS)

    for idx, traj_path in enumerate(TRAJ_PATHS, 1):
        print(f"\n[{idx}/{total}] {traj_path}")

        if not os.path.exists(traj_path):
            print(f"  SKIP — file not found")
            continue

        traj = Trajectory(traj_path)
        print(f"  Opened: {len(traj)} frames")

        # Use folder name + thermostat directly for exp_density_nvt paths
        parts = traj_path.replace('\\', '/').split('/')
        folder = os.path.basename(os.path.dirname(traj_path))
        thermostat = parts[-3] if len(parts) >= 3 else "unknown"
        gif_name = f"{thermostat}_{folder}_3d.gif"
        out_path = os.path.join(OUTPUT_DIR, gif_name)
        print(f"  Parsed: {thermostat} | {folder}")
        print(f"  Output: {gif_name}")

        save_gif_3d(traj, output_path=out_path,
                    label=os.path.splitext(gif_name)[0],
                    **GIF_PARAMS)

    print(f"\nDone. {total} trajectory/ies processed → {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
