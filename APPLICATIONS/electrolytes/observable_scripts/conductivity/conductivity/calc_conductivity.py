"""
Minimal script to load the NVT trajectory from the Transport run and compute
ionic conductivity via the Onsager formalism.

Reproduces results.json:  conductivity_onsager = 12.50 mS/cm  (paper: ~10-12 mS/cm)
                          conductivity_NE      = 18.27 mS/cm
                          Dself (DMC, EC, PF6, LI) = [4.30, 4.61, 3.23, 2.30] × 10⁻¹⁰ m²/s

System: 505 DMC + 345 EC + 69 PF6 + 69 LI  (10062 atoms, cubic box ~48.22 Å)
"""

import sys
sys.path.insert(0, '/home/yuejian/project/byteff2')

import numpy as np
import pandas as pd
from MDAnalysis.lib.formats.libdcd import DCDFile

from byteff2.md_utils.onsager_conductivity import onsager_calc

# ── paths ──────────────────────────────────────────────────────────────────────
RUN_DIR   = '/home/yuejian/project/byteff2/yuejian/bytemol/md_runs/transport_results'
DCD_PATH  = f'{RUN_DIR}/nvt.dcd'
CSV_PATH  = f'{RUN_DIR}/nvt_state.csv'

# ── system description (from system.top / itp files) ──────────────────────────
# Atom ordering in the DCD matches system.top: DMC → EC → PF6 → LI

SPECIES_ORDER = ['DMC', 'EC', 'PF6', 'LI']

SPECIES_MASS = {
    'DMC': [12.011, 15.999, 12.011, 15.999, 15.999, 12.011,
             1.008,  1.008,  1.008,  1.008,  1.008,  1.008],   # 12 atoms/mol
    'EC':  [15.999, 12.011, 15.999, 12.011, 12.011, 15.999,
             1.008,  1.008,  1.008,  1.008],                    # 10 atoms/mol
    'PF6': [18.998, 30.974, 18.998, 18.998, 18.998, 18.998, 18.998],  # 7 atoms/mol
    'LI':  [6.941],                                              # 1 atom/mol
}

SPECIES_NUMBER = {'DMC': 505, 'EC': 345, 'PF6': 69, 'LI': 69}

# Net integer charge per molecule
SPECIES_CHARGE = {'DMC': 0, 'EC': 0, 'PF6': -1, 'LI': 1}

# ── simulation parameters (read from NVT state CSV, matching protocol.py volume_calc) ─────
VISCOSITY_CP = 2.9159330758843414  # cP (from viscosity.csv via the transport run)

_df = pd.read_csv(CSV_PATH)
VOLUME_ANG3   = _df['Box Volume (nm^3)'].mean() * 1000   # nm³ → Å³
TEMPERATURE_K = _df['Temperature (K)'].mean()            # trajectory-average T

# ── load trajectory (unwrapped coords, Å) ────────────────────────────────────
print(f'Loading {DCD_PATH} ...')
frames = []
with DCDFile(DCD_PATH) as dcd:
    for frame in dcd:
        frames.append(frame.xyz.copy())
positions = np.array(frames)          # shape: (nframes, natoms, 3)  [Å, unwrapped]
print(f'  {positions.shape[0]} frames, {positions.shape[1]} atoms')

# ── compute ───────────────────────────────────────────────────────────────────
print('Running onsager_calc ...')
results = onsager_calc(
    species_order   = SPECIES_ORDER,    # Molecule/ion order in the trajectory topology.
    species_mass    = SPECIES_MASS,     # Per-atom masses for each species, used for COM motion.
    species_number  = SPECIES_NUMBER,   # Number of molecules/ions of each species in this system.
    species_charge  = SPECIES_CHARGE,   # Net integer charge per molecule/ion.
    volume_angstrom3= VOLUME_ANG3,      # Average simulation box volume in Å^3.
    viscosity_cP    = VISCOSITY_CP,     # Dynamic viscosity in cP for Stokes-Einstein correction.
    T_K             = TEMPERATURE_K,    # Average trajectory temperature in Kelvin.
    positions       = positions,        # Unwrapped atomic positions, shape (frames, atoms, 3), in Å.
)

# ── report ────────────────────────────────────────────────────────────────────
print()
print('=== Results ===')
print(f"Conductivity (Onsager): {results['conductivity_onsager']:.2f} mS/cm   "
      f"[paper reference: ~10-12 mS/cm, reproduced: 12.50]")
print(f"Conductivity (NE):      {results['conductivity_NE']:.2f} mS/cm   "
      f"[reproduced: 18.27]")
print(f"Ionicity (Onsager/NE):  {results['conductivity_onsager']/results['conductivity_NE']:.3f}  "
      f"[reproduced: 0.684]")
print(f"Dself (10⁻¹⁰ m²/s):    DMC={results['Dself_inf'][0]:.3f}  EC={results['Dself_inf'][1]:.3f}  "
      f"PF6={results['Dself_inf'][2]:.3f}  LI={results['Dself_inf'][3]:.3f}")
print(f"Box length:             {results['cubic_box_length']:.4f} Å")
