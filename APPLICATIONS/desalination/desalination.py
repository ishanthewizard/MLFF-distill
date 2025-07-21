#!/usr/bin/env python3
import os
import re
import subprocess
from collections import defaultdict
import numpy as np

from ase.io.lammpsdata import read_lammps_data
from utils import write_lammps_data_with_angles_dihedrals  # your custom writer

# ---------------- User Controls ----------------
INPUT_STRUCTURE = "APPLICATIONS/geometry.dat"
TMP_DIR         = "lmp_tmp_GRAPHENE_SALT"
OUTPUT_TRAJ     = "salt_graphene.lammpstrj"   # dump file
LOG_FILE        = "log.lammps"
LAMMPS_EXE      = "lmp"

TIMESTEP_FS     = 0.5  # Reduced from 1.0 to 0.5 fs for stability
STEPS           = 3000         # e.g. 100 ps at 1 fs; adjust
THERMO_EVERY    = 1000
DUMP_EVERY      = 2000
T_INITIAL       = 300.0
THERMOSTAT_TAU  = 100.0           # damping (fs in 'real' units)
SEED_VELOCITY   = 4928459

RELAX           = True  # Enable energy minimization
FREEZE_GRAPHENE = True
PBC             = "p p p"         # keep full PBC (adjust if using a membrane + vacuum)

# Optional: shift box origin to zero (your writer already sets 0..L)
# ----------------------------------------------

os.makedirs(TMP_DIR, exist_ok=True)

# ------------- 1. Read data file ---------------
atoms = read_lammps_data(INPUT_STRUCTURE, style='full')
n_atoms = len(atoms)
print(f"[INFO] Loaded {n_atoms} atoms.")

# ------------- 2. Extract per-atom metadata ----
types_arr = atoms.arrays['type']
unique_types = np.unique(types_arr)
print(f"[INFO] Found {len(unique_types)} unique type IDs: {unique_types.tolist()}")

# Attempt to infer labels from comment tokens in original file (if available)
# Your custom reader stores only the first file comment in atoms.info['comment'].
# To retain per-atom labels you’d need to modify the reader earlier; instead we
# approximate via masses+counts or (if you have stored nothing) let user supply.
#
# We will build a type->label map heuristically from masses if multiple carbons etc.
mass_by_type = {}
for tid in unique_types:
    idx = np.flatnonzero(types_arr == tid)[0]
    mass_by_type[tid] = atoms.get_masses()[idx]

# Attempt semantic labeling by approximate atomic masses:
def guess_symbol(m):
    # crude matching
    close = {
        12.01: "C",
        1.008: "H",
        15.999: "O",
        22.99: "Na",
        35.45: "Cl"
    }
    for ref, sym in close.items():
        if abs(m - ref) < 0.2:
            return sym
    return f"T{int(round(m))}"

type_label = {}
label_counts = defaultdict(int)
for tid, m in mass_by_type.items():
    sym = guess_symbol(m)
    # disambiguate repeating elements (e.g., multiple carbon “flavors”)
    label_counts[sym] += 1
    type_label[tid] = sym

print("[INFO] Provisional type labels:")
for tid in sorted(type_label):
    print(f"  type {tid}: {type_label[tid]} (mass ~ {mass_by_type[tid]:.4f})")


# ------------- 4. Write cleaned data file -------
data_file = os.path.join(TMP_DIR, "system.data")
write_lammps_data_with_angles_dihedrals(data_file, atoms)
print(f"[INFO] Wrote LAMMPS data: {data_file}")

# ------------- 5. Force field parameters --------
# LJ eps (kcal/mol) and sigma (Å) for like-like; cross via mixing rule.
# Placeholder TIP3P and generic carbon / ion values.
params_lj = {
    "O":  (0.1521, 3.1507),
    "H":  (0.0000, 0.0000),  # TIP3P H has no LJ interactions
    "C":  (0.0700, 3.4000),
    "Na": (0.1301, 2.5900),
    "Cl": (0.1000, 4.8300),
}
# Bond & angle (TIP3P)
bond_coeff = (1, 450.0, 0.9572)      # (type, k, r0)
angle_coeff = (1, 55.0, 104.52)      # (type, k, theta0)

# ------------- 6. Generate LAMMPS input script --
input_file = os.path.join(TMP_DIR, "in.lammps")

def _pair_coeff_lines():
    lines = []
    for tid, elem_name in type_label.items():
        if elem_name in params_lj:
            eps, sig = params_lj[elem_name]
            lines.append(f"pair_coeff {tid} {tid} {eps:.6f} {sig:.6f}  # {elem_name}")
    return "\n".join(lines)

freeze_group_line = ""
freeze_fix_line = ""
if FREEZE_GRAPHENE:
    # right after you build type_label:
    carbon_tids = [tid for tid, sym in type_label.items() if sym == "C"]
    freeze_group_line = f"group graphene type {' '.join(map(str, carbon_tids))}"
    freeze_fix_line   = "fix freeze graphene setforce 0.0 0.0 0.0"
    mobile_group_line = "group mobile subtract all graphene"
else:
    freeze_group_line = "group mobile all"

mobile_group_line = "group mobile subtract all graphene" if freeze_group_line.startswith("group graphene") else "group mobile all"

min_lines = "minimize 1.0e-4 1.0e-6 1000 10000" if RELAX else ""

lammps_in = f"""
# ---------- Simulation: Salt water through graphene ----------
units           real
atom_style      full
boundary        {PBC}

read_data       system.data

# (Masses read from data file)
# -- Pair style & electrostatics --
pair_style      lj/cut/coul/long 10.0
pair_modify     mix geometric
kspace_style    pppm 1.0e-4

# Self pair coefficients (cross via mix rule)
{_pair_coeff_lines()}

# Bonded
bond_style      harmonic
angle_style     harmonic
bond_coeff      {bond_coeff[0]} {bond_coeff[1]:.3f} {bond_coeff[2]:.4f}
angle_coeff     {angle_coeff[0]} {angle_coeff[1]:.3f} {angle_coeff[2]:.2f}

special_bonds   lj/coul 0.0 0.0 0.5

# Neighbor list settings for stability
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# Groups
{freeze_group_line}
{mobile_group_line}
{freeze_fix_line}

# Optional minimization
{min_lines}

# Reset timestep after minimization
reset_timestep  0

# Velocities
velocity        mobile create {T_INITIAL:.1f} {SEED_VELOCITY} dist gaussian

# Integrator / Thermostat
timestep        {TIMESTEP_FS:.3f}
fix             nvt mobile nvt temp {T_INITIAL:.1f} {T_INITIAL:.1f} {THERMOSTAT_TAU:.1f}

thermo_style    custom step temp etotal pe ke press density
thermo          {THERMO_EVERY}

dump            traj all custom {DUMP_EVERY} {os.path.basename(OUTPUT_TRAJ)} id type x y z q
dump_modify     traj sort id

run             {STEPS}

write_data      final_system.data
"""

with open(input_file, "w") as f:
    f.write(lammps_in.strip() + "\n")

print(f"[INFO] Wrote LAMMPS input script: {input_file}")

# ------------- 7. Run LAMMPS -------------------
print("[INFO] Running LAMMPS...")
try:
    subprocess.run(f"{LAMMPS_EXE} -in in.lammps",
                   cwd=TMP_DIR, shell=True, check=True)
    print("[INFO] LAMMPS finished.")
except subprocess.CalledProcessError as e:
    print(f"[ERROR] LAMMPS exited with code {e.returncode}. See {os.path.join(TMP_DIR, LOG_FILE)}")
    raise

# Move trajectory to main dir
produced_traj = os.path.join(TMP_DIR, os.path.basename(OUTPUT_TRAJ))
if os.path.exists(produced_traj):
    os.replace(produced_traj, OUTPUT_TRAJ)
    print(f"[INFO] Trajectory moved to: {OUTPUT_TRAJ}")

print("[DONE] Simulation complete. Files in:", TMP_DIR)
