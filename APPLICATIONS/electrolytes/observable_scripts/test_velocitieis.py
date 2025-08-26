#!/usr/bin/env python3
import numpy as np
from ase.io.trajectory import Trajectory

# --- constants (SI) ---
k_B = 1.380649e-23            # J/K
amu_to_kg = 1.66053906660e-27 # kg per amu
A_to_m = 1e-10
ps_to_s = 1e-12
fs_to_s = 1e-15
eV_to_J = 1.602176634e-19

SOLVENTS = ["DME", "DG", "DMC", "TGDME", "PC", "THF"]
TRAJS = [f"/projects/beye/iamin/trajs/napf6_{s}_1ns.traj" for s in SOLVENTS]
SPECIES = "Na"

def temp_from_velocities(atoms, vel, assume="A/ps"):
    """Compute temperature from velocities using SI, honoring constraints DOF."""
    N = len(atoms)
    dof = atoms.get_number_of_degrees_of_freedom()  # accounts for constraints
    m = atoms.get_masses()[:, None] * amu_to_kg     # (N,1) kg

    if assume == "A/ps":
        v_si = vel * A_to_m / ps_to_s
    elif assume == "A/fs":
        v_si = vel * A_to_m / fs_to_s
    else:
        raise ValueError("assume must be 'A/ps' or 'A/fs'")

    Ek_J = 0.5 * np.sum(m * (v_si**2))  # total J
    T = 2.0 * Ek_J / (dof * k_B)        # K
    return T, Ek_J

for path, solv in zip(TRAJS, SOLVENTS):
    with Trajectory(path) as tr:
        # Use a later frame to avoid early transients; fallback to first if short
        i = min(50, len(tr)-1) if len(tr) > 0 else 0
        atoms = tr[i]

    vel = atoms.get_velocities()  # shape (N,3) or None
    if vel is None:
        print(f"{solv}: velocities not stored in trajectory.")
        continue

    # ASE-reported values
    T_reported = atoms.get_temperature()                           # K
    Ek_eV = atoms.get_kinetic_energy()                             # eV
    Ek_J_from_ase = Ek_eV * eV_to_J                                # J

    # Our two interpretations
    T_ps,  Ek_J_ps  = temp_from_velocities(atoms, vel, "A/ps")
    T_fs,  Ek_J_fs  = temp_from_velocities(atoms, vel, "A/fs")

    # Magnitudes for sanity
    vmag = np.linalg.norm(vel, axis=1)
    med_comp = np.median(np.abs(vel))      # median component magnitude
    med_mag  = np.median(vmag)             # median speed magnitude

    print(f"\n=== {solv} @ frame {i} ===")
    print(f"ASE report:  T = {T_reported:8.2f} K | Ek = {Ek_eV:10.3f} eV ({Ek_J_from_ase:.3e} J)")
    print(f"Assume Å/ps: T = {T_ps:8.2f} K | Ek = {Ek_J_ps:.3e} J")
    print(f"Assume Å/fs: T = {T_fs:8.2f} K | Ek = {Ek_J_fs:.3e} J")
    print(f"vel stats: median |component| = {med_comp:.3e} (raw units), median |v| = {med_mag:.3e} (raw units)")
    # Hint: whichever T (Å/ps vs Å/fs) matches ASE's T is the correct unit interpretation for this trajectory.
