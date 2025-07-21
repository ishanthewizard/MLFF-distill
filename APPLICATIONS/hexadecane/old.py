#!/usr/bin/env python3
"""
Corrected script for liquid n-hexadecane diffusion using TraPPE-UA in ASE+LAMMPS.
"""

import numpy as np
from ase import Atoms, Atom
from ase.io import write
import os
from ase.calculators.lammpsrun import LAMMPS
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import ase.units as units

# Ensure the LAMMPS executable is in your PATH or set the environment variable
# os.environ['ASE_LAMMPSRUN_COMMAND'] = 'lmp'

# --- 1) Build a single United-Atom (UA) n-hexadecane molecule ---
# In TraPPE-UA, n-hexadecane is 16 pseudo-atoms.
n_ua_atoms = 16
symbols_ua = ['X'] * n_ua_atoms # Use a placeholder symbol like 'X'
positions_ua = np.zeros((n_ua_atoms, 3))
bonds_ua = []
angles_ua = []
dihedrals_ua = []
atom_types_ua = []

# Define bond length and angle for a simple straight chain (will be relaxed by MD)
bond_length = 1.54  # Å
angle_deg = 114.0

# Create a simple linear chain geometry
for i in range(n_ua_atoms):
    if i == 0:
        positions_ua[i] = [0, 0, 0]
    elif i == 1:
        positions_ua[i] = [bond_length, 0, 0]
    else:
        # Add atoms using previous two to define angle
        p1, p2 = positions_ua[i-2], positions_ua[i-1]
        v1 = p2 - p1
        v1 /= np.linalg.norm(v1)
        # Simple rotation for the next atom
        theta = np.deg2rad(180 - angle_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
        if i % 2 == 0: # Alternate rotation axis to avoid flat chain
            positions_ua[i] = p2 + rot.T @ v1 * bond_length
        else:
            positions_ua[i] = p2 + v1 * bond_length


# Define topology and atom types (1-based for LAMMPS)
# Atom types: 1 for CH3 (ends), 2 for CH2 (middle)
for i in range(n_ua_atoms):
    # Atom types
    if i == 0 or i == n_ua_atoms - 1:
        atom_types_ua.append(1)  # CH3
    else:
        atom_types_ua.append(2)  # CH2
    # Bonds
    if i < n_ua_atoms - 1:
        bonds_ua.append((i, i + 1))
    # Angles
    if i < n_ua_atoms - 2:
        angles_ua.append((i, i + 1, i + 2))
    # Dihedrals
    if i < n_ua_atoms - 3:
        dihedrals_ua.append((i, i + 1, i + 2, i + 3))

single_mol_ua = Atoms(symbols=symbols_ua, positions=positions_ua)
masses_ua = [15.035 if atype == 1 else 14.027 for atype in atom_types_ua]
single_mol_ua.set_masses(masses_ua)

# --- 2) Build liquid box by replicating the UA molecules ---
n_x, n_y, n_z = 4, 4, 3  # 48 molecules
n_mol = n_x * n_y * n_z
box_length = 35.0 # Adjusted for UA model density ~0.77 g/cm3

all_atoms = Atoms()
for i in range(n_x):
    for j in range(n_y):
        for k in range(n_z):
            # Create a new molecule and rotate it randomly
            mol_copy = single_mol_ua.copy()
            mol_copy.rotate(np.random.rand() * 360, 'x', rotate_cell=False)
            mol_copy.rotate(np.random.rand() * 360, 'y', rotate_cell=False)
            mol_copy.rotate(np.random.rand() * 360, 'z', rotate_cell=False)
            
            # Place it in the box
            center = [i * box_length/n_x, j * box_length/n_y, k * box_length/n_z]
            mol_copy.translate(center)
            all_atoms.extend(mol_copy)

all_atoms.set_cell([box_length] * 3)
all_atoms.set_pbc(True)

# --- 3) Set up LAMMPS calculator with correct parameters ---
# Combine all parameters into one dictionary
parameters = {
    'units': 'real',
    'atom_style': 'full',
    'log': 'log.lammps',
    
    # Force Field Styles - THIS WAS THE MISSING PIECE
    'pair_style': 'lj/cut 14.0',
    'bond_style': 'harmonic',
    'angle_style': 'harmonic',
    'dihedral_style': 'opls',
    
    # Mixing rule for LJ parameters
    'pair_modify': 'mix geometric', # Or arithmetic, check FF documentation
    
    # Force Field Coefficients
    # pair_coeff: atom_type1 atom_type2 epsilon(kcal/mol) sigma(Å)
    'pair_coeff': [
        '1 1 0.1947 3.75',  # CH3-CH3
        '2 2 0.0914 3.95'   # CH2-CH2
    ],
    # bond_coeff: bond_type K(kcal/mol/Å^2) r0(Å)
    'bond_coeff': ['1 120.0 1.54'],
    # angle_coeff: angle_type K(kcal/mol/rad^2) theta0(deg)
    'angle_coeff': ['1 62.0 114.0'],
    # dihedral_coeff: dihedral_type F1 F2 F3 F4 (kcal/mol)
    'dihedral_coeff': ['1 1.411 -0.271 3.145 0.0'],
}

# The LAMMPS calculator
calc = LAMMPS(
    specorder=['X'],  # Tell LAMMPS to treat our placeholder 'X'
    atom_types={'X': 1}, # This is a placeholder, we will overwrite types below
    tmp_dir='lmp_tmp',
    keep_tmp_files=True, # Keep files for debugging
    parameters=parameters
)

atoms = all_atoms # Use the fully constructed system
atoms.calc = calc

# --- 4) Equilibrate & Production MD ---
# Set initial temperature
MaxwellBoltzmannDistribution(atoms, temperature_K=500) # Start hot to relax structure
Stationary(atoms)

# Setup Langevin dynamics
dyn = Langevin(atoms, timestep=2 * units.fs, temperature_K=298.15, friction=0.01)

def log_progress(a=atoms):
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = ekin / (1.5 * len(a) * units.kB)
    print(f'Energy per atom: Epot = {epot/len(a):.3f} eV, Ekin = {ekin/len(a):.3f} eV, Temp = {temp:.1f} K')

# Attach logger to dynamics
dyn.attach(log_progress, interval=100)

print("Starting NVT equilibration...")
dyn.run(10000) # Run for 20 ps to equilibrate

print("\nStarting production run...")
# Reset dynamics for production run at target temperature
MaxwellBoltzmannDistribution(atoms, temperature_K=298.15)
Stationary(atoms)
dyn = Langevin(atoms, timestep=2 * units.fs, temperature_K=298.15, friction=0.01)

# Trajectory analysis setup
coms = []
times = []
n_atoms_per_mol = n_ua_atoms
tags = np.repeat(np.arange(1, n_mol + 1), n_atoms_per_mol)
masses = atoms.get_masses()

def compute_and_save_coms(a=atoms):
    """Computes and stores the center-of-mass of each molecule."""
    pos = a.get_positions()
    current_coms = np.array([
        np.average(pos[tags == m], axis=0, weights=masses[tags == m])
        for m in range(1, n_mol + 1)
    ])
    coms.append(current_coms)
    times.append(a.get_time())

# Attach the analysis function to the dynamics
dyn.attach(compute_and_save_coms, interval=500) # Save every 1 ps

# Run production dynamics
production_steps = 50000 # 100 ps for demo; use 5,000,000 for 10 ns
dyn.run(production_steps)

# --- 5) Compute Diffusion Coefficient ---
coms = np.array(coms)
times_s = np.array(times) / (1000 * units.fs) # Convert ASE time (fs) to seconds

# Account for periodic boundary conditions in displacement
cell = atoms.get_cell()
inv_cell = np.linalg.inv(cell)
disp_unwrapped = np.zeros_like(coms)
disp_unwrapped[0] = coms[0]

for i in range(1, len(coms)):
    # Calculate displacement in scaled coordinates
    delta_scaled = (coms[i] - coms[i-1]) @ inv_cell
    # Apply minimum image convention
    delta_scaled -= np.round(delta_scaled)
    # Convert back to Cartesian and accumulate
    disp_unwrapped[i] = disp_unwrapped[i-1] + delta_scaled @ cell

msd = np.mean(np.sum((disp_unwrapped - disp_unwrapped[0])**2, axis=2), axis=1)

# Convert from Å^2 to m^2
msd_m2 = msd * (1e-10)**2

# Fit line to the last half of the data to find diffusion coefficient
fit_start_index = len(times_s) // 2
coeffs = np.polyfit(times_s[fit_start_index:], msd_m2[fit_start_index:], 1)
slope = coeffs[0]
D = slope / 6.0

print(f"\nFinal TraPPE-UA diffusion coefficient: {D:.3e} m^2/s")