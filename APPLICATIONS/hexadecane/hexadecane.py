#!/usr/bin/env python3
"""
All-atom simulation of liquid n-hexadecane diffusion using OPLS-AA in ASE+LAMMPS.
"""
import numpy as np
from ase import Atoms
from ase.io import write
import os
os.environ['ASE_LAMMPSRUN_COMMAND'] = 'lmp'
from ase.calculators.lammpsrun import LAMMPS
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import ase.units as units

# RDKit for initial molecule generation
from rdkit import Chem
from rdkit.Chem import AllChem

# --- 1) Define OPLS-AA Force Field Parameters for n-Alkanes ---
# These parameters are from the original OPLS-AA papers by Jorgensen et al.
# We will assign integer types for LAMMPS:
# 1: opls_135 (Alkane CH3 Carbon)
# 2: opls_136 (Alkane CH2 Carbon)
# 3: opls_140 (Alkane Hydrogen)

opls_parameters = {
    # Atom types, mass, charge, and LJ parameters (sigma, epsilon)
    'atom_info': {
        1: {'mass': 12.011, 'charge': -0.18, 'sigma': 3.50, 'epsilon': 0.066}, # CH3-C
        2: {'mass': 12.011, 'charge': -0.12, 'sigma': 3.50, 'epsilon': 0.066}, # CH2-C
        3: {'mass': 1.008,  'charge':  0.06, 'sigma': 2.50, 'epsilon': 0.030}, # H
    },
    # Bond parameters (K_b, r_0)
    'bonds': {
        'C-C': {'k': 268.0, 'r0': 1.529}, # kcal/mol/A^2
        'C-H': {'k': 340.0, 'r0': 1.090}, # kcal/mol/A^2
    },
    # Angle parameters (K_theta, theta_0)
    'angles': {
        'C-C-C': {'k': 58.35, 'theta0': 112.7}, # kcal/mol/rad^2
        'C-C-H': {'k': 37.5,  'theta0': 110.7},
        'H-C-H': {'k': 33.0,  'theta0': 107.8},
    },
    # Dihedral parameters (V1, V2, V3, V4) - OPLS style
    'dihedrals': {
        'C-C-C-C': {'v': [1.411, -0.271, 3.145, 0.0]}, # kcal/mol
        'H-C-C-C': {'v': [0.0, 0.0, 0.355, 0.0]},
        'H-C-C-H': {'v': [0.0, 0.0, 0.318, 0.0]},
    }
}

# --- 2) Generate a single All-Atom n-Hexadecane Molecule ---
smiles = "CCCCCCCCCCCCCCCC"
mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol)

symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
coords = np.array([mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
masses_single = np.array([atom.GetMass() for atom in mol.GetAtoms()])
com = np.average(coords, axis=0, weights=masses_single)
coords -= com  # shift center of mass to (0,0,0)
n_atoms_per_mol = len(symbols)

# Assign OPLS atom types and charges
atom_types = np.zeros(n_atoms_per_mol, dtype=int)
charges = np.zeros(n_atoms_per_mol)
for atom in mol.GetAtoms():
    idx = atom.GetIdx()
    sym = atom.GetSymbol()
    if sym == 'C':
        # Terminal CH3 carbon if it has only one carbon neighbor
        c_neighbors = [n.GetSymbol() for n in atom.GetNeighbors()].count('C')
        if c_neighbors == 1:
            atom_types[idx] = 1 # opls_135
            charges[idx] = opls_parameters['atom_info'][1]['charge']
        else:
            atom_types[idx] = 2 # opls_136
            charges[idx] = opls_parameters['atom_info'][2]['charge']
    elif sym == 'H':
        atom_types[idx] = 3 # opls_140
        charges[idx] = opls_parameters['atom_info'][3]['charge']
# Build single molecule Atoms object
single_mol = Atoms(symbols=symbols, positions=coords)
single_mol.set_array('atom_types', atom_types)
single_mol.set_array('charge', charges)

# --- 3) Build Liquid Box and Topology ---
# Get topology templates (bonds, angles, dihedrals) from the RDKit molecule
single_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
single_angles = []
for atom in mol.GetAtoms():
    idx = atom.GetIdx()
    neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            single_angles.append((neighbors[i], idx, neighbors[j]))
single_dihedrals = []
for bond in mol.GetBonds():
    begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    begin_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(begin).GetNeighbors() if n.GetIdx() != end]
    end_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(end).GetNeighbors() if n.GetIdx() != begin]
    for i in begin_neighbors:
        for j in end_neighbors:
            single_dihedrals.append((i, begin, end, j))

# Generate a template of topology *types* from the single molecule based on OPLS-AA
# Bond types: 1 (C-C), 2 (C-H)
single_bond_types = [1 if 'H' not in (symbols[b1], symbols[b2]) else 2
                   for b1, b2 in single_bonds]

# Angle types: 1 (C-C-C), 2 (C-C-H), 3 (H-C-H)
single_angle_types = []
for a1, a2, a3 in single_angles:
    s_outer = sorted([symbols[a1], symbols[a3]])
    if symbols[a2] == 'C':
        if s_outer == ['C', 'C']:
            single_angle_types.append(1)  # C-C-C
        elif s_outer == ['C', 'H']:
            single_angle_types.append(2)  # C-C-H
        else:  # H-C-H
            single_angle_types.append(3)

# Dihedral types: 1 (C-C-C-C), 2 (H-C-C-C), 3 (H-C-C-H)
single_dihedral_types = []
for d1, d2, d3, d4 in single_dihedrals:
    s_outer = sorted([symbols[d1], symbols[d4]])
    if symbols[d2] == 'C' and symbols[d3] == 'C':
        if s_outer == ['C', 'C']:
            single_dihedral_types.append(1)  # C-C-C-C
        elif s_outer == ['C', 'H']:
            single_dihedral_types.append(2)  # H-C-C-C
        else:  # H-C-C-H
            single_dihedral_types.append(3)

n_x, n_y, n_z = 3, 3, 2  # 18 molecules to keep atom count reasonable
n_mol = n_x * n_y * n_z
box_length = 30.0  # Å, adjust for density (~0.77 g/cm^3)

all_atoms = Atoms()
all_bonds_list, all_angles_list, all_dihedrals_list = [], [], []
all_bond_types_list, all_angle_types_list, all_dihedral_types_list = [], [], []

for i in range(n_x):
    for j in range(n_y):
        for k in range(n_z):
            global_atom_offset = len(all_atoms)
            mol_copy = single_mol.copy()
            mol_copy.rotate(np.random.rand() * 360, 'x', rotate_cell=False)
            mol_copy.rotate(np.random.rand() * 360, 'y', rotate_cell=False)
            mol_copy.rotate(np.random.rand() * 360, 'z', rotate_cell=False)
            
            offset = [i * box_length/n_x, j * box_length/n_y, k * box_length/n_z]
            mol_copy.translate(offset)
            all_atoms.extend(mol_copy)

            # Add topology for the new molecule using global indices
            for b1, b2 in single_bonds:
                all_bonds_list.append((b1 + global_atom_offset, b2 + global_atom_offset))
            for a1, a2, a3 in single_angles:
                all_angles_list.append((a1 + global_atom_offset, a2 + global_atom_offset, a3 + global_atom_offset))
            for d1, d2, d3, d4 in single_dihedrals:
                all_dihedrals_list.append((d1 + global_atom_offset, d2 + global_atom_offset, d3 + global_atom_offset, d4 + global_atom_offset))
            
            # Add topology types
            all_bond_types_list.extend(single_bond_types)
            all_angle_types_list.extend(single_angle_types)
            all_dihedral_types_list.extend(single_dihedral_types)

all_atoms.set_cell([box_length] * 3)
all_atoms.set_pbc(True)
# Set molecule tags for analysis
all_atoms.set_tags(np.repeat(np.arange(1, n_mol + 1), n_atoms_per_mol))

# Attach the generated topology and types to the Atoms object
all_atoms.info['bonds'] = all_bonds_list
all_atoms.info['bond_types'] = all_bond_types_list
all_atoms.info['angles'] = all_angles_list
all_atoms.info['angle_types'] = all_angle_types_list
all_atoms.info['dihedrals'] = all_dihedrals_list
all_atoms.info['dihedral_types'] = all_dihedral_types_list

# --- 4) Set up LAMMPS Calculator ---
# This setup passes all parameters directly to the ASE calculator, which is more robust.
# ASE will handle writing the necessary data and input files in a temporary directory.
parameters = {
    'units': 'real',
    'atom_style': 'full',
    'log': 'log.lammps',
    'boundary': 'p p p',

    # Mass for each LAMMPS atom type
    'mass': [
        '1 12.011', # CH3-C
        '2 12.011', # CH2-C
        '3 1.008'   # H
    ],

    # Force Field Styles
    'pair_style': 'lj/cut/coul/long 12.0',
    'kspace_style': 'pppm 1.0e-4',
    'bond_style': 'harmonic',
    'angle_style': 'harmonic',
    'dihedral_style': 'opls',
    
    'pair_modify': 'mix arithmetic',
    
    # Force Field Coefficients
    'pair_coeff': [
        '1 1 0.066 3.50', # CH3-C
        '2 2 0.066 3.50', # CH2-C
        '3 3 0.030 2.50', # H
    ],
}
# Bond/angle/dihedral coeffs must be passed as extra commands
extra_commands = [
    'bond_coeff 1 268.0 1.529', # C-C
    'bond_coeff 2 340.0 1.090', # C-H
    'angle_coeff 1 58.35 112.7', # C-C-C
    'angle_coeff 2 37.5 110.7',  # C-C-H
    'angle_coeff 3 33.0 107.8',  # H-C-H
    'dihedral_coeff 1 1.411 -0.271 3.145 0.0', # C-C-C-C
    'dihedral_coeff 2 0.0 0.0 0.355 0.0',      # H-C-C-C
    'dihedral_coeff 3 0.0 0.0 0.318 0.0',      # H-C-C-H
]

calc = LAMMPS(
    lammps_command='lmp',
    **parameters,      # ← pass the dict you built
    extra_commands=extra_commands,
    tmp_dir='lmp_tmp_aa',
    keep_tmp_files=True,
    # specorder=['C', 'C', 'H']
)

# Assign the calculator to the atoms object
atoms = all_atoms
atoms.calc = calc

# --- 5) Minimize, Equilibrate, & Production MD ---
from ase.optimize import LBFGS

# First, minimize the structure to relax overlaps
print("Starting energy minimization...")
optimizer = LBFGS(atoms, logfile='min.log')
optimizer.run(fmax=0.1) # Stop when max force is below 0.1 eV/Angstrom
print("Minimization finished.")

# Now, set up the MD
print("Setting up MD simulation...")
MaxwellBoltzmannDistribution(atoms, temperature_K=298.15, force_temp=True)
Stationary(atoms)

dyn = Langevin(atoms, timestep=1 * units.fs, temperature_K=298.15, friction=0.01)

def log_progress(a=atoms):
    """Function to print progress during the simulation."""
    step = dyn.get_number_of_steps()
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = ekin / (1.5 * len(a) * units.kB)
    print(f'Step: {step:6d} | Epot/atom: {epot/len(a):.3f} eV | Temp: {temp:.1f} K')

dyn.attach(log_progress, interval=500) # Log progress every 500 steps

print("Running NVT equilibration...")
dyn.run(10000) # 10 ps equilibration

print("\nStarting production run...")
# Analysis setup
coms = []
times = []
tags = atoms.get_tags()
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

dyn.attach(compute_and_save_coms, interval=500) # Save every 0.5 ps

# Production run
dyn.run(100000) # 100 ps for demo, increase for better stats

# --- 6) Compute Diffusion Coefficient --- ⚛️
coms = np.array(coms)
times_s = np.array(times) / (1000 * units.fs)

# Unwrap coordinates to handle periodic boundary conditions
cell = atoms.get_cell()
inv_cell = np.linalg.inv(cell)
disp_unwrapped = np.zeros_like(coms)
disp_unwrapped[0] = coms[0]
for i in range(1, len(coms)):
    delta_scaled = (coms[i] - coms[i-1]) @ inv_cell
    delta_scaled -= np.round(delta_scaled)
    disp_unwrapped[i] = disp_unwrapped[i-1] + (delta_scaled @ cell)

msd = np.mean(np.sum((disp_unwrapped - disp_unwrapped[0])**2, axis=2), axis=1)
msd_m2 = msd * (1e-10)**2

# Fit line to the last half of the data
fit_start_index = len(times_s) // 2
coeffs = np.polyfit(times_s[fit_start_index:], msd_m2[fit_start_index:], 1)
slope = coeffs[0]
D = slope / 6.0

print(f"\nFinal OPLS-AA diffusion coefficient: {D:.3e} m^2/s")

# Write final structure for visualization
write('final_structure.xyz', atoms)