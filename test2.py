# Load structure
import ase.io
import matplotlib.pyplot as plt
from ase.io import read
from fairchem.core.datasets.ase_datasets import AseDBDataset
import os
import glob
from ase.io import Trajectory
import matplotlib.pyplot as plt
from ase.io import Trajectory
from ase.visualize import view
from ase.io import Trajectory
import nglview as nv
import numpy as np
from APPLICATIONS.electrolytes.get_calc import get_uma_calc
from copy import deepcopy

path = "/data/ishan-amin/OMOL/electrolytes_application/all_trajs_min50ps/md_omol_naotf_dme_s1p1_omol.traj"
atoms = read(path, index=0)

# Original forces
true_forces = atoms.get_forces().copy()

idx = 0
atoms = read(path, index=idx)
pos0 = atoms.get_positions().copy()

# Move ONLY atom 0 by one lattice vector a1
print(atoms.positions[0, 0])
atoms.positions[0] = atoms.positions[0] + 5.0 * np.array([atoms.cell[0, 0], 0.0, 0.0]) # <-- note [0], not [:,0] or + atoms.cell
breakpoint()
# Recompute forces
calc = get_uma_calc("/data/ishan-amin/OMOL/ESEN_OMol_ckpts/uma-s-1p1.pt")
atoms.calc = calc
calc.calculate(atoms)
breakpoint()
shifted_forces = calc.results["forces"].copy()

# Compare
diff = np.abs(true_forces[idx] - shifted_forces[idx]).max()
print(f"Max force difference after PBC shift: {diff:.6e} eV/Å")