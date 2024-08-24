import os
import numpy as np
from tqdm import tqdm
from ase import Atoms
from ase.io import write
from fairchem.core.common.registry import registry

# Configuration
dataset_type = 'train'
dataset_path = f'/data/ishan-amin/post_data/md17/ethanol/1k/{dataset_type}'
destination_path = '/data/ishan-amin/xyzs/md17/aspirin'
file_name = f'1k_{dataset_type}.xyz'

# Ensure the destination directory exists
os.makedirs(destination_path, exist_ok=True)
xyz_file_path = os.path.join(destination_path, file_name)

# Load the dataset
config = {"src": dataset_path}
dataset = registry.get_dataset_class("lmdb")(config)

# List to hold all Atoms objects
atoms_list = []

# Iterate through the dataset and create Atoms objects
for sample in tqdm(dataset):
    positions = sample.pos.numpy()
    atomic_numbers = sample.atomic_numbers.numpy()
    forces = sample.force.numpy()
    energy = sample.y.item()

    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    atoms.info['energy'] = energy
    atoms.arrays['forces'] = forces
    atoms_list.append(atoms)

# Write all Atoms objects to a single XYZ file
write(xyz_file_path, atoms_list, write_results=True)

print(f"XYZ file saved at: {xyz_file_path}")
