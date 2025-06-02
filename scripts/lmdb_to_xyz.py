import os
import numpy as np
from tqdm import tqdm
from ase import Atoms
from ase.io import write
from fairchem.core.common.registry import registry

# Function to process each dataset type and molecule type
def process_dataset(lmdb_pth, dst_path):
    # dataset_path = os.path.join(base_dataset_path, molecule_type, dataset_type)
    # destination_path = os.path.join(base_destination_path, molecule_type, dataset_type)
    # os.makedirs(destination_path, exist_ok=True)
    
    dataset = registry.get_dataset_class("lmdb")({"src": lmdb_pth})

    # List to hold all Atoms objects
    atoms_list = []

    # Iterate through the dataset and create Atoms objects
    for sample in tqdm(dataset, desc=f"Processing {lmdb_pth}"):
        breakpoint()
        positions = sample.pos.numpy()
        atomic_numbers = sample.atomic_numbers.numpy()
        forces = sample.force.numpy()
        energy = sample.y.item()

        atoms = Atoms(numbers=atomic_numbers, positions=positions, cell=None, pbc=True)
        atoms.info['energy'] = energy
        atoms.info['forces'] = forces
        # atoms.arrays['forces'] = forces
        atoms_list.append(atoms)

    # Write all Atoms objects to a single XYZ file
    write(dst_path, atoms_list, write_results=True)
    print(f"XYZ file saved at: {dst_path}")

lmdb_path = "/data/ishan-amin/hessian_proj_data/SPICE/spice_separated/Solvated_Amino_Acids/train"
dst_path = "tester_data/solvated/train.xyz"
process_dataset(lmdb_path, dst_path)
