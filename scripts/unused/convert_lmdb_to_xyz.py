import os
import numpy as np
from tqdm import tqdm
from ase import Atoms
from ase.io import write
from fairchem.core.common.registry import registry

# Configuration
dataset_types = ["train", "val", "test"]
molecule_types = ["Perovskites_noIons", "Yttrium", "Bandgap_greater_than_5"]
base_dataset_path = "/data/ishan-amin/MPtrj/MPtrj_seperated_all_splits"
base_destination_path = "/data/ishan-amin/xyzs/MPtrj/xyz_MPtrj_seperated_all_splits"


# Function to process each dataset type and molecule type
def process_dataset(molecule_type, dataset_type):
    dataset_path = os.path.join(base_dataset_path, molecule_type, dataset_type)
    destination_path = os.path.join(base_destination_path, molecule_type, dataset_type)
    # os.makedirs(destination_path, exist_ok=True)

    # Define the XYZ file path
    file_name = "data.xyz"
    xyz_file_path = os.path.join(destination_path, file_name)

    # Load the dataset
    config = {"src": dataset_path}
    dataset = registry.get_dataset_class("lmdb")(config)

    # List to hold all Atoms objects
    atoms_list = []

    # Iterate through the dataset and create Atoms objects
    for sample in tqdm(dataset, desc=f"Processing {molecule_type} - {dataset_type}"):
        positions = sample.pos.numpy()
        atomic_numbers = sample.atomic_numbers.numpy()
        forces = sample.force.numpy()
        energy = sample.corrected_total_energy.item()

        atoms = Atoms(
            numbers=atomic_numbers,
            positions=positions,
            cell=sample.cell.numpy()[0],
            pbc=True,
        )
        atoms.info["energy"] = energy
        atoms.arrays["forces"] = forces
        atoms_list.append(atoms)

    # Write all Atoms objects to a single XYZ file
    write(xyz_file_path, atoms_list, write_results=True)
    print(f"XYZ file saved at: {xyz_file_path}")


# Iterate over all molecule types and dataset types
for molecule_type in molecule_types:
    for dataset_type in dataset_types:
        process_dataset(molecule_type, dataset_type)
