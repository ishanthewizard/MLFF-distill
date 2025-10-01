import os
import numpy as np
from ase import Atoms
from mace.calculators import mace_off, mace_mp
from tqdm import tqdm
from torch.utils.data import Subset
import torch
from fairchem.core.common.registry import registry
import lmdb 
from src.distill_utils import print_cuda_memory_usage
import argparse
from ase.io import read

def get_mace_lmdb_forces(sample, mace_model):
    atoms = Atoms(numbers=sample.atomic_numbers.numpy(), positions=sample.pos.numpy(), cell=sample.cell.numpy()[0], pbc=True)
    atoms.calc = mace_model
    return atoms.get_forces()

def get_accuracy(dataset_path, model='large'):
    dataset = registry.get_dataset_class("lmdb")({"src": dataset_path})
    folder_path = '/data/shared/MPtrj/original'

    # Get a list of all extxyz files
    files = [f for f in os.listdir(folder_path) if f.endswith('.extxyz')]

    print(f"Number of samples in dataset: {len(dataset)}")
    mace_model = mace_mp(model="large", dispersion=False, default_dtype="float32", device="cuda")
    i = 0
    for sample in tqdm(dataset):
        i += 1
        mace_lmdb_forces = get_mace_lmdb_forces(sample, mace_model)
        # Assuming sample.mp_od contains the mp number, construct the file name
        mp_number = sample.mp_id  # Extract mp number from the sample
        file_name = f"{mp_number}.extxyz"  # Construct the file name
        file_path = os.path.join(folder_path, file_name)
        print(sample.mp_id, sample.sid)
        if os.path.exists(file_path):
            all_atoms = []
            for atoms in read(file_path, index=":"):
                all_atoms.append(atoms)
            atom = all_atoms[0]
            # print(f"Loaded file: {file_path}")
            # print("POS DIFF:", ((all_atoms[0].get_positions() - sample.pos.numpy())**2).sum())
            # print("FORCE LABEL DIFF:", ((all_atoms[0].get_forces() - sample.force.numpy())**2).sum())
            atom.calc = mace_model 

            # print("DIFFERENCE IN MACE FORCES:",  ((mace_lmdb_forces - atom.get_forces())**2).sum())
        else:
            print(f"File not found for sample with mp_id: {mp_number}")
            raise Exception("UH OH!!")
        
        # Breakpoint for debugging if needed

    print("Finished processing dataset.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Get accuracy of a dataset.")
    parser.add_argument('--dataset', type=str, help="Path to the dataset")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call get_accuracy with the provided dataset path
    get_accuracy(args.dataset)
