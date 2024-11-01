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


def get_accuracy(dataset_path, model='large'):
    # Load model
    calc = mace_mp(model=model, dispersion=False,  default_dtype="float32", device="cuda")
    # Counting parameters
    # total_params = sum(p.numel() for p in calc.parameters())
    # print(f"Total trainable parameters: {total_params}")
    # print("TOTAL_PARAMS:", calc.parameters)
    
    # Load the dataset
    dataset = registry.get_dataset_class("lmdb")({"src": dataset_path})
    

    # indxs = np.random.default_rng(seed=123).choice(len(dataset), 1000, replace=False)
    # dataset = Subset(dataset, torch.tensor(indxs))
    
    print(len(dataset))
    num_atoms = []
    force_mae_loss = 0
    energy_mae_loss = 0
    numel = 0
    for sample in tqdm(dataset):
        true_energy = sample.corrected_total_energy.item()
        true_force = sample.force.numpy()
        
        atomic_numbers = sample.atomic_numbers.numpy()
        num_atoms.append(len(atomic_numbers))
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy(), cell=sample.cell.numpy()[0], pbc=True)
        atoms.calc = calc

        predicted_energy = atoms.get_potential_energy()
        predicted_force = atoms.get_forces()
        force_mae_loss += np.abs(predicted_force - true_force).sum()
        energy_mae_loss += np.abs(predicted_energy - true_energy)  / len(atomic_numbers)
        numel += len(sample.pos) * 3
    force_mae_loss /= numel
    print("FORCE MAE LOSS:", force_mae_loss) 
    print("ENERGY MAE:", energy_mae_loss / len(dataset))   
    
    num_atoms = np.array(num_atoms)
    print("MEAN:", np.mean(num_atoms), "MIN:", min(num_atoms), "MAX:", max(num_atoms))
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Get accuracy of a dataset.")
    parser.add_argument('--dataset', type=str, help="Path to the dataset")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call get_accuracy with the provided dataset path
    get_accuracy(args.dataset)
