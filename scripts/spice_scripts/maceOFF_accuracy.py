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

def calculate_l2mae(true, predicted):
    return np.mean(np.abs(true - predicted))

def calculate_l2mae_forces(true_forces, predicted_forces):
    l2mae_forces = []
    for true, pred in zip(true_forces, predicted_forces):
        l2mae_forces.append(np.mean(np.abs(true - pred)))
    return np.mean(l2mae_forces)

def get_accuracy(dataset_path, model='large'):
    # Load model
    calc = mace_off(model=model, dispersion=False,  default_dtype="float32", device="cuda")
    # Counting parameters
    # total_params = sum(p.numel() for p in calc.parameters())
    # print(f"Total trainable parameters: {total_params}")
    # print("TOTAL_PARAMS:", calc.parameters)
    
    # Load the dataset
    index = 0
    dataset = registry.get_dataset_class("lmdb")({"src": dataset_path})
    

    indxs = np.random.default_rng(seed=123).choice(len(dataset), 1000, replace=False)
    # dataset = Subset(dataset, torch.tensor(indxs))
        

    print(len(dataset))
    # indx = np.random.default_rng(seed=0).choice(
    #     len(dataset), 
    #     500, 
    #     replace=False
    # )
    # dataset = Subset(dataset, torch.tensor(indx))
    print(len(dataset))
    true_energies = []
    true_forces = []
    predicted_energies = []
    predicted_forces = []
    num_atoms = []
    for sample in tqdm(dataset):
        true_energies.append(sample.uncorrected_total_energy.item())
        true_forces.append(sample.force.numpy())
        atomic_numbers = sample.atomic_numbers.numpy()
        num_atoms.append(len(atomic_numbers))
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy())
        atoms.calc = calc

        predicted_energies.append(atoms.get_potential_energy())
        predicted_forces.append(atoms.get_forces())
    
    l2mae_energy = calculate_l2mae(np.array(true_energies), np.array(predicted_energies))
    l2mae_forces = calculate_l2mae_forces(true_forces, predicted_forces)
    num_atoms = np.array(num_atoms)
    print("MEAN:", np.mean(num_atoms), "MIN:", min(num_atoms), "MAX:", max(num_atoms))
    print(f"TRUE STD:",  np.std(np.concatenate(true_forces)))
    print(f"Predicted STD:", np.std(np.concatenate(predicted_forces)))
    print(f"L2MAE Energy: {l2mae_energy * 1000:.2f}")
    print(f"L2MAE Forces: {l2mae_forces * 1000:.2f} meV")
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Get accuracy of a dataset.")
    parser.add_argument('--dataset', type=str, help="Path to the dataset")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call get_accuracy with the provided dataset path
    get_accuracy(args.dataset)
