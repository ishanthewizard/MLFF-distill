import os
import numpy as np
from ase import Atoms
from mace.calculators import mace_off, mace_mp
from tqdm import tqdm
from torch.utils.data import Subset
import torch
from fairchem.core.common.registry import registry
import lmdb 

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
    
    # Load the dataset
    index = 0
    train_dataset = registry.get_dataset_class("lmdb")({"src": dataset_path})
    # train_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, 'train')})
    # val_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, 'val')})
    # test_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, 'test')})
    # Create a subset of 200 samples with a fixed seed
    print(len(train_dataset))
    # indx = np.random.default_rng(seed=0).choice(
    #     len(train_dataset), 
    #     500, 
    #     replace=False
    # )
    # train_dataset = Subset(train_dataset, torch.tensor(indx))
    true_energies = []
    true_forces = []
    predicted_energies = []
    predicted_forces = []

    unique_dataset_names = set()
    #     # Iterate through the dataset and collect 'dataset_name' keys
    # for sample in tqdm(train_dataset, desc="Processing samples"):
    #     dataset_name = sample['dataset_name']
    #     unique_dataset_names.add(dataset_name)

    # Print all unique dataset names
    print(len(train_dataset))
    num_atoms = []
    for sample in tqdm(train_dataset):
        true_energies.append(sample.y.item())
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
    
    print(f"L2MAE Energy: {l2mae_energy}")
    print(f"L2MAE Forces: {l2mae_forces}")

if __name__ == "__main__":

    # dataset_path = '/data/ishan-amin/post_data/md17/ethanol/1k/' 
    # dataset_path = '/data/ishan-amin/WATER/1k'
    # dataset_path = '/data/ishan-amin/dipeptides/all/train'
    # dataset_path = '/data/ishan-amin/lmdb_w_ref2/'
    dataset_path = '/data/ishan-amin/spice_separated/Solvated_Amino_Acids/train'
    # dataset_path = '/data/ishan-amin/post_data/md17/aspirin/1k/'
    # dataset_path = '/data/ishan-amin/post_data/md22/AT_AT/1k/'

    get_accuracy(dataset_path)
