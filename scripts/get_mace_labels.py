from fairchem.core.common.registry import registry
import numpy as np
import lmdb
from ase.io import read
from ase import Atoms
from mace.calculators import mace_off, mace_mp
from sklearn.metrics import mean_absolute_error
import torch
from tqdm import tqdm

def calculate_l2mae(true, predicted):
    return np.mean((true - predicted) ** 2)

def convert_to_lmdb(hessians, lmdb_path):
    env = lmdb.open(lmdb_path, map_size=int(1e9))
    with env.begin(write=True) as txn:
        for i, hessian in enumerate(hessians):
            txn.put(f"hessian_{i}".encode(), hessian.tobytes())

def process_xyz(xyz_path, lmdb_path, model="large", dispersion=False, default_dtype="float64", device='cuda'):
    atoms_list = read(xyz_path, index=':')
    calc = mace_off(model=model, dispersion=dispersion, default_dtype=default_dtype, device=device)
    # calc = mace_mp(model=model, dispersion=dispersion, default_dtype=default_dtype, device=device)
    
    true_energies = []
    true_forces = []
    predicted_energies = []
    predicted_forces = []
    hessians = []
    

    # Load the dataset
    dataset_type = 'train'
    # dataset_path = f'/data/ishan-amin/post_data/md17/ethanol/1k/{dataset_type}' 
    dataset_path = f'/data/ishan-amin/post_data/md22/AT_AT/1k/{dataset_type}'
    # dataset_path = f'/data/ishan-amin/post_data/md17/aspirin/1k/{dataset_type}'
    config = {"src": dataset_path}
    dataset = registry.get_dataset_class("lmdb")(config)
    # 
    for sample in tqdm(dataset):
        true_energies.append(sample.y.item())
        true_forces.append(sample.force.numpy())

        atomic_numbers = sample.atomic_numbers.numpy()
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy())
        
        atoms.calc = calc
        predicted_energies.append(atoms.get_potential_energy())
        predicted_forces.append(atoms.get_forces())
        # hessian = calc.get_hessian(atoms=atoms)
        # hessians.append(hessian)
    
    l2mae_energy = calculate_l2mae(np.array(true_energies), np.array(predicted_energies))
    l2mae_forces = calculate_l2mae(np.array(true_forces), np.array(predicted_forces))
    
    print(f"L2MAE Energy: {l2mae_energy}")
    print(f"L2MAE Forces: {l2mae_forces}")
    
    convert_to_lmdb(hessians, lmdb_path)

# Example usage:
xyz_path = '/data/ishan-amin/xyzs/md17/aspirin/1k_train.xyz'
lmdb_path = 'hessians.lmdb'
process_xyz(xyz_path, lmdb_path)
