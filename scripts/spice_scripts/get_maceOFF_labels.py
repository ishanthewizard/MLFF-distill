import os
from fairchem.core.common.registry import registry
import numpy as np
import lmdb
from ase.io import read
from ase import Atoms
from mace.calculators import mace_off, mace_mp
from tqdm import tqdm
from torch.utils.data import Subset
import torch
def record_and_save(dataset, file_path, fn):
    # Assuming train_loader is your DataLoader
    avg_num_atoms = dataset[0].natoms.item()
    map_size = 1099511627776 * 2

    env = lmdb.open(file_path, map_size=map_size)
    env_info = env.info()
    with env.begin(write=True) as txn:
        for sample in tqdm(dataset):
            sample_id = str(int(sample.id))
            sample_output = fn(sample)  # this function needs to output an array where each element correponds to the label for an entire molecule
            txn.put(sample_id.encode(), sample_output.tobytes())
    env.close()
    print(f"All tensors saved to LMDB:{file_path}")

def record_labels(labels_folder, dataset_path, model="large"):
    os.makedirs(labels_folder, exist_ok=True)
    # Load the dataset
    train_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, 'train')})
    val_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, 'val')})
    
    # Load the model
    calc = mace_off(model=model, dispersion=False, default_dtype="float32", device='cuda')

    # calc = mace_mp(model=model, dispersion=dispersion, default_dtype=default_dtype, device=device)
    def get_forces(sample):
        atomic_numbers = sample.atomic_numbers.numpy()
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy())
        atoms.calc = calc
        return atoms.get_forces()
    def get_hessians(sample):
        atomic_numbers = sample.atomic_numbers.numpy()
        natoms = sample.natoms
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy())
        atoms.calc = calc
        # sys.exit()
        hessian = calc.get_hessian(atoms=atoms)
        return - 1 * hessian.reshape(natoms, 3, natoms, 3) # this is SUPER IMPORTANT!!! multiply by -1
        
    record_and_save(train_dataset, os.path.join(labels_folder, 'force_jacobians', 'force_jacobians.lmdb'), get_hessians)
    record_and_save(train_dataset, os.path.join(labels_folder, 'train_forces', 'train_forces.lmdb'), get_forces)
    record_and_save(val_dataset, os.path.join(labels_folder, 'val_forces', 'val_forces.lmdb'), get_forces)

if __name__ == "__main__":
    labels_folder = 'labels/mace_off_large_Yttrium'
    if os.path.isdir(labels_folder):
        raise Exception('folder already exists')
    dataset_path = '/data/ishan-amin/MPtrj/MPtrj_seperated/Yttrium'

    record_labels(labels_folder, dataset_path)
