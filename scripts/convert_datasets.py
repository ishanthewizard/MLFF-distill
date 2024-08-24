import os
from fairchem.core.common.registry import registry
import numpy as np
import lmdb
from ase.io import read
from ase import Atoms
from mace.calculators import mace_off, mace_mp
from tqdm import tqdm
from src.distill_datasets import SimpleDataset
from torch.utils.data import Subset
import torch
def record_and_save(dataset, file_path, fn):
    # Assuming train_loader is your DataLoader
    map_size = 1099511627776 * 2

    env = lmdb.open(file_path, map_size=map_size)
    env_info = env.info()
    with env.begin(write=True) as txn:
        i = 0
        while i < len(dataset):
            sample = dataset[i]
            sample_id = str(i)
            sample_output = fn(sample)  # this function needs to output an array where each element correponds to the label for an entire molecule
            # Convert tensor to bytes and write to LMDB
            txn.put(sample_id.encode(), sample_output.tobytes())
            i += 1
    env.close()
    print(f"All tensors saved to LMDB:{file_path}")

def convert_labels(labels_folder, new_labels_folder, model="medium"):
    os.makedirs(new_labels_folder)
    force_jac_dataset = SimpleDataset(os.path.join(labels_folder, 'force_jacobians.lmdb'))
    teacher_force_train_dataset = SimpleDataset(os.path.join(labels_folder,  'train_forces.lmdb'  ))
    teacher_force_val_dataset = SimpleDataset(os.path.join(labels_folder,  'val_forces.lmdb'  ))
    array = [1,2,3]
    for item in array:
        print(item)
    breakpoint()
    identity = lambda x: x.numpy()
    flip = lambda x: -x.numpy()
    record_and_save(force_jac_dataset, os.path.join(new_labels_folder, 'force_jacobians.lmdb'), flip)
    record_and_save(teacher_force_train_dataset, os.path.join(new_labels_folder, 'train_forces.lmdb'), identity)
    record_and_save(teacher_force_val_dataset, os.path.join(new_labels_folder, 'val_forces.lmdb'), identity)
if __name__ == "__main__":
    labels_folder = 'labels/mace_off_large_SpiceMonomers'
    new_labels_folder = 'labels/mace_off_large_SpiceMonomersNEW'

    convert_labels(labels_folder, new_labels_folder)
