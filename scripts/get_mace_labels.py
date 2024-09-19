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
            # Convert tensor to bytes and write to LMDB
            txn.put(sample_id.encode(), sample_output.tobytes())
    env.close()
    print(f"All tensors saved to LMDB:{file_path}")

def record_labels(labels_folder, dataset_path, model="large"):
    # os.makedirs(labels_folder, exist_ok=True)
    # Load the dataset
    train_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, 'train')})
    
    val_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, 'val')})
    
    # Load the model
    if "mace_mp" in labels_folder:
        calc = mace_mp(model=model, dispersion=False, default_dtype="float32", device='cuda')
    elif "mace_off" in labels_folder:
        calc = mace_off(model=model, dispersion=False, default_dtype="float32", device='cuda')

    def get_forces(sample):
        atomic_numbers = sample.atomic_numbers.numpy()
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy(), cell=sample.cell.numpy()[0], pbc=True)
        atoms.calc = calc
        return atoms.get_forces()
    def get_hessians(sample):
        atomic_numbers = sample.atomic_numbers.numpy()
        natoms = sample.natoms
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy(), cell=sample.cell.numpy()[0], pbc=True)
        atoms.calc = calc

        hessian = calc.get_hessian(atoms=atoms)
        return - 1 * hessian.reshape(natoms, 3, natoms, 3) # this is SUPER IMPORTANT!!! multiply by -1
    
    def get_final_node_features(sample):
        atomic_numbers = sample.atomic_numbers.numpy()
        natoms = sample.natoms
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy(), cell=sample.cell.numpy()[0], pbc=True)
        atoms.calc = calc
        return calc.get_final_node_features(atoms=atoms)


    def get_atom_embeddings():
        return calc.get_atom_embeddings()
    
    atom_embeddings = get_atom_embeddings()
    np.save(os.path.join(labels_folder, 'atom_embeddings.npy'), get_atom_embeddings())
    record_and_save(train_dataset, os.path.join(labels_folder, 'train_final_node_features', 'final_node_features.lmdb'), get_final_node_features)
    record_and_save(val_dataset, os.path.join(labels_folder, 'val_final_node_features', 'final_node_features.lmdb'), get_final_node_features)
    # record_and_save(train_dataset, os.path.join(labels_folder, 'train_forces.lmdb'), get_forces)
    # record_and_save(train_dataset, os.path.join(labels_folder, 'force_jacobians.lmdb'), get_hessians)
    # record_and_save(val_dataset, os.path.join(labels_folder, 'val_forces.lmdb'), get_forces)

if __name__ == "__main__":
    '''Mace-MP on MPTraj'''
    # labels_folder = '/data/shared/ishan_stuff/labels_unlocked/mace_mp_all_splits_Bandgap_greater_5'
    # dataset_path = '/data/shared/ishan_stuff/mptraj_seperated_all_splits_unlocked/Bandgap_greater_than_5'

    # record_labels(labels_folder, dataset_path)

    # labels_folder = '/data/shared/ishan_stuff/labels_unlocked/mace_mp_all_splits_Yttrium'
    # dataset_path = '/data/shared/ishan_stuff/mptraj_seperated_all_splits_unlocked/Yttrium'

    # record_labels(labels_folder, dataset_path)


    # labels_folder = '/data/shared/ishan_stuff/labels_unlocked/mace_mp_all_splits_Perovskites_noIons'
    # dataset_path = '/data/shared/ishan_stuff/mptraj_seperated_all_splits_unlocked/Perovskites_noIons'


    # record_labels(labels_folder, dataset_path)

    '''Mace-OFF on SPICE'''
    labels_folder = '/data/shared/ishan_stuff/labels_unlocked/mace_off_large_SpiceAminos'
    dataset_path = '/data/shared/ishan_stuff/spice_separated/Solvated_Amino_Acids'
    record_labels(labels_folder, dataset_path)

    labels_folder = '/data/shared/ishan_stuff/labels_unlocked/mace_off_large_SpiceIodine'
    dataset_path = '/data/shared/ishan_stuff/spice_separated/Iodine'

    record_labels(labels_folder, dataset_path)


    labels_folder = '/data/shared/ishan_stuff/labels_unlocked/mace_off_large_SpiceMonomers'
    dataset_path = '/data/shared/ishan_stuff/spice_separated/DES370K_Monomers'


    record_labels(labels_folder, dataset_path)





