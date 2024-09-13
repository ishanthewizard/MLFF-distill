import os
import pickle
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs
import numpy as np
import lmdb
from ase.io import read
from ase import Atoms
from mace.calculators import mace_off, mace_mp
from tqdm import tqdm
from torch.utils.data import Subset
import torch

def record_and_save(dataset, new_file_path):
    # Assuming train_loader is your DataLoader
    db = lmdb.open(
        os.path.join(new_file_path, "data.lmdb"),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    

    lengths = np.ones(3)[None, :] * 30.
    angles = np.ones(3)[None, :] * 90.
    a2g = AtomsToGraphs(
        max_neigh=1000,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_distances=False,
        r_edges=False,
    )
    i = 0
    for sample in tqdm(dataset):
        # Convert tensor to bytes and write to LMDB
        natoms = np.array([len(sample.pos)] * 1, dtype=np.int64)
        positions = sample.pos.float()
        assert len(sample.atomic_numbers) == len(positions)
        data = a2g.convert(natoms, positions, sample.atomic_numbers, 
                        lengths, angles, sample.corrected_total_energy, sample.force)
        data.sid = sample.sid
        data.fid = sample.id

        txn = db.begin(write=True)
        txn.put(f"{i}".encode("ascii"), pickle.dumps(sample, protocol=-1))
        txn.commit()
        i += 1


 # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(i, protocol=-1))
    txn.commit()
    db.sync()
    # close environment
    db.close()

    print(f"All tensors saved to LMDB: {new_file_path}")

if __name__ == "__main__":
    # new_dataset_path= '/data/ishan-amin/maceoff_split/train'
    new_dataset_path = '/data/ishan-amin/MPtraj/mptraj_seperated/Perovskites/train'
    # new_dataset_path = '/data/ishan-amin/MPtraj/mace_mp_split_natoms/train'
    os.makedirs(new_dataset_path)

    # dataset_path = '/data/ishan-amin/post_data/md17/ethanol/1k/' 
    # dataset_path = '/data/ishan-amin/post_data/md17/aspirin/1k/'
    # dataset_path = '/data/ishan-amin/post_data/md22/AT_AT/1k/train'
    dataset_path = '/data/ishan-amin/MPtraj/mptraj_seperated_old/Perovskites/train'
    # dataset_path = '/data/ishan-amin/MPtraj/mace_mp_split_old/train'
    # dataset_path = '/data/shared/MPTrj/lmdb/train'
    dataset = registry.get_dataset_class("lmdb")({"src": dataset_path})
    record_and_save(dataset, new_dataset_path)
