import os
import pickle
from fairchem.core.common.registry import registry
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
    i = 0
    num_ions = 0
    for sample in tqdm(dataset):
        # Convert tensor to bytes and write to LMDB
        # if len(sample.atomic_numbers) < 3:
        #     num_ions += 1
        #     continue
        sample.natoms = torch.tensor(len(sample.atomic_numbers)).unsqueeze(0)

        sample.fixed = torch.zeros(len(sample.atomic_numbers))
        sample.pos = sample.pos.float() 
        txn = db.begin(write=True)
        txn.put(f"{i}".encode("ascii"), pickle.dumps(sample, protocol=-1))
        txn.commit()
        i += 1

    print("NUM IONS:", num_ions)
    print("TOTAL:", i)
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
    new_dataset_path = '/data/ishan-amin/MPtrj/MPtrj_seperated/Sulfides/val'
    # new_dataset_path = '/data/ishan-amin/MPtrj/mace_mp_split_natoms/train'
    os.makedirs(new_dataset_path)

    # dataset_path = '/data/ishan-amin/post_data/md17/ethanol/1k/' 
    # dataset_path = '/data/ishan-amin/post_data/md17/aspirin/1k/'
    # dataset_path = '/data/ishan-amin/post_data/md22/AT_AT/1k/train'
    dataset_path = '/data/ishan-amin/MPtrj/MPtrj_seperated_old/Sulfides/val'
    # dataset_path = '/data/ishan-amin/MPtrj/mace_mp_split_old/train'
    # dataset_path = '/data/shared/MPtrj/lmdb/train'
    dataset = registry.get_dataset_class("lmdb")({"src": dataset_path})
    record_and_save(dataset, new_dataset_path)
