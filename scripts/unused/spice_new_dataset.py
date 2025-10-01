import os
import numpy as np
from ase import Atoms
from mace.calculators import mace_off, mace_mp
from tqdm import tqdm
from torch.utils.data import Subset
import torch
from pathlib import Path
import pickle
import lmdb
from fairchem.core.common.registry import registry
def create_dup_dataset(dataset, output_dir):
    lmdb_envs = {}
    db = lmdb.open(
        os.path.join(output_dir, 'data.lmdb'),
        map_size= 1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,)

    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        txn  = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(sample, protocol=-1))
        txn.commit()

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    print(f"Finished sorting and saving samples to LMDB.")

# Usage
dataset_path = "/data/ishan-amin/lmdb_w_ref2"  # Update this with your actual path
output_dir = "/data/ishan-amin/spice_dup"   # Update this with your desired output directory
train_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, 'train')})
indx = np.random.default_rng(seed=0).choice(
    len(train_dataset), 
    5000, 
    replace=False
)
train_dataset = Subset(train_dataset, torch.tensor(indx))
create_dup_dataset(train_dataset, output_dir)
