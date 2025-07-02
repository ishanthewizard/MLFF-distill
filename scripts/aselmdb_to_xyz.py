import os
import numpy as np
from tqdm import tqdm
from ase import Atoms
from ase.io import write
from fairchem.core.datasets.ase_datasets import AseDBDataset

def process_dataset_split(lmdb_pth, dst_dir, val_fraction=0.05, seed=42):
    # Prepare dataset loader
    a2g_args = {
        "r_energy": True,
        "r_forces": True,
    }
    dataset = AseDBDataset({
        "src": lmdb_pth,
        "a2g_args": a2g_args,
    })

    # Determine split indices
    N = len(dataset)
    indices = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_val = int(np.floor(val_fraction * N))
    val_idx = set(indices[:n_val])
    train_idx = indices[n_val:]

    # Prepare output folders
    train_dir = os.path.join(dst_dir, "train")
    val_dir   = os.path.join(dst_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    # Collect Atoms objects
    train_atoms = []
    val_atoms   = []

    for i in tqdm(range(N), desc="Loading atoms"):
        atoms = dataset.get_atoms(i)
        if i in val_idx:
            val_atoms.append(atoms)
        else:
            train_atoms.append(atoms)

    # Write out
    train_xyz = os.path.join(train_dir, "train.xyz")
    val_xyz   = os.path.join(val_dir,   "val.xyz")

    write(train_xyz, train_atoms, write_results=True)
    write(val_xyz,   val_atoms,   write_results=True)

    print(f"Saved {len(train_atoms)} structures to {train_xyz}")
    print(f"Saved {len(val_atoms)} structures to {val_xyz}")

if __name__ == "__main__":
    lmdb_path = "/data/ishan-amin/OMOL/TOY/AuNaFHO_DFT_database.db"
    dst_root  = "tester_data/AuNaFHO_DFT"
    process_dataset_split(lmdb_path, dst_root)
