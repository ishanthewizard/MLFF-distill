from fairchem.core.datasets import AseDBDataset
from tqdm import tqdm
import numpy as np
import sys
import os

def compute_metadata(dataset_path: str) -> None:
    """
    Compute and save metadata.npz for the dataset at the given path.
    Args:
        dataset_path (str): Path to the dataset directory.
    Returns:
        None. Saves metadata.npz in the dataset directory.
    """
    metadata_path = os.path.join(dataset_path, 'metadata.npz')
    if os.path.exists(metadata_path):
        print(f"Metadata file already exists at {metadata_path}. Exiting to avoid overwriting.")
        return
    a2g_args = {  
        "molecule_cell_size": 120.0,
        "r_energy": True,
        "r_forces": True,
        # "r_stress": True,
        "r_data_keys": [ 'spin','charge', "data_id"],
        # 'sid': 'data_id',
    }
    dataset = AseDBDataset({
        "src": dataset_path,
        "a2g_args": a2g_args,
    })
    natoms_list   = []            # number of atoms per structure
    data_ids_list = []
    for i in tqdm(range(len(dataset))):
        natoms_list.append(dataset[i].natoms.item())
        data_ids_list.append(dataset[i].data_id)
    natoms   = np.array(natoms_list,   dtype=int).reshape(-1)
    data_ids = np.array(data_ids_list, dtype='<U15').reshape(-1)
    out_path = os.path.join(dataset_path,'metadata.npz')
    np.savez(out_path,
            natoms=natoms,
            data_ids=data_ids)
    print(f"Saved NPZ to {out_path} with keys: {list(np.load(out_path).keys())}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_metadata.py /path/to/dataset")
        sys.exit(1)
    compute_metadata(sys.argv[1])