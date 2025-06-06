from fairchem.core.datasets import AseDBDataset
from tqdm import tqdm
import numpy as np
import sys
import os



if __name__ == "__main__":
    # load dataset
    dataset_path = "/data/yuejian/OMol_subset/toy_train"

    # detect if metadata.npz already exists
    metadata_path = os.path.join(dataset_path, 'metadata.npz')
    if os.path.exists(metadata_path):
        print(f"Metadata file already exists at {metadata_path}. Exiting to avoid overwriting.")
        sys.exit(0)
    
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

    # --- your data ---
    # e.g., lists of values; replace with your actual data
    natoms_list   = []            # number of atoms per structure
    data_ids_list = []

    # extracting metadata from the dataset
    for i in tqdm(range(len(dataset))):
        natoms_list.append(dataset[i].natoms.item())
        data_ids_list.append(dataset[i].data_id)
        
    # convert to numpy arrays
    natoms   = np.array(natoms_list,   dtype=int).reshape(-1)  # adjust dtype/shape as needed
    data_ids = np.array(data_ids_list, dtype='<U15').reshape(-1)  # adjust dtype/length as needed

    # --- save to .npz ---
    out_path = os.path.join(dataset_path,'metadata.npz')
    np.savez(out_path,
            natoms=natoms,
            data_ids=data_ids)

    print(f"Saved NPZ to {out_path} with keys: {list(np.load(out_path).keys())}")