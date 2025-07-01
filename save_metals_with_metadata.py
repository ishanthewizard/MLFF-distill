import numpy as np
import os
from tqdm import tqdm
import torch
import ase.db
from fairchem.core.datasets.ase_datasets import AseDBDataset
from torch.utils.data import Subset

def save_subset_to_aselmdb(
    dataset,  # Your AseDBDataset
    indices,  # Your indices['metal_complexes'] indices
    dst_path: str,
    overwrite: bool = False
):
    """
    Save a subset of AseDBDataset to an ASE database (.aselmdb) file.
    
    Args:
        dataset: Your AseDBDataset instance
        indices: List of indices to save (e.g., test['metal_complexes'])
        dst_path: Path to the destination .aselmdb file
        overwrite: If True, overwrite the destination file if it exists
    """
    
    # Check if destination file exists
    if os.path.exists(dst_path) and not overwrite:
        raise FileExistsError(f"Destination file already exists: {dst_path}. Set overwrite=True to replace.")

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    # Connect to the destination ASE database
    dst_db = ase.db.connect(dst_path) #, use_lock_file=False)

    print(f"Saving {len(indices)} samples to {dst_path}")

    # Iterate through the indices and save each atoms object
    for idx in tqdm(indices):
        atoms = dataset.get_atoms(idx)
        dst_db.write(
            atoms=atoms,
            data=atoms.info,  # Store all metadata from atoms.info
        )
    
    # Cleanup
    if hasattr(dst_db, 'close'):
        dst_db.close()

    print(f"ASE database saved at: {dst_path}")
    
def save_subset_to_sharded_aselmdb(
    dataset,
    indices,
    dst_dir: str,
    samples_per_shard: int = 10000,
    overwrite: bool = False
):
    """
    Save a subset to multiple sharded aselmdb files.
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    total_samples = len(indices)
    num_shards = (total_samples + samples_per_shard - 1) // samples_per_shard
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = min(start_idx + samples_per_shard, total_samples)
        shard_indices = indices[start_idx:end_idx]
        
        shard_path = os.path.join(dst_dir, f"data{shard_idx:04d}.aselmdb")
        
        if os.path.exists(shard_path) and not overwrite:
            print(f"Skipping existing shard: {shard_path}")
            continue
            
        # Connect to shard database
        dst_db = ase.db.connect(shard_path, use_lock_file=False)
        
        print(f"Processing shard {shard_idx + 1}/{num_shards}: {len(shard_indices)} samples")
        
        for idx in tqdm(shard_indices, desc=f"Shard {shard_idx + 1}"):
            atoms = dataset.get_atoms(idx)
            dst_db.write(
                atoms=atoms,
                data=atoms.info,
            )
        
        if hasattr(dst_db, 'close'):
            dst_db.close()
            
    print(f"Saved {total_samples} samples across {num_shards} shards in {dst_dir}")
    
def create_metadata(dataset_dir: str):
    """
    Create metadata.npz file for a directory containing aselmdb files.
    """
    metadata_path = os.path.join(dataset_dir, 'metadata.npz')
    
    # Check if metadata already exists
    if os.path.exists(metadata_path):
        print(f"Metadata file already exists at {metadata_path}")
        return
    
    # Configure the dataset
    a2g_args = {  
        "molecule_cell_size": 120.0,
        "r_energy": True,
        "r_forces": True,
        "r_data_keys": ['spin', 'charge', 'data_id'],
    }
    
    # Load the dataset
    dataset = AseDBDataset({
        "src": dataset_dir,
        "a2g_args": a2g_args,
    })

    # Extract metadata
    natoms_list = []
    data_ids_list = []
    neighbors_list = []
    
    print(f"Computing metadata for {len(dataset)} samples...")
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        natoms_list.append(data.natoms.item())
        
        # Extract data_id if available
        if hasattr(data, 'data_id'):
            data_ids_list.append(data.data_id)
        
        # Extract neighbor count if available
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            neighbors_list.append(data.edge_index.shape[1])
        else:
            neighbors_list.append(None)
    
    # Convert to numpy arrays
    natoms = np.array(natoms_list, dtype=np.int32)
    
    # Save metadata
    save_dict = {"natoms": natoms}
    
    if data_ids_list:
        data_ids = np.array(data_ids_list, dtype='<U15')
        save_dict["data_ids"] = data_ids
    
    if None not in neighbors_list:
        neighbors = np.array(neighbors_list, dtype=np.int32)
        save_dict["neighbors"] = neighbors
    
    np.savez(metadata_path, **save_dict)
    print(f"Metadata saved to {metadata_path} with keys: {list(save_dict.keys())}")
    
    
# dataset_path = '/data/ishan-amin/OMOL/4M/train_4M/'
dataset_path = '/data/ishan-amin/OMOL/ALL/val'
dataset = AseDBDataset({"src": dataset_path})
indices = torch.load('/data/christine/OMOL/yue_indices/val/subsets_indices_dict.pt')

# Usage example:
dst_dir = "/data/christine/OMOL/ALL/val/metal_complexes"

# Usage for sharded approach:
save_subset_to_sharded_aselmdb(
    dataset=dataset,
    indices=indices['metal_complexes'],
    dst_dir=dst_dir,
    samples_per_shard=50000,
    overwrite=True  # Set to False if you don't want to overwrite existing files
)

# Usage:
create_metadata("/data/christine/OMOL/ALL/val/metal_complexes")
