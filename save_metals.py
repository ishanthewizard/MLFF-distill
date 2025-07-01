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
    
    
# dataset_path = '/data/ishan-amin/OMOL/4M/train_4M/'
dataset_path = '/data/ishan-amin/OMOL/ALL/val'
dataset = AseDBDataset({"src": dataset_path})
indices = torch.load('/data/christine/OMOL/yue_indices/val/subsets_indices_dict.pt')

dst_dir = "/data/christine/OMOL/ALL/val/metal_complexes"

save_subset_to_sharded_aselmdb(
    dataset=dataset,
    indices=indices['metal_complexes'],
    dst_dir=dst_dir,
    samples_per_shard=50000,
    overwrite=True
)
