import re
from fairchem.core.units.mlip_unit.mlip_unit import load_inference_model
from fairchem.core.common import registry
from copy import deepcopy
import torch
import logging
import os
from tqdm import tqdm
import  lmdb
import shutil

def get_inverse_indices(original_indices: list[int]) -> list[int]:
    """
    A O(n) method for inverse permutation of indices with duplicates handled.
    Given a list of indices, 
    return the inverse indices to collate the shuffled databack to its original order.
    E.g., if original_indices = [3,3,4,4,5,4,2,0,1], which is created by sampler
    that shuffles the original dataset, then the inverse indices will be
    [7, 8, 6, 1, 5, 4].
    duplicates are handled by taking the last occurrence of the index.
    """
    # handling the case when duplicates are present
    right_cut = max(original_indices) + 1
    
    # O(n) approach to find inverse indices with handling duplicates
    inverse_indices = [0] * len(original_indices)
    for new_pos, original_idx in enumerate(original_indices):
        inverse_indices[original_idx] = new_pos
        
    return inverse_indices[:right_cut]


def initialize_finetuning_model(
    checkpoint_location: str, use_ema: True, overrides: dict | None = None, heads: dict | None = None
) -> torch.nn.Module:
    model, _ = load_inference_model(checkpoint_location, overrides, use_ema=use_ema)

    logging.warning(
        f"initialize_finetuning_model starting from checkpoint_location: {checkpoint_location}"
    )

    return model

def initialize_model_that_have_same_structure_with_checkpoint_but_no_load_weight(
    checkpoint_location: str, overrides: dict | None = None, heads: dict | None = None
) -> torch.nn.Module:
    """
    Initialize a model with the same structure as the checkpoint but without loading weights.
    This is useful for fine-tuning or transfer learning scenarios. Also, when you want to load a student model same as a checkpoint.
    """
    # Load the model architecture from the checkpoint
    model, _ = load_inference_model(checkpoint_location, overrides)

    # logging
    logging.warning(
        f"initialize_model_that_have_same_structure_with_checkpoint_but_no_load_weight starting from checkpoint_location: {checkpoint_location}"
    )

    return model


def merge_lmdb_shards(input_dir, output_path, map_size=1099511627776 * 2):
    """
    Merge multiple LMDB folders (shards) into a single LMDB database.

    Args:
        input_dir (str): Directory containing shard folders (e.g., "/path_to", containing, data.0000.lmdb, data.0001.lmdb, ...).
        output_path (str): Path to the output LMDB folder (e.g., '/path_to/merged_data.lmdb').
        map_size (int): Max size of the LMDB (default 2 TB).
    """
    shard_dirs = sorted([os.path.join(input_dir, d) for d in os.listdir(input_dir)
                        if d.endswith('.lmdb') and os.path.isdir(os.path.join(input_dir, d))])
    if os.path.exists(output_path):
        print(f"Removing existing LMDB at {output_path}")
        shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)
    env_out = lmdb.open(output_path, map_size=map_size)

    idx = 0
    for shard in tqdm(shard_dirs, desc="Merging shards"):
        init_idx = idx
        # print(f"Processing {shard},starting index: {init_idx}")
        env_in = lmdb.open(shard, readonly=True, lock=False)
        with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
            cursor = txn_in.cursor()
            for key, value in cursor:

                # BUG: !!! this line could be extremely buggy when switch to new dataset, or when changing
                # the index encode in label generation
                new_key = str(int(key.decode())+init_idx).encode()  # Reindexing, keep the original key index order
                txn_out.put(new_key, value)
                idx += 1
        env_in.close()

    env_out.close()
    print(f" Done. Total entries merged: {idx}")
    print(f" Output LMDB saved at: {output_path}")
    

def merge_indices_pt_shards(input_dir, output_path):
    """
    Merge multiple PyTorch tensor files (shards) into a single tensor file.

    Args:
        input_dir (str): Directory containing shard files (e.g., "/path_to", containing, data.0000.pt, data.0001.pt, ...).
        output_path (str): Path to the output tensor file (e.g., '/path_to/merged_data.pt').
    """
    # List all files in the directory
    all_files = os.listdir(input_dir)  # or replace '.' with your path

    # Filter files that start with 'train_indices_' and end with '.pt'
    train_files = [f for f in all_files if f.startswith('train_indices_') and f.endswith('.pt')]
    val_files = [f for f in all_files if f.startswith('val_indices_') and f.endswith('.pt')]
    # Extract numeric index using regex and sort based on it
    sorted_train_files = sorted(train_files, key=lambda x: int(re.search(r'(\d+)', x).group()))
    sorted_val_files = sorted(val_files, key=lambda x: int(re.search(r'(\d+)', x).group()))

    # Output result
    train_set_idx = []
    for f in sorted_train_files:
        path_to_f = os.path.join(input_dir, f)
        idx_list = torch.load(path_to_f,weights_only=False)
        train_set_idx+= [ idx for sublist in idx_list for idx in sublist]  # Flatten the list of lists
    
    val_set_idx = []
    for f in sorted_val_files:
        path_to_f = os.path.join(input_dir, f)
        idx_list = torch.load(path_to_f,weights_only=False)
        val_set_idx+= [ idx for sublist in idx_list for idx in sublist]  # Flatten the list of lists
    
    torch.save(train_set_idx, os.path.join(output_path, "train_indices.pt"))
    torch.save(val_set_idx, os.path.join(output_path, "val_indices.pt"))
