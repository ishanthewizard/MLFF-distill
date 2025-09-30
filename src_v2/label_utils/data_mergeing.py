"""
Data Merging Script for Sharded ASE LMDB Datasets and Labels

This script merges sharded ASE LMDB datasets and LMDB label databases back into a single directory.
It takes a directory containing multiple shards (e.g., toy_dir, toy_dir1, toy_dir2) and combines
the train, val, and label data while preserving order and re-indexing LMDB keys sequentially.

Usage:
    python src_v2/label_utils/data_mergeing.py --shards_dir <shards_directory> --target_dir <output_directory>

Example:
    python src_v2/label_utils/data_mergeing.py \
        --shards_dir /path/to/all_shards \
        --target_dir /path/to/merged_output

Expected input structure:
    shards_dir/
    ├── toy_dir/
    │   ├── train/data.aselmdb
    │   ├── val/data.aselmdb
    │   └── label/
    │       ├── force_jacobians/data.lmdb/
    │       ├── train_forces/data.lmdb/
    │       └── val_forces/data.lmdb/
    ├── toy_dir1/
    │   ├── train/data.aselmdb
    │   ├── val/data.aselmdb
    │   └── label/
    │       └── ...
    └── toy_dir2/
        └── ...

Output structure:
    target_dir/
    ├── train/data.aselmdb
    ├── val/data.aselmdb
    └── label/
        ├── force_jacobians/data.lmdb/
        ├── train_forces/data.lmdb/
        └── val_forces/data.lmdb/

Key Features:
- Preserves data order across shards
- Re-indexes LMDB keys sequentially (0-9 from shard1, 10-19 from shard2, etc.)
- Handles both integer keys and special keys ending with "_idxs"
- Only merges existing label databases (force_jacobians, train_forces, val_forces)

Requirements:
    - ase
    - numpy
    - tqdm
    - lmdb
    - Standard Python libraries (argparse, os, glob, logging, shutil)
"""

import argparse
import os
import glob
import ase.db
import numpy as np
from tqdm import tqdm
import logging
import lmdb
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_aselmdb(shards_dir, target_dir, split):
    """
    Merges all .aselmdb files from a specific split across all shards into a single database.

    Args:
        shards_dir (str): The root directory containing the shard subdirectories.
        target_dir (str): The directory where the merged data will be stored.
        split (str): The data split to process (e.g., 'train', 'val').
    """
    shard_dirs = sorted(glob.glob(os.path.join(shards_dir, 'toy_dir*')))
    if not shard_dirs:
        logging.warning(f"No shard directories found in {shards_dir}.")
        return

    merged_db_path = os.path.join(target_dir, split, 'data.aselmdb')
    if os.path.exists(merged_db_path):
        logging.warning(f"Removing existing merged database at {merged_db_path}.")
        os.remove(merged_db_path)

    all_db_files = []
    for shard_dir in shard_dirs:
        path = os.path.join(shard_dir, split)
        if os.path.isdir(path):
            all_db_files.extend(glob.glob(os.path.join(path, '*.aselmdb')))
    
    if not all_db_files:
        logging.warning(f"No .aselmdb files found for the '{split}' split in any shard, skipping.")
        return

    logging.info(f"Merging {len(all_db_files)} database files for the '{split}' split.")
    
    with ase.db.connect(merged_db_path, use_lock_file=False) as dst_db:
        for db_file in tqdm(all_db_files, desc=f"Merging {split} databases"):
            try:
                with ase.db.connect(db_file) as src_db:
                    for row in src_db.select():
                        atoms = row.toatoms()
                        data = dict(row.data) if hasattr(row, 'data') else {}
                        dst_db.write(atoms, data=data)
            except Exception as e:
                logging.error(f"Failed to process {db_file}: {e}")


def merge_lmdb_labels(shards_dir, target_dir):
    """
    Merges LMDB databases from the 'label' directory across all shards,
    re-indexing keys to be sequential to maintain order.

    Args:
        shards_dir (str): The root directory containing the shard subdirectories.
        target_dir (str): The directory where the merged labels will be stored.
    """
    shard_dirs = sorted(glob.glob(os.path.join(shards_dir, 'toy_dir*')))
    if not shard_dirs:
        logging.warning(f"No shard directories found in {shards_dir} for label merging.")
        return
        
    label_dirs_to_merge = ['force_jacobians', 'val_forces', 'train_forces']

    for label_name in label_dirs_to_merge:
        source_lmdb_paths = []
        for shard_dir in shard_dirs:
            search_path = os.path.join(shard_dir, 'label', label_name, '*.lmdb')
            found_paths = glob.glob(search_path)
            if found_paths:
                source_lmdb_paths.append(found_paths[0])

        if not source_lmdb_paths:
            logging.warning(f"No LMDB database found for label '{label_name}' in any shard. Skipping.")
            continue

        logging.info(f"Found {len(source_lmdb_paths)} LMDB databases for '{label_name}'. Merging...")

        target_lmdb_path = os.path.join(target_dir, 'label', label_name,'data.lmdb')
        if os.path.exists(target_lmdb_path):
            logging.warning(f"Removing existing merged LMDB at {target_lmdb_path}.")
            shutil.rmtree(target_lmdb_path)
        os.makedirs(target_lmdb_path, exist_ok=True)
        
        total_size = sum(os.path.getsize(os.path.join(p, 'data.mdb')) for p in source_lmdb_paths)
        map_size = max(int(total_size * 1.5), 1024**3)

        dest_env = lmdb.open(target_lmdb_path, map_size=map_size, writemap=True)
        
        total_entries = 0
        for path in source_lmdb_paths:
             with lmdb.open(path, readonly=True, lock=False) as env:
                total_entries += env.stat()['entries']
        with dest_env.begin(write=True) as dest_txn, tqdm(total=total_entries, desc=f"Merging {label_name}") as pbar:
            key_offset = 0
            for src_path in source_lmdb_paths:
                temp_key_offset = 0
                src_env = lmdb.open(src_path, readonly=True, lock=False)
                with src_env.begin() as src_txn:
                    keys_to_process = []
                    non_int_keys = []
                    for key_bytes, _ in src_txn.cursor():
                        try:
                            if key_bytes.decode().isdigit():
                                keys_to_process.append(int(key_bytes.decode()))
                            elif key_bytes.decode().endswith("_idxs"):
                                non_int_keys.append(key_bytes.decode())
                        except:
                            print("unknown key: ", key_bytes.decode())

                    keys_to_process.sort()
                    # dealing with int keys
                    for key_int in keys_to_process:
                        original_key_bytes = str(key_int).encode()
                        value = src_txn.get(original_key_bytes)
                        
                        if value:
                            new_key_bytes = str(key_offset+key_int).encode()
                            dest_txn.put(new_key_bytes, value)
                            temp_key_offset += 1
                            pbar.update(1)
                            
                    # dealing with non-int keys
                    for key_non_int in non_int_keys:
                        original_key_bytes = key_non_int.encode()
                        value = src_txn.get(original_key_bytes)
                        index = key_non_int.split("_")[0]
                        if value:
                            new_key_bytes = (str(key_offset+ int(index))+ "_idxs").encode()
                            dest_txn.put(new_key_bytes, value)
                            pbar.update(1)
                key_offset += temp_key_offset
                src_env.close()
        dest_env.close()
        logging.info(f"Successfully merged '{label_name}' into {target_lmdb_path}.")


def main():
    """
    Main function to parse arguments and initiate the merging process.
    """
    parser = argparse.ArgumentParser(description="Merge sharded ASE LMDB and label datasets back into a single directory.")
    parser.add_argument(
        "--shards_dir", 
        required=True, 
        help="Root directory containing the sharded data (e.g., 'shard1', 'shard2', ...)."
    )
    parser.add_argument(
        "--target_dir", 
        required=True, 
        help="Target directory to store the merged data."
    )
    args = parser.parse_args()

    logging.info(f"Starting dataset merging process.")
    logging.info(f"  Shards directory: {args.shards_dir}")
    logging.info(f"  Target directory: {args.target_dir}")

    # Create target directories
    os.makedirs(os.path.join(args.target_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'label'), exist_ok=True)

    # Merge train and val splits
    # for split in ['train', 'val']:
    #     merge_aselmdb(args.shards_dir, args.target_dir, split)
        
    # Merge labels
    merge_lmdb_labels(args.shards_dir, args.target_dir)

    logging.info("Dataset merging process completed.")

if __name__ == "__main__":
    main()
