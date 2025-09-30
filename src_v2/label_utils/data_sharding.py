"""
Data Sharding Script for ASE LMDB Datasets

This script shards ASE LMDB datasets into N pieces for distributed training or processing.
It takes a root directory containing 'train' and 'val' subdirectories with .aselmdb files
and splits them into multiple shards while preserving data order.

Usage:
    python src_v2/label_utils/data_sharding.py --root_dir <input_dir> --target_dir <output_dir> --num_shards <N>

Example:
    python src_v2/label_utils/data_sharding.py \
        --root_dir /path/to/dataset/to_be_sharded \
        --target_dir /path/to/output/shards \
        --num_shards 3

Expected input structure:
    root_dir/
    ├── train/
    │   ├── data.aselmdb (or multiple .aselmdb files)
    └── val/
        ├── data.aselmdb (or multiple .aselmdb files)

Output structure:
    target_dir/
    ├── shard1/
    │   ├── train/data.aselmdb
    │   └── val/data.aselmdb
    ├── shard2/
    │   ├── train/data.aselmdb
    │   └── val/data.aselmdb
    └── shard3/
        ├── train/data.aselmdb
        └── val/data.aselmdb

Requirements:
    - ase
    - tqdm
    - Standard Python libraries (argparse, os, glob, itertools, logging)
"""

import argparse
import os
import glob
import ase.db
from tqdm import tqdm
import itertools
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def shard_split(root_dir, target_dir, num_shards, split):
    """
    Shards the .aselmdb files for a given data split (e.g., 'train', 'val') into multiple smaller databases.

    Args:
        root_dir (str): The root directory containing the split directory.
        target_dir (str): The base directory where the sharded data will be stored.
        num_shards (int): The number of shards to create.
        split (str): The name of the data split to process (e.g., 'train', 'val').
    """
    source_dir = os.path.join(root_dir, split)
    db_files = sorted(glob.glob(os.path.join(source_dir, '*.aselmdb')))
    
    if not db_files:
        logging.warning(f"No .aselmdb files found in {source_dir}, skipping.")
        return

    logging.info(f"Found {len(db_files)} database files in {source_dir}.")

    db_connections = [ase.db.connect(f) for f in db_files]
    total_len = sum(len(db) for db in db_connections)
    
    if total_len == 0:
        logging.warning(f"No data found in .aselmdb files in {source_dir}, skipping.")
        return

    logging.info(f"Total entries in '{split}' split: {total_len}.")

    shard_size = total_len // num_shards
    remainder = total_len % num_shards

    row_iterators = [db.select() for db in db_connections]
    all_rows = itertools.chain(*row_iterators)

    with tqdm(total=total_len, desc=f"Sharding {split} data") as pbar:
        for i in range(1, num_shards + 1):
            target_shard_dir = os.path.join(target_dir, f'shard{i}', split)
            os.makedirs(target_shard_dir, exist_ok=True)
            target_db_path = os.path.join(target_shard_dir, 'data.aselmdb')
            
            if os.path.exists(target_db_path):
                logging.warning(f"Removing existing database at {target_db_path}.")
                os.remove(target_db_path)
                
            with ase.db.connect(target_db_path, use_lock_file=False) as dst_db:
                current_shard_size = shard_size + (remainder if i == num_shards else 0)
                
                for _ in range(current_shard_size):
                    try:
                        row = next(all_rows)
                        atoms = row.toatoms()
                        data = dict(row.data) if hasattr(row, 'data') else {}
                        dst_db.write(atoms, data=data)
                        pbar.update(1)
                    except StopIteration:
                        logging.error("Iterator stopped unexpectedly. There might be a mismatch in data length.")
                        break
    
    for db in db_connections:
        # The ase.db.connect context manager should handle closing, but being explicit doesn't hurt.
        pass

def main():
    """
    Main function to parse arguments and initiate the sharding process.
    """
    parser = argparse.ArgumentParser(
        description="Shard ASE LMDB datasets into N pieces for distributed training.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--root_dir", 
        required=True, 
        help="Root directory containing 'train' and 'val' subdirectories with .aselmdb files.\n"
             "Example: /path/to/my/dataset_to_be_sharded"
    )
    parser.add_argument(
        "--target_dir", 
        required=True, 
        help="Target directory to store the sharded data.\n"
             "Example: /path/to/my/sharded_dataset"
    )
    parser.add_argument(
        "--num_shards", 
        required=True, 
        type=int, 
        help="The number of shards to create.\n"
             "Example: 3"
    )
    args = parser.parse_args()

    logging.info(f"Starting dataset sharding process.")
    logging.info(f"  Root directory: {args.root_dir}")
    logging.info(f"  Target directory: {args.target_dir}")
    logging.info(f"  Number of shards: {args.num_shards}")

    if not os.path.isdir(args.root_dir):
        logging.error(f"Root directory not found: {args.root_dir}")
        return

    for split in ['train', 'val']:
        split_path = os.path.join(args.root_dir, split)
        if os.path.isdir(split_path):
            logging.info(f"Processing '{split}' split...")
            shard_split(args.root_dir, args.target_dir, args.num_shards, split)
        else:
            logging.warning(f"Directory for '{split}' split not found, skipping: {split_path}")

    logging.info("Dataset sharding process completed.")

if __name__ == "__main__":
    main()
