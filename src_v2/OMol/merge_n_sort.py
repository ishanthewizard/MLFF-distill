"""
Usage:
-----
This script merges and sorts LMDB label shards for force_jacobians, train_forces, and val_forces in place.
It merges all shards for each label type (if present), sorts them using the provided indices, and replaces the original shards with the sorted LMDBs.

Example:
    python merge_n_sort.py --root_dir /path/to/labels

Arguments:
    --root_dir: Root directory containing label shards (default: /home/yuejian/project/MLFF-distill/OMOL/TOY/temp_strain/labels)

After running, only the sorted LMDBs (data0000.lmdb) will remain in each label type folder.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add project root to sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import shutil
from src_v2.dataset_utils import merge_lmdb_shards, merge_indices_pt_shards, get_inverse_indices
from src_v2.distill_datasets import LmdbDataset
import torch
from torch.utils.data import Subset
import lmdb
from tqdm import tqdm
import tempfile

def has_lmdb_files(folder: str) -> bool:
    """
    Check if a folder contains any .lmdb directories.
    Args:
        folder (str): Path to the folder to check.
    Returns:
        bool: True if any .lmdb directories are found, False otherwise.
    """
    return os.path.isdir(folder) and any(f.endswith('.lmdb') for f in os.listdir(folder))

def merge_and_sort_labels(root_dir: str) -> None:
    """
    Merge LMDB shards for each label type, sort them using indices, and replace original shards with sorted LMDBs.
    Args:
        root_dir (str): Root directory containing label shards (force_jacobians, train_forces, val_forces, indices)
    Returns:
        None. The original shards are replaced with sorted LMDBs in place.
    """
    label_types = ["force_jacobians", "train_forces", "val_forces"]
    indices_type = "indices"
    print(f"Merging and sorting in root: {root_dir}")

    # 1. Merge LMDB shards for each label type to a temp location
    temp_merged = {}
    temp_dirs = {}
    for label_type in label_types:
        label_dir = os.path.join(root_dir, label_type)
        if not has_lmdb_files(label_dir):
            print(f"Skipping merge for {label_type}: no LMDB files found in {label_dir}")
            continue
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, "data0000.lmdb")
        print(f"Merging LMDB shards in {label_dir} -> {output_file}")
        try:
            merge_lmdb_shards(input_dir=label_dir, output_path=output_file)
            temp_merged[label_type] = tmpdir  # Store the parent dir, not the LMDB dir
            temp_dirs[label_type] = tmpdir
        except Exception as e:
            print(f"Failed to merge {label_type} in {label_dir}: {e}")
            shutil.rmtree(tmpdir)
            continue

    # 2. Merge indices
    indices_dir = os.path.join(root_dir, indices_type)
    if os.path.isdir(indices_dir):
        print(f"Merging indices in {indices_dir}")
        try:
            merge_indices_pt_shards(indices_dir, output_path=indices_dir)
        except Exception as e:
            print(f"Failed to merge indices in {indices_dir}: {e}")
    else:
        print(f"No indices directory found at {indices_dir}, skipping indices merge.")

    # 3. Sort and replace original shards with final sorted LMDBs
    # Load indices
    train_indices_path = os.path.join(indices_dir, "train_indices.pt")
    val_indices_path = os.path.join(indices_dir, "val_indices.pt")
    train_indices = torch.load(train_indices_path) if os.path.isfile(train_indices_path) else None
    val_indices = torch.load(val_indices_path) if os.path.isfile(val_indices_path) else None
    inverse_train_indices = get_inverse_indices(train_indices) if train_indices is not None else None
    inverse_val_indices = get_inverse_indices(val_indices) if val_indices is not None else None

    for label_type in label_types:
        if label_type not in temp_merged:
            print(f"Skipping sort for {label_type}: no merged LMDB available.")
            continue
        merged_lmdb_parent = temp_merged[label_type]
        print(f"Sorting {label_type} and replacing original shards...")
        dataset = LmdbDataset(folder_path=merged_lmdb_parent)
        if label_type == "force_jacobians" and inverse_train_indices is not None:
            sorted_subset = Subset(dataset=dataset, indices=inverse_train_indices)
        elif label_type == "train_forces" and inverse_train_indices is not None:
            sorted_subset = Subset(dataset=dataset, indices=inverse_train_indices)
        elif label_type == "val_forces" and inverse_val_indices is not None:
            sorted_subset = Subset(dataset=dataset, indices=inverse_val_indices)
        else:
            print(f"No valid indices for {label_type}, skipping sort.")
            continue
        # Remove all original shards in the label_dir
        for item in os.listdir(os.path.join(root_dir, label_type)):
            item_path = os.path.join(root_dir, label_type, item)
            if os.path.isdir(item_path) and item.endswith('.lmdb'):
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path) and (item.endswith('.mdb') or item.endswith('.pt')):
                os.remove(item_path)
        # Save sorted LMDB as data0000.lmdb in the original label_dir
        out_dir = os.path.join(root_dir, label_type, "data0000.lmdb")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, "data.mdb")
        map_size = 1099511627776 * 2
        env = lmdb.open(out_dir, map_size=map_size)
        with env.begin(write=True) as txn:
            for idx, data in enumerate(sorted_subset):
                txn.put(str(int(idx)).encode(), data.numpy().tobytes())
        env.close()
        print(f"Saved sorted LMDB for {label_type} to {out_dir}")
    print(f"Done. All original shards replaced with sorted LMDBs in {root_dir}")

    # 4. Clean up temp dirs
    for tmpdir in temp_dirs.values():
        shutil.rmtree(tmpdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and sort LMDB label shards in place, replacing original shards with sorted results.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/yuejian/project/MLFF-distill/OMOL/TOY/temp_strain/labels",
        help="Root directory containing label shards (force_jacobians, train_forces, val_forces, indices)",
    )
    args = parser.parse_args()
    merge_and_sort_labels(args.root_dir)
