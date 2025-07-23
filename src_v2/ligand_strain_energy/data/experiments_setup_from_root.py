"""
User Guide:
-----------
This script sets up experiment directories, splits, and aselmdb files for ligand strain energy workflows.

Usage:
    python experiments_setup_from_root.py \
        --root_data_dir /path/to/ligandboundconf3.0corrected/subset_200 \
        --output_dir /path/to/OMOL/strain_energy_application \
        --parquet_path /path/to/original_ligboundconf_2_molecules_and_energies.parquet \
        --train_ratio 0.8 \
        --subroot_name my_subroot

Arguments:
    --root_data_dir   : Root directory containing ligand subdirectories (required)
    --output_dir      : Output directory for experiment setup (required)
    --parquet_path    : Path to the original ligand/energy parquet file (required)
    --train_ratio     : Fraction of data for training (default: 0.8)
    --subroot_name    : Name for the experiment subdirectory (default: my_subroot)

This script will:
- Create all necessary output directories
- Generate train/val/test splits and save as .pkl
- Compose lists of .sdf paths for each split
- Call process_sdf_paths_to_aselmdb to create aselmdb files for each split
- Dump filtered parquet files for each split to the appropriate directory
"""

import os
import sys
# Add the project root to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
print(f"Adding project root to sys.path: {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import chain
import argparse
from sdf2aselmdb import process_sdf_paths_to_aselmdb
from src_v2.OMol.compute_metadata import compute_metadata

def dedup(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def main():
    parser = argparse.ArgumentParser(description="Setup experiment directories, splits, and aselmdb files for ligand strain energy workflows.")
    parser.add_argument('--root_data_dir', type=str, required=False, default="/home/yuejian/project/MLFF-distill/data/ligandboundconf3.0corrected/subset_200", help='Root directory containing ligand subdirectories')
    parser.add_argument('--output_dir', type=str, required=False, default="/home/yuejian/project/MLFF-distill/OMOL/strain_energy_application", help='Output directory for experiment setup')
    parser.add_argument('--parquet_path', type=str, required=False, default="/home/yuejian/project/MLFF-distill/data/original_ligboundconf_2_molecules_and_energies.parquet", help='Path to the original ligand/energy parquet file')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Fraction of data for training (default: 0.8)')
    parser.add_argument('--subroot_name', type=str, default='local_200_global_200', help='Name for the experiment subdirectory (default: my_subroot)')
    args = parser.parse_args()

    root_data_dir = args.root_data_dir
    output_dir = args.output_dir
    parquet_path = args.parquet_path
    train_ratio = args.train_ratio
    subroot_name = args.subroot_name

    # 1. Create necessary directories
    subroot_dir = os.path.join(output_dir, subroot_name)
    os.makedirs(subroot_dir, exist_ok=True)

    path_dict = {}
    path_dict["subroot_dir"] = os.path.abspath(subroot_dir)
    required_subdirs = ["train_val_test_aselmdb", "label_aselmdb", "strain_energy_inputs", "train_val_splits"]
    for sub in required_subdirs:
        sub_path = os.path.join(subroot_dir, sub)
        os.makedirs(sub_path, exist_ok=True)
        path_dict[sub] = os.path.abspath(sub_path)

    train_val_test_dir = os.path.join(subroot_dir, "train_val_test_aselmdb")
    for split in ["train", "val", "test"]:
        split_path = os.path.join(train_val_test_dir, split)
        os.makedirs(split_path, exist_ok=True)
        path_dict[f"train_val_test_{split}"] = os.path.abspath(split_path)

    strain_energy_inputs_dir = path_dict["strain_energy_inputs"]
    for split in ["train", "val", "trainval"]:
        split_path = os.path.join(strain_energy_inputs_dir, split)
        os.makedirs(split_path, exist_ok=True)
        path_dict[f"strain_energy_inputs_{split}"] = os.path.abspath(split_path)

    # 2. Find all subdirectories in root_data_dir
    subdirs = [d for d in os.listdir(root_data_dir) if os.path.isdir(os.path.join(root_data_dir, d))]
    subdir_paths = [os.path.join(root_data_dir, d) for d in subdirs if os.path.isdir(os.path.join(root_data_dir, d))]

    # 3. Get ligand IDs (assume all subdirs have the same ligand IDs)
    subdir_ligand_ids = []
    for path in subdir_paths:
        ligand_ids = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        subdir_ligand_ids.append(sorted(ligand_ids))
    all_equal = all(ids == subdir_ligand_ids[0] for ids in subdir_ligand_ids)
    if all_equal and len(subdir_ligand_ids) > 0:
        ligand_ids = subdir_ligand_ids[0]
    else:
        raise ValueError("Ligand IDs are not consistent across subdirectories; cannot proceed with train/val split.")

    # 4. Train/val/test split
    ligand_ids_train, ligand_ids_test = train_test_split(ligand_ids, train_size=train_ratio, random_state=42, shuffle=True)

    # Save splits
    train_val_splits_dir = path_dict['train_val_splits']
    os.makedirs(train_val_splits_dir, exist_ok=True)
    with open(os.path.join(train_val_splits_dir, "ligand_ids_train.pkl"), "wb") as f:
        pickle.dump(ligand_ids_train, f)
    with open(os.path.join(train_val_splits_dir, "ligand_ids_test.pkl"), "wb") as f:
        pickle.dump(ligand_ids_test, f)
    
    # Filter subdirs to only those that have DFT results for all ligand IDs
    valid_subdirs = []
    removed_subdirs = []
    for subdir in subdir_paths:
        subdir_path = os.path.join(root_data_dir, subdir)
        ligand_ids_in_subdir = [
            d for d in os.listdir(subdir_path)
            if os.path.isdir(os.path.join(subdir_path, d))
        ]
        all_success = True
        for ligand_id in ligand_ids_in_subdir:
            success_path = os.path.join(subdir_path, ligand_id, "DFT", "success.txt")
            if not os.path.isfile(success_path):
                all_success = False
                break
        if all_success:
            valid_subdirs.append(subdir)
        else:
            removed_subdirs.append(subdir)

    if removed_subdirs:
        print(f"Removed subdirs without complete DFT results: {removed_subdirs}")
    else:
        print("All subdirs have complete DFT results.")

    # Use only valid_subdirs from now on
    subdir_paths = valid_subdirs
    
    # 5. Compose lists of .sdf paths for each split
    subdir_ligand_sdf_paths_train = []
    subdir_ligand_sdf_paths_test = []
    for path in subdir_paths:
        ligand_ids_in_subdir = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        ligand_sdf_paths_train = []
        ligand_sdf_paths_test = []
        for ligand_id in ligand_ids_in_subdir:
            ligand_dir = os.path.join(path, ligand_id)
            sdf_path = os.path.join(ligand_dir, f"{ligand_id}.sdf")
            if ligand_id in ligand_ids_train:
                ligand_sdf_paths_train.append(sdf_path)
            if ligand_id in ligand_ids_test:
                ligand_sdf_paths_test.append(sdf_path)
        subdir_ligand_sdf_paths_train.append(ligand_sdf_paths_train)
        subdir_ligand_sdf_paths_test.append(ligand_sdf_paths_test)
    flat_train_sdf_paths = list(chain.from_iterable(subdir_ligand_sdf_paths_train))
    flat_test_sdf_paths = list(chain.from_iterable(subdir_ligand_sdf_paths_test))
    flat_train_sdf_paths_dedup = dedup(flat_train_sdf_paths)
    flat_test_sdf_paths_dedup = dedup(flat_test_sdf_paths)

    # 6. Write aselmdb for each split
    train_aselmdb_path = os.path.join(path_dict["train_val_test_train"], "data0000.aselmdb")
    val_aselmdb_path = os.path.join(path_dict["train_val_test_val"], "data0000.aselmdb")
    test_aselmdb_path = os.path.join(path_dict["train_val_test_test"], "data0000.aselmdb")

    process_sdf_paths_to_aselmdb(flat_train_sdf_paths_dedup, train_aselmdb_path, "train")
    process_sdf_paths_to_aselmdb(flat_test_sdf_paths_dedup, val_aselmdb_path, "val")
    process_sdf_paths_to_aselmdb(flat_test_sdf_paths_dedup, test_aselmdb_path, "test")

    # 6b. Compute metadata for each aselmdb directory
    print("Computing metadata for train/val/test aselmdb directories...")
    compute_metadata(path_dict["train_val_test_train"])
    compute_metadata(path_dict["train_val_test_val"])
    compute_metadata(path_dict["train_val_test_test"])

    # 7. Dump filtered parquet files for each split
    ligboundconf_df = pd.read_parquet(parquet_path)
    ligboundconf_df_train = ligboundconf_df[ligboundconf_df["ligand_id"].isin(ligand_ids_train)].copy()
    ligboundconf_df_val = ligboundconf_df[ligboundconf_df["ligand_id"].isin(ligand_ids_test)].copy()
    ligand_ids_trainval = set(ligand_ids_train) | set(ligand_ids_test)
    ligboundconf_df_trainval = ligboundconf_df[ligboundconf_df["ligand_id"].isin(ligand_ids_trainval)].copy()

    for split_name, df in [
        ("train", ligboundconf_df_train),
        ("val", ligboundconf_df_val),
        ("trainval", ligboundconf_df_trainval)
    ]:
        split_dir = path_dict.get(f"strain_energy_inputs_{split_name}", None)
        if split_dir is None:
            raise RuntimeError(f"Could not find directory for split '{split_name}' in path_dict")
        out_path = os.path.join(split_dir, f"ligboundconf_{split_name}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"Saved {split_name} DataFrame ({len(df)} rows) to {out_path}")

    print("Experiment setup complete.")

if __name__ == "__main__":
    main()
