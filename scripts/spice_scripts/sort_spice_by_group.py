import os
import pickle
import random
from tqdm import tqdm
import lmdb
from pathlib import Path
from fairchem.core.common.registry import registry
import numpy as np
from torch.utils.data import Subset
import torch


def categorize_samples_by_name(dataset, output_dir, test=False):
    categorized_samples = {}

    # Iterate through the dataset and categorize samples by 'dataset_name'
    for sample in tqdm(dataset, desc="Processing samples"):
        dataset_name = sample.dataset_name
        if dataset_name not in categorized_samples:
            categorized_samples[dataset_name] = []
        categorized_samples[dataset_name].append(sample)

    # Create separate LMDB datasets for each category
    for dataset_name, samples in categorized_samples.items():
        if test:
            _save_to_lmdb(samples, os.path.join(output_dir, dataset_name, "test"))
        else:
            save_samples_to_lmdb(samples, output_dir, dataset_name)


def save_samples_to_lmdb(samples, output_dir, dataset_name, split_ratio=0.85):
    # Shuffle the samples
    random.shuffle(samples)

    # Split the samples into train and val sets
    split_index = int(len(samples) * split_ratio)
    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    # Define paths for train and val datasets
    train_db_path = os.path.join(output_dir, dataset_name, "train")
    val_db_path = os.path.join(output_dir, dataset_name, "val")

    # Save the train samples
    _save_to_lmdb(train_samples, train_db_path)

    # Save the val samples
    _save_to_lmdb(val_samples, val_db_path)


def _save_to_lmdb(samples, db_path):
    # Check if the directory exists
    if os.path.exists(db_path):
        print(f"The directory '{db_path}' exists.")
        raise Exception("The directory already exists")

    os.makedirs(db_path, exist_ok=True)
    db = lmdb.open(
        os.path.join(db_path, "data.lmdb"),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    for idx, sample in enumerate(tqdm(samples)):
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(sample, protocol=-1))
        txn.commit()

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(samples), protocol=-1))
    txn.commit()

    db.sync()
    db.close()
    print(f"Saved {len(samples)} samples to {db_path}")


# Example usage:
# save_samples_to_lmdb(samples, output_dir, dataset_name)
# Usage
dataset_path = "/data/ishan-amin/maceoff_split"  # Update this with your actual path
output_dir = (
    "/data/ishan-amin/spice_separated"  # Update this with your desired output directory
)
dataset_type = "train"
train_dataset = registry.get_dataset_class("lmdb")(
    {"src": os.path.join(dataset_path, dataset_type)}
)
categorize_samples_by_name(train_dataset, output_dir, test=(dataset_type == "test"))
