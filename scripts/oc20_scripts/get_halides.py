# main_path = '/data/shared/MLFF/OC20/s2ef/2M/'
# train = os.path.join(main_path, 'train')
# # test = os.path.join(main_path, 'test')
# pickle_file = 'oc20_data_mapping.pkl'
# with open(pickle_file, 'rb') as f:
#     data_dict = pickle.load(f)
# train_dataset = registry.get_dataset_class("lmdb")({"src": train})
# num = 0
# samples = [0,0,0,0]
# for sample in tqdm(train_dataset):
#     key = "random" + str(sample.sid)
#     samples[data_dict[key]['class']] += 1


# # test_dataset = registry.get_dataset_class("lmdb")({"src": test})
# total =  len(train_dataset) 
# print(total, samples)



import os
import pickle
import random
from tqdm import tqdm
import lmdb
from fairchem.core.common.registry import registry

def categorize_samples_by_name(dataset, output_dir, test=False):
    # class:
    # 0 - intermetallics
    # 1 - metalloids
    # 2 - non-metals
    # 3 - halides
    # tags:
    # 0 - Sub-surface slab atoms
    # 1 - Surface slab atoms
    # 2 - Adsorbate atoms
    categorized_samples = {"Halides": [], "Nonmetals_N_in_adsorbate": [], "Metalloids_O_in_adsorbate": []}
    pickle_file = 'labels/oc20_data_mapping.pkl'
    with open(pickle_file, 'rb') as f:
        data_dict = pickle.load(f)
    # Iterate through the dataset and categorize samples by 'dataset_name'
    for sample in tqdm(dataset, desc="Processing samples"):
        key = "random" + str(sample.sid)

        if data_dict[key]['class'] == 3: # is a halide
            categorized_samples["Halides"].append(sample)
        if data_dict[key]['class'] == 2: # is a nonmetal
            nitrogen_in_adsorbate = (sample.atomic_numbers[sample.tags == 2] == 7).any()
            if nitrogen_in_adsorbate:
                categorized_samples["Nonmetals_N_in_adsorbate"].append(sample)
        if data_dict[key]['class'] == 1: # is a metalloid
            oxygen_in_adsorbate = (sample.atomic_numbers[sample.tags == 2] == 8).any()
            if oxygen_in_adsorbate:
                categorized_samples["Metalloids_O_in_adsorbate"].append(sample)

    # Create separate LMDB datasets for each category
    for dataset_name, samples in categorized_samples.items():
        if test:
            _save_to_lmdb(samples, os.path.join(output_dir, dataset_name, "test"))
        else:
            save_samples_to_lmdb(samples, output_dir, dataset_name)

def save_samples_to_lmdb(samples, output_dir, dataset_name, split_ratio=0.90):
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
        os.path.join(db_path, 'data.lmdb'),
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

dataset_path = "/data/shared/MLFF/OC20/s2ef/all/val_id"
output_dir = "/data/ishan-amin/OC20_seperated/"   # Update this with your desired output directory
dataset_type = "test" # if set to train, then it'll split into a train and val set
train_dataset = registry.get_dataset_class("lmdb")({"src": dataset_path})
categorize_samples_by_name(train_dataset, output_dir, test=(dataset_type == "test"), )
