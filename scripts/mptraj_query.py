import os
from fairchem.core.common.registry import registry
from mp_api.client import MPRester
from tqdm import tqdm
import os
import pickle
import random
import lmdb



def count_stuff(dataset):
    mptraj_sulphur_samples = []
    mpproj_sulphur_ids = set()
    with MPRester("PZ6Gx8dJTmeySErT5xuuaGhypYyg86p4") as mpr:
        docs = mpr.summary.search(elements=["S"], 
                                fields=["material_id"])
        mpproj_sulphur_ids.update([sample.material_id for sample in docs])
    count = 0
    seen_ids = set()
    for sample in tqdm(dataset):
        seen_ids.add(sample.mp_id)
        if sample.mp_id in mpproj_sulphur_ids:
            count += 1
            # mptraj_sulphur_samples.append(sample)
    print(f"number of unique ids:{len(seen_ids)}")
    print("COUNT:", count)



def get_different_splits(dataset, output_dir, test):
    categorized_samples = {"Perovskites": []}
    mpproj_query_ids = set()
    with MPRester("PZ6Gx8dJTmeySErT5xuuaGhypYyg86p4") as mpr:
        docs = mpr.summary.search(spacegroup_symbol= 'Pm-3m', 
                                fields=["material_id", "symmetry"])
        mpproj_query_ids.update([sample.material_id for sample in docs])
    count = 0
    for sample in tqdm(dataset):
        if sample.mp_id in mpproj_query_ids:
            categorized_samples["Perovskites"].append(sample)
            count += 1

    for dataset_name, samples in categorized_samples.items():
        if test:
            _save_to_lmdb(samples, os.path.join(output_dir, dataset_name, "test"))
        else:
            save_samples_to_lmdb(samples, output_dir, dataset_name)
    
    print("COUNT:", count)

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



if __name__ == "__main__":
    # dataset_path = "/data/ishan-amin/MPtraj/mace_mp_split"  # Update this with your actual path
    dataset_path = "/data/shared/MPTrj/lmdb"
    output_dir = "/data/ishan-amin/mptraj_seperated_eric"   # Update this with your desired output directory
    dataset_type = "train"
    dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, dataset_type)})
    get_different_splits(dataset, output_dir, test=(dataset_type == "test"))