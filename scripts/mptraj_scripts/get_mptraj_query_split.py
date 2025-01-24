import os
from fairchem.core.common.registry import registry
from mp_api.client import MPRester
from tqdm import tqdm
import os
import pickle
import random
import lmdb

def get_different_splits(dataset, output_dir, test):
    categorized_samples = {"Bandgap_greater_than_5": []}
    mpproj_query_ids = set()
    key = 'Z6Gx8dJTmeySErT5xuuaGhypYyg86p4' # NOTE: use your own key! don't use mine
    with MPRester(key) as mpr:
        docs = mpr.summary.search(band_gap=(5, 3000), fields=["material_id"])
        mpproj_query_ids.update([sample.material_id for sample in docs])
    count = 0
    for sample in tqdm(dataset):
        if sample.mp_id in mpproj_query_ids:
            if len(sample.pos) < 2:
                print("BRUH")
                continue
            categorized_samples["Bandgap_greater_than_5"].append(sample)
            count += 1

    for dataset_name, samples in categorized_samples.items():
        save_samples_to_lmdb(samples, output_dir, dataset_name)
    
    print("COUNT:", count)

def save_samples_to_lmdb(samples, output_dir, dataset_name, train_ratio=0.7, val_ratio=0.15):
    # NOTE: WE also made a test set here, different than the other ones
    # Shuffle the samples
    random.shuffle(samples)

    # Split the samples into train, val, and test sets
    total_samples = len(samples)
    train_index = int(total_samples * train_ratio)
    val_index = int(total_samples * (train_ratio + val_ratio))

    train_samples = samples[:train_index]
    val_samples = samples[train_index:val_index]
    test_samples = samples[val_index:]

    # Define paths for train, val, and test datasets
    train_db_path = os.path.join(output_dir, dataset_name, "train")
    val_db_path = os.path.join(output_dir, dataset_name, "val")
    test_db_path = os.path.join(output_dir, dataset_name, "test")

    # Save the train, val, and test samples
    _save_to_lmdb(train_samples, train_db_path)
    _save_to_lmdb(val_samples, val_db_path)
    _save_to_lmdb(test_samples, test_db_path)

    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}, Test samples: {len(test_samples)}")


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
    # dataset_path = "/data/ishan-amin/MPtrj/mace_mp_split"  # Update this with your actual path
    dataset_path = "/data/ishan-amin/MPtrj/mace_mp_split/"
    output_dir = "/data/ishan-amin/MPtrj/MPtrj_seperated/Bandgap_greater_than_5"   # Update this with your desired output directory
    dataset_type = "train"
    dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(dataset_path, dataset_type)})
    get_different_splits(dataset, output_dir, test=(dataset_type == "test"))