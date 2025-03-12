import os
import lmdb
import pickle
import random
import shutil
from tqdm import tqdm
from fairchem.core.common.registry import registry
from src.distill_datasets import SimpleDataset

# Paths
main_path = "/data/ishan-amin/MPtrj/MPtrj_seperated/Yttrium/"
labels_folder = "labels/mace_mp_large_Yttrium"
output_path = "/data/ishan-amin/MPtrj/MPtrj_seperated_all_splits/Yttrium/"
output_label_path = "labels/mace_mp_all_splits_Yttrium"


os.makedirs(output_path)
os.makedirs(output_label_path)
# Load datasets
val_dataset = registry.get_dataset_class("lmdb")(
    {"src": os.path.join(main_path, "val")}
)
train_dataset = registry.get_dataset_class("lmdb")(
    {"src": os.path.join(main_path, "train")}
)
force_jac_dataset = SimpleDataset(os.path.join(labels_folder, "force_jacobians"))
teacher_force_train_dataset = SimpleDataset(os.path.join(labels_folder, "train_forces"))

# Ensure dataset lengths match
assert len(train_dataset) == len(force_jac_dataset)
assert len(train_dataset) == len(teacher_force_train_dataset)

# Copy val dataset from main_path to output_path
val_output_path = os.path.join(output_path, "val")
val_label_output_path = os.path.join(output_label_path, "val_forces")


shutil.copytree(os.path.join(main_path, "val"), val_output_path)
print(f"Validation dataset copied to {val_output_path}")

shutil.copytree(os.path.join(labels_folder, "val_forces"), val_label_output_path)
print(f"Validation dataset copied to {val_label_output_path}")

# Split percentage and set random seed for reproducibility

total_samples = len(train_dataset)
test_size = len(val_dataset)

# Generate random indices for test and train splits
indices = list(range(total_samples))

random_seed = 42  # Set your seed value for reproducibility
random.seed(random_seed)
random.shuffle(indices)  # Random but reproducible shuffle

test_idxs = indices[:test_size]
train_idxs = indices[test_size:]


# Helper function to save LMDB datasets
def save_lmdb_dataset(dataset, indices, new_file_path):
    os.makedirs(new_file_path)
    db = lmdb.open(
        os.path.join(new_file_path, "data.lmdb"),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    i = 0
    for idx in tqdm(indices):
        txn = db.begin(write=True)
        sample = dataset[idx]
        txn.put(f"{i}".encode("ascii"), pickle.dumps(sample, protocol=-1))
        i += 1
        txn.commit()

    print(f"SAVED {i} samples to {new_file_path}")
    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(i, protocol=-1))
    txn.commit()
    db.sync()
    # close environment
    db.close()


# Save the train dataset
save_lmdb_dataset(train_dataset, train_idxs, os.path.join(output_path, "train"))
save_lmdb_dataset(train_dataset, test_idxs, os.path.join(output_path, "test"))


# Helper function to save force Jacobians or teacher forces
def save_tensor_dataset(dataset, indices, file_path):
    os.makedirs(file_path)
    env = lmdb.open(file_path, map_size=1099511627776 * 2)
    with env.begin(write=True) as txn:
        i = 0
        for idx in tqdm(indices):
            sample = dataset[idx]
            sample_id = str(i)
            txn.put(sample_id.encode(), sample.numpy().tobytes())
            i += 1
        print(f"SAVED {i} samples to {file_path}")
    env.close()
    print("SAVED:", i)
    print(f"All tensors saved to LMDB:{file_path}")


# Save the force Jacobians and teacher forces datasets
save_tensor_dataset(
    force_jac_dataset,
    train_idxs,
    os.path.join(output_label_path, "force_jacobians", "data.lmdb"),
)
save_tensor_dataset(
    teacher_force_train_dataset,
    train_idxs,
    os.path.join(output_label_path, "train_forces", "data.lmdb"),
)

print(
    "All datasets have been successfully split, saved, and validation dataset copied."
)
