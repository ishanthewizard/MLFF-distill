from fairchem.core.common.registry import registry
from tqdm import tqdm
import numpy as np

# dataset_path = '/data/shared/MLFF/MD22/95_lmdb/Ac-Ala3-NHMe/train/'
# dataset_path = '/data/ishan-amin/spice_separated/Solvated_Amino_Acids/train'
# dataset_path = '/data/ishan-amin/MPtraj/mptraj_seperated_all_splits/Perovskites_noIons/train'
# dataset_path = '/data/ishan-amin/MPtraj/mptraj_seperated_all_splits/Bandgap_greater_than_5/train'
dataset_path = '/data/ishan-amin/MPtraj/mptraj_seperated_all_splits/Yttrium/train'
print(registry)
config = {"src": dataset_path}
dataset = registry.get_dataset_class("lmdb")(config)

# Initialize lists to store the values
y_values = []
forces_values = []

# Iterate over all samples to gather data
for sample in tqdm(dataset):
    y_values.append(sample.corrected_total_energy.numpy().flatten())
    forces_values.append(sample['force'].numpy().flatten())

# Convert lists to numpy arrays
y_values = np.concatenate(y_values)
forces_values = np.concatenate(forces_values)

# Calculate means and standard deviations
y_mean = np.mean(y_values)
y_std = np.std(y_values)

forces_mean = np.mean(forces_values)
forces_std = np.std(forces_values)

print(f"Mean of dataset.y: {y_mean}")
print(f"Standard Deviation of dataset.y: {y_std}")
print(f"Mean of dataset['forces']: {forces_mean}")
print(f"Standard Deviation of dataset['forces']: {forces_std}")
