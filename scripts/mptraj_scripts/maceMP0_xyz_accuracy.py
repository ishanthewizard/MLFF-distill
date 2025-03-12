import os
import random
import numpy as np
from ase.io import read, iread
from mace.calculators import mace_mp
from tqdm import tqdm

# Path to the directory containing the extxyz files
folder_path = "/data/shared/MPtrj/original"

# Get a list of all extxyz files
files = [f for f in os.listdir(folder_path) if f.endswith(".extxyz")]

# Randomly select 300 files
random_subset = random.sample(files, 1000)

# Initialize the Mace-mp0 model
mace_model = mace_mp(
    model="large", dispersion=False, default_dtype="float32", device="cuda"
)


# Function to compute L2 mean absolute error
def compute_mae(true_forces, predicted_forces):
    l2mae_forces = []
    for true, pred in zip(true_forces, predicted_forces):
        l2mae_forces.append(np.mean(np.abs(true - pred)))
    return np.mean(l2mae_forces)


# Initialize a list to store L2MAE for each configuration
l2mae_list = []

# Iterate through the random subset
total = 0
for file in tqdm(random_subset):
    file_path = os.path.join(folder_path, file)

    # Load the configuration using ASE's iread
    i = 0
    for atoms in read(file_path, index=":"):
        i += 1
        # print(f"Position len: {len(atoms.positions)}")
        # Extract true forces from the atoms object
        true_forces = atoms.get_forces()

        atoms.calc = mace_model
        # Evaluate Mace-mp0 on the configuration to get predicted forces
        predicted_forces = atoms.get_forces()

        # Compute L2MAE for the forces
        l2mae = compute_mae(true_forces, predicted_forces)

        # Store the result
        l2mae_list.append(l2mae)
    # print(f"Len of file: {i}")
    total += i
# Print the average L2MAE over the 300 configurations
average_l2mae = np.mean(l2mae_list)
print(f"Average L2MAE over {total} configurations: {average_l2mae}")
