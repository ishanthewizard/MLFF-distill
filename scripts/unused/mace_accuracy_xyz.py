from itertools import islice
import os
import numpy as np
from ase.io import iread
from ase import Atoms
from mace.calculators import mace_off, mace_anicc
from tqdm import tqdm

def calculate_l2mae(true, predicted):
    return np.mean(np.abs(true - predicted))

def calculate_l2mae_forces(true_forces, predicted_forces):
    l2mae_forces = []
    for true, pred in zip(true_forces, predicted_forces):
        l2mae_forces.append(np.mean(np.abs(true - pred)))
    return np.mean(l2mae_forces)

def get_accuracy(xyz_file_path, model='large'):
    # Load model
    calc = mace_off(model=model, dispersion=False, default_dtype="float32", device="cuda")
    # calc = mace_mp(model=model, dispersion=dispersion, default_dtype=default_dtype, device=device)
    # calc = mace_anicc(device="cuda")

    # Load the dataset from the .xyz file

    true_energies = []
    true_forces = []
    predicted_energies = []
    predicted_forces = []
    subsample_rate = 1000  # This will take every 10th item

    for atoms in tqdm(islice(iread(xyz_file_path), 1000)):
        true_energies.append(atoms.info['MACE_energy'])
        true_forces.append(atoms.get_forces())

        # atoms.calc = calc

        predicted_energies.append(atoms.get_potential_energy())
        predicted_forces.append(atoms.get_forces())
    
    l2mae_energy = calculate_l2mae(np.array(true_energies), np.array(predicted_energies))
    l2mae_forces = calculate_l2mae_forces(true_forces, predicted_forces)
    
    print(f"L2MAE Energy: {l2mae_energy}")
    print(f"L2MAE Forces: {l2mae_forces}")

if __name__ == "__main__":
    # xyz_file_path = '/data/ishan-amin/train_large_neut_no_bad_clean.xyz'
    xyz_file_path = '/data/ishan-amin/test_large_neut_all.xyz'
    get_accuracy(xyz_file_path)
