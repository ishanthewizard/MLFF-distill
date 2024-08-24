import os
import numpy as np
from ase.io import read
from ase import Atoms
from mace.calculators import mace_off, mace_anicc
from tqdm import tqdm

def calculate_l2mae(true, predicted):
    return np.mean(np.abs(true - predicted))

def get_accuracy(xyz_file_path, model='large'):
    # Load model
    calc = mace_off(model=model, dispersion=False, default_dtype="float64", device="cuda")
    # calc = mace_mp(model=model, dispersion=dispersion, default_dtype=default_dtype, device=device)
    # calc = mace_anicc(device="cuda")

    # Load the dataset from the .xyz file
    conformations = read(xyz_file_path, index=':')

    true_energies = []
    true_forces = []
    predicted_energies = []
    predicted_forces = []
    
    for atoms in tqdm(conformations):
        true_energies.append(atoms.info['TotEnergy'])
        true_forces.append(atoms.arrays['dft_total_gradient'])

        atoms.calc = calc

        predicted_energies.append(atoms.get_potential_energy())
        predicted_forces.append(atoms.get_forces())
    
    l2mae_energy = calculate_l2mae(np.array(true_energies), np.array(predicted_energies))
    l2mae_forces = calculate_l2mae(np.array(true_forces), np.array(predicted_forces))
    
    print(f"L2MAE Energy: {l2mae_energy}")
    print(f"L2MAE Forces: {l2mae_forces}")

if __name__ == "__main__":
    xyz_file_path = '/data/ishan-amin/train_large_neut_no_bad_clean.xyz'
    get_accuracy(xyz_file_path)
