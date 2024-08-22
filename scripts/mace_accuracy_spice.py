import os
from fairchem.core.common.registry import registry
import numpy as np
import lmdb
from ase.io import read
from ase import Atoms
from mace.calculators import mace_off, mace_anicc
import h5py
from tqdm import tqdm


def calculate_l2mae(true, predicted):
    return np.mean(np.abs(true - predicted))



def get_accuracy(file, model='large'):
    # Load model
    # calc = mace_off(model=model, dispersion=True,  default_dtype="float64", device="cuda")
    # calc = mace_mp(model=model, dispersion=dispersion, default_dtype=default_dtype, device=device)
    calc = mace_anicc(device="cuda")
    # Load the dataset
    dataset = file["water"]


    true_energies = []
    true_forces = []
    predicted_energies = []
    predicted_forces = []
    
    atomic_numbers = dataset["atomic_numbers"][:]
    for i in tqdm(range(len(dataset["conformations"]))):
        true_energies.append(dataset['dft_total_energy'][i])
        true_forces.append(dataset['dft_total_gradient'][i])

        atoms = Atoms(numbers=atomic_numbers, positions=dataset['conformations'][i])
        atoms.calc = calc

        predicted_energies.append(atoms.get_potential_energy())
        predicted_forces.append(atoms.get_forces())
    
    l2mae_energy = calculate_l2mae(np.array(true_energies), np.array(predicted_energies))
    l2mae_forces = calculate_l2mae(np.array(true_forces), np.array(predicted_forces))
    
    print(f"L2MAE Energy: {l2mae_energy}")
    print(f"L2MAE Forces: {l2mae_forces}")

if __name__ == "__main__":

    # dataset_path = '/data/ishan-amin/post_data/md17/ethanol/1k/' 
    file = h5py.File('/data/ishan-amin/SPICE-2.0.1.hdf5', 'r')
    # dataset_path = '/data/ishan-amin/post_data/md17/aspirin/1k/'
    # dataset_path = '/data/ishan-amin/post_data/md22/AT_AT/1k/'

    get_accuracy(file)

    
