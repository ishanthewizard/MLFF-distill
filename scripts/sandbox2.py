import os
import pickle
from fairchem.core.common.registry import registry
# from mp_api.client import MPRester
# from ase.io import read
from src.distill_datasets import SimpleDataset
from tqdm import tqdm
import torch
import numpy as np
import lmdb
from torch.utils.data import Subset
from matbench_discovery.energy import get_e_form_per_atom

from mp_api.client import MPRester
import json

# def get_mp_reference_energies(api_key: str) -> dict:
#     """
#     Retrieve elemental reference energies from Materials Project.
    
#     Args:
#         api_key: Materials Project API key
        
#     Returns:
#         Dictionary mapping atomic numbers to reference energies
#     """
#     with MPRester(api_key) as mpr:
#         # Get all elements data
#         # elements = mpr.get_entries_in_chemsys(["Fe"])[0].energy_per_atom  # Just to get the units
#         # print(f"Note: Energies will be in {elements.units}")
        
#         # Dictionary to store reference energies
#         reference_energies = {}
        
#         # List of elements to query (you can extend this)
#         elements_list = ["C", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
#                         "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca"]
        
#         for element in elements_list:
#             try:
#                 # Get the most stable elemental entry
#                 entries = mpr.get_entries_in_chemsys([element])
#                 breakpoint()
#                 if entries:
#                     lowest_energy_entry = min(entries, key=lambda e: e.energy_per_atom)
#                     atomic_number = mpr.get_materials_by_formula(element)[0].composition[0].Z
#                     reference_energies[atomic_number] = float(lowest_energy_entry.energy_per_atom)
#             except Exception as e:
#                 print(f"Could not get data for {element}: {e}")
        
#         return reference_energies

# def save_reference_energies(energies: dict, filename: str = "mp_reference_energies.json"):
#     """Save reference energies to a JSON file"""
#     with open(filename, 'w') as f:
#         json.dump(energies, f, indent=4)

# def main():
#     # Replace with your API key
#     API_KEY = "PZ6Gx8dJTmeySErT5xuuaGhypYyg86p4"
    
#     try:
#         energies = get_mp_reference_energies(API_KEY)
#         save_reference_energies(energies)
        
#         print("\nReference energies retrieved:")
#         for atomic_num, energy in energies.items():
#             print(f"Z={atomic_num}: {energy:.6f} eV")
            
#     except Exception as e:
#         print(f"Error: {e}")
#         print("\nTo use this script:")
#         print("1. Get an API key from https://materialsproject.org/open")
#         print("2. Replace 'YOUR_API_KEY' with your actual API key")

# if __name__ == "__main__":
#     main()



# main_path = 'data/MPtraj/mptraj_seperated_all_splits/Yttrium/'
main_path = '/data/shared/MPTrj/lmdb/'
train_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(main_path, 'train')})
breakpoint()

