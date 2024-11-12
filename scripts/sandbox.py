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
# breakpoint()


# # List of main paths and corresponding DB paths
# main_paths = {
#     'DES370K_Monomers': 'data/SPICE/spice_seperated_w_energies/DES30K_Monomers',
#     'Solvated_Amino_Acids': 'data/SPICE/spice_seperated_w_energies/Solvated_Amino_Acids',
#     'Iodine': 'data/SPICE/spice_seperated_w_energies/Iodine'
# }

# # Function to process a given dataset and save to LMDB
# def process_dataset(main_path, db_path, split):
#     os.makedirs(os.path.join(db_path, split))
#     dataset_path = os.path.join(main_path, split)
#     dataset = registry.get_dataset_class("lmdb")({"src": dataset_path})
    
#     output_db_path = os.path.join(db_path, split, 'data.lmdb')
#     env = lmdb.open(output_db_path, 
#                     map_size=int(1099511627776 * 2),
#                     subdir=False,
#                     meminit=False,
#                     map_async=True,)  # 2TB map size
#     for i, sample in tqdm(enumerate(dataset), desc=f"Processing {split} dataset for {main_path.split('/')[-1]}"):
#         # Compute energy per atom and update sample output (assuming `sample.y` is energy and `sample.natoms` is number of atoms)
#         sample.energy_per_atom = sample.y / sample.natoms.item()
#         # Convert the sample to bytes and store in LMDB
#         txn = env.begin(write=True)
#         txn.put(f"{i}".encode("ascii"), pickle.dumps(sample, protocol=-1))
#         txn.commit()
        

    
#     env.sync()
#     env.close()


# # Iterate over each main path and process both train and val datasets
# for dataset_name, db_path in main_paths.items():
#     main_path = f'data/SPICE/spice_separated/{dataset_name}'
#     for split in ['train', 'val']:
#         process_dataset(main_path, db_path, split)



# main_path = '/pscratch/sd/i/ishan_a/OC20_seperated/Halides'

# train_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(main_path, 'train')})
# val_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(main_path, 'val')})
# normal_label_path = '/pscratch/sd/i/ishan_a/OC20_labels/Nonmetals_N_in_adsorbate_gemnet_oc_large_s2ef_all_md_final'

# force_jac_dataset = SimpleDataset(os.path.join(normal_label_path, 'force_jacobians'))


# teacher_val_forces = SimpleDataset(os.path.join(normal_label_path, 'val_forces'))
# teacher_train_forces = SimpleDataset(os.path.join(normal_label_path, 'train_forces'))

# print(len(train_dataset))
# # print(len(force_jac_dataset))
# # print(len(teacher_train_forces))
# print(" ")
# # print(len(teacher_val_forces))
# print(len(val_dataset))
# breakpoint()


# db_path = '/pscratch/sd/i/ishan_a/OC20_labels/Nonmetals_N_in_adsorbate_gemnet_oc_large_s2ef_all_md_ALL_HESSIANS'

# env = lmdb.open(os.path.join(db_path, 'data.lmdb'), map_size=int(1099511627776 * 2)) 
# with env.begin(write=True) as txn:
#     for i, sample in tqdm(enumerate(samples)):
#         sample_id = str(i)
#         sample_output = sample.numpy()  # this function needs to output an array where each element correponds to the label for an entire molecule
#             # Convert tensor to bytes and write to LMDB
#         txn.put(sample_id.encode(), sample_output.tobytes())
#         txn.commit()

#     db.sync()
#     db.close()

# Close the new LMDB environment







    