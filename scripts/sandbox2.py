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


# main_path = '/pscratch/sd/i/ishan_a/OC20_seperated/Nonmetals_N_in_adsorbate'
main_path = '/data/ishan-amin/OC20_seperated/Halides'
# val = os.path.join(main_path, 'val')
# # test = os.path.join(main_path, 'test')

train_dataset = registry.get_dataset_class("lmdb")({"src": os.path.join(main_path, 'train')})
test_dataset =   registry.get_dataset_class("lmdb")({"src": os.path.join(main_path, 'test')})

num = 0
seen_systemids = set()
for sample in tqdm(train_dataset):
    seen_systemids.add(sample.sid)


print(len(seen_systemids))


# for sample in train_dataset:
#     categorized_samples = {"Halides": [], "Nonmetals_N_in_adsorbate": [], "Metalloids_O_in_adsorbate": []}
#     pickle_file = 'labels/oc20_data_mapping.pkl'
#     with open(pickle_file, 'rb') as f:
#         data_dict = pickle.load(f)
#     # Iterate through the dataset and categorize samples by 'dataset_name'
#     for sample in tqdm(dataset, desc="Processing samples"):
#         key = "random" + str(sample.sid)