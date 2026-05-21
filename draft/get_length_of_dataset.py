
import sys
sys.path.append("../..")
sys.path.append("..")
import torch
from src_v2.distill_datasets import LmdbDataset, LmdbHessianIndexDataset
import matplotlib.pyplot as plt
import lmdb
import pickle
from tqdm import tqdm
from fairchem.core.datasets import AseDBDataset
import numpy as np
from torch.utils.data import Subset
from IPython.display import Image, display
import ase




# load dataset
dataset_path = "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/lmdb_for_distillation/all_salt_all_concentration_first_50ps/train"
a2g_args = {  
    # "molecule_cell_size": 120.0,
    "r_energy": True,
    "r_forces": True,
    # "r_stress": True,
    "r_data_keys": [ 'spin','charge', "data_id"],
    # 'sid': 'data_id',
}
# select_args ={ "data_id":"ani2x"}
# select_args ={"selection": [("charge", "=", -1)] }
# select_args = {
#     "filter": lambda row: row._data.get("data_id") == "orbnet_denali"
# }
dataset = AseDBDataset({
    "src": dataset_path,
    "a2g_args": a2g_args,
    # "select_args": select_args,
})

# # How to know the length of the dataset:
print("Length of dataset:", len(dataset))