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

toby_path = "/data/shared/MLFF/SPICE/maceoff_split/test"

toby_spice_dataset = registry.get_dataset_class("lmdb")({"src": toby_path})
fairchem_dataset = registry.get_dataset_class("lmdb")({"src": toby_path})

# dataset_path = os.path.join(main_path)
# dataset = registry.get_dataset_class("ase_db")({"src": dataset_path, 'a2g_args': { 'r_energy': True, 'r_forces': True, 'r_stress': True}})
# a = dataset[0]
breakpoint()
print(fairchem_dataset[0])
breakpoint()
