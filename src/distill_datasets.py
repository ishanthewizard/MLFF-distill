import bisect
import os
from torch.utils.data import Dataset, DataLoader
import lmdb
import torch
import numpy as np
import logging 
from torch.utils.data import Sampler

class CombinedDataset(Dataset):
    def __init__(self, main_dataset, teach_force_dataset, force_jac_dataset=None):
        if  len(main_dataset) != len(teach_force_dataset):
            logging.info("WARNING: TEACH FORCE DATASET DIFFERENT SIZE")
        if force_jac_dataset and len(main_dataset) != len(force_jac_dataset):
            logging.info("WARNING: FORCE JACOBIAN DIFFERENT SIZE")
        self.main_dataset = main_dataset
        self.teach_force_dataset = teach_force_dataset
        self.force_jac_dataset = force_jac_dataset 

    def __len__(self):
        return len(self.main_dataset)  # Assuming both datasets are the same size

    def __getitem__(self, idx):
        main_batch = self.main_dataset[idx]
        num_atoms = main_batch.natoms
        
        teacher_forces = self.teach_force_dataset[idx].reshape(num_atoms, 3)
        if self.force_jac_dataset:
            num_free_atoms = (main_batch.fixed == 0).sum().item()
            # force_jacs  = self.force_jac_dataset[idx].reshape(num_free_atoms, 3, num_atoms, 3) 
            force_jacs = self.force_jac_dataset[idx] # DON'T RESHAPE! We'll do it later, easier for atoms of different lengths
        else: 
            force_jacs = None
        main_batch.teacher_forces = teacher_forces
        main_batch.force_jacs = force_jacs
        return main_batch

    def close_db(self):
        self.main_dataset.close_db()
        self.teach_force_dataset.close_db()
        if self.force_jac_dataset:
            self.force_jac_dataset.close_db() 


class SimpleDataset(Dataset):
    def __init__(self, folder_path, dtype=np.float32):
        self.folder_path = folder_path
        self.dtype = dtype
        
        # List all LMDB files in the folder
        self.db_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.lmdb')])
        assert len(self.db_paths) > 0, "No LMDB files found in the specified folder."

        self.envs = []
        self._keys = []
        self._keylen_cumulative = []

        total_entries = 0
        for db_path in self.db_paths:
            env = lmdb.open(db_path, readonly=True, lock=False)
            self.envs.append(env)
            with env.begin() as txn:
                num_entries = txn.stat()['entries']
                total_entries += num_entries
                self._keylen_cumulative.append(total_entries)

    def __len__(self):
        return self._keylen_cumulative[-1] if self._keylen_cumulative else 0

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()  # Convert tensor to integer

        # Find which database to access
        db_idx = bisect.bisect_right(self._keylen_cumulative, index)

        with self.envs[db_idx].begin() as txn:
            byte_data = txn.get(str(index).encode())
            if byte_data:
                tensor = torch.from_numpy(np.frombuffer(byte_data, dtype=self.dtype))
                return tensor
            else:
                raise Exception(f"Data not found for index {index} in LMDB file.")

    def close_db(self):
        for env in self.envs:
            env.close()

class SubsetDistributedSampler(Sampler):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)