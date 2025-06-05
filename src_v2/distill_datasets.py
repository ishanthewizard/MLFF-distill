import bisect
from functools import cached_property
import os,sys
from typing import Any, Callable
from fairchem.core.datasets.base_dataset import BaseDataset
from fairchem.core.datasets.ase_datasets import apply_one_tags, AseDBDataset
from torch.utils.data import Dataset, DataLoader
import lmdb
import torch
import numpy as np
import logging 
from torch.utils.data import Sampler
from tqdm import tqdm
import shutil
import ase
from fairchem.core.common.registry import registry

class HessianSampler():
    
    def sample_with_mask(self, num_samples, mask):     
        # Calculate total available columns after applying the mask
        # Only rows where mask is True are considered
        assert num_samples <= len(mask), "System too small for the number of samples. Need to account for this case" 
        valid_rows = torch.where(mask)[0]  # Get indices of rows that are True
        if valid_rows.numel() == 0:
            raise ValueError("No valid rows available according to the mask")

        # Each valid row contributes 3 indices
        valid_indices = valid_rows.repeat_interleave(3) * 3 + torch.tensor([0, 1, 2]).repeat(valid_rows.size(0)).to(mask.device)

        # Sample unique indices from the valid indices
        chosen_indices = valid_indices[torch.randperm(valid_indices.size(0))[:num_samples]]

        # Convert flat indices back to row and column indices
        row_indices = chosen_indices // 3
        col_indices = chosen_indices % 3

        # Combine into 2-tuples
        samples = torch.stack((row_indices, col_indices), dim=1) # (num_samples, 2)
        
        return samples
    
    def sample_hessian(self, samples, num_atoms, force_jacs):
        force_jacs = force_jacs.reshape(num_atoms, 3, num_atoms, 3) # IMPORTANT: Assumes no masking
        force_jacs = force_jacs[samples[:, 0], samples[:, 1], :]# num_samples, num_atoms, 3
        force_jacs = force_jacs.permute(1, 0, 2).reshape(num_atoms, -1) # num_atoms, num_samples * 3 
        
        return force_jacs

class CombinedDataset(AseDBDataset):
    def __init__(
            self,
            config: dict,
            dataset_type: str,
            atoms_transform: Callable[[ase.Atoms, Any, ...], ase.Atoms] = apply_one_tags,
    ):
        super().__init__(config, atoms_transform)
        self.labels_folder = config['teacher_labels_folder']
        self.teacher_force_dataset = SimpleDataset(os.path.join(config['teacher_labels_folder'], f'{dataset_type}_forces'))
        
        if dataset_type == 'train':
            self.hessian_dataset = SimpleDataset(os.path.join(config['teacher_labels_folder'], 'force_jacobians'))
            self.hessian_sampler = HessianSampler()
        else:
            self.hessian_dataset = None
    
    def __getitem__(self, idx):
        main_batch = super().__getitem__(idx)
        # print("main_batch",main_batch)
        num_atoms = main_batch.natoms
        teacher_forces = self.teacher_force_dataset[idx].reshape(num_atoms, 3)
        num_samples = 2
        if self.hessian_dataset:
            # force_jacs  = self.force_jac_dataset[idx].reshape(num_free_atoms, 3, num_atoms, 3) 
            force_jacs = self.hessian_dataset[idx] # DON'T RESHAPE! We'll do it later, easier for atoms of different lengths
            samples = self.hessian_sampler.sample_with_mask(num_samples, torch.ones(num_atoms))
            force_jacs = self.hessian_sampler.sample_hessian(samples, num_atoms, force_jacs) # num_atoms, num_samples * 3 
            main_batch.forces_jac = force_jacs
            main_batch.samples = samples
            main_batch.num_samples = torch.tensor(num_samples)
        else: 
            main_batch.forces_jac = torch.zeros((num_atoms, num_samples * 3))
            main_batch.num_samples = torch.tensor(num_samples)
            
        main_batch.teacher_forces = teacher_forces
        
        return main_batch

class CombinedDatasetTrain(CombinedDataset):
    def __init__(self, config, atoms_transform=apply_one_tags):
        super().__init__(config, 'train', atoms_transform)
        
class CombinedDatasetVal(CombinedDataset):
    def __init__(self, config, atoms_transform=apply_one_tags):
        super().__init__(config, 'val', atoms_transform)





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
        # print(self._keylen_cumulative)
        print(f"Total entries across all LMDB files jacs: {total_entries}")

    def __len__(self):
        return self._keylen_cumulative[-1] if self._keylen_cumulative else 0

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()  # Convert tensor to integer

        # Find which database to access
        db_idx = bisect.bisect_right(self._keylen_cumulative, index)
        with self.envs[db_idx].begin() as txn:
            # print("txn",txn.stat()['entries'])
            # print(f"Accessing LMDB file {self.db_paths[db_idx]} for index {index}")
            # index = 10
            byte_data = txn.get(str(index).encode())
            if byte_data:
                tensor = torch.from_numpy(np.frombuffer(byte_data, dtype=self.dtype))
                # print(byte_data,tensor)
                return tensor
            else:
                raise Exception(f"Data not found for index {index} in LMDB file.")
            