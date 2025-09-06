import bisect
from typing import Any, Callable
from fairchem.core.datasets.base_dataset import BaseDataset
from fairchem.core.datasets.ase_datasets import apply_one_tags, AseDBDataset
from torch.utils.data import Dataset
import lmdb
import torch
import numpy as np
import os
import ase
from fairchem.core.common.registry import registry


class CombinedDataset(AseDBDataset):
    def __init__(
        self,
        config: dict,
        dataset_type: str,
        atoms_transform: Callable[[ase.Atoms, Any, ...], ase.Atoms] = apply_one_tags,
    ):
        super().__init__(config, atoms_transform)
        self.labels_folder = config['teacher_labels_folder']
        self.num_hessian_samples = int(config['num_hessian_samples'])
        # self.teacher_force_dataset = LmdbDataset(
        #     os.path.join(config['teacher_labels_folder'], f'{dataset_type}_forces')
        # )
        self.num_lmdb_rows = int(config['num_lmdb_rows'])
        if dataset_type == 'train':
            self.hessian_dataset = LmdbDataset(
                os.path.join(config['teacher_labels_folder'], 'force_jacobians'), div_2=True
            )
            self.grad_outputs_dataset = LmdbGradOutputsDataset(
                os.path.join(config['teacher_labels_folder'], 'force_jacobians')
            )
        else:
            self.hessian_dataset = None

    def __getitem__(self, idx):
        # pid = os.getpid()
        # 1) Get the main AtomicData (always on CPU at this point)
        main_batch = super().__getitem__(idx)
        num_atoms = main_batch.natoms
        num_samples = self.num_hessian_samples
        # 3) Load teacher_forces (CPU)
        # teacher_forces = self.teacher_force_dataset[idx].reshape(num_atoms, 3)
        
        if self.hessian_dataset is not None:
            sampled_rows = torch.randperm(self.num_lmdb_rows)[:num_samples]
            
            # 4) Load the raw force_jacobian vector (CPU)
            grad_outputs = self.grad_outputs_dataset[idx].reshape(self.num_lmdb_rows, num_atoms, 3)
            force_jacs = self.hessian_dataset[idx].reshape(self.num_lmdb_rows, num_atoms, 3)
            
            force_jacs =  force_jacs[sampled_rows].permute(1, 0, 2).reshape(num_atoms, -1) # (n_samples, natoms, 3) ->(natoms, nsamples, 3) -> (natoms, nsamples*3)
            grad_outputs = grad_outputs[sampled_rows].permute(1, 0, 2).reshape(num_atoms, -1) # (n_samples, natoms, 3) ->(natoms, nsamples, 3) -> (natoms, nsamples*3)
            
            main_batch.forces_jac = force_jacs
            main_batch.grad_outputs = grad_outputs
            main_batch.num_samples = torch.tensor(num_samples)
        else:
            # 6) If no Hessian, just fill zeros on CPU
            main_batch.forces_jac = torch.zeros((num_atoms, num_samples * 3))
            main_batch.grad_outputs = torch.tensor((num_atoms, num_samples * 3))
            main_batch.num_samples = torch.tensor(num_samples)
        # main_batch.teacher_forces = teacher_forces
        return main_batch

class CombinedDatasetTrain(CombinedDataset):
    def __init__(self, config, atoms_transform=apply_one_tags):
        super().__init__(config, 'train', atoms_transform)

class CombinedDatasetVal(CombinedDataset):
    def __init__(self, config, atoms_transform=apply_one_tags):
        super().__init__(config, 'val', atoms_transform)
    
    
class LmdbDataset(Dataset):
    def __init__(self, folder_path, dtype=np.float32, div_2 = False):
        self.folder_path = folder_path
        self.dtype = dtype
        
        # List all LMDB files in the folder
        self.db_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.lmdb')])
        assert len(self.db_paths) > 0, f"No LMDB files found in the specified folder: {folder_path}."

        self.envs = []
        self._keys = []
        self._keylen_cumulative = []
        self.div_2 = div_2

        total_entries = 0
        for db_path in self.db_paths:
            env = lmdb.open(db_path, readonly=True, lock=False)
            self.envs.append(env)
            with env.begin() as txn:
                num_entries = txn.stat()['entries'] if not self.div_2 else txn.stat()['entries'] // 2
                total_entries += num_entries
                self._keylen_cumulative.append(total_entries)
        print(f"Total entries across all LMDB files jacs: {total_entries}")

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
                # tensor = torch.from_numpy(np.frombuffer(byte_data, dtype=self.dtype))
                arr = np.frombuffer(byte_data, dtype=self.dtype).copy()   # now writable
                tensor = torch.from_numpy(arr)
                return tensor
            else:
                raise Exception(f"Data not found for index {index} in LMDB file.")



class LmdbGradOutputsDataset(LmdbDataset):
    def __init__(self, folder_path, dtype=np.float32, div_2=True):
        super().__init__(folder_path, dtype=dtype, div_2=div_2)
        
    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()  # Convert tensor to integer

        # Find which database to access
        db_idx = bisect.bisect_right(self._keylen_cumulative, index)
        with self.envs[db_idx].begin() as txn:
            byte_data = txn.get((str(index) + "_grad_outputs").encode())
            if byte_data:
                arr = np.frombuffer(byte_data, dtype=self.dtype).copy()   # now writable
                tensor = torch.from_numpy(arr)
                return tensor
            else:
                raise Exception(f"Data not found for index {index} in LMDB file.")