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


class HessianSampler:
    def sample_with_mask(self, num_samples, mask):
        assert num_samples <= len(mask), "System too small for the number of samples."
        valid_rows = torch.where(mask)[0]
        if valid_rows.numel() == 0:
            raise ValueError("No valid rows available according to the mask")

        # Each valid row contributes 3 indices
        valid_indices = valid_rows.repeat_interleave(3) * 3 + torch.tensor([0, 1, 2]).repeat(valid_rows.size(0)).to(mask.device)
        chosen_indices = valid_indices[torch.randperm(valid_indices.size(0))[:num_samples]]
        row_indices = chosen_indices // 3
        col_indices = chosen_indices % 3
        return torch.stack((row_indices, col_indices), dim=1)  # (num_samples, 2)

    def sample_hessian(self, samples, num_atoms, force_jacs):
        # force_jacs: flat 1D tensor of length num_atoms*3*num_atoms*3
        force_jacs = force_jacs.reshape(num_atoms, 3, num_atoms, 3)
        force_jacs = force_jacs[samples[:, 0], samples[:, 1], :, :]  # (num_samples, num_atoms, 3)
        force_jacs = force_jacs.permute(1, 0, 2).reshape(num_atoms, -1)  # (num_atoms, num_samples*3)
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
        self.teacher_force_dataset = LmdbDataset(
            os.path.join(config['teacher_labels_folder'], f'{dataset_type}_forces')
        )

        if dataset_type == 'train':
            self.hessian_dataset = LmdbDataset(
                os.path.join(config['teacher_labels_folder'], 'force_jacobians')
            )
            self.hessian_sampler = HessianSampler()
        else:
            self.hessian_dataset = None

    def __getitem__(self, idx):
        pid = os.getpid()
        # 1) Get the main AtomicData (always on CPU at this point)
        main_batch = super().__getitem__(idx)

        # 2) Pull out natoms from the CPU‐side data
        num_atoms = main_batch.natoms

        # 3) Load teacher_forces (CPU)
        teacher_forces = self.teacher_force_dataset[idx].reshape(num_atoms, 3)
        # print(f"[CombinedDataset __getitem__] PID={pid}, teacher_forces.device = {teacher_forces.device}", flush=True)

        num_samples = 1
        if self.hessian_dataset is not None:
            # 4) Load the raw force_jacobian vector (CPU)
            raw_jac = self.hessian_dataset[idx]
            # print(f"[CombinedDataset __getitem__] PID={pid}, raw_jac.device = {raw_jac.device}", flush=True)

            # 5) Sample one Hessian entry per atom (still on CPU)
            samples = self.hessian_sampler.sample_with_mask(num_samples, torch.ones(num_atoms))
            force_jacs = self.hessian_sampler.sample_hessian(samples, num_atoms, raw_jac)
            # print(f"[CombinedDataset __getitem__] PID={pid}, sampled force_jacs.device = {force_jacs.device}", flush=True)

            main_batch.forces_jac = force_jacs
            main_batch.samples = samples
            main_batch.num_samples = torch.tensor(num_samples)
        else:
            # 6) If no Hessian, just fill zeros on CPU
            main_batch.forces_jac = torch.zeros((num_atoms, num_samples * 3))
            main_batch.num_samples = torch.tensor(num_samples)
            print(f"[CombinedDataset __getitem__] PID={pid}, forces_jac.zeros.device = {main_batch.forces_jac.device}", flush=True)

        main_batch.teacher_forces = teacher_forces
        return main_batch

class CombinedDatasetTrain(CombinedDataset):
    def __init__(self, config, atoms_transform=apply_one_tags):
        super().__init__(config, 'train', atoms_transform)

class CombinedDatasetVal(CombinedDataset):
    def __init__(self, config, atoms_transform=apply_one_tags):
        super().__init__(config, 'val', atoms_transform)
    
    
class LmdbDataset(Dataset):
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


