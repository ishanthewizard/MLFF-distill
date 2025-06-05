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
        force_jacs = force_jacs[samples].flatten() # num_samples, batch.natoms, 3 
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
        if self.hessian_dataset:
            # force_jacs  = self.force_jac_dataset[idx].reshape(num_free_atoms, 3, num_atoms, 3) 
            force_jacs = self.hessian_dataset[idx] # DON'T RESHAPE! We'll do it later, easier for atoms of different lengths
            num_samples = 4
            samples = self.hessian_sampler.sample_with_mask(num_samples, torch.ones(num_atoms))
            force_jacs = self.hessian_sampler.sample_hessian(samples, num_atoms, force_jacs)
            main_batch.force_jacs = force_jacs
            main_batch.samples = samples
            main_batch.num_samples = torch.tensor(samples.shape[0])
        else: 
            force_jacs = None
            
        main_batch.teacher_forces = teacher_forces
        main_batch.force_jacs = force_jacs
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
        
    
    
# class CombinedDataset(Dataset):
#     def __init__(self, main_dataset, teach_force_dataset, force_jac_dataset=None, final_node_feature_dataset=None):
#         if  len(main_dataset) != len(teach_force_dataset):
#             logging.info("WARNING: TEACH FORCE DATASET DIFFERENT SIZE")
#             raise Exception("DIFF SIZE!!")
#         if force_jac_dataset and len(main_dataset) != len(force_jac_dataset):
#             logging.info("WARNING: FORCE JACOBIAN DIFFERENT SIZE")
#             raise Exception("DIFF SIZE!!")
#         if final_node_feature_dataset and len(main_dataset) != len(final_node_feature_dataset):
#             logging.info("WARNING: FINAL NODE FEATURE DATASET DIFFERENT SIZE")
#             raise Exception("DIFF SIZE!!")
#         self.main_dataset = main_dataset
#         self.teach_force_dataset = teach_force_dataset
#         self.force_jac_dataset = force_jac_dataset 
#         # breakpoint()
#         self._metadata = self.main_dataset._metadata
#         self.metadata_hasattr = self.main_dataset.metadata_hasattr
        
#         self.final_node_feature_dataset = final_node_feature_dataset
#         self.get_metadata = self.main_dataset.get_metadata

#     def __len__(self):
#         return len(self.main_dataset)  # Assuming both datasets are the same size

#     def __getitem__(self, idx):
#         main_batch = self.main_dataset[idx]
#         # print("main_batch",main_batch)
#         num_atoms = main_batch.natoms
#         teacher_forces = self.teach_force_dataset[idx].reshape(num_atoms, 3)
#         # print("where is my force jacobian")
#         # print(self.force_jac_dataset)
#         # import sys
#         # sys.exit()
#         if self.force_jac_dataset:
#             # force_jacs  = self.force_jac_dataset[idx].reshape(num_free_atoms, 3, num_atoms, 3) 
#             force_jacs = self.force_jac_dataset[idx] # DON'T RESHAPE! We'll do it later, easier for atoms of different lengths
#         else: 
#             force_jacs = None
            
#         if self.final_node_feature_dataset:
#             final_node_features = self.final_node_feature_dataset[idx]
#         else:
#             final_node_features = None
        
#         if not hasattr(main_batch, 'cell'):
#             main_batch.cell =  torch.eye(3, device=main_batch.pos.device).unsqueeze(0)
#         if not hasattr(main_batch, 'fixed'):
#             main_batch.fixed = torch.zeros(main_batch.pos.shape[0], device=main_batch.pos.device) 
#         main_batch.teacher_forces = teacher_forces
#         main_batch.force_jacs = force_jacs
#         if final_node_features is not None:
#             main_batch.final_node_features = final_node_features.reshape(teacher_forces.shape[0], -1)
#         return main_batch

#     def close_db(self):
#         self.main_dataset.close_db()
#         self.teach_force_dataset.close_db()
#         if self.force_jac_dataset:
#             self.force_jac_dataset.close_db() 


# class Dataset_with_Hessian_Masking(Dataset):
#     def __init__(self, dataset, hessian_mask=None):
#         self.hessian_mask = hessian_mask
#         self.dataset = dataset
#         if len(dataset) != len(hessian_mask):
#             logging.info("WARNING: HESSIAN MASK DATASET DIFFERENT SIZE")
#             raise Exception("DIFF SIZE!!")
        
#         self._metadata = self.dataset._metadata
#         self.metadata_hasattr = self.dataset.metadata_hasattr
#         self.get_metadata = self.dataset.get_metadata
        
#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, idx):
#         data = self.dataset[idx]
#         data.hessian_mask = self.hessian_mask[idx]
#         return data
    


# class SimpleDataset(Dataset):
#     def __init__(self, folder_path, dtype=np.float32):
#         self.folder_path = folder_path
#         self.dtype = dtype
        
#         # List all LMDB files in the folder
#         self.db_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.lmdb')])
#         assert len(self.db_paths) > 0, "No LMDB files found in the specified folder."

#         self.envs = []
#         self._keys = []
#         self._keylen_cumulative = []

#         total_entries = 0
#         for db_path in self.db_paths:
#             env = lmdb.open(db_path, readonly=True, lock=False)
#             self.envs.append(env)
#             with env.begin() as txn:
#                 num_entries = txn.stat()['entries']
#                 total_entries += num_entries
#                 self._keylen_cumulative.append(total_entries)
#         # print(self._keylen_cumulative)
#         print(f"Total entries across all LMDB files jacs: {total_entries}")

#     def __len__(self):
#         return self._keylen_cumulative[-1] if self._keylen_cumulative else 0

#     def __getitem__(self, index):
#         if isinstance(index, torch.Tensor):
#             index = index.item()  # Convert tensor to integer

#         # Find which database to access
#         db_idx = bisect.bisect_right(self._keylen_cumulative, index)
#         with self.envs[db_idx].begin() as txn:
#             # print("txn",txn.stat()['entries'])
#             # print(f"Accessing LMDB file {self.db_paths[db_idx]} for index {index}")
#             # index = 10
#             byte_data = txn.get(str(index).encode())
#             if byte_data:
#                 tensor = torch.from_numpy(np.frombuffer(byte_data, dtype=self.dtype))
#                 # print(byte_data,tensor)
#                 return tensor
#             else:
#                 raise Exception(f"Data not found for index {index} in LMDB file.")

#     def merge_lmdb_files(self, output_file, map_size=1099511627776 * 2):
#         # Initialize new LMDB environment for output
#         input_folder = self.folder_path
#         os.makedirs(os.path.join(output_file),exist_ok=True)
#         env = lmdb.open(os.path.join(output_file, 'data.lmdb'), map_size=int(map_size))  # Adjust map_size as needed

#         # Iterate through each LMDB file in the folder
#         for db_path in tqdm(sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.lmdb')])):
#             print(f"Processing {db_path}")
#             with lmdb.open(db_path, readonly=True, lock=False) as env_in:
#                 with env_in.begin() as txn_in, env.begin(write=True) as txn_out:
#                     cursor = txn_in.cursor()
#                     for key, value in cursor:
#                         # Use the original key from the source LMDB file
#                         txn_out.put(key, value)

#         # Close the new LMDB environment
#         env.close()

#     def close_db(self):
#         for env in self.envs:
#             env.close()



# class SubsetDistributedSampler(Sampler):
#     def __init__(self, dataset, indices):
#         self.dataset = dataset
#         self.indices = indices

#     def __iter__(self):
#         return iter(self.indices)

#     def __len__(self):
#         return len(self.indices)

# def dump_dataset_to_lmdb(dataset, lmdb_path, map_size=1099511627776 * 2):
#     """
#     Dump a PyTorch Dataset into a single LMDB file.

#     Args:
#         dataset (torch.utils.data.Dataset): The dataset to dump.
#         lmdb_path (str): Target path for dumping LMDB database.
#         map_size (float): Max size of the LMDB database in bytes (default: 2 TB).
#     """
#     if os.path.exists(lmdb_path):
#         print(f"Removing existing LMDB at: {lmdb_path}")
#         import shutil
#         shutil.rmtree(lmdb_path)

#     env = lmdb.open(lmdb_path, map_size=int(map_size))
#     print("len dataset", len(dataset))
#     # breakpoint()
#     with env.begin(write=True) as txn:
#         for idx, data in enumerate(dataset):
#             if idx == len(dataset):
#                 breakpoint()
#             # print(data)
#             # sys.exit()
#             if isinstance(data, torch.Tensor):
#                 arr = data.numpy()
#             elif isinstance(data, np.ndarray):
#                 arr = data
#             else:
#                 raise TypeError(f"Unsupported data type at index {idx}: {type(data)}")

#             # Convert array to bytes
#             byte_data = arr.astype(np.float32).tobytes()

#             # Use the index as the key
#             txn.put(str(idx).encode(), byte_data)

#             if (idx + 1) % 1000 == 0:
#                 print(f"Written {idx + 1} entries...")

#     env.close()
#     print(f"LMDB written to: {lmdb_path}")


# def merge_lmdb_shards(shard_dirs, output_path, map_size=1099511627776 * 2):
#     """
#     Merge multiple LMDB folders (shards) into a single LMDB database.

#     Args:
#         input_dir (str): Directory containing shard folders (e.g., data.0000.lmdb, data.0001.lmdb, ...).
#         output_path (str): Path to the output LMDB folder (e.g., 'merged_data.lmdb').
#         map_size (int): Max size of the LMDB (default 2 TB).
#     """
#     if os.path.exists(output_path):
#         print(f"Removing existing LMDB at {output_path}")
#         shutil.rmtree(output_path)

#     os.makedirs(output_path, exist_ok=True)
#     env_out = lmdb.open(output_path, map_size=map_size)

#     idx = 0
#     for shard in tqdm(shard_dirs, desc="Merging shards"):
#         print(f"Processing {shard}")
#         env_in = lmdb.open(shard, readonly=True, lock=False)
#         with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
#             cursor = txn_in.cursor()
#             for key, value in cursor:
#                 new_key = str(idx).encode()  # Reindexing
#                 txn_out.put(new_key, value)
#                 idx += 1
#         env_in.close()

#     env_out.close()
#     print(f" Done. Total entries merged: {idx}")
#     print(f" Output LMDB saved at: {output_path}")


# def merge_lmdb_shards(shard_dirs, output_path, map_size=1099511627776 * 2):
#     """
#     Merge multiple LMDB folders (shards) into a single LMDB database.

#     Args:
#         input_dir (str): Directory containing shard folders (e.g., data.0000.lmdb, data.0001.lmdb, ...).
#         output_path (str): Path to the output LMDB folder (e.g., 'merged_data.lmdb').
#         map_size (int): Max size of the LMDB (default 2 TB).
#     """
#     if os.path.exists(output_path):
#         print(f"Removing existing LMDB at {output_path}")
#         shutil.rmtree(output_path)

#     os.makedirs(output_path, exist_ok=True)
#     env_out = lmdb.open(output_path, map_size=map_size)

#     base_idx = 0
#     for shard in tqdm(shard_dirs, desc="Merging shards"):
#         print(f"Processing {shard}")
#         counter = 0
#         env_in = lmdb.open(shard, readonly=True, lock=False)
#         with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
#             cursor = txn_in.cursor()
#             for key, value in cursor:
#                 # print("original key:", key.decode())
#                 # print("new key:", idx)
#                 new_key = str(int(key.decode())+base_idx).encode()  # Reindexing
#                 txn_out.put(new_key, value)
#                 counter += 1
#         env_in.close()
#         base_idx += counter

#     env_out.close()
#     print(f" Done. Total entries merged: {base_idx}")
#     print(f" Output LMDB saved at: {output_path}")