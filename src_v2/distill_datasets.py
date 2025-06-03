import bisect
import os,sys
from torch.utils.data import Dataset, DataLoader
import lmdb
import torch
import numpy as np
import logging 
from torch.utils.data import Sampler
from tqdm import tqdm
import shutil


class TeachingDataset(Dataset):
    def __init__(self, main_dataset, teach_force_dataset = None, force_jac_dataset=None, final_node_feature_dataset=None):
        if  teach_force_dataset and len(main_dataset) != len(teach_force_dataset):
            logging.info("WARNING: TEACH FORCE DATASET DIFFERENT SIZE")
            raise Exception("DIFF SIZE!!")
        # breakpoint()
        # print("check dataset length",len(main_dataset),len(force_jac_dataset))
        if force_jac_dataset and len(main_dataset) != len(force_jac_dataset):
            logging.info("WARNING: FORCE JACOBIAN DIFFERENT SIZE")
            raise Exception("DIFF SIZE!!")
        if final_node_feature_dataset and len(main_dataset) != len(final_node_feature_dataset):
            logging.info("WARNING: FINAL NODE FEATURE DATASET DIFFERENT SIZE")
            raise Exception("DIFF SIZE!!")
        self.main_dataset = main_dataset
        self.teach_force_dataset = teach_force_dataset
        self.force_jac_dataset = force_jac_dataset 

        self._metadata = self.main_dataset._metadata
        self.metadata_hasattr = self.main_dataset.metadata_hasattr
        
        self.final_node_feature_dataset = final_node_feature_dataset
        self.get_metadata = self.main_dataset.get_metadata

    def __len__(self):
        return len(self.main_dataset)  # Assuming both datasets are the same size

    def __getitem__(self, idx):
        main_batch = self.main_dataset[idx]
        num_atoms = main_batch.natoms
        # teacher_forces = self.teach_force_dataset[idx].reshape(num_atoms, 3)

        if self.force_jac_dataset:
            force_jacs = self.force_jac_dataset[idx] # DON'T RESHAPE! We'll do it later, easier for atoms of different lengths
        else: 
            force_jacs = None
            
        if self.final_node_feature_dataset:
            final_node_features = self.final_node_feature_dataset[idx]
        else:
            final_node_features = None
        
        if not hasattr(main_batch, 'cell'):
            main_batch.cell =  torch.eye(3, device=main_batch.pos.device).unsqueeze(0)
        if not hasattr(main_batch, 'fixed'):
            main_batch.fixed = torch.zeros(main_batch.pos.shape[0], device=main_batch.pos.device) 
        # main_batch.teacher_forces = teacher_forces
        main_batch.force_jacs = force_jacs
        # if final_node_features is not None:
        #     main_batch.final_node_features = final_node_features.reshape(teacher_forces.shape[0], -1)
        return main_batch


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


def merge_lmdb_shards(input_dir, output_path, map_size=1099511627776 * 2):
    """
    Merge multiple LMDB folders (shards) into a single LMDB database.

    Args:
        input_dir (str): Directory containing shard folders (e.g., "/path_to", containing, data.0000.lmdb, data.0001.lmdb, ...).
        output_path (str): Path to the output LMDB folder (e.g., '/path_to/merged_data.lmdb').
        map_size (int): Max size of the LMDB (default 2 TB).
    """
    shard_dirs = sorted([os.path.join(input_dir, d) for d in os.listdir(input_dir)
                        if d.endswith('.lmdb') and os.path.isdir(os.path.join(input_dir, d))])
    if os.path.exists(output_path):
        print(f"Removing existing LMDB at {output_path}")
        shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)
    env_out = lmdb.open(output_path, map_size=map_size)

    idx = 0
    for shard in tqdm(shard_dirs, desc="Merging shards"):
        init_idx = idx
        print(f"Processing {shard},starting index: {init_idx}")
        env_in = lmdb.open(shard, readonly=True, lock=False)
        with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
            cursor = txn_in.cursor()
            for key, value in cursor:

                # BUG: !!! this line could be extremely buggy when switch to new dataset, or when changing
                # the index encode in label generation
                new_key = str(int(key.decode())+init_idx).encode()  # Reindexing, keep the original key index order
                txn_out.put(new_key, value)
                idx += 1
        env_in.close()

    env_out.close()
    print(f" Done. Total entries merged: {idx}")
    print(f" Output LMDB saved at: {output_path}")



if __name__ == "__main__":
    pass