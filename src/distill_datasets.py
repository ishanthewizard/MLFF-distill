from torch.utils.data import Dataset, DataLoader
import lmdb
import torch
import numpy as np

class CombinedDataset(Dataset):
    def __init__(self, main_dataset, teach_force_dataset, force_jac_dataset=None):
        assert len(main_dataset) == len(teach_force_dataset), "Datasets must be of the same size"
        if force_jac_dataset:
            assert len(main_dataset) == len(force_jac_dataset), "Datasets must be of the same size"
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
            force_jacs  = self.force_jac_dataset[idx].reshape(num_free_atoms, 3, num_atoms, 3) 
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
    def __init__(self, file_path):
        self.file_path = file_path
        self.env = lmdb.open(file_path, readonly=True, lock=False)  # Optimize LMDB by disabling lock for readonly
        with self.env.begin() as txn:
            stat = txn.stat()  # Get statistics of the database
            self.length  =  stat['entries']  # Number of entries in the databas

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin() as txn:
            byte_data = txn.get(str(index).encode())
            if byte_data:
                tensor = torch.from_numpy(np.frombuffer(byte_data, dtype=np.float32))
                return tensor
            else:
                raise Exception('blah blah blah')

    def close_db(self):
        self.env.close()