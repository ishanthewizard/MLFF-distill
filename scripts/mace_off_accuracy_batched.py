import os
from fairchem.core.common.data_parallel import BalancedBatchSampler
from fairchem.core.common.data_parallel import OCPCollater
import numpy as np
from ase import Atoms
from mace.calculators import mace_off, mace_mp
from tqdm import tqdm
from torch.utils.data import Subset
import torch
from fairchem.core.common.registry import registry
import lmdb 
from torch.utils.data import DataLoader

def calculate_l2mae(true, predicted):
    return np.mean(np.abs(true - predicted))

def calculate_l2mae_forces(true_forces, predicted_forces):
    l2mae_forces = []
    for true, pred in zip(true_forces, predicted_forces):
        l2mae_forces.append(np.mean(np.abs(true - pred)))
    return np.mean(l2mae_forces)

def get_accuracy(dataset_path, batch_size, model='large'):
    # Load model
    model = mace_off(model=model, dispersion=False,  default_dtype="float32", device="cuda", return_raw_model=True).float()
    # Counting parameters
    # total_params = sum(p.numel() for p in calc.parameters())
    # print(f"Total trainable parameters: {total_params}")
    # print("TOTAL_PARAMS:", calc.parameters)
    
    # Load the dataset
    index = 0
    dataset = registry.get_dataset_class("lmdb")({"src": dataset_path})
    print(len(dataset))
    # indx = np.random.default_rng(seed=0).choice(
    #     len(dataset), 
    #     500, 
    #     replace=False
    # )
    # dataset = Subset(dataset, torch.tensor(indx))
    print(len(dataset))
    sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=1,
            rank=0,
            device="cuda",
            mode=False,
            shuffle=True,
            force_balancing=False,
            seed=0,
        )
    dataloader = DataLoader(
            dataset,
            collate_fn=OCPCollater(True),
            num_workers=4,
            pin_memory=True,
            batch_sampler=sampler,
        )

    true_energies = []
    true_forces = []
    predicted_energies = []
    predicted_forces = []
    num_atoms = []
    for batch in tqdm(dataloader):
        batch.to('cuda')
        batch.node_attrs = torch.zeros(len(batch.atomic_numbers), 10, dtype=torch.float32).to('cuda')
        batch.positions = batch.pos
        out = model(batch)
        true_energies.append(sample.y.item())
        true_forces.append(sample.force.numpy())

        atomic_numbers = sample.atomic_numbers.numpy()
        num_atoms.append(len(atomic_numbers))
        atoms = Atoms(numbers=atomic_numbers, positions=sample.pos.numpy())
        atoms.calc = calc

        predicted_energies.append(atoms.get_potential_energy())
        predicted_forces.append(atoms.get_forces())
    
    l2mae_energy = calculate_l2mae(np.array(true_energies), np.array(predicted_energies))
    l2mae_forces = calculate_l2mae_forces(true_forces, predicted_forces)
    num_atoms = np.array(num_atoms)
    print("MEAN:", np.mean(num_atoms), "MIN:", min(num_atoms), "MAX:", max(num_atoms))

    print(f"L2MAE Energy: {l2mae_energy * 1000:.2f}")
    print(f"L2MAE Forces: {l2mae_forces * 1000:.2f} meV")
if __name__ == "__main__":
    # dataset_path = '/data/ishan-amin/spice_separated/DES370K_Monomers/test'
    dataset_path = '/data/ishan-amin/maceoff_split/test'
    # dataset_path = '/data/ishan-amin/dipeptides/all/test'
    batch_size = 3
    get_accuracy(dataset_path, batch_size)
