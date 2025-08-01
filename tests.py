
from fairchem.core.units.mlip_unit import load_predict_unit
from src_v2.distill_datasets import CombinedDataset, LmdbDataset
import torch
from fairchem.core.datasets.atomic_data import AtomicData
from functools import partial
from fairchem.core.datasets import data_list_collater
from APPLICATIONS.electrolytes.get_calc import get_uma_calc
import numpy as np
import os

def test_hessian_generated(dataset, data_idx, sample_idxs, hessian_labels):
    datapoint = dataset[data_idx]
    data_atoms = dataset.datasets[0].dataset.get_atoms(data_idx)
    calc = get_uma_calc()
    calc.calculate(data_atoms)
    output_forces = calc.results['forces'].copy()
    atoms_perturbed = data_atoms.copy()
    atoms_perturbed.positions[sample_idxs[10, 0], sample_idxs[10, 1]] += 0.001
    calc.calculate(atoms_perturbed)
    perturbed_forces = calc.results['forces'].copy()
    true_hessian = (perturbed_forces - output_forces) / 0.001
    # print(hessian)
    print("Comparing hessian with datapoint.forces_jac")
    compare = hessian_labels.reshape(60, datapoint.natoms, 3)[10, :, :]
    # Find indices where abs(compare) > 0.01
    print(np.abs(true_hessian - compare.numpy()).max())
    print(np.abs(true_hessian - compare.numpy()).mean() / np.abs(true_hessian).mean())
    breakpoint()
    
    
def compare_hessian_twice(datapoint, data_atoms, calc, samp_idx):
    print("Calculating forces...")
    calc.calculate(data_atoms)
    output_forces = calc.results['forces'].copy()
    print("Calculating sampled_hessian... 1")
    atoms_perturbed = data_atoms.copy()
    atoms_perturbed.positions[datapoint.samples[samp_idx, 0], datapoint.samples[samp_idx, 1]] += 0.001
    calc.calculate(atoms_perturbed)
    perturbed_forces = calc.results['forces'].copy()
    hessian = (perturbed_forces - output_forces) / 0.001
    # print(hessian)
    print("Calculating sampled hessian 2....")
    calc.calculate(data_atoms)
    output_forces2 = calc.results['forces'].copy()
    calc.calculate(atoms_perturbed)
    perturbed_forces2 = calc.results['forces'].copy()
    hessian2 = (perturbed_forces2 - output_forces2) / 0.001
    print(f"Hessian double forward diff: {np.abs(hessian2 - hessian).mean() / np.abs(hessian).mean()}")
    compare = datapoint.forces_jac.reshape(-1, config['num_hessian_samples'], 3)[:, samp_idx, :]
    # Find indices where abs(compare) > 0.01

    # print(np.abs(hessian - compare.numpy()).max())
    print(np.abs(hessian - compare.numpy()).mean() / np.abs(hessian).mean())
    
if __name__ == "__main__":
    # get datasets
    config = {
        "src": "/data/ishan-amin/OMOL/electrolytes_application/aselmdb_data/NAPF6/s1p1/train",
        "teacher_labels_folder": "/data/ishan-amin/OMOL/electrolytes_application/labels/NAPF6/s1p1",
        "num_hessian_samples": 3
    }
    
    print("Loading dataset...")
    train_dataset = CombinedDataset(config, dataset_type="train")
    train_forces_dataset = LmdbDataset(os.path.join(config['teacher_labels_folder'], 'train_forces'), dtype=np.float32, div_2=False)


    print("Loading calculator...")
    calc = get_uma_calc("/data/ishan-amin/OMOL/ESEN_OMol_ckpts/uma-s-1p1.pt")
    for idx in range(10):
        # datapoint = train_dataset[idx + 1]
        data_atoms = train_dataset.get_atoms(idx) 
        calc.calculate(data_atoms)
        output_forces = calc.results['forces'].copy()
        stored_forces  = train_forces_dataset[idx + 1].reshape(-1, 3).numpy()
        breakpoint()
        print("FORCE DIFF:", np.abs((output_forces - stored_forces )).mean())
    breakpoint()
    idx = 1
    samp_idx = 0
    # compare_hessian_twice(datapoint, data_atoms, calc, samp_idx)
    
    # true_hessian = get_sampled_hessian(datapoint, output)
    # breakpoint()
    # print(true_hessian)

# true_hessian = get_sampled_hessian(datapoint, predicted_forces)
# print((true_hessian - datapoint.force_jacs).abs().max())