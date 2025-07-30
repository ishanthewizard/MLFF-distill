
from fairchem.core.units.mlip_unit import load_predict_unit
from src_v2.distill_datasets import CombinedDataset, LmdbDataset
import torch
from src_v2.distill_utils import get_jacobian, get_jacobian_finite_difference
from fairchem.core.datasets.atomic_data import AtomicData
from functools import partial
from fairchem.core.datasets import data_list_collater
from APPLICATIONS.electrolytes.get_calc import get_uma_calc
import numpy as np

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
    
    
    
    
    
if __name__ == "__main__":
    # get datasets
    config = {
        "src": "/data/ishan-amin/OMOL/electrolytes_application/aselmdb_data/NAPF6/s1p1/train",
        "teacher_labels_folder": "/data/ishan-amin/OMOL/electrolytes_application/labels/NAPF6/s1p1",
        "num_hessian_samples": 3
    }
    
    print("Loading dataset...")
    train_dataset = CombinedDataset(config, dataset_type="train")
    train_forces_dataset = LmdbDataset(config['teacher_labels_folder'], dtype=np.float32, div_2=False)
    idx = 101
    datapoint = train_dataset[idx]
    data_atoms = train_dataset.get_atoms(idx) 
    print("Loading calculator...")
    calc = get_uma_calc()
    print("Calculating forces...")
    calc.calculate(data_atoms)
    output_forces = calc.results['forces'].copy()
    print("Calculating sampled_hessian...")
    atoms_perturbed = data_atoms.copy()
    print(atoms_perturbed.positions[datapoint.samples[0, 0], datapoint.samples[0, 1]])
    print(datapoint.pos[datapoint.samples[0, 0], datapoint.samples[0, 1]])
    breakpoint()
    atoms_perturbed.positions[datapoint.samples[0, 0], datapoint.samples[0, 1]] += 0.001
    calc.calculate(atoms_perturbed)
    perturbed_forces = calc.results['forces'].copy()
    hessian = (perturbed_forces - output_forces) / 0.001
    # print(hessian)
    print("Comparing hessian with datapoint.forces_jac")
    compare = datapoint.forces_jac.reshape(-1, config['num_hessian_samples'], 3)[:, 0, :]
    # Find indices where abs(compare) > 0.01

    print(np.abs(hessian - compare.numpy()).max())
    print(np.abs(hessian - compare.numpy()).mean() / np.abs(hessian).mean())
    
    mask = np.abs(compare.numpy()) > 0.01
    idxs = np.argwhere(mask)
    print("Indices and values of compare where abs(compare) > 0.01:")
    for idx in idxs:
        print(f"Index: {tuple(idx)}, Value: {compare.numpy()[tuple(idx)]}")
        print(f"                   Hessian: {hessian[tuple(idx)]}")
        
    breakpoint()
    
    # true_hessian = get_sampled_hessian(datapoint, output)
    # breakpoint()
    # print(true_hessian)

# true_hessian = get_sampled_hessian(datapoint, predicted_forces)
# print((true_hessian - datapoint.force_jacs).abs().max())