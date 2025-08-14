from fairchem.core.units.mlip_unit import load_predict_unit
from src_v2.distill_datasets import CombinedDataset
import torch
from src_v2.distill_utils import get_jacobian, get_jacobian_finite_difference
from fairchem.core.datasets.atomic_data import AtomicData
from functools import partial
from fairchem.core.datasets import data_list_collater
from APPLICATIONS.electrolytes.get_calc import get_uma_calc
import numpy as np

# get datasets
config = {
    "src": "/u/czhang31/data/all_NAPF6/train",
    "teacher_labels_folder": "/u/czhang31/data/all_NAPF6/labels",
    "num_hessian_samples": 3
}
uma_path = '/u/czhang31/data/uma-s-1p1.pt'

print("Loading dataset...")
train_dataset = CombinedDataset(config, dataset_type="train")
idx = 11
datapoint = train_dataset[idx]
data_atoms = train_dataset.get_atoms(idx) 
print("Loading calculator...")
calc = get_uma_calc(uma_path)
print("Calculating forces...")
calc.calculate(data_atoms)
output_forces = calc.results['forces'].copy()
print("Calculating sampled_hessian...")
atoms_perturbed = data_atoms.copy()
atoms_perturbed.positions[datapoint.samples[0, 0], datapoint.samples[0, 1]] += 0.001
calc.calculate(atoms_perturbed)
perturbed_forces = calc.results['forces'].copy()
hessian = (perturbed_forces - output_forces) / 0.001
# print(hessian)
print("Comparing hessian with datapoint.forces_jac")
compare = datapoint.forces_jac.reshape(-1, 3, 3)[:, 0, :]
print(np.abs(hessian - compare.numpy()).max())
print(np.abs(hessian - compare.numpy()).mean() / np.abs(hessian).mean())

# true_hessian = get_sampled_hessian(datapoint, output)
# breakpoint()
# print(true_hessian)

# true_hessian = get_sampled_hessian(datapoint, predicted_forces)
# print((true_hessian - datapoint.force_jacs).abs().max())