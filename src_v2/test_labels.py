
from fairchem.core.units.mlip_unit import load_predict_unit
from src_v2.distill_datasets import CombinedDataset, LmdbDataset
import torch
from fairchem.core.datasets.atomic_data import AtomicData
from functools import partial
from fairchem.core.datasets import data_list_collater
from APPLICATIONS.electrolytes.get_calc import get_uma_calc
import numpy as np

def test_hessian_generated(dataset, data_idx, grad_outputs, hessian_labels, total_num_samples, eps=1e-3):
    sel_idx = 2
    datapoint = dataset[data_idx]
    data_atoms = dataset.datasets[0].dataset.get_atoms(data_idx)

    calc = get_uma_calc("/data/ishan-amin/OMOL/ESEN_OMol_ckpts/uma-s-1p1.pt")

    # Base forces
    calc.calculate(data_atoms)
    F0 = calc.results['forces'].copy()

    # Perturb along the selected probe
    v = grad_outputs[sel_idx].cpu().numpy()  # (N,3)
    atoms_p = data_atoms.copy()
    atoms_p.positions += eps * v

    calc.calculate(atoms_p)
    Fp = calc.results['forces'].copy()

    true_hessian = (Fp - F0) / eps

    # Compare to stored label
    compare = hessian_labels.reshape(total_num_samples, datapoint.natoms, 3)[sel_idx]
    diff = np.abs(true_hessian - compare.numpy())
    print("scale ratio ~", np.abs(compare).mean() / (np.abs(true_hessian).mean() + 1e-12))
    print("Comparison vs stored label:")
    print("  max abs diff:", diff.max())
    print("  mean rel diff:", diff.mean() / (np.abs(true_hessian).mean() + 1e-12))

    # ---------------------------------------------------------
    # Repeatability check: rerun calculator on same points
    # ---------------------------------------------------------
    calc.calculate(data_atoms)
    F0b = calc.results['forces'].copy()

    atoms_p2 = data_atoms.copy()
    atoms_p2.positions += eps * v
    calc.calculate(atoms_p2)
    Fp2 = calc.results['forces'].copy()

    true_hessian2 = (Fp2 - F0b) / eps

    repeat_diff = np.abs(true_hessian2 - true_hessian)
    print("Repeatability check (same perturbation twice):")
    print("  max abs diff:", repeat_diff.max())
    print("  mean rel diff:", repeat_diff.mean() / (np.abs(true_hessian).mean() + 1e-12))


def test_forces_generated(dataset, idx, out):
    datapoint = dataset[idx]
    print("ENTERING CALCULATURE")
    data_atoms = dataset.datasets[0].dataset.get_atoms(idx)
    calc = get_uma_calc("/data/ishan-amin/OMOL/ESEN_OMol_ckpts/uma-s-1p1.pt")
    calc.calculate(data_atoms)
    calc_forces = calc.results['forces'].copy()
    
    true_forces = datapoint.forces
    
    label_vs_calc_err = np.abs((calc_forces - out.cpu().numpy())).mean()
    true_vs_calc_err = np.abs((calc_forces - true_forces.cpu().numpy())).mean()
    true_vs_label_err = (out.cpu() - true_forces.cpu()).abs().mean().item()  # convert tensor to float

    print(f"label_vs_calc_err: {label_vs_calc_err:.8f}")
    print(f"true_vs_calc_err: {true_vs_calc_err:.8f}")
    print(f"true_vs_label_err: {true_vs_label_err:.8f}")

    
    
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
    # train_dataset = CombinedDataset(config, dataset_type="train")
    train_dataset = AseDBDataset(config)
    train_forces_dataset = LmdbDataset(config['teacher_labels_folder'], dtype=np.float32, div_2=False)
    idx = 1
    samp_idx = 0
    for i in range(10):
        idx = i
        datapoint = train_dataset[idx]
        data_atoms = train_dataset.get_atoms(idx) 
        print("Loading calculator...")
        calc = get_uma_calc("/data/ishan-amin/OMOL/ESEN_OMol_ckpts/uma-s-1p1.pt")
    
    compare_hessian_twice(datapoint, data_atoms, calc, samp_idx)
    
    # true_hessian = get_sampled_hessian(datapoint, output)
    # breakpoint()
    # print(true_hessian)

# true_hessian = get_sampled_hessian(datapoint, predicted_forces)
# print((true_hessian - datapoint.force_jacs).abs().max())