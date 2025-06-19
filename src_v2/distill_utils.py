import torch 
import numpy as np
import time
import logging
# from fairchem.core.common.data_parallel import  OCPCollater
from fairchem.core.common import distutils
from tqdm import tqdm

def print_cuda_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert bytes to GB
    logging.info(f"CUDA memory allocated: {allocated:.2f} GB")
    logging.info(f"CUDA memory reserved: {reserved:.2f} GB")
    print(f"CUDA memory allocated: {allocated:.2f} GB")
    print((f"CUDA memory reserved: {reserved:.2f} GB"))


def get_teacher_jacobian(batch, vectorize=True, should_mask=True, approximation="disabled", forward=None, collater=None, device="cuda"):
    device = device
    natoms = batch.natoms
    total_num_atoms = sum(batch.natoms)
    cumulative_sums = [0] + torch.cumsum(natoms, 0).tolist()
    mask = batch.fixed == 0
    if not should_mask:
        mask = torch.ones(total_num_atoms, dtype=torch.bool)
    
    mask_per_mol = [mask[cum_sum:cum_sum + nat] for cum_sum, nat in zip(cumulative_sums[:-1], natoms)]
    # print("mask_per_mol", mask_per_mol)
    
    num_free_atoms_per_mol = [sum(sub_mask) for sub_mask in mask_per_mol]
    max_free_atom_per_mol = max(num_free_atoms_per_mol)
    grad_outputs = torch.zeros((max_free_atom_per_mol, 3, total_num_atoms, 3)).to(device)
    for i, free_atoms_in_mol in enumerate(num_free_atoms_per_mol):
        indices = torch.arange(free_atoms_in_mol)
        offset_indices = torch.nonzero(mask_per_mol[i]).flatten() + cumulative_sums[i]
        assert len(offset_indices) == free_atoms_in_mol
        grad_outputs[indices, :, offset_indices, :] = torch.eye(3)[None, :, :].to(device)
    grad_outputs = grad_outputs.reshape(max_free_atom_per_mol * 3, total_num_atoms, 3)
    if approximation == "forward":
        # Forward Difference
        forces = forward(batch)['forces']['forces'].detach()
        # print("device", forces.device, batch.pos.device,grad_outputs.device)
        jac = get_jacobian_finite_difference(forces, batch, grad_outputs, forward=forward, detach=True, collater = collater, looped=(not vectorize))
        jac = jac.reshape(max_free_atom_per_mol, 3, total_num_atoms, 3)
        jacs_per_mol = [jac[:n_fr_at, :,  cum_sum:cum_sum + nat, :] for cum_sum, n_fr_at, nat in zip(cumulative_sums[:-1], num_free_atoms_per_mol, natoms)]
        pass
    elif approximation == "disabled":
        # Vector Jacobian Product
        forces = forward(batch)['forces']['forces']
        jac = get_jacobian(forces, batch.pos, grad_outputs, looped=(not vectorize)) # outputs a max_free_atom_per_mol x 3 x total_num_atoms x 3 matrix.
        jac = jac.reshape(max_free_atom_per_mol, 3, total_num_atoms, 3)
        jacs_per_mol = [jac[:n_fr_at, :,  cum_sum:cum_sum + nat, :].cpu() for cum_sum, n_fr_at, nat in zip(cumulative_sums[:-1], num_free_atoms_per_mol, natoms)]
    # print("JACS PER MOL", [(jacs.reshape(12,12),jacs.shape) for jacs in jacs_per_mol])
    return jacs_per_mol


def sample_with_mask(n, num_samples, mask):
    if mask.shape[0] != n:
        raise ValueError("Mask length must be equal to the number of rows in the grid (n)")
    
    # Calculate total available columns after applying the mask
    # Only rows where mask is True are considered
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
    samples = torch.stack((row_indices, col_indices), dim=1)
    
    return samples

def get_jacobian(forces, pos, grad_outputs, create_graph=False, looped=False):
    # This function should get the VJP of forces with respect to positions with the vectors being the row sof grad_outputs.
    def compute_grad(grad_output):
        return torch.autograd.grad(
                outputs=forces,
                inputs=pos,
                grad_outputs=grad_output,
                create_graph=create_graph,
                retain_graph=True
            )[0]
    if not looped:
        return torch.vmap(compute_grad)(grad_outputs)
    else:
        num_atoms = forces.shape[0]
        full_jac = torch.zeros(grad_outputs.shape[0], num_atoms, 3).to(forces.device)
        for i in range(grad_outputs.shape[0]):
                full_jac[i] = compute_grad(grad_outputs[i])
        return full_jac


def get_jacobian_finite_difference(forces, batch, grad_outputs, forward, detach, collater, looped=False, h=0.001): 

    original_pos = batch.pos.clone()
    perturbed_batches = []

    total_num_atoms = batch.pos.shape[0]
    for output in grad_outputs:
        perturbed_batch_forward = batch.clone()
        perturbed_batch_forward.pos = (original_pos + h * output).detach()
        perturbed_batches.append(perturbed_batch_forward)

    if not looped:
        large_batch = collater(perturbed_batches)
        perturbed_forces = forward(large_batch)['forces']
    else:
        perturbed_forces = []
        for batch in perturbed_batches:
            pert_force = forward(batch)['forces']['forces'].detach() if detach else forward(batch)['forces']['forces']
            perturbed_forces.append(pert_force)
        perturbed_forces = torch.cat(perturbed_forces, dim=0)
    # Split the large batch's forces into individual forward and backward forces
    hessian_columns = []
    for i in range(len(perturbed_batches)):
        forward_force = perturbed_forces[i * total_num_atoms:(i + 1) * total_num_atoms]
        hessian_col = (forward_force - forces.detach()) / h if detach else (forward_force - forces) / h 
        # print("HESSIAN", hessian_col.shape)
        hessian_columns.append(hessian_col)

    return torch.stack(hessian_columns, dim=0)  # NOTE: this is technically the transpose of the hessian, not the hessian
