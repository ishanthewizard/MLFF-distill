import torch 
import numpy as np
from fairchem.core.modules.loss import L2MAELoss
import time

def print_cuda_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert bytes to GB
    print(f"CUDA memory allocated: {allocated:.2f} GB")
    print(f"CUDA memory reserved: {reserved:.2f} GB")

def custom_sigmoid(x, threshold):
    # Shift and scale the input to create a steep transition
    center = threshold / 2
    z1 = center - x
    z2 = x - center
    e_z1 = np.exp(z1)
    e_z2 = np.exp(z2)
    sum_e = e_z1 + e_z2
    return e_z2 / sum_e

def get_teacher_jacobian(forces, batch, vectorize=True):
    natoms = batch.natoms
    total_num_atoms = sum(batch.natoms)
    max_atom_per_mol = max(batch.natoms)
    cumulative_sums = [0] + torch.cumsum(natoms, 0).tolist()
    grad_outputs = torch.zeros((max_atom_per_mol, 3, total_num_atoms, 3)).to(forces.device)
    for i, atoms_in_mol in enumerate(batch.natoms):
        indices = torch.arange(atoms_in_mol)
        offset_indices = indices + cumulative_sums[i]
        grad_outputs[indices, :, offset_indices, :] = torch.eye(3)[None, :, :].to(forces.device)
    jac = get_jacobian(forces, batch.pos, grad_outputs, looped=(not vectorize)) # outputs a max_atom_per_mol x 3 x total_num_atoms x 3 matrix. 
    
    jacs_per_mol = [jac[:nat, :,  cum_sum:cum_sum + nat, :] for cum_sum, nat in zip(cumulative_sums[:-1], natoms)]
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
    # This function should: take the derivatives of forces with respect to positions. 
    # Grad_outputs should be supplied. if it's none, then 
    def compute_grad(grad_output):
        return torch.autograd.grad(
                outputs=forces,
                inputs=pos,
                grad_outputs=grad_output,
                create_graph=create_graph,
                retain_graph=True
            )[0]
    if not looped:
        if len(grad_outputs.shape) == 4:
            compute_jacobian = torch.vmap(torch.vmap(compute_grad))
        else:
            compute_jacobian = torch.vmap(compute_grad)
        return compute_jacobian(grad_outputs)
    else:
        num_atoms = forces.shape[0]
        if len(grad_outputs.shape) == 4:
            full_jac = torch.zeros(grad_outputs.shape[0], 3, num_atoms, 3).to(forces.device)
            for i in range(grad_outputs.shape[0]):
                for j in range(3):
                    full_jac[i, j] = compute_grad(grad_outputs[i, j])
        else:
            full_jac = torch.zeros(grad_outputs.shape[0], num_atoms, 3).to(forces.device)
            for i in range(grad_outputs.shape[0]):
                    full_jac[i] = compute_grad(grad_outputs[i])
        return full_jac

def get_force_jac_loss(out, batch, num_samples, mask, should_mask, looped=False):
    forces = out['forces']
    natoms = batch.natoms
    total_num_atoms = forces.shape[0]
    if not should_mask:
        mask = torch.ones(total_num_atoms, dtype=torch.bool)
    cumulative_sums = [0] + torch.cumsum(natoms, 0).tolist()
    
    by_molecule = []
    grad_outputs = torch.zeros((num_samples, total_num_atoms, 3)).to(forces.device)
    for i, atoms_in_mol in enumerate(batch.natoms):
        submask = mask[cumulative_sums[i]:cumulative_sums[i+1]]
        samples = sample_with_mask(atoms_in_mol, num_samples, submask)
        samples[:, 0] += cumulative_sums[i]  # offset to the correct molecule
        by_molecule.append(samples)
        
        # Vectorized assignment to grad_outputs
        grad_outputs[torch.arange(min(num_samples, atoms_in_mol*3)), samples[:, 0], samples[:, 1]] = 1
    # Compute the jacobian using grad_outputs
    jac = get_jacobian(forces, batch.pos, grad_outputs, create_graph=True, looped=looped)
    
    # Decomposing the Jacobian tensor by molecule in a batch
    jacs_per_mol = [jac[:len(mol_samps), cum_sum:cum_sum + nat, :] for mol_samps, cum_sum, nat in zip(by_molecule, cumulative_sums[:-1], natoms)]
    
    # Preparing the true jacobians in batch (we're gonna have to change this later most likely)
    true_jacs_per_mol = [batch['force_jacs'][samples[:, 0], samples[:, 1]] for samples in by_molecule]

    loss_fn = L2MAELoss()
    total_loss = sum(loss_fn(jac, true_jac) for jac, true_jac in zip(jacs_per_mol, true_jacs_per_mol)) / len(jacs_per_mol)
    assert hasattr(total_loss, "grad_fn")
    return total_loss