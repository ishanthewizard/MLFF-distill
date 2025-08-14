from typing import final
import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
# from fairchem.core.common.data_parallel import  OCPCollater
from fairchem.core.common import distutils
from tqdm import tqdm


def get_diverse_idxs(x, natoms, num_samples):
    """
    Selects a diverse subset of atoms from each molecule using farthest point sampling.
    Args:
        x (torch.Tensor): Node embeddings for all atoms in the batch.
                          Expected shape: (total_num_atoms, num_channels, ...),
                          where the scalar features are in x[:, 0, :].
        natoms (torch.Tensor): A tensor where each element is the number of atoms
                               in a molecule. Shape: (num_molecules,).
        num_samples (int): The number of diverse samples to select from each molecule.
    Returns:
        torch.Tensor: A 1D tensor containing the global indices of the selected
                      diverse atoms for the entire batch.
    """
    # Assuming scalar features are at index 0 of the second dimension.
    # Shape of scalar_x: (total_num_atoms, num_sphere_channels)
    scalar_x = x[:, 0, :]

    # Split embeddings by molecule
    cumulative_sums = torch.cumsum(natoms, 0) - natoms
    x_by_mol = [scalar_x[start:start + nat, :]for start, nat in zip(cumulative_sums, natoms)]

    all_diverse_indices = []

    for mol_embeddings in x_by_mol:
        num_atoms_in_mol = mol_embeddings.shape[0]

        if num_atoms_in_mol == 0:
            continue

        # Determine number of points to select for this molecule
        num_to_select = min(num_samples, num_atoms_in_mol)

        if num_to_select < num_samples:
            raise Exception("num_atoms_in_mol < num_samples not currently supported. num_atoms_in_mol: {}, num_samples: {}".format(num_atoms_in_mol, num_samples))

        # Farthest Point Sampling (FPS)
        # Step 1: Compute pairwise distance matrix
        dists = torch.cdist(mol_embeddings, mol_embeddings)

        # Step 2: Iteratively select farthest points
        selected_indices = torch.zeros(
            num_to_select, dtype=torch.long, device=x.device
        )

        # Start with a random point
        current_idx = torch.randint(0, num_atoms_in_mol, (1,), device=x.device).item()
        selected_indices[0] = current_idx

        # Initialize distances from all points to the selected set
        min_dists = dists[:, current_idx]

        for j in range(1, num_to_select):
            # Find the point that is farthest from the current set of selected points
            current_idx = torch.argmax(min_dists)
            selected_indices[j] = current_idx

            # Update min_dists with the distances to the new point
            min_dists = torch.minimum(min_dists, dists[:, current_idx])

        # Convert local indices to global indices
        repeated_indices = selected_indices.repeat_interleave(3)
        col = torch.arange(3, device=selected_indices.device).repeat(selected_indices.shape[0])
        final_indices = torch.stack([repeated_indices, col], dim=1)
        
        
        all_diverse_indices.append(final_indices)


    return all_diverse_indices



def get_teacher_jac_diverse(data, forward, n_diverse_samples,  force_keyword='forces', vectorize=True,  approximation="disabled", collater=None):
    out = forward(data)
    forces = out[force_keyword]['forces'].detach()
    node_embedding = out['node_embedding'] #['node_embedding']
    natoms = data.natoms
    total_atoms = forces.shape[0]
    num_samples = n_diverse_samples # NOTE: this is the number of atoms that will be sampled from each molecule, so the number of force jac rows is actually 3x this 
    cumulative_sums = torch.cumsum(natoms, 0) - natoms # 0... sum(natoms) - natoms


    diverse_indices = get_diverse_idxs(node_embedding, natoms, num_samples) #array of len num_molecules, each item is a tensor of (num_samples, 2)
    offset_samples = torch.cat(diverse_indices) 
    offsets = cumulative_sums.repeat_interleave(num_samples * 3)   # offset the samples so that they correspond to the correct start position of molecule in the batch
    offset_samples[:, 0] += offsets
        
    n_rows = num_samples * 3
    grad_outputs = torch.zeros((n_rows, total_atoms, 3)).to(forces.device) # (num_samples, total_num_atoms, 3)
    
    sample_idxs = torch.arange(n_rows, device=forces.device).repeat(len(natoms)) # repeat for each molecule
    grad_outputs[sample_idxs, offset_samples[:, 0], offset_samples[:, 1]] = 1

    jac = get_jacobian_finite_difference(forces, data, grad_outputs, forward=forward, detach=True, force_keyword=force_keyword, collater=None, looped=True, h= 0.001)
    jacs_per_mol = [jac[:, cum_sum:cum_sum + nat, :].cpu() for cum_sum,  nat in zip(cumulative_sums, natoms)]

    return zip(jacs_per_mol, diverse_indices)


def get_teacher_jacobian(batch, vectorize=True,  approximation="disabled", forward=None, collater=None, device="cuda"):
    natoms = batch.natoms                 # e.g. tensor([49,21], device=...)
    max_atoms = int(natoms.max().item())   # maximum per‐mol atom count
    total_atoms = int(natoms.sum().item())

    ranges = [torch.arange(n, device=batch.pos.device) for n in natoms.tolist()]
    sample_indices = torch.cat(ranges, dim=0)   # shape [total_atoms]

    cumsum = torch.cumsum(natoms, 0) - natoms
    offset_indices = sample_indices + cumsum.repeat_interleave(natoms)

    grad_outputs = torch.zeros((max_atoms, 3, total_atoms, 3)).to(batch.pos.device)
    eye3 = torch.eye(3, device=batch.pos.device)[None, :, :]  # [1×3×3]
    grad_outputs[sample_indices, :, offset_indices, :] = eye3
    grad_outputs = grad_outputs.reshape(max_atoms* 3, total_atoms, 3)

    forces = forward(batch)['forces']['forces']

    jac = get_jacobian_finite_difference(forces, batch, grad_outputs, forward=forward, detach=True, collater = collater, looped=(not vectorize))
    jac = jac.reshape(max_atoms, 3, total_atoms, 3)
    jacs_per_mol = [jac[:nat, :,  cum_sum:cum_sum + nat, :].cpu() for cum_sum,  nat in zip(cumsum, natoms)]
    
    diverse_indices = [torch.arange(nat, device=natoms.device) for nat in natoms]
    diverse_x3 = diverse_indices.repeat_interleave(3)
    col = [torch.arange(3, device=diverse_indices.device).repeat(div_idxs.shape[0]) for div_idxs in diverse_indices]
    diverse_indices = [torch.stack([diverse_x3, col], dim=1) for diverse_x3, col in zip(diverse_x3, col)]
    
    return zip(jacs_per_mol, diverse_indices)

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


def get_jacobian_finite_difference(forces, batch, grad_outputs, forward, detach, collater, looped=False, force_keyword='forces', h=0.001):

    original_pos = batch.pos.clone()
    perturbed_batches = []

    total_num_atoms = batch.pos.shape[0]
    for output in grad_outputs:
        perturbed_batch_forward = batch.clone()
        perturbed_batch_forward.pos = (original_pos + h * output).detach()
        perturbed_batches.append(perturbed_batch_forward)

    if not looped:
        large_batch = collater(perturbed_batches)
        perturbed_forces = forward(large_batch)[force_keyword]['forces']
    else:
        perturbed_forces = []
        for batch in perturbed_batches:
            pert_force = forward(batch)[force_keyword]['forces'].detach() if detach else forward(batch)[force_keyword]['forces']
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

def print_cuda_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert bytes to GB
    logging.info(f"CUDA memory allocated: {allocated:.2f} GB")
    logging.info(f"CUDA memory reserved: {reserved:.2f} GB")
    print(f"CUDA memory allocated: {allocated:.2f} GB")
    print((f"CUDA memory reserved: {reserved:.2f} GB")) 