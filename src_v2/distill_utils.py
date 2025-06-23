import torch 
import numpy as np
import time
import logging
# from fairchem.core.common.data_parallel import  OCPCollater
from fairchem.core.common import distutils
from tqdm import tqdm


def get_teacher_jacobian(batch, vectorize=True,  approximation="disabled", forward=None, collater=None, device="cuda"):
    natoms = batch.natoms                 # e.g. tensor([49,21], device=...)
    max_atoms = int(natoms.max().item())   # maximum per‐mol atom count
    total_atoms = int(natoms.sum().item())
    
    ranges = [torch.arange(n, device=batch.pos.device) for n in natoms.tolist()] 
    sample_indices = torch.cat(ranges, dim=0)   # shape [total_atoms] 
    
    cumsum = torch.cat([torch.tensor([0], device=batch.pos.device, dtype=natoms.dtype), torch.cumsum(natoms, 0)], dim=0)      
    offset_indices = sample_indices + cumsum[:-1].repeat_interleave(natoms)
     
    grad_outputs = torch.zeros((max_atoms, 3, total_atoms, 3)).to(batch.pos.device) 
    eye3 = torch.eye(3, device=batch.pos.device)[None, :, :]  # [1×3×3]
    grad_outputs[sample_indices, :, offset_indices, :] = eye3
    grad_outputs = grad_outputs.reshape(max_atoms* 3, total_atoms, 3)
    
    forces = forward(batch)['forces']['forces']
    
    jac = get_jacobian_finite_difference(forces, batch, grad_outputs, forward=forward, detach=True, collater = collater, looped=(not vectorize))
    jac = jac.reshape(max_atoms, 3, total_atoms, 3)
    jacs_per_mol = [jac[:nat, :,  cum_sum:cum_sum + nat, :].cpu() for cum_sum,  nat in zip(cumsum[:-1], natoms)]
    
    return jacs_per_mol

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

def print_cuda_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert bytes to GB
    logging.info(f"CUDA memory allocated: {allocated:.2f} GB")
    logging.info(f"CUDA memory reserved: {reserved:.2f} GB")
    print(f"CUDA memory allocated: {allocated:.2f} GB")
    print((f"CUDA memory reserved: {reserved:.2f} GB"))
    