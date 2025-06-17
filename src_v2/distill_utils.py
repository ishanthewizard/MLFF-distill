import torch 
import numpy as np
import time
import logging
# from fairchem.core.common.data_parallel import  OCPCollater
from fairchem.core.common import distutils


def create_hessian_mask(dataset, mask_out_percentage=0.2):
    init_mask = torch.ones(len(dataset))
    
    hessian_mask = torch.bernoulli(init_mask*(1 - mask_out_percentage))
    
    logging.info(f"\nMasking out {(1-hessian_mask.sum()*100)/len(dataset):.2f}% of the Hessian\n")
    
    return hessian_mask




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
        jac = get_jacobian_finite_difference(forces, batch, grad_outputs, forward=forward, collater = collater, looped=(not vectorize))
        jac = jac.reshape(max_free_atom_per_mol, 3, total_num_atoms, 3)
        jacs_per_mol = [jac[:n_fr_at, :,  cum_sum:cum_sum + nat, :] for cum_sum, n_fr_at, nat in zip(cumulative_sums[:-1], num_free_atoms_per_mol, natoms)]
    elif approximation == "central":
        # Central Difference
        jacs_per_mol = get_jacobian_central_difference(batch, forward=forward)
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
        return torch.vmap(compute_grad)
    else:
        num_atoms = forces.shape[0]
        full_jac = torch.zeros(grad_outputs.shape[0], num_atoms, 3).to(forces.device)
        for i in range(grad_outputs.shape[0]):
                full_jac[i] = compute_grad(grad_outputs[i])
        return full_jac



def get_force_jac_loss(out, batch, num_samples, mask, should_mask, looped=False, finite_differences=False, forward=None, collater=None):
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
        
        by_molecule.append(samples) # swap below and above line, crucial
        offset_samples = samples.clone()  # Create a copy of the samples array to avoid modifying the original
        offset_samples[:, 0] += cumulative_sums[i]
        # Vectorized assignment to grad_outputs
        grad_outputs[torch.arange(samples.shape[0]), offset_samples[:, 0], offset_samples[:, 1]] = 1
    # Compute the jacobian using grad_outputs
    if not finite_differences:
        jac = get_jacobian(forces, batch.pos, grad_outputs, create_graph=True, looped=looped)
    else:
        jac = get_jacobian_finite_difference(
            forces=forces, 
            batch= batch, 
            grad_outputs=grad_outputs, 
            collater=collater,  
            forward= forward, 
            looped=looped)
    # Decomposing the Jacobian tensor by molecule in a batch
    mask_per_mol = [mask[cum_sum:cum_sum + nat] for cum_sum, nat in zip(cumulative_sums[:-1], natoms)]
    num_free_atoms_per_mol = torch.tensor([sum(sub_mask) for sub_mask in mask_per_mol], device=natoms.device)
    cum_jac_indexes = [0] +  torch.cumsum((num_free_atoms_per_mol * natoms)*9, dim=0).tolist()
    
    jacs_per_mol = [jac[:len(mol_samps), cum_sum:cum_sum + nat, :] for mol_samps, cum_sum, nat in zip(by_molecule, cumulative_sums[:-1], natoms)]
    jacs_per_mol = [mol_jac[:, mask, :] for mol_jac, mask in  zip(jacs_per_mol, mask_per_mol)] # do the same for te student hessians

    if torch.any(torch.isnan(jac)):
        raise Exception("FORCE JAC IS NAN")
    

    true_jacs_per_mol = []
    for i, samples in enumerate(by_molecule):
        fixed_atoms = batch.fixed[cumulative_sums[i]:cumulative_sums[i+1]]
        fixed_cumsum = torch.cumsum(fixed_atoms, dim=0)
        num_free_atoms = num_free_atoms_per_mol[i]
        curr = batch['force_jacs'][cum_jac_indexes[i]:cum_jac_indexes[i+1]].reshape(num_free_atoms, 3, natoms[i], 3)
        curr = curr[:, :, mask_per_mol[i], :] # filter out the masked columns 
        subsampled_curr = curr[(samples[:, 0] - fixed_cumsum[samples[:, 0]]).long(), samples[:, 1]] # get the sampled rows
        true_jacs_per_mol.append(subsampled_curr)

    # just copying what DDPLoss does for our special case
    custom_loss = lambda jac, true_jac: torch.norm(jac - true_jac, p=2, dim=-1).sum(dim=1).mean(dim=0)
    # YUE JIAN: added hessian mask to the loss, maybe should deduct num_samples by masked out hessians
    losses = [custom_loss(jac, true_jac) for jac, true_jac in zip(jacs_per_mol, true_jacs_per_mol)]
    valid_losses = [loss * 1e-8 if true_jac.abs().max().item() > 10000 else loss for loss, true_jac in zip(losses, true_jacs_per_mol)]  # filter weird hessians

    loss = sum(valid_losses)
    
    num_samples = sum(num_free_atoms_per_mol)
    num_samples = distutils.all_reduce(num_samples, device=forces.device)
    # Multiply by world size since gradients are averaged
    # across DDP replicas
    loss  = loss * distutils.get_world_size() / num_samples
    return loss 

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


def get_jacobian_central_difference(batch, forward, h=0.0001):
    
    batch_forward = batch.clone().detach()
    batch_forward.pos = batch_forward.pos + h
    forward_forces = forward(batch_forward)['forces']
    
    batch_backward = batch.clone().detach()
    batch_backward.pos = batch_backward.pos - h
    backward_forces = forward(batch_backward)['forces']

    del batch_forward
    del batch_backward
    del batch
    
    # calculate Forces+ - Forces- / 2h
    delta_forces_delta_2_h = (forward_forces - backward_forces)/ (2*h)
    
    del forward_forces
    del backward_forces
    
    # batch to list using batch index
    batch_idx = batch.batch
    delta_forces_delta_2_h_list = [delta_forces_delta_2_h[batch_idx == i] for i in batch_idx.unique()]
    print("DELTA FORCES DELTA 2 H LIST", [d.shape for d in delta_forces_delta_2_h_list])

    # loop through the list and create jacobian
    jacobian_list = []
    for dfd2h in delta_forces_delta_2_h_list:
        (n_atom, n_dim) = dfd2h.shape
        jacobian = dfd2h.reshape(-1)
        pass
    return
