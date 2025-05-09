import torch 
import numpy as np
import time
import logging
from fairchem.core.common.data_parallel import  OCPCollater
from fairchem.core.common import distutils

def print_cuda_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert bytes to GB
    logging.info(f"CUDA memory allocated: {allocated:.2f} GB")
    logging.info(f"CUDA memory reserved: {reserved:.2f} GB")
    print(f"CUDA memory allocated: {allocated:.2f} GB")
    print((f"CUDA memory reserved: {reserved:.2f} GB"))


def get_teacher_jacobian(forces, batch, vectorize=True, should_mask=True, finite_differences=False, forward=None, collater=None):
    natoms = batch.natoms
    total_num_atoms = sum(batch.natoms)
    cumulative_sums = [0] + torch.cumsum(natoms, 0).tolist()
    mask = batch.fixed == 0
    if not should_mask:
        mask = torch.ones(total_num_atoms, dtype=torch.bool)
    
    mask_per_mol = [mask[cum_sum:cum_sum + nat] for cum_sum, nat in zip(cumulative_sums[:-1], natoms)]
    num_free_atoms_per_mol = [sum(sub_mask) for sub_mask in mask_per_mol]
    max_free_atom_per_mol = max(num_free_atoms_per_mol)
    grad_outputs = torch.zeros((max_free_atom_per_mol, 3, total_num_atoms, 3)).to(forces.device)
    for i, free_atoms_in_mol in enumerate(num_free_atoms_per_mol):
        indices = torch.arange(free_atoms_in_mol)
        offset_indices = torch.nonzero(mask_per_mol[i]).flatten() + cumulative_sums[i]
        assert len(offset_indices) == free_atoms_in_mol
        grad_outputs[indices, :, offset_indices, :] = torch.eye(3)[None, :, :].to(forces.device)

    grad_outputs = grad_outputs.reshape(max_free_atom_per_mol * 3, total_num_atoms, 3)
    if finite_differences:
        jac = get_jacobian_finite_difference(forces, batch, grad_outputs, forward=forward, collater = collater, looped=(not vectorize))
    else:
        jac = get_jacobian(forces, batch.pos, grad_outputs, looped=(not vectorize)) # outputs a max_free_atom_per_mol x 3 x total_num_atoms x 3 matrix.
    jac = jac.reshape(max_free_atom_per_mol, 3, total_num_atoms, 3)
    jacs_per_mol = [jac[:n_fr_at, :,  cum_sum:cum_sum + nat, :] for cum_sum, n_fr_at, nat in zip(cumulative_sums[:-1], num_free_atoms_per_mol, natoms)]

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

def get_samples_biased(true_jac, num_samples):
    temperature = 7
    num_free_atoms = true_jac.shape[0]
    num_samples = min(num_samples, 3*num_free_atoms)
    true_jac_reshaped = true_jac.reshape(3*num_free_atoms, 3*num_free_atoms)

    # Calculate the norm of each row (L2 norm)
    row_norms = torch.norm(true_jac_reshaped, dim=1)
    
    # Apply softmax with temperature
    probabilities = torch.softmax(row_norms / temperature, dim=0)
    chosen_indices = torch.multinomial(probabilities, num_samples)

    # Convert flat indices back to block (atom) and component (coordinate) indices
    block_row_indices = chosen_indices // 3
    component_row_indices = chosen_indices % 3

    # Combine into 2-tuples (atom index, component index)
    samples = torch.stack((block_row_indices, component_row_indices), dim=1)

    return samples


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
    losses = [custom_loss(jac, true_jac) for jac, true_jac in zip(jacs_per_mol, true_jacs_per_mol)]
    valid_losses = [loss * 1e-8 if true_jac.abs().max().item() > 10000 else loss for loss, true_jac in zip(losses, true_jacs_per_mol)]  # filter weird hessians
    
    loss = sum(valid_losses)
    
    num_samples = sum(num_free_atoms_per_mol)
    num_samples = distutils.all_reduce(num_samples, device=forces.device)
        # Multiply by world size since gradients are averaged
        # across DDP replicas
    loss  = loss * distutils.get_world_size() / num_samples
    return loss 


def get_energy_jac_loss(out, batch, energy_std):
    neg_force_label = -1 * batch['forces']
    neg_force_label_normed = neg_force_label / energy_std # we've been training our energies to normalized targets (labels /= std), so we need to do the same here
    
    energy_jac = torch.autograd.grad(out['energy'].sum(), batch.pos, create_graph=True, retain_graph=True)[0]
    
    custom_loss = lambda jac, true_jac: torch.norm(jac - true_jac, p=2, dim=-1).sum(dim=0) 
    loss = custom_loss(energy_jac, neg_force_label_normed)
    
    loss  = loss / sum(batch.natoms) # different options here, currently weighting all systems the same regardless of size
    
    return loss
    

def get_jacobian_finite_difference(forces, batch, grad_outputs, forward, collater, looped=False, h=0.0001): 
    # currently a right difference scheme, not a central difference scheme.
    
    # Store original positions
    original_pos = batch.pos.clone()

    # Create a list to store all perturbed batches
    perturbed_batches = []

    # Total number of atoms
    total_num_atoms = batch.pos.shape[0]
    for output in grad_outputs:
        # Create forward perturbation
        perturbed_batch_forward = batch.clone()
        # perturbed_batch_forward.pos = (original_pos + h * output)
        perturbed_batch_forward.pos = (original_pos + h * output).detach().requires_grad_()
        # Append both perturbed batches to the list
        perturbed_batches.append(perturbed_batch_forward)

    # Combine all perturbed batches into one large batch
    if not looped:
        large_batch = collater(perturbed_batches)
        # Perform forward pass for all perturbed batches at once
        perturbed_forces = forward(large_batch)['forces']
    else:
        perturbed_forces = []
        for batch in perturbed_batches:
            perturbed_forces.append(forward(batch)['forces'])
        perturbed_forces = torch.cat(perturbed_forces, dim=0)
    # Split the large batch's forces into individual forward and backward forces
    hessian_columns = []
    for i in range(len(perturbed_batches)):
        forward_force = perturbed_forces[i * total_num_atoms:(i + 1) * total_num_atoms]
        hessian_col = (forward_force - forces) / h
        hessian_columns.append(hessian_col)

    # Stack columns to form the Jacobian matrix
    #technically, dim should be 1 here since they're columns...but since the hessian is symmetric it shouldn't matter hopefully
    return torch.stack(hessian_columns, dim=0) 

    
    

# def get_jacobian_forward_mode(forces, batch, grad_outputs, collater, forward, looped=False, h=0.0001):
#     """
#     Compute Jacobian-vector products using forward-mode autodiff.
#     Args:
#         forces: Predicted forces tensor
#         batch: Input batch containing positions
#         grad_outputs: Vectors to compute JVP with (shape: [..., num_atoms, 3])
#         collater: Batch collation function
#         forward: Forward pass function
#     Returns:
#         JVPs stacked along first dimension
#     """
#     # Prepare function that computes forces given positions
#     def forces_fn(positions):
#         # Clone the batch to avoid mutating it
#         # batch_copy = batch.clone()  # Ensure you have a clone method or copy mechanism for your batch
#         # batch_copy.pos = positions  # Assign the positions to the new batch
#         batch.pos = positions
#         def simple_tester(batch):
#             print(batch.pos.shape)
#             return batch.pos ** 2
        
#         return simple_tester(batch)
    
#     # Function to compute single JVP
#     def compute_single_jvp(tangent):
#         _, jvp_out = jvp(
#             func=forces_fn,
#             primals=(batch.pos,),
#             tangents=(tangent,)
#         )
#         return jvp_out

#     # Vectorize over grad_outputs
#     if len(grad_outputs.shape) == 4:
#         # Case: grad_outputs is [batch, atom, atom, 3]
#         compute_jvp = torch.vmap(torch.vmap(compute_single_jvp))
#     else:
#         # Case: grad_outputs is [atom, atom, 3]
#         compute_jvp = torch.vmap(compute_single_jvp)
        
#     return compute_jvp(grad_outputs)
    

    
