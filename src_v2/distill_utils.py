import torch 
import numpy as np
import time
import logging
# from fairchem.core.common.data_parallel import  OCPCollater

def print_cuda_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert bytes to GB
    logging.info(f"CUDA memory allocated: {allocated:.2f} GB")
    logging.info(f"CUDA memory reserved: {reserved:.2f} GB")
    print(f"CUDA memory allocated: {allocated:.2f} GB")
    print((f"CUDA memory reserved: {reserved:.2f} GB"))
    
def get_teacher_jacobian(batch, vectorize=True, should_mask=True, approximation="disabled", forward=None, collater=None, device="cuda"):
    # build a list of aranges
        # 1) pull out sizes
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
    
    jac = get_jacobian(forces, batch.pos, grad_outputs, looped=(not vectorize))
    jac = jac.reshape(max_atoms, 3, total_atoms, 3)
    jacs_per_mol = [jac[:nat, :,  cum_sum:cum_sum + nat, :] for cum_sum,  nat in zip(cumsum[:-1], natoms)]
    print("FINISHED AUTOGRAD JAC")
    
    jac2 = get_jacobian_finite_difference(forces, batch, grad_outputs, forward=forward, collater = collater, looped=(not vectorize))
    jac2 = jac2.reshape(max_atoms, 3, total_atoms, 3)
    jacs_per_mol2 = [jac2[:nat, :,  cum_sum:cum_sum + nat, :] for cum_sum,  nat in zip(cumsum[:-1], natoms)]
    
    for i in range(2):
        rel_err = (jacs_per_mol[i] - jacs_per_mol2[i].permute(2,3,0,1)).abs().mean() / jacs_per_mol[i].abs().mean()
        autograd_symm = (jacs_per_mol[i] - jacs_per_mol[i].permute(2,3,0,1)).abs().mean() / jacs_per_mol[i].abs().mean()
        finite_diff_symm = (jacs_per_mol2[i] - jacs_per_mol2[i].permute(2,3,0,1)).abs().mean() / jacs_per_mol2[i].abs().mean()
        print(f"Mean relative error: {100*rel_err:.2f}%")
        print(f"Autograd asymmetry: {100*autograd_symm:.2f}%")
        print(f"Finite Difference assymetry: {100*finite_diff_symm:.2f}% \n")
    
    
    breakpoint()
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
        return torch.vmap(compute_grad)
    else:
        num_atoms = forces.shape[0]
        full_jac = torch.zeros(grad_outputs.shape[0], num_atoms, 3).to(forces.device)
        for i in range(grad_outputs.shape[0]):
                full_jac[i] = compute_grad(grad_outputs[i])
        return full_jac

    
def get_jacobian_finite_difference(forces, batch, grad_outputs, forward, collater, looped=False, h=0.001): 
    # currently a right difference scheme, not a central difference scheme.
    # Store original positions
    original_pos = batch.pos.clone()

    # Create a list to store all perturbed batches
    perturbed_batches = []

    # Total number of atoms
    total_num_atoms = batch.pos.shape[0]
    for output in grad_outputs:
        # print("OUTPUT SHAPE", output.shape)
        # Create forward perturbation
        perturbed_batch_forward = batch.clone()
        # perturbed_batch_forward.pos = (original_pos + h * output)
        perturbed_batch_forward.pos = (original_pos + h * output).detach()
        # print("PERTURBED BATCH FORWARD SHAPE", perturbed_batch_forward.pos.shape)
        # Append both perturbed batches to the list
        perturbed_batches.append(perturbed_batch_forward)

    # Combine all perturbed batches into one large batch
    if not looped:
        # print("herere")
        large_batch = collater(perturbed_batches)
        # Perform forward pass for all perturbed batches at once
        perturbed_forces = forward(large_batch)['forces']
    else:
        # print("here")
        perturbed_forces = []
        for batch in perturbed_batches:
            # perturbed_output = forward(batch)
            # save memory
            # perturbed_output['energy'] = perturbed_output['energy'].detach()
            # perturbed_output['forces'] = perturbed_output['forces'].detach()
            # print("PERTURBED OUTPUT", perturbed_output.keys())
            perturbed_forces.append(forward(batch)['forces']['forces'].detach())
        perturbed_forces = torch.cat(perturbed_forces, dim=0)
    # Split the large batch's forces into individual forward and backward forces
    hessian_columns = []
    for i in range(len(perturbed_batches)):
        forward_force = perturbed_forces[i * total_num_atoms:(i + 1) * total_num_atoms]
        hessian_col = (forward_force - forces.detach()) / h
        # print("HESSIAN", hessian_col.shape)
        hessian_columns.append(hessian_col)

    # Stack columns to form the Jacobian matrix
    # technically, dim should be 1 here since they're columns...but since the hessian is symmetric it shouldn't matter hopefully
    # print("HESSIAN SHAPE", torch.stack(hessian_columns, dim=0).shape)
    return torch.stack(hessian_columns, dim=0) 

# def get_teacher_jacobian(batch, vectorize=True, should_mask=True, approximation="disabled", forward=None, collater=None, device="cuda"):
#     device = device
#     natoms = batch.natoms
#     total_num_atoms = sum(batch.natoms)
#     cumulative_sums = [0] + torch.cumsum(natoms, 0).tolist()
#     mask = batch.fixed == 0
#     if not should_mask:
#         mask = torch.ones(total_num_atoms, dtype=torch.bool)
    
#     mask_per_mol = [mask[cum_sum:cum_sum + nat] for cum_sum, nat in zip(cumulative_sums[:-1], natoms)]
#     # print("mask_per_mol", mask_per_mol)
    
#     num_free_atoms_per_mol = [sum(sub_mask) for sub_mask in mask_per_mol]
#     max_free_atom_per_mol = max(num_free_atoms_per_mol)
#     grad_outputs = torch.zeros((max_free_atom_per_mol, 3, total_num_atoms, 3)).to(device)
#     for i, free_atoms_in_mol in enumerate(num_free_atoms_per_mol):
#         indices = torch.arange(free_atoms_in_mol)
#         offset_indices = torch.nonzero(mask_per_mol[i]).flatten() + cumulative_sums[i]
#         assert len(offset_indices) == free_atoms_in_mol
#         grad_outputs[indices, :, offset_indices, :] = torch.eye(3)[None, :, :].to(device)
#     grad_outputs = grad_outputs.reshape(max_free_atom_per_mol * 3, total_num_atoms, 3)
#     if approximation == "forward":
#         # Forward Difference
#         forces = forward(batch)['forces']['forces'].detach()
#         # print("device", forces.device, batch.pos.device,grad_outputs.device)
#         jac = get_jacobian_finite_difference(forces, batch, grad_outputs, forward=forward, collater = collater, looped=(not vectorize))
#         jac = jac.reshape(max_free_atom_per_mol, 3, total_num_atoms, 3)
#         jacs_per_mol = [jac[:n_fr_at, :,  cum_sum:cum_sum + nat, :] for cum_sum, n_fr_at, nat in zip(cumulative_sums[:-1], num_free_atoms_per_mol, natoms)]
#     elif approximation == "disabled":
#         # Vector Jacobian Product
#         forces = forward(batch)['forces']['forces']
#         # jac = torch.zeros((max_free_atom_per_mol * 3, total_num_atoms, 3), device=forces.device)
#         jac = get_jacobian(forces, batch.pos, grad_outputs, looped=(not vectorize)) # outputs a max_free_atom_per_mol x 3 x total_num_atoms x 3 matrix.
#         print("FINISHED AUTOGRAD JAC")
#         jac = jac.reshape(max_free_atom_per_mol, 3, total_num_atoms, 3)
#         jacs_per_mol = [jac[:n_fr_at, :,  cum_sum:cum_sum + nat, :] for cum_sum, n_fr_at, nat in zip(cumulative_sums[:-1], num_free_atoms_per_mol, natoms)]
        
#         jac2 = get_jacobian_finite_difference(forces, batch, grad_outputs, forward=forward, collater = collater, looped=(not vectorize))
#         jac2 = jac2.reshape(max_free_atom_per_mol, 3, total_num_atoms, 3)
#         jacs_per_mol2 = [jac2[:n_fr_at, :,  cum_sum:cum_sum + nat, :] for cum_sum, n_fr_at, nat in zip(cumulative_sums[:-1], num_free_atoms_per_mol, natoms)]
#         breakpoint()
#         rel_err = (jacs_per_mol[0] - jacs_per_mol2[0]).abs().mean() \
#           / jacs_per_mol[0].abs().mean()
#         print(f"Mean relative error: {100*rel_err:.2f}%")
#         print((jacs_per_mol[0] - jacs_per_mol2[0]).abs().mean() / jacs_per_mol[0].abs().mean())
        
#     # print("JACS PER MOL", [(jacs.reshape(12,12),jacs.shape) for jacs in jacs_per_mol])
#     return jacs_per_mol
