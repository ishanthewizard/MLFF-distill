from fairchem.core.models.base import HydraModel
import torch
from src.distill_utils import get_jacobian
from src.dataset_utils import initialize_finetuning_model

# class HessianModelWrapper(HydraModel):
#     def get_sampled_hessian(self, data, out):
#         # CURRENTLY DOES NOT SUPPORT MASKING
#         ###### OTHER ARGS THAT NEED TO BE ADDED TO THE CONFIG #######
#         looped = False
#         finite_differences = False
#         #############################################################
#         # breakpoint()
#         forces = out['forces']['forces']
#         natoms = data.natoms 
#         total_num_atoms = forces.shape[0]
#         num_samples = data.num_samples[0]
#         cumulative_sums = torch.cat([torch.tensor([0], device=natoms.device), torch.cumsum(natoms, 0)]) # 0... sum(natoms)
#         grad_outputs = torch.zeros((num_samples, total_num_atoms, 3)).to(forces.device) # (num_samples, total_num_atoms, 3)

#         offset_samples = data.samples.clone() #(num_systems x num_samples, 2)
#         offsets = cumulative_sums[:-1].repeat_interleave(data.num_samples)   # shape = (num_systems × num_samples,)
#         offset_samples[:, 0] += offsets

#         grad_outputs[:, offset_samples[:, 0], offset_samples[:, 1]] = 1
        
#         jac = get_jacobian(forces, data.pos, grad_outputs, create_graph=True, looped=looped) # num_samples, num_atoms, 
#         jacs_per_mol = [jac[:, cum_sum:cum_sum + nat, :] for cum_sum, nat in zip(cumulative_sums[:-1], natoms)] # arr where each elem is (num_samples, num_atoms, 3)
#         jacs_per_mol = [jac.permute(1, 0, 2).reshape(nat, -1) for jac, nat in zip(jacs_per_mol, natoms)] # (arr where each elem is (num_atoms, num_samples *3))

#         return torch.cat(jacs_per_mol)
        
#     def forward(self, data):
#         # breakpoint()
#         data.pos = data.pos.detach().requires_grad_(True)
#         out = super().forward(data)
#         is_validating = torch.all(data['forces_jac'] == 0)
#         force_jacs = torch.zeros((sum(data.natoms), data.num_samples[0] * 3), device=data.pos.device)  if is_validating else  self.get_sampled_hessian(data, out)#self.get_sampled_hessian(data, out) # torch.zeros((sum(data.natoms), data.num_samples[0] * 3))
#         out['forces_jac'] = {'forces_jac': force_jacs}
#         return out

class HessianModelWrapper(HydraModel):
    def get_sampled_hessian(self, data, out):
        # CURRENTLY DOES NOT SUPPORT MASKING
        ###### OTHER ARGS THAT NEED TO BE ADDED TO THE CONFIG #######
        looped = False
        finite_differences = False
        #############################################################
        # breakpoint()
        forces = out['forces']['forces']
        natoms = data.natoms 
        total_num_atoms = forces.shape[0]
        num_samples = data.num_samples[0]
        cumulative_sums = torch.cat([torch.tensor([0], device=natoms.device), torch.cumsum(natoms, 0)]) # 0... sum(natoms)
        grad_outputs = torch.zeros((num_samples, total_num_atoms, 3)).to(forces.device) # (num_samples, total_num_atoms, 3)

        offset_samples = data.samples.clone() #(num_systems x num_samples, 2)
        offsets = cumulative_sums[:-1].repeat_interleave(data.num_samples)   # shape = (num_systems × num_samples,)
        offset_samples[:, 0] += offsets

        grad_outputs[:, offset_samples[:, 0], offset_samples[:, 1]] = 1
        
        jac = get_jacobian(forces, data.pos, grad_outputs, create_graph=True, looped=looped) # num_samples, num_atoms, 
        jacs_per_mol = [jac[:, cum_sum:cum_sum + nat, :] for cum_sum, nat in zip(cumulative_sums[:-1], natoms)] # arr where each elem is (num_samples, num_atoms, 3)
        jacs_per_mol = [jac.permute(1, 0, 2).reshape(nat, -1) for jac, nat in zip(jacs_per_mol, natoms)] # (arr where each elem is (num_atoms, num_samples *3))

        return torch.cat(jacs_per_mol)
        
    def forward(self, data):
        # breakpoint()
        temp = initialize_finetuning_model("/home/yuejian/project/MLFF-distill/OMol_Whole/ckpt/esen_sm_direct_all.pt").to("cuda")
        data.pos = data.pos.detach().requires_grad_(True)
        out = temp.forward(data)
        is_validating = torch.all(data['forces_jac'] == 0)
        force_jacs = torch.zeros((sum(data.natoms), data.num_samples[0] * 3), device=data.pos.device)  if is_validating else  self.get_sampled_hessian(data, out)#self.get_sampled_hessian(data, out) # torch.zeros((sum(data.natoms), data.num_samples[0] * 3))
        out['forces_jac'] = {'forces_jac': force_jacs}
        return out
# class HessianModelWrapper(HydraModel):
#     def __init__(self, model, backbone=None, heads=None):
#         super().__init__(
#             heads=heads,
#             backbone=backbone,
#         )
#         self.model = model
#     def get_sampled_hessian(self, data, out):
#         # CURRENTLY DOES NOT SUPPORT MASKING
#         ###### OTHER ARGS THAT NEED TO BE ADDED TO THE CONFIG #######
#         looped = False
#         finite_differences = False
#         #############################################################
#         # breakpoint()
#         forces = out['forces']['forces']
#         natoms = data.natoms 
#         total_num_atoms = forces.shape[0]
#         num_samples = data.num_samples[0]
#         cumulative_sums = torch.cat([torch.tensor([0], device=natoms.device), torch.cumsum(natoms, 0)]) # 0... sum(natoms)
#         grad_outputs = torch.zeros((num_samples, total_num_atoms, 3)).to(forces.device) # (num_samples, total_num_atoms, 3)

#         offset_samples = data.samples.clone() #(num_systems x num_samples, 2)
#         offsets = cumulative_sums[:-1].repeat_interleave(data.num_samples)   # shape = (num_systems × num_samples,)
#         offset_samples[:, 0] += offsets

#         grad_outputs[:, offset_samples[:, 0], offset_samples[:, 1]] = 1
        
#         jac = get_jacobian(forces, data.pos, grad_outputs, create_graph=True, looped=looped) # num_samples, num_atoms, 
#         jacs_per_mol = [jac[:, cum_sum:cum_sum + nat, :] for cum_sum, nat in zip(cumulative_sums[:-1], natoms)] # arr where each elem is (num_samples, num_atoms, 3)
#         jacs_per_mol = [jac.permute(1, 0, 2).reshape(nat, -1) for jac, nat in zip(jacs_per_mol, natoms)] # (arr where each elem is (num_atoms, num_samples *3))

#         return torch.cat(jacs_per_mol)
        
#     def forward(self, data):
#         # breakpoint()
#         data.pos = data.pos.detach().requires_grad_(True)
#         out = self.model.forward(data)
#         # breakpoint()
#         is_validating = torch.all(data['forces_jac'] == 0)
#         force_jacs = torch.zeros((sum(data.natoms), data.num_samples[0] * 3), device=data.pos.device)  if is_validating else  self.get_sampled_hessian(data, out)#self.get_sampled_hessian(data, out) # torch.zeros((sum(data.natoms), data.num_samples[0] * 3))
#         out['forces_jac'] = {'forces_jac': force_jacs}
#         return out