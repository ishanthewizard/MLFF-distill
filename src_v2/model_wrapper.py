from fairchem.core.models.base import HydraModel
import torch
from src_v2.distill_utils import get_jacobian, get_jacobian_finite_difference


class HessianModelWrapper(HydraModel):
    def get_sampled_hessian(self, data, out):
        # CURRENTLY DOES NOT SUPPORT MASKING
        ###### OTHER ARGS THAT NEED TO BE ADDED TO THE CONFIG #######
        looped = False
        finite_differences = False
        #############################################################
        
        forces = out['forces']['forces']
        natoms = data.natoms 
        total_num_atoms = forces.shape[0]
        num_samples = data.num_samples[0] 
        cumulative_sums = torch.cat([torch.tensor([0], device=natoms.device), torch.cumsum(natoms, 0)]) # 0... sum(natoms)
        
        offset_samples = data.samples.clone() #(num_systems x num_samples, 2)
        offsets = cumulative_sums[:-1].repeat_interleave(data.num_samples)   # offset the samples so that they correspond to the correct start position of molecule in the batch
        offset_samples[:, 0] += offsets 
        
        grad_outputs = torch.zeros((num_samples, total_num_atoms, 3)).to(forces.device) # (num_samples, total_num_atoms, 3)
        for i in range(len(natoms)):
            curr_samples = offset_samples[num_samples * i : num_samples * (i + 1)]
            grad_outputs[torch.arange(num_samples, device=forces.device), curr_samples[:, 0], curr_samples[:, 1]] = 1

        
        jac = get_jacobian(forces, data.pos, grad_outputs, create_graph=True, looped=looped) # num_samples, num_atoms, 
        # jac = get_jacobian_finite_difference(forces, data, grad_outputs, super().forward, detach=False, collater=None, looped=True, h= 0.001)
        jacs_per_mol = [jac[:, cum_sum:cum_sum + nat, :] for cum_sum, nat in zip(cumulative_sums[:-1], natoms)] # arr where each elem is (num_samples, num_atoms, 3)
        jacs_per_mol = [jac.permute(1, 0, 2).reshape(nat, -1) for jac, nat in zip(jacs_per_mol, natoms)] # (arr where each elem is (num_atoms, num_samples *3))

        return torch.cat(jacs_per_mol)
        
    def forward(self, data):
        data.pos = data.pos.detach().requires_grad_(True)
        out = super().forward(data)
        is_validating = torch.all(data['forces_jac'] == 0)
        force_jacs = torch.zeros((sum(data.natoms), data.num_samples[0] * 3), device=data.pos.device)  if is_validating else  self.get_sampled_hessian(data, out)#self.get_sampled_hessian(data, out) # torch.zeros((sum(data.natoms), data.num_samples[0] * 3))
        out['forces_jac'] = {'forces_jac': force_jacs}
        return out
        