from fairchem.core.models.base import HydraModel
import torch
from src_v2.distill_utils import get_jacobian

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
        grad_outputs = torch.zeros((num_samples, total_num_atoms, 3)).to(forces.device)
        offset_samples = data.samples.clone()
        offset_samples[:, 0] = cumulative_sums[:-1].repeat_interleave(data.num_samples)
        grad_outputs[:, offset_samples] = 1
        
        jac = get_jacobian(forces, data.pos, grad_outputs, create_graph=True, looped=looped)
        jacs_per_mol = [jac[:, cum_sum:cum_sum + nat, :].flatten() for cum_sum, nat in zip(cumulative_sums[:-1], natoms)]

        return torch.cat(jacs_per_mol)
        
    def forward(self, data):
        data.pos = data.pos.detach().requires_grad_(True)
        out = super().forward(data)
        force_jacs = self.get_sampled_hessian(data, out)
        out['force_jacs'] = force_jacs
        return out
        