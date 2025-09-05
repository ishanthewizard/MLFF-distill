from fairchem.core.models.base import HydraModel
import torch
from src_v2.distill_utils import get_jacobian, get_jacobian_finite_difference
from fairchem.core.models.base import HeadInterface
import torch.nn as nn

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
        grad_outputs = data.grad_outputs.reshape(total_num_atoms, -1, 3).permute(1,0,2) # (num_samples, total_num_atoms, 3)

        
        # jac = get_jacobian(forces, data.pos, grad_outputs, create_graph=True, looped=looped) # num_samples, num_atoms, 
        jac = get_jacobian_finite_difference(forces, data, grad_outputs, super().forward, detach=False, collater=None, looped=True, h= 0.001)
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


class Node_Embedding_Head(nn.Module, HeadInterface):
    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
        return {"node_embedding": emb["node_embedding"]}