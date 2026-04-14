import torch
import torch.nn as nn
from fairchem.core.models.base import HeadInterface
from fairchem.core.models.uma.outputs import (
    compute_energy as uma_compute_energy,
    compute_forces_and_stress as uma_compute_forces_and_stress,
)
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear
from fairchem.core.common import gp_utils
from fairchem.core.common.utils import conditional_grad

def get_l_component_range(tensor, l_min, l_max):
    """
    Standard spherical harmonic layout: 
    L=0: 1 component (idx 0)
    L=1: 3 components (idx 1, 2, 3)
    ...
    Start index of L is L^2, end is (L+1)^2
    """
    return tensor.narrow(1, l_min**2, (l_max+1)**2 - l_min**2)

def reduce_node_to_system(node_values, batch, num_systems, natoms=None, reduce="sum"):
    """
    Reduces node-wise values to system-wise values based on batch.
    """
    energies = torch.zeros(num_systems, device=node_values.device, dtype=node_values.dtype)
    energies.index_add_(0, batch, node_values.view(-1))
    if reduce == "mean" and natoms is not None:
        energies = energies / natoms
    return energies

def compute_energy(emb, energy_block, batch, num_systems, natoms=None, reduce="sum"):
    """
    Computes system energy from node embeddings.
    """
    # Extract L=0 (scalar) component for energy prediction
    node_embedding = get_l_component_range(emb["node_embedding"], 0, 0).squeeze(1)
    node_energies = energy_block(node_embedding)
    energy = reduce_node_to_system(node_energies, batch, num_systems, natoms, reduce)
    
    if gp_utils.initialized():
        energy = gp_utils.reduce_from_model_parallel_region(energy)
        
    return energy, node_energies

def compute_forces_and_stress(energy_part, pos, displacement=None, cell=None, batch=None, training=True):
    """
    Computes forces and stress using autograd.
    """
    if not energy_part.requires_grad:
        return None, None

    targets = []
    if pos.requires_grad: targets.append(pos)
    if displacement is not None and displacement.requires_grad: targets.append(displacement)
    
    if not targets:
        return None, None

    grads = torch.autograd.grad(
        energy_part.sum(),
        targets,
        create_graph=training,
        retain_graph=training,  # Must be False for compiled models during inference
        allow_unused=True
    )
    
    # Map grads back to pos and displacement
    forces = None
    stress = None
    
    grad_idx = 0
    if pos.requires_grad:
        forces = -grads[grad_idx]
        grad_idx += 1
    if displacement is not None and displacement.requires_grad:
        # Virial based stress conversion
        virial = grads[grad_idx]
        # In eSCN, virial might be [natoms, 3, 3] or [nbatch, 3, 3] depending on implementation
        # Usually it's already summed if we differentiated energy_sum
        volume = torch.det(cell).abs().view(-1, 1, 1) if cell is not None else 1.0
        stress = (virial / volume).view(-1, 9)
        grad_idx += 1
        
    return forces, stress

class Direct_Force_Head(nn.Module, HeadInterface):
    """
    A custom head for predicting forces directly from equivariant embeddings.
    Supports both Linear and MLP-like architectures.
    """
    def __init__(
        self,
        backbone,
        prefix: str | None = None,
        wrap_property: bool = True,
        hidden_channels: int | None = None,
    ):
        super().__init__()
        self.prefix = prefix
        self.wrap_property = wrap_property
        
        channels = backbone.sphere_channels
        self.hidden_channels = hidden_channels or backbone.hidden_channels
        
        # We use one SO3_Linear to get the vector component.
        # If hidden_channels is provided and different from channels, we might want an MLP.
        # For simplicity and standard practice in eSCN, a single SO3_Linear is often used 
        # as the 'direct force' head if the backbone is already deep.
        self.force_block = SO3_Linear(channels, 1, lmax=1)

    def forward(self, data, emb):
        forces_key = f"{self.prefix}_forces" if self.prefix else "forces"
        
        # SO3_Linear with lmax=1 requires both L=0 and L=1 as input
        l0_l1_embedding = get_l_component_range(emb["node_embedding"], l_min=0, l_max=1)
        forces_output = self.force_block(l0_l1_embedding)

        # Extract L=1 (vector) component from the output
        forces = get_l_component_range(forces_output, l_min=1, l_max=1)
        forces = forces.view(-1, 3).contiguous()

        if gp_utils.initialized():
            forces = gp_utils.gather_from_model_parallel_region(
                forces, data["atomic_numbers_full"].shape[0]
            )

        return {forces_key: {"forces": forces} if self.wrap_property else forces}

class Energy_Head(nn.Module, HeadInterface):
    """
    A custom head for predicting energy using an MLP.
    Also supports autograd-based stress if requested.
    """
    def __init__(
        self,
        backbone,
        prefix: str | None = None,
        wrap_property: bool = True,
        reduce: str = "sum",
    ):
        super().__init__()
        self.prefix = prefix
        self.wrap_property = wrap_property
        self.reduce = reduce
        self.regress_stress = getattr(backbone, "regress_stress", False)
        self.direct_stress = getattr(backbone, "direct_stress", False)
        
        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb):
        energy_key = f"{self.prefix}_energy" if self.prefix else "energy"
        stress_key = f"{self.prefix}_stress" if self.prefix else "stress"
        
        energy, energy_part = uma_compute_energy(
            emb,
            self.energy_block,
            data["batch"],
            len(data["natoms"]),
            natoms=data["natoms"],
            reduce=self.reduce,
        )
        
        outputs = {energy_key: {"energy": energy} if self.wrap_property else energy}
        
        # If stress is requested, compute via autograd.
        # conditional_grad(torch.enable_grad()) ensures this works even under
        # inference no_grad context (which happens when direct_forces=True).
        if self.regress_stress and not self.direct_stress:
            _, stress = uma_compute_forces_and_stress(
                energy_part,
                data["pos"],
                data["cell"],
                batch=data["batch_full"],
                training=self.training,
            )
            outputs[stress_key] = {"stress": stress} if self.wrap_property else stress
            
        return outputs
