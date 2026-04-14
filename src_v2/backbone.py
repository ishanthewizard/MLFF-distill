from fairchem.core.common.registry import registry
from fairchem.core.models.uma.escn_md import eSCNMDBackbone


@registry.register_model("custom_escnmd_backbone")
class CustomESCNMDBackbone(eSCNMDBackbone):
    """
    A custom subclass of eSCNMDBackbone that resolves a conflict where autograd-based
    stress computation fails when direct_forces=True.

    Root cause: MLIPPredictUnit._run_inference uses torch.no_grad() when
    backbone.direct_forces=True, which prevents the autograd graph from being built for
    stress. Additionally, the backbone's _get_displacement_and_cell skips displacement
    setup when direct_forces=True.

    Fix: Override the direct_forces property to return False when regress_stress=True.
    This causes:
      1. MLIPPredictUnit to use nullcontext() (not no_grad) during inference
      2. The backbone's _get_displacement_and_cell to set up displacement + requires_grad

    The actual force values are still predicted fast via Direct_Force_Head
    (which reads equivariant embeddings directly), so there is no performance regression
    for force prediction.
    """

    @property
    def direct_forces(self) -> bool:
        # When stress is needed, report False so that:
        # - MLIPPredictUnit._run_inference uses nullcontext() (not no_grad)
        # - _get_displacement_and_cell sets up displacement with requires_grad=True
        # The actual force computation uses Direct_Force_Head regardless.
        if self.regress_config.stress:
            return False
        return self.regress_config.direct_forces
