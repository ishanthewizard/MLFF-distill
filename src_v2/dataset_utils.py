from fairchem.core.units.mlip_unit.mlip_unit import load_inference_model
from fairchem.core.common import registry
from copy import deepcopy
import torch
import logging

def get_inverse_indices(original_indices: list[int]) -> list[int]:
    """
    Given a list of indices, 
    return the inverse indices to collate the shuffled databack to its original order.
    """
    
    inverse_indices = [0] * len(original_indices)
    for new_pos, original_idx in enumerate(original_indices):
        inverse_indices[original_idx] = new_pos
        
    return inverse_indices


def initialize_finetuning_model(
    checkpoint_location: str, overrides: dict | None = None, heads: dict | None = None
) -> torch.nn.Module:
    model, _ = load_inference_model(checkpoint_location, overrides)

    logging.warning(
        f"initialize_finetuning_model starting from checkpoint_location: {checkpoint_location}"
    )



    return model