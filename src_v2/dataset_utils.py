from fairchem.core.units.mlip_unit.mlip_unit import load_inference_model
from fairchem.core.common import registry
from copy import deepcopy
import torch
import logging

def get_inverse_indices(original_indices: list[int]) -> list[int]:
    """
    A O(n) method for inverse permutation of indices with duplicates handled.
    Given a list of indices, 
    return the inverse indices to collate the shuffled databack to its original order.
    E.g., if original_indices = [3,3,4,4,5,4,2,0,1], which is created by sampler
    that shuffles the original dataset, then the inverse indices will be
    [7, 8, 6, 1, 5, 4].
    duplicates are handled by taking the last occurrence of the index.
    """
    # handling the case when duplicates are present
    right_cut = max(original_indices) + 1
    
    # O(n) approach to find inverse indices with handling duplicates
    inverse_indices = [0] * len(original_indices)
    for new_pos, original_idx in enumerate(original_indices):
        inverse_indices[original_idx] = new_pos
        
    return inverse_indices[:right_cut]


def initialize_finetuning_model(
    checkpoint_location: str, overrides: dict | None = None, heads: dict | None = None
) -> torch.nn.Module:
    model, _ = load_inference_model(checkpoint_location, overrides)

    logging.warning(
        f"initialize_finetuning_model starting from checkpoint_location: {checkpoint_location}"
    )

    return model

def initialize_model_that_have_same_structure_with_checkpoint_but_no_load_weight(
    checkpoint_location: str, overrides: dict | None = None, heads: dict | None = None
) -> torch.nn.Module:
    """
    Initialize a model with the same structure as the checkpoint but without loading weights.
    This is useful for fine-tuning or transfer learning scenarios. Also, when you want to load a student model same as a checkpoint.
    """
    # Load the model architecture from the checkpoint
    model, _ = load_inference_model(checkpoint_location, overrides)

    # logging
    logging.warning(
        f"initialize_model_that_have_same_structure_with_checkpoint_but_no_load_weight starting from checkpoint_location: {checkpoint_location}"
    )

    return model