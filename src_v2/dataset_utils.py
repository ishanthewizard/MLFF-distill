

def get_inverse_indices(original_indices: list[int]) -> list[int]:
    """
    Given a list of indices, 
    return the inverse indices to collate the shuffled databack to its original order.
    """
    
    inverse_indices = [0] * len(original_indices)
    for new_pos, original_idx in enumerate(original_indices):
        inverse_indices[original_idx] = new_pos
        
    return inverse_indices