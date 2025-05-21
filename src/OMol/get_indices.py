import os
from fairchem.core.datasets import AseDBDataset
from tqdm import tqdm
import torch



if __name__ == "__main__":
    
    # arguments
    dataset_path = "/data/yuejian/OMol_Whole/neutral_train/"
    dump_path = "/home/yuejian/project/MLFF-distill/OMol_subset/train_neutral"
    
    
    a2g_args = {  
        "molecule_cell_size": 120.0,
        "r_energy": True,
        "r_forces": True,
        "r_data_keys": ['spin', 'charge', "data_id"],
    }
    
    dataset = AseDBDataset({
                        "src": dataset_path,
                        "a2g_args": a2g_args,
                        })
    
    indices_dict = {}
    for i,data in tqdm(enumerate(dataset),total=len(dataset)):
        data_id = data.data_id
        if data_id not in indices_dict:
            indices_dict[data_id] = [i]
        else:
            indices_dict[data_id].append(i)
    
    
    torch.save(indices_dict, os.path.join(dump_path, "subsets_indices_dict.pt"))