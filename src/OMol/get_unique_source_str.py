from fairchem.core.datasets import AseDBDataset
from tqdm import tqdm
import torch
from torch.utils.data import Subset


if __name__ == "__main__":

    dataset_name = "neutral_val"
    dataset_path = "/data/yuejian/OMol_Whole/"+dataset_name
    indices = torch.load("/home/yuejian/project/MLFF-distill/OMol_subset/"+dataset_name+"/subsets_indices_dict.pt")
    save_path = "/home/yuejian/project/MLFF-distill/OMol_subset/"+dataset_name+"/biomolecules/unique_source_str.pt"
    
    a2g_args = {  
        "molecule_cell_size": 120.0,
        "r_energy": True,
        "r_forces": True,
        # "r_stress": True,
        "r_data_keys": [ 'spin','charge', "data_id"],
        'sid': 'data_id',

    }

    select_args = {
        "filter": lambda row: row._data.get("data_id") == "orbnet_denali"
    }
    dataset = AseDBDataset({
                            "src": dataset_path,
                            "a2g_args": a2g_args,
                            # "select_args": select_args,
                            })
    
    sub_dataset = Subset(dataset, indices['biomolecules'])
    

    unique_source = []
    for i, _ in tqdm(enumerate(sub_dataset),total=len(sub_dataset)):
        # if i > 100:
        #     break
        if not sub_dataset[i].source[0] in unique_source:
            unique_source.append(sub_dataset[i].source[0])
    
    torch.save(unique_source, save_path)