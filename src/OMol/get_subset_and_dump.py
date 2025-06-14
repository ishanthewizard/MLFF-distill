import sys
sys.path.append("../..")  # Adjust path to import fairchem
sys.path.append("./")  # Adjust path to import src_v2
from src.OMol.datasets import AseDBDataset_io as AseDBDataset
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Subset
from IPython.display import Image, display
import ase
import os


def open_shard(idx: int):
    fname = f"data{idx:04d}.aselmdb"
    path  = os.path.join(dst_dir, fname)
    return ase.db.connect(path, use_lock_file=False)


if __name__ == "__main__":
    # ── CONFIG ────────────────────────────────────────────────────────────────────
    dataset_group_name = "train"  # e.g. "train_4M", "val", "train_100K"
    subset_name = "toy"  # e.g. "protein_ligand_pockets", "protein_ligand_fragments", "protein_nucleic_acids", "protein_core", "protein_interface", "ML-MD"
    unique_src_path = "/home/yuejian/project/MLFF-distill/OMol_subset/"+dataset_group_name+"/biomolecules/unique_source_str.pt"
    subset_indices_path = "/home/yuejian/project/MLFF-distill/OMol_subset/"+dataset_group_name+"/subsets_indices_dict.pt"
    dataset_path = "/data/yuejian/OMol_copy/"+dataset_group_name
    
    dst_dir     = "/data/yuejian/OMol_subset/"+subset_name+"_"+dataset_group_name
    shard_size  = 50000
    # map_size    = 2**40  # adjust if you hit LMDB map-size limits
    # ───────────────────────────────────────────────────────────────────────────────
    
    subset_name_dictionary = {
        "protein_ligand_pockets": ["pdb_pockets_300K", "pdb_pockets_400K"],
        "protein_ligand_fragments": ["pdb_fragments_300K", "pdb_fragments_400K"],
        "protein_nucleic_acids": ["dna", "rna","nakb"], # nakb refers to Nucleic Acid Knowledge Base dataset
        "protein_core":["protein_core"],
        "protein_interface":["protein_interface"],
        "ML-MD":["omol","ml_protein_interface"]
    # "ML-MD":["omol/solvated_protein","omol/torsion_profiles","ml_protein_interface"],# could be wrong?
    }
    subset_indices = {
        "protein_ligand_pockets": [],
        "protein_ligand_fragments": [],
        "protein_nucleic_acids": [],
        "protein_core": [],
        "protein_interface": [],
        "ML-MD": []
    }


    # loading source str list
    unique_src = torch.load(unique_src_path)


    # filtering out the string
    first_1 = []
    first_2 = []
    print("Filtering out the source strings...")
    for dir in tqdm(unique_src):
        parts = dir.split('/')
        # print(parts)
        first_1.append(parts[:1])  # Get the first part of the path
        first_2.append(parts[:2])  # Get the last part of the path
        # break
    first_1 = np.array(first_1)
    first_2 = np.array(first_2)

    # for pdb fragments
    print("gathering indices for the subset from parts")
    part = []
    for keyname in subset_name_dictionary[subset_name]:
        part+=np.where(first_1[:,0] == keyname)[0].tolist()
    # part_1 = np.where(first_1[:,0] == "pdb_fragments_300K")[0].tolist()
    # part_2 = np.where(first_1[:,0] == "pdb_fragments_400K")[0].tolist()
    protein_ligand_fragments_indices = part

    # load indices list
    print("Loading indices from the subset_indices_path...")
    print("can take a while...")
    indices  = torch.load(subset_indices_path)
    # load dataset

    a2g_args = {  
        "molecule_cell_size": 120.0,
        "r_energy": True,
        "r_forces": True,
        # "r_stress": True,
        "r_data_keys": [ 'spin','charge', "data_id"],
        # 'sid': 'data_id',

    }

    select_args = {
        "filter": lambda row: row._data.get("data_id") == "orbnet_denali"
    }
    dataset = AseDBDataset({
                            "src": dataset_path,
                            "a2g_args": a2g_args,
                            # "select_args": select_args,
                            })

    protein_ligand_fragments_indices_in_train_4M = [ indices['biomolecules'][i] for i in protein_ligand_fragments_indices ]

    # sanity check
    print("sanity check: checking if the source names match the subset name")
    # breakpoint()
    for i in tqdm(protein_ligand_fragments_indices_in_train_4M):
        assert (dataset[i][0].source[0].split("/")[0] in subset_name_dictionary[subset_name]) == True, f"Unexpected source name: {dataset[i][0].source[0].split('/')[0]}"
    
    # breakpoint()
    os.makedirs(dst_dir, exist_ok=True)

    # initialize
    shard_idx   = 0
    entry_count = 0
    dst_db      = open_shard(shard_idx)
    print(f"Writing {len(protein_ligand_fragments_indices_in_train_4M)} entries to {dst_dir} in shards of size {shard_size}...")
    for id in tqdm(protein_ligand_fragments_indices_in_train_4M, total=len(protein_ligand_fragments_indices_in_train_4M)):
        _,row = dataset[id]
        # rotate shard if needed
        if entry_count and entry_count % shard_size == 0:
            dst_db.close()
            shard_idx  += 1
            dst_db      = open_shard(shard_idx)

        # extract atoms + metadata
        atoms = row.toatoms()
        data  = dict(row.data) if hasattr(row, 'data') else {}
        key   = getattr(row, 'key', None)

        # write (omit key if None)
        if key is not None:
            dst_db.write(atoms, key=key, data=data)
        else:
            dst_db.write(atoms, data=data)

        entry_count += 1

    # final close
    dst_db.close()
    print(f"Wrote {shard_idx+1} shard(s), up to {shard_size} entries each, into {dst_dir}")