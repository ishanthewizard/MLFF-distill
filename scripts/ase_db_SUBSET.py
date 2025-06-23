import os
import numpy as np
from tqdm import tqdm
import ase
from ase import Atoms
from ase.db import connect
from fairchem.core.common.registry import registry
import logging
import torch
from ase.io import read
from fairchem.core.datasets import AseDBDataset

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_dataset(
    src_path: str,
    dst_path: str,
    subset_len: int,
    overwrite: bool = False
):
    """
    Convert an ASE LMDB dataset to an ASE LMDB database (.aselmdb) compatible with AseDBDataset.
    Args:
        src_path (str): Path to the source LMDB dataset directory.
        dst_path (str): Path to the destination .aselmdb file.
        overwrite (bool): If True, overwrite the destination file if it exists.
    """

    a2g_args = {  
        "molecule_cell_size": 120.0,
        "r_energy": True,
        "r_forces": True,
        "r_data_keys": [ 'spin','charge', "data_id"],
    }
    src_db = AseDBDataset({
                            "src": src_path,
                            "a2g_args": a2g_args,
                            # "select_args": select_args,
                            })
    
    # Check if destination file exists
    if os.path.exists(dst_path) and not overwrite:
        raise FileExistsError(f"Destination file already exists: {dst_path}. Set overwrite=True to replace.")

    # Connect to the destination ASE database
    dst_db = connect(dst_path, use_lock_file=False)


    logger.info(f"Processing dataset from {src_path} to {dst_path}")
    logger.info(f"Iterating through {subset_len} samples")

    # Iterate through the dataset and convert to ASE Atoms
    for i in tqdm(range(subset_len)):
        atoms = src_db.get_atoms(i)
        dst_db.write(
            atoms=atoms,
            data=atoms.info, # atoms.info,  # Store metadata in ASE's data field
        )
    # Cleanup
    if hasattr(dst_db, 'close'):
        dst_db.close()

    logger.info(f"ASE LMDB database saved at: {dst_path}")

def main():
    # Configuration
    src_path = "/data/ishan-amin/OMOL/4M/subsets/OMol_subset/protein_ligand_pockets_train_4M"
    # src_path = "/data/ishan-amin/OMOL/4M/subsets/OMol_subset/protein_ligand_pockets_val"
    dst_path = "/data/ishan-amin/OMOL/TOY/ligand_pocket_300/train/"
    os.makedirs(dst_path)
    dst_path = os.path.join(dst_path, 'data0000.aselmdb')
    subset_length = 300
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    try:
        process_dataset(
            src_path=src_path,
            dst_path=dst_path,
            subset_len= subset_length,
            overwrite=False
        )
    except Exception as e:
        logger.error(f"Failed to process dataset: {e}")
        raise

if __name__ == "__main__":
    main()