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

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_dataset(
    src_path: str,
    dst_path: str,
    overwrite: bool = False
):
    """
    Convert an LMDB dataset to an ASE LMDB database (.aselmdb) compatible with AseDBDataset.

    Args:
        src_path (str): Path to the source LMDB dataset directory.
        dst_path (str): Path to the destination .aselmdb file.
        overwrite (bool): If True, overwrite the destination file if it exists.
    """
    # Ensure source path exists
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source path does not exist: {src_path}")

    # Check if destination file exists
    if os.path.exists(dst_path) and not overwrite:
        raise FileExistsError(f"Destination file already exists: {dst_path}. Set overwrite=True to replace.")

    # Connect to the destination ASE database
    try:
        dst_db = connect(dst_path, use_lock_file=False)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to ASE database at {dst_path}: {e}")


    logger.info(f"Processing dataset from {src_path} to {dst_path}")

    # Iterate through the dataset and convert to ASE Atoms
    for atoms in read(src_path, index=":"):
        # Extract data from the sample (assuming Fairchem LMDB format)
        # Store energy and metadata in atoms.info
        # Write to ASE database
        dst_db.write(
            atoms=atoms,
            data=None, # atoms.info,  # Store metadata in ASE's data field
        )


    # Cleanup
    if hasattr(dst_db, 'close'):
        dst_db.close()

    logger.info(f"ASE LMDB database saved at: {dst_path}")

def main():
    # Configuration
    src_path = "tester_data/solvated/xyz/test.xyz"
    dst_path = "tester_data/solvated/test/data.aselmdb"
    
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    try:
        process_dataset(
            src_path=src_path,
            dst_path=dst_path,
            overwrite=False
        )
    except Exception as e:
        logger.error(f"Failed to process dataset: {e}")
        raise

if __name__ == "__main__":
    main()