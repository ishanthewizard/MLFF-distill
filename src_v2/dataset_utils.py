import re
from fairchem.core.units.mlip_unit.mlip_unit import load_inference_model
from fairchem.core.common import registry
from copy import deepcopy
import torch
import logging
import os
from tqdm import tqdm
import  lmdb
import shutil


def initialize_finetuning_model(
    checkpoint_location: str, use_ema: True, overrides: dict | None = None, heads: dict | None = None
) -> torch.nn.Module:
    model, _ = load_inference_model(checkpoint_location, overrides, use_ema=use_ema)

    logging.warning(
        f"initialize_finetuning_model starting from checkpoint_location: {checkpoint_location}"
    )
    return model

def load_existing_indices(lmdb_path: str) -> set[int]:
    if not os.path.isdir(lmdb_path):
        return set()
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        idxs = set()
        with env.begin() as txn:
            for k, _ in txn.cursor():
                idxs.add(int(k.decode()))  # Decode bytes to string, then to int
        env.close()
        return idxs
    except lmdb.Error as e:
        logging.warning(f"Error opening LMDB at {lmdb_path}: {e}")
        return set()
