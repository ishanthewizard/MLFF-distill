"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Callable
import shutil
import gc
import lmdb
from tqdm import tqdm
import contextlib
from typing import TYPE_CHECKING, Optional, Union

import torch
from fairchem.core.common import distutils
from fairchem.core.components.runner import Runner
from .distill_utils import get_teacher_jacobian

if TYPE_CHECKING:
    from torch.distributed.checkpoint.stateful import Stateful
    from torchtnt.framework import EvalUnit, TrainUnit

class TeacherLabelGenerator(Runner):
    """
    A class to generate labels for training and evaluation datasets in a distributed manner.
    Labels are stored with keys corresponding to their original dataset indices, and LMDBs
    from multiple workers are merged into a single database after processing.
    """
    def __init__(
        self,
        train_dataloader: torch.utils.data.dataloader,
        eval_dataloader: torch.utils.data.dataloader,
        eval_unit: Union[TrainUnit, EvalUnit, Stateful],
        label_folder: str = None,
    ):  
        # Initialize the class
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_eval_unit = eval_unit
        self.device = self.train_eval_unit.model.device
        
        # Check if the dataloaders are using a deterministic sampler
        if self.train_dataloader.batch_sampler.shuffle:
            raise ValueError("TrainSampler should not shuffle for deterministic indexing.")
        if self.eval_dataloader.batch_sampler.shuffle:
            raise ValueError("EvalSampler should not shuffle for deterministic indexing.")
        
        # Create the label folder if it does not exist
        self.label_folder = label_folder
        if not os.path.exists(self.label_folder):
            os.makedirs(self.label_folder, exist_ok=True)
        
        # Create subfolders for unmerged and merged data
        logging.info(f"Creating Label folder at: {self.label_folder}")
        os.makedirs(os.path.join(self.label_folder, "unmerged_val_forces"), exist_ok=True)
        os.makedirs(os.path.join(self.label_folder, "unmerged_force_jacobians"), exist_ok=True)
        os.makedirs(os.path.join(self.label_folder, "val_forces"), exist_ok=True)
        os.makedirs(os.path.join(self.label_folder, "force_jacobians"), exist_ok=True)
        
        # Log the sizes of the dataloaders
        logging.info(f"Train Dataloader size {len(self.train_dataloader)}")
        logging.info(f"Eval Dataloader size {len(self.eval_dataloader)}")

    def run(self) -> None:
        """Generate labels for train and val datasets and merge LMDBs on rank 0."""
        self.record_labels_parallel(self.label_folder, 'train')
        self.record_labels_parallel(self.label_folder, 'val')
        
        # Synchronize all workers before merging
        distutils.barrier()
        
        # Merge LMDBs on rank 0
        if distutils.get_rank() == 0:
            # Merge validation forces
            val_forces_dir = os.path.join(self.label_folder, "unmerged_val_forces")
            merged_val_forces_path = os.path.join(self.label_folder, "val_forces")
            self.merge_lmdb_shards(val_forces_dir, merged_val_forces_path)
            
            # Merge training force Jacobians
            train_jacobians_dir = os.path.join(self.label_folder, "unmerged_force_jacobians")
            merged_train_jacobians_path = os.path.join(self.label_folder, "force_jacobians")
            self.merge_lmdb_shards(train_jacobians_dir, merged_train_jacobians_path)

    def record_labels_parallel(self, labels_folder: str, dataset_type: str) -> None:
        """
        Record labels in parallel, storing them in LMDBs with original dataset indices as keys.
        """
        # Select the appropriate dataloader
        if dataset_type == 'train':
            dataloader = self.train_dataloader
        elif dataset_type == 'val':
            dataloader = self.eval_dataloader
        else:
            raise ValueError("Invalid dataset type provided")
        
        # Define LMDB file path and function based on dataset type
        if dataset_type == 'val':
            lmdb_path = os.path.join(labels_folder, "unmerged_val_forces", f"data.{distutils.get_rank():04d}.lmdb")
            def get_separated_forces(batch):
                all_forces = self.train_eval_unit.model(batch)['forces']['forces']
                natoms = batch.natoms
                return [all_forces[sum(natoms[:i]):sum(natoms[:i+1])] for i in range(len(natoms))]
            self.record_and_save(dataloader, lmdb_path, get_separated_forces)
        
        if dataset_type == 'train':
            jac_lmdb_path = os.path.join(labels_folder, "unmerged_force_jacobians", f"data.{distutils.get_rank():04d}.lmdb")
            def get_separated_force_jacs(batch): 
                batch.pos.detach().requires_grad_()
                jacs = get_teacher_jacobian(
                    batch, 
                    vectorize=False,
                    approximation="forward",
                    forward=self.train_eval_unit.model,
                    collater=None,
                    device=self.device
                )
                return jacs
            self.record_and_save(dataloader, jac_lmdb_path, get_separated_force_jacs)

    def record_and_save(self, dataloader: torch.utils.data.dataloader, file_path: str, fn: Callable) -> None:
        """Save computed labels to LMDB using original dataset indices as keys."""
        map_size = 1099511627776 * 2  # 2TB map size
        env = lmdb.open(file_path, map_size=map_size)
        
        no_sync_context = self.train_eval_unit.model.no_sync() if distutils.get_world_size() > 1 else contextlib.nullcontext()
        
        # Iterate over batch sampler to get original indices
        batch_sampler_iter = iter(dataloader.batch_sampler)
        with no_sync_context:
            for batch in tqdm(dataloader, total=len(dataloader)):
                indices = next(batch_sampler_iter)
                with env.begin(write=True) as txn:
                    batch_output = fn(batch.to(self.device))
                    for idx, output in zip(indices, batch_output):
                        txn.put(str(idx).encode(), output.detach().cpu().numpy().tobytes())
        
        env.close()
        logging.info(f"All tensors saved to LMDB: {file_path}")

    def merge_lmdb_shards(self, input_dir: str, output_path: str, map_size: int = 1099511627776 * 2) -> None:
        """
        Merge multiple LMDB shards into a single LMDB database, preserving original keys.

        Args:
            input_dir (str): Directory containing shard folders (e.g., "data.0000.lmdb").
            output_path (str): Path to the merged LMDB folder (e.g., "val_forces").
            map_size (int): Max size of the LMDB (default 2 TB).
        """
        shard_dirs = sorted([os.path.join(input_dir, d) for d in os.listdir(input_dir)
                            if d.endswith('.lmdb') and os.path.isdir(os.path.join(input_dir, d))])
        if os.path.exists(output_path):
            print(f"Removing existing LMDB at {output_path}")
            shutil.rmtree(output_path)

        os.makedirs(output_path, exist_ok=True)
        env_out = lmdb.open(os.path.join(output_path, "data.0000.lmdb"), map_size=map_size)

        total_entries = 0
        for shard in tqdm(shard_dirs, desc="Merging shards"):
            env_in = lmdb.open(shard, readonly=True, lock=False)
            with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
                cursor = txn_in.cursor()
                for key, value in cursor:
                    txn_out.put(key, value)
                    total_entries += 1
            env_in.close()

        env_out.close()
        print(f"Done. Total entries merged: {total_entries}")
        print(f"Output LMDB saved at: {os.path.join(output_path, 'data.0000.lmdb')}")

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        pass

    def load_state(self, checkpoint_location: Optional[str]) -> None:
        pass