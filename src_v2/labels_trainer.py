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

from src_v2.dataset_utils import load_existing_indices
import torch
from fairchem.core.common import distutils
from fairchem.core.components.runner import Runner
from .distill_utils import get_teacher_jacobian

if TYPE_CHECKING:
    from torch.distributed.checkpoint.stateful import Stateful
    from torchtnt.framework import EvalUnit, TrainUnit
    
os.umask(0o022)  # Results in 755 for dirs, 644 for files


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
        label_folder: str,
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
        self.record_labels_parallel(self.label_folder, 'val')
        self.record_labels_parallel(self.label_folder, 'train')
        
        
        # Synchronize all workers before merging
        distutils.synchronize()
        
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

    def _get_label_fn(self, dataset_type):
        if dataset_type == 'train':
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
            return get_separated_force_jacs
        if dataset_type == 'val':
            def get_separated_forces(batch):
                all_forces = self.train_eval_unit.model(batch)['forces']['forces']
                natoms = batch.natoms
                return [all_forces[sum(natoms[:i]):sum(natoms[:i+1])] for i in range(len(natoms))]
            return get_separated_forces
    
        
    
    def record_labels_parallel(self, labels_folder: str, dataset_type: str) -> None:
        # pick dataloader & get_fn exactly as before…
        if dataset_type == 'val':
            dataloader = self.eval_dataloader
            subdir = "unmerged_val_forces"
        else:  # 'train'
            dataloader = self.train_dataloader
            subdir = "unmerged_force_jacobians"
            
        get_fn = self._get_label_fn(dataset_type)

        shard_dir = os.path.join(labels_folder, subdir)
        os.makedirs(shard_dir, exist_ok=True)
        shard_name = f"data.{distutils.get_rank():04d}.lmdb"
        shard_path = os.path.join(shard_dir, shard_name)

        # 1) gather *all* seen indices from every shard in shard_dir
        distutils.synchronize()
        seen = set()
        for name in os.listdir(shard_dir):
            if name.endswith(".lmdb"):
                path = os.path.join(shard_dir, name)
                seen |= load_existing_indices(path)

        # 2) record only the *new* ones
        self._record_and_save_filtered(
            dataloader=dataloader,
            file_path=shard_path,
            fn=get_fn,
            skip_indices=seen,
        )

    def _record_and_save_filtered(
        self,
        dataloader: torch.utils.data.DataLoader,
        file_path: str,
        fn: Callable,
        skip_indices: set[int],
    ) -> None:
        """
        Iterate through dataloader.batch_sampler to get the *original* indices,
        drop any already in skip_indices, collate the remaining samples into
        a mini-batch, run fn(mini_batch) once, and write out.
        """
        map_size = 2 * 1099511627776  # 2 TB
        env = lmdb.open(file_path, map_size=map_size, create=True)

        # for multi-GPU no_sync
        sync_ctx = (
            self.train_eval_unit.model.no_sync()
            if distutils.get_world_size() > 1
            else contextlib.nullcontext()
        )

        batch_iter = iter(dataloader.batch_sampler)
        collate_fn = getattr(dataloader, 'collate_fn')
        dataset = dataloader.dataset

        with sync_ctx:
            for _ in tqdm(range(len(dataloader)), desc=f"shard {file_path}"):
                # full_batch = next(dataloader)            # you can drop this if you rebuild from samples
                indices = next(batch_iter)               # list of ints

                # figure out which ones are truly new
                new_idxs = [i for i in indices if i not in skip_indices]
                if not new_idxs:
                    continue

                # build a tiny batch of only those samples
                samples = [dataset[i] for i in new_idxs]
                mini_batch = collate_fn(samples)
                mini_batch = mini_batch.to(self.device) if hasattr(mini_batch, "to") else mini_batch

                # expensive forward call only on new samples
                outs = fn(mini_batch)

                # write each result under its original key
                with env.begin(write=True) as txn:
                    for idx, out in zip(new_idxs, outs):
                        txn.put(str(idx).encode(),
                                out.detach().cpu().numpy().tobytes())
                        skip_indices.add(idx)

        env.close()
        logging.info(f"Finished writing new labels to {file_path}")

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