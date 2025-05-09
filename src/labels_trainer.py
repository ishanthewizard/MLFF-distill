"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import datetime
import errno
import logging
import os
import random
from abc import ABC, abstractmethod
from itertools import chain
import sys
from typing import TYPE_CHECKING
import time
from fairchem.core.common.utils import load_config
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import lmdb
import copy
from torch_geometric.data import Data, Batch

from fairchem.core import __version__
from fairchem.core.common import distutils
from fairchem.core.common.registry import registry
from fairchem.core.trainers.ocp_trainer import OCPTrainer
from . import get_jacobian, get_force_jac_loss, print_cuda_memory_usage, get_teacher_jacobian
from . import CombinedDataset, SimpleDataset
from fairchem.core.common import distutils
if TYPE_CHECKING:
    from collections.abc import Sequence


# Set the GPU device (if needed)
device = torch.device("cuda:0")

@registry.register_trainer("labels")
class LabelsTrainer(OCPTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(len(self.train_dataset))
        print(len(self.val_dataset))
        print(self.train_dataset[0])
        # sys.exit(0)
        self.load_teacher_model_and_record(self.config['dataset']['teacher_labels_folder'])
        sys.exit(0)
    
    
    def load_teacher_model_and_record(self, labels_folder):
        model_attributes_holder = self.config['model']

        checkpoint = torch.load(self.config["dataset"]["teacher_checkpoint_path"], map_location=torch.device("cpu"))
        self.teacher_config = checkpoint["config"]

        if self.config["dataset"].get("teacher_scale_file", None):
            self.teacher_config["model_attributes"]["scale_file"] = self.config["dataset"]["teacher_scale_file"]

        self.config['model'] = self.teacher_config['model']
        #Load teacher config from teacher checkpoint
        self.config['model'] =  self.teacher_config['model']
        self.config['model'].pop('scale_file', None)
        self.normalizers = {}  # This SHOULD be okay since it gets overridden later (in tasks, after datasets), but double check
        
        self.load_task()
        self.load_model()
        # reload ema to prevent wrongful initialization
        self.load_extras()
        print("len of model parameters", len(list(self.model.parameters())))
        print("len of ema parameters", len(list(self.ema.shadow_params)))

        self.load_checkpoint(self.config["dataset"]["teacher_checkpoint_path"])
        
        os.makedirs(os.path.join(labels_folder, "train_forces"),exist_ok=True)
        os.makedirs(os.path.join(labels_folder, "val_forces"),exist_ok=True)
        os.makedirs(os.path.join(labels_folder, "force_jacobians"),exist_ok=True)
        
        self.launch_record_tasks(labels_folder, 'train')
        
        self.launch_record_tasks(labels_folder, 'val')

        self.config['model'] = model_attributes_holder

    def launch_record_tasks(self, labels_folder, dataset_type):
        """
        This function launches the recording tasks across distributed workers without
        the need for manually spawning processes since they are already distributed.
        """
        rank = distutils.get_rank()
        world_size = distutils.get_world_size()

        dataset_size = len(self.train_dataset if dataset_type == 'train' else self.val_dataset)
        indices_per_worker = dataset_size // world_size
        start_idx = rank * indices_per_worker
        end_idx = start_idx + indices_per_worker if rank < world_size - 1 else dataset_size

        # Each worker calls the record_labels_parallel function with its assigned indices
        self.record_labels_parallel(labels_folder, dataset_type, start_idx, end_idx)
    
    def record_labels_parallel(self, labels_folder, dataset_type, start_idx, end_idx):
        """
        This function records labels and saves them in separate LMDB files for parallel processing.
        Each worker handles its segment of the dataset.
        """
        # print("\n\n\n\n\n getting here")
        # Subset the datasets based on the provided indices
        if dataset_type == 'train':
            dataset = self.train_dataset
        elif dataset_type == 'val':
            dataset = self.val_dataset
        else:
            raise ValueError("Invalid dataset type provided")

        # Indices for the subset this process will handle
        subset_indices = list(range(start_idx, end_idx))
        subset_dataset = Subset(dataset, subset_indices)
        dataloader = DataLoader(
            subset_dataset,
            collate_fn=self.collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            batch_size=self.config["dataset"]["label_force_batch_size"],
        )
        
        # Define LMDB file path for this particular worker
        lmdb_path = os.path.join(labels_folder, f"{dataset_type}_forces", f"data.{distutils.get_rank():04d}.lmdb")

        # Function to calculate forces
        def get_seperated_forces(batch):
            # with torch.no_grad():
            all_forces = self._forward(batch)['forces']
            natoms = batch.natoms
            return [all_forces[sum(natoms[:i]):sum(natoms[:i+1])] for i in range(len(natoms))]
        
        # Record and save the data
        # self.record_and_save(dataloader, lmdb_path, get_seperated_forces)

        if dataset_type == 'train':
            # Only for training dataset, save jacobians as well
            jac_dataloader = DataLoader(
            subset_dataset,
            collate_fn=self.collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            batch_size=self.config["dataset"]["label_jac_batch_size"],
            )
            
            jac_lmdb_path = os.path.join(labels_folder, "force_jacobians", f"data.{distutils.get_rank():04d}.lmdb")
            should_mask = self.output_targets['forces']["train_on_free_atoms"]
            def get_seperated_force_jacs(batch): 
                batch.pos.detach().requires_grad_()
                # print(self._forward,self.collater)
                jacs = get_teacher_jacobian(
                                            self._forward(batch)['forces'], 
                                            batch, vectorize=self.config["dataset"]["vectorize_teach_jacs"], 
                                            should_mask=should_mask,
                                            finite_differences=True,
                                            forward = self._forward,
                                            collater=self.collater,
                                            )
                return jacs
            self.record_and_save(jac_dataloader, jac_lmdb_path, get_seperated_force_jacs)

    def record_and_save(self, dataloader, file_path, fn):
        # Assuming train_loader is your DataLoader
        data0 = next(iter(dataloader))
        map_size= 1099511627776 * 2

        env = lmdb.open(file_path, map_size=map_size)
        env_info = env.info()
        
        # Choose the appropriate context manager based on world size
        no_sync_context = self.model.no_sync() if distutils.get_world_size() > 1 else contextlib.nullcontext()
        
        # t = next(iter(dataloader))
        # print("t", t)
        # large = Batch.from_data_list([t]*220)
        # large.cell = t.cell
        # print("large", large)
        # sys.exit()
        id = 0
        with no_sync_context:
            for batch in tqdm(dataloader):
                with env.begin(write=True) as txn:
                    # batch = large
                    batch_ids = [str(i+id) for i in range(len(batch))]
                    id = int(batch_ids[-1]) + 1  
                    batch_output = fn(batch)
                    for i in range(len(batch_ids)):
                        txn.put(batch_ids[i].encode(), batch_output[i].detach().cpu().numpy().tobytes())
        env.close()
        logging.info(f"All tensors saved to LMDB:{file_path}")
