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

from fairchem.core import __version__
from fairchem.core.common import distutils
from fairchem.core.common.registry import registry
from fairchem.core.trainers.ocp_trainer import OCPTrainer
from . import get_jacobian, get_force_jac_loss, print_cuda_memory_usage, get_teacher_jacobian
from . import CombinedDataset, SimpleDataset
from fairchem.core.common import distutils
if TYPE_CHECKING:
    from collections.abc import Sequence


@registry.register_trainer("labels")
class LabelsTrainer(OCPTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Compute teacher MAE
        self.load_teacher_model_and_record(self.config['dataset']['teacher_labels_folder'])
        sys.exit(0)
    
    
    def load_teacher_model_and_record(self, labels_folder):
        model_attributes_holder = self.config['model']

        checkpoint = torch.load(self.config["dataset"]["teacher_checkpoint_path"], map_location=torch.device("cpu"))
        # print("keys in checkpoint", checkpoint.keys())
        # print("keys in ema", checkpoint['ema'].keys())
        # print("keys in state", checkpoint['state_dict'].keys())
        # print("len of state", len(checkpoint['state_dict']))
        # print("shadow ema",checkpoint['ema']["shadow_params"], len(list(checkpoint['ema']["shadow_params"])))
        # print("shadow state", checkpoint['state_dict']["shadow_params"],len(list(checkpoint['state_dict']["shadow_params"])))
        # sys.exit(0)
        self.teacher_config = checkpoint["config"]
        # print("Teacher config loaded from checkpoint",self.teacher_config)
        # sys.exit()
        # if self.teacher_config['model'].endswith('EfficientGraphAttentionPotential'):
        #     self.teacher_config['model_attributes']['atten_name'] = 'scaled_dot_product'
        if self.config["dataset"].get("teacher_scale_file", None):
            self.teacher_config["model_attributes"]["scale_file"] = self.config["dataset"]["teacher_scale_file"]

        self.config['model'] = self.teacher_config['model']
        #Load teacher config from teacher checkpoint
        self.config['model'] =  self.teacher_config['model']
        self.config['model'].pop('scale_file', None)
        self.normalizers = {}  # This SHOULD be okay since it gets overridden later (in tasks, after datasets), but double check
        
        # test
        # model_config_copy = copy.deepcopy(self.config["model"])
        # print("this is the preloaded model config", model_config_copy)
        
        # model_name = model_config_copy.pop("name")
        # self.model = registry.get_model_class(model_name)(
        #     **model_config_copy,
        # ).to(self.device)
        # print("len of model parameters", len(list(self.model.parameters())))
        
        
        
        self.load_task()
        self.load_model()
        # reload ema to prevent wrongful initialization
        self.load_extras()
        print("len of model parameters", len(list(self.model.parameters())))
        print("len of ema parameters", len(list(self.ema.shadow_params)))
        # sys.exit(0)
        self.load_checkpoint(self.config["dataset"]["teacher_checkpoint_path"])
        
        os.makedirs(os.path.join(labels_folder, "train_forces"),exist_ok=True)
        os.makedirs(os.path.join(labels_folder, "val_forces"),exist_ok=True)
        os.makedirs(os.path.join(labels_folder, "force_jacobians"),exist_ok=True)
        
        self.launch_record_tasks(labels_folder, 'train')
        # sys.exit()
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
            all_forces = self._forward(batch)['forces']
            natoms = batch.natoms
            return [all_forces[sum(natoms[:i]):sum(natoms[:i+1])] for i in range(len(natoms))]
        
        # Record and save the data
        self.record_and_save(dataloader, lmdb_path, get_seperated_forces)

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
                batch.pos.requires_grad_(True)
                return get_teacher_jacobian(self._forward(batch)['forces'], batch, vectorize=self.config["dataset"]["vectorize_teach_jacs"], should_mask=should_mask)
            self.record_and_save(jac_dataloader, jac_lmdb_path, get_seperated_force_jacs)

    def record_and_save(self, dataloader, file_path, fn):
        # Assuming train_loader is your DataLoader
        data0 = next(iter(dataloader))
        map_size= 1099511627776 * 2

        env = lmdb.open(file_path, map_size=map_size)
        env_info = env.info()
        
        # Choose the appropriate context manager based on world size
        no_sync_context = self.model.no_sync() if distutils.get_world_size() > 1 else contextlib.nullcontext()

        with no_sync_context:
            for batch in tqdm(dataloader):
                with env.begin(write=True) as txn:
                    batch_ids = [str(int(i)) for i in batch.id]
                    batch_output = fn(batch)
                    for i in range(len(batch_ids)):
                        txn.put(batch_ids[i].encode(), batch_output[i].detach().cpu().numpy().tobytes())

        env.close()
        logging.info(f"All tensors saved to LMDB:{file_path}")
