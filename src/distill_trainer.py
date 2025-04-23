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
import numpy as np

from src.distill_utils import get_energy_jac_loss
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from fairchem.core import __version__
from fairchem.core.common import distutils 
from fairchem.core.common.data_parallel import OCPCollater
from fairchem.core.common.registry import registry


from fairchem.core.trainers.ocp_trainer import OCPTrainer
from . import get_jacobian, get_force_jac_loss, print_cuda_memory_usage, get_teacher_jacobian
from . import CombinedDataset, SimpleDataset
from fairchem.core.common import distutils
if TYPE_CHECKING:
    from collections.abc import Sequence


@registry.register_trainer("distill")
class DistillTrainer(OCPTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_validating = False
        self.force_mae =  None
        self.modify_train_val_datasets()
        self.start_time = time.time()

        # Compute teacher MAE
        self.calculate_teacher_loss()
        self.force_jac_loss_fn = self.loss_functions.pop()
        self.teacher_force_loss_fn = self.loss_functions.pop()
        assert self.force_jac_loss_fn[0] == 'force_jacs'
        assert self.teacher_force_loss_fn[0] == 'teacher_forces' 
        self.original_fjac_coeff = self.force_jac_loss_fn[1]['coefficient']
    
    def calculate_teacher_loss(self):
        # print("\n\n\n\n\n")
        # print("TEACHER LOSS")
        # print("\n\n\n\n")
        self.teacher_force_mae = 0
        for datapoint in tqdm(self.val_dataset):
            true_label = datapoint['forces'].to(self.device)
            if 'forces' in self.normalizers:
                true_label = self.normalizers['forces'].norm(true_label)
            self.teacher_force_mae += torch.abs(datapoint['teacher_forces'].to(self.device) - true_label).mean().item()
        self.teacher_force_mae /= len(self.val_dataset)
        print("TEACHER FORCE MAE:", self.teacher_force_mae)
        
    def modify_train_val_datasets(self):
        train_indxs = val_indxs = None
        if "split" in self.config["dataset"]:
            train_indxs = np.random.default_rng(seed=0).choice(
                    len(self.train_dataset), 
                    self.config["dataset"]["split"], 
                    replace=False
            )
        if "split" in self.config["val_dataset"]:
                    # to make sampling deterministic, seed rng
            val_indxs = np.random.default_rng(seed=0).choice(
                len(self.val_dataset), 
                self.config["val_dataset"]["split"], 
                replace=False
            )
        # print("before",next(iter(self.train_dataset)))
        self.train_dataset = self.insert_teach_datasets(self.train_dataset, 'train', train_indxs ) # ADDED LINE
        # print("after",next(iter(self.train_dataset)))
        self.train_sampler = self.get_sampler(
            self.train_dataset,
            self.config["optim"]["batch_size"],
            shuffle=True,
        )
        self.train_loader = self.get_dataloader(
            self.train_dataset,
            self.train_sampler,
        )

        self.val_dataset = self.insert_teach_datasets(self.val_dataset, 'val', val_indxs)

        self.val_sampler = self.get_sampler(
            self.val_dataset,
            self.config["optim"].get(
                "eval_batch_size", self.config["optim"]["batch_size"]
            ),
            shuffle=False,
        )
        self.val_loader = self.get_dataloader(
            self.val_dataset,
            self.val_sampler,
        )

    def insert_teach_datasets(self, main_dataset, dataset_type, indxs=None):
        #dataset_type either equals 'train' or 'val'

        labels_folder = self.config['dataset']['teacher_labels_folder']
        # print("labeldir \n\n\n")
        # print(os.path.join(labels_folder,  f'{dataset_type}_forces'  ))
        teacher_force_dataset = SimpleDataset(os.path.join(labels_folder,  f'{dataset_type}_forces'  ))
        # print("here\n\n\n",len(teacher_force_dataset))
        if self.config['optim'].get('final_node_distill', False):
            final_node_feature_dataset = SimpleDataset(os.path.join(labels_folder, f'{dataset_type}_final_node_features' ))
        else:
            final_node_feature_dataset = None
        if indxs is not None:
            # print("\n\n\n\nINDXS:", indxs)
            teacher_force_dataset = Subset(teacher_force_dataset, torch.tensor(indxs))
            final_node_feature_dataset = Subset(final_node_feature_dataset, torch.tensor(indxs))
        if dataset_type == 'train':
            force_jac_dataset = SimpleDataset(os.path.join(labels_folder, 'force_jacobians'))
            if indxs is not None:
                force_jac_dataset = Subset(force_jac_dataset, torch.tensor(indxs))
        else: 
            force_jac_dataset = None
        # print("\n\n\n\n")
        # print(main_dataset, teacher_force_dataset, force_jac_dataset, final_node_feature_dataset)
        # print(len(main_dataset))
        # print(len(teacher_force_dataset))
        # # print(len(force_jac_dataset))
        # print(len(final_node_feature_dataset))
        return CombinedDataset(main_dataset,  teacher_force_dataset, force_jac_dataset, final_node_feature_dataset)

    def update_loss_coefficients(self):
        # self.force_mae, self.teacher_force_mae are good to go
        if self.force_mae == None:
            self.force_mae = float('inf')
        if self.step % 20 == 0:
            if self.force_mae < self.teacher_force_mae:
                logging.info("EXCEEDED TEACHER ACCURACY!!")

            percent_higher = ((self.force_mae - self.teacher_force_mae) / self.teacher_force_mae) *100
            threshold = 5
            if percent_higher < threshold: 
                self.force_jac_loss_fn[1]['coefficient'] = 0.5 * self.original_fjac_coeff
           
    def _forward(self, batch):
        print("\n\n\n\n\n")
        print("FORWARD")
        print("\n\n\n\n")
        if not self.is_validating:
            # print("here")
            batch.pos.requires_grad_(True)
        print(batch)
        res = super()._forward(batch)
        print(res)
        return res
        # return super()._forward(batch)

    def validate(self, split: str = "val", disable_tqdm: bool = False):
        print("\n\n\n\n\n")
        print("VALIDATING")
        print("\n\n\n\n")
        self.is_validating =True 

        total_rate =  self.step / (time.time() - self.start_time)  * 60
        print("TOTAL RATE:", total_rate, "iter /min " )

        val_metrics = super().validate(split, disable_tqdm)

        self.force_mae = val_metrics['forces_mae']['metric']

        self.is_validating = False
        return val_metrics

    def _compute_loss(self, out, batch):
        print("\n\n\n\n\n")
        print("COMPUTING LOSS")
        print("\n\n\n\n")
        if self.is_validating:
            return torch.tensor([0])
        batch_size = batch.natoms.numel()
        fixed = batch.fixed
        mask = fixed == 0
        should_mask = self.output_targets['forces']["train_on_free_atoms"]
        self.update_loss_coefficients()
        
        loss = [super()._compute_loss(out, batch)]
        batch['force_jac_loss'] = torch.tensor(0)
        
        
        # NEW!!! Energy stuff to see if this even works. if it works we'll make it efficient 
        
        if self.config['optim'].get('energy_distill_coeff', 0) > 0:
            energy_jac_loss = get_energy_jac_loss(
                out=out,
                batch=batch,
                energy_std = self.normalizers['energy'].rmsd
            )
            
            en_dist_coeff = self.config['optim']['energy_distill_coeff']
            loss.append(en_dist_coeff* energy_jac_loss)
        
        
        
        if self.force_jac_loss_fn[1]['coefficient'] > 0:
            force_jac_loss = get_force_jac_loss(
                out=out, 
                batch=batch, 
                num_samples=self.config['optim']['force_jac_sample_size'], 
                mask= mask, 
                should_mask=should_mask, 
                finite_differences= self.config['optim'].get('finite_differences', False),
                looped=(not self.config['optim']["vectorize_jacs"]),
                collater = self.collater,
                forward = self._forward
            )
            if self.config['optim'].get("print_memory_usage", False):
                print_cuda_memory_usage()
            loss.append(force_jac_loss * self.force_jac_loss_fn[1]['coefficient'])
            batch['force_jac_loss'] = force_jac_loss

        if self.teacher_force_loss_fn[1]['coefficient'] != 0:
            batch_size = batch.natoms.numel()
            natoms = torch.repeat_interleave(batch.natoms, batch.natoms)
            
            mult = self.teacher_force_loss_fn[1]["coefficient"]
            curr_loss = self.teacher_force_loss_fn[1]["fn"](
                        out['forces'],
                        batch['teacher_forces'],
                        natoms=natoms,
            )
            loss.append(
                mult
                * curr_loss
            )
        for lc in loss:
            if torch.any(torch.isnan(lc)):
                raise Exception("loss is nan")
            assert hasattr(lc, "grad_fn")
        return sum(loss)
    
    def _compute_metrics(self, out, batch, evaluator, metrics=None):
        print("\n\nCOMPUTING METRICS\n\n")
        # print(out)
        # print(batch)
        # print(evaluator)
        # print(metrics)
        # print("\n\n\n\n",self.output_targets)
        # del self.output_targets['final_node_features']
        print(out,"energy",batch['energy'],"forces" ,batch['forces'])
        metrics = super()._compute_metrics(out, batch, evaluator, metrics)
        if not self.is_validating and self.original_fjac_coeff > 0:
            avg_force_jac_loss = distutils.all_reduce(batch["force_jac_loss"], average=True )
            metrics['force_jac_loss'] = {}
            metrics['force_jac_loss']['metric'] = avg_force_jac_loss.item()
            metrics['force_jac_loss']['total'] = avg_force_jac_loss.item()
        return metrics