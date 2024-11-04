"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
import time
import numpy as np

from src.distill_utils import get_energy_jac_loss
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from fairchem.core import __version__
from fairchem.core.common import distutils 
from fairchem.core.common.data_parallel import  OCPCollater
from fairchem.core.common.registry import registry


from fairchem.core.trainers.ocp_trainer import OCPTrainer
from . import get_jacobian, get_force_jac_loss, print_cuda_memory_usage, get_teacher_jacobian, get_atomic_number_hashes
from . import CombinedDataset, SimpleDataset
from fairchem.core.common import distutils
if TYPE_CHECKING:
    from collections.abc import Sequence


@registry.register_trainer("distill")
class DistillTrainer(OCPTrainer):
    def __init__(
        self,
        task,
        model,
        outputs,
        dataset,
        optimizer,
        loss_functions,
        evaluation_metrics,
        identifier,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        print_every=100,
        seed=None,
        logger="wandb",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm=None,
        noddp=False,
        name="ocp",
        gp_gpus=None,
    ):
        if slurm is None:
            slurm = {}
        super().__init__(
            task=task,
            model=model,
            outputs=outputs,
            dataset=dataset,
            optimizer=optimizer,
            loss_functions=loss_functions,
            evaluation_metrics=evaluation_metrics,
            identifier=identifier,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            slurm=slurm,
            noddp=noddp,
            name=name,
            gp_gpus=gp_gpus,
        )
        
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

        self.force_jac_epoch_start = self.config['optim'].get('force_jac_epoch_start', 0)
    
    def calculate_teacher_loss(self):
        self.teacher_force_mae = 0
        for datapoint in tqdm(self.val_dataset):
            true_label = datapoint['forces']
            # if 'forces' in self.normalizers:
            #     true_label = self.normalizers['forces'].norm(true_label)
            self.teacher_force_mae += torch.abs(datapoint['teacher_forces'] - true_label).mean().item()
        self.teacher_force_mae /= len(self.val_dataset)
        print("TEACHER FORCE MAE:", self.teacher_force_mae)
        # initialize hash map to store running average of force jacobian loss per system
        self.force_jac_loss_dict = {}
        
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
        
        self.train_dataset = self.insert_teach_datasets(self.train_dataset, 'train', train_indxs ) # ADDED LINE

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
        
        teacher_force_dataset = SimpleDataset(os.path.join(labels_folder,  f'{dataset_type}_forces'  ))
        final_node_feature_dataset = SimpleDataset(os.path.join(labels_folder, f'{dataset_type}_final_node_features'))
        if indxs is not None:
            teacher_force_dataset = Subset(teacher_force_dataset, torch.tensor(indxs))
            final_node_feature_dataset = Subset(final_node_feature_dataset, torch.tensor(indxs))
        if dataset_type == 'train':
            force_jac_dataset = SimpleDataset(os.path.join(labels_folder, 'force_jacobians'))
            if indxs is not None:
                force_jac_dataset = Subset(force_jac_dataset, torch.tensor(indxs))
        else: 
            force_jac_dataset = None
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
        if not self.is_validating:
            batch.pos.requires_grad_(True)
        return super()._forward(batch)

    def validate(self, split: str = "val", disable_tqdm: bool = False):
        self.is_validating =True 

        total_rate =  self.step / (time.time() - self.start_time)  * 60
        print("TOTAL RATE:", total_rate, "iter /min " )

        val_metrics = super().validate(split, disable_tqdm)

        self.force_mae = val_metrics['forces_mae']['metric']

        self.is_validating = False
        return val_metrics

    def _compute_loss(self, out, batch):
        if self.is_validating:
            return torch.tensor([0])
        batch_size = batch.natoms.numel()
        fixed = batch.fixed
        mask = fixed == 0
        should_mask = self.output_targets['forces']["train_on_free_atoms"]
        self.update_loss_coefficients()
        
        loss = [super()._compute_loss(out, batch)]
        batch['force_jac_loss'] = torch.tensor(0)
        
        
        # # NEW!!! Energy stuff to see if this even works. if it works we'll make it efficient 
        # energy_jac_loss = get_energy_jac_loss(
        #     out=out,
        #     batch=batch,
        #     energy_std = self.normalizers['energy'].std
        # )
        # loss.append(15* energy_jac_loss)
        
        if self.force_jac_loss_fn[1]['coefficient'] !=0 and self.epoch >= self.force_jac_epoch_start:
            force_jac_loss, per_sample_loss, sampled_hessian_idxs = get_force_jac_loss(
                out=out, 
                batch=batch, 
                num_samples=self.config['optim']['force_jac_sample_size'], 
                mask= mask, 
                should_mask=should_mask, 
                force_normalizer=self.normalizers['forces'],
                finite_differences= self.config['optim'].get('finite_differences', False),
                looped=(not self.config['optim']["vectorize_jacs"]),
                force_jac_hash_map=self.force_jac_loss_dict,
                hard_mining=self.config['optim'].get('hard_mining', False),
                hard_mining_visited_threshold=self.config['optim'].get('hard_mining_visited_threshold', 0.1),
                hard_mining_temperature=self.config['optim'].get('hard_mining_temperature', 0.1),
                collater = self.ocp_collater,
                forward = self._forward
            )
            if self.config['optim'].get("print_memory_usage", False):
                print_cuda_memory_usage()
            
            epochs_since_hard_mining_start = int(self.epoch - self.config['optim'].get("hard_mining_epoch_start", 0))
            if epochs_since_hard_mining_start  % self.config['optim'].get("hard_mining_epoch_reset_frequency", 50) == 0 and epochs_since_hard_mining_start > 0:
                # reset hash map
                self.force_jac_loss_dict = {}
            if self.config['optim'].get("hard_mining", False) and self.epoch >= self.config['optim'].get("hard_mining_epoch_start", 0):
                atomic_number_hashes = get_atomic_number_hashes(batch)
                for atomic_number_hash, hessian_idx, _loss in zip(atomic_number_hashes, sampled_hessian_idxs, per_sample_loss):
                    # TODO: the loss is a mean over the sampled rows, so it's kind of weird for s !=1
                    _loss = _loss.detach()
                    if atomic_number_hash not in self.force_jac_loss_dict:                        
                        count = torch.zeros((len(atomic_number_hash), 3)).to(self.device)
                        data = torch.zeros((len(atomic_number_hash), 3)).to(self.device)
                        count[hessian_idx[:, 0], hessian_idx[:, 1]] = 1
                        data[hessian_idx[:, 0], hessian_idx[:, 1]] = _loss
                    else:
                        count, data = self.force_jac_loss_dict[atomic_number_hash]
                        
                        if self.config['optim'].get('hard_mining_ema', False):
                            # Update the EMA for the specific Hessian indices
                            beta = self.config['optim'].get('hard_mining_ema_beta', 0.9)
                            if count[hessian_idx[:, 0], hessian_idx[:, 1]] == 0:
                                data[hessian_idx[:, 0], hessian_idx[:, 1]] = _loss
                            else:     
                                data[hessian_idx[:, 0], hessian_idx[:, 1]] = beta * data[hessian_idx[:, 0], hessian_idx[:, 1]] + (1 - beta) * _loss
                        else:
                            data[hessian_idx[:, 0], hessian_idx[:, 1]] = (count[hessian_idx[:, 0], hessian_idx[:, 1]] * data[hessian_idx[:, 0], hessian_idx[:, 1]] + _loss) / (count[hessian_idx[:, 0], hessian_idx[:, 1]] + 1)
                        count[hessian_idx[:, 0], hessian_idx[:, 1]] += 1
                    self.force_jac_loss_dict[atomic_number_hash] = (count, data)


            loss.append(force_jac_loss * self.force_jac_loss_fn[1]['coefficient'])
            batch['force_jac_loss'] = force_jac_loss
        if self.teacher_force_loss_fn[1]['coefficient'] != 0:
            batch_size = batch.natoms.numel()
            natoms = torch.repeat_interleave(batch.natoms, batch.natoms)
            
            mult = self.teacher_force_loss_fn[1]["coefficient"]
            curr_loss = self.teacher_force_loss_fn[1]["fn"](
                        out['forces'],
                        self.normalizers['forces'].norm(batch['teacher_forces']),
                        natoms=natoms,
                        batch_size=batch_size,
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
        metrics = super()._compute_metrics(out, batch, evaluator, metrics)
        if not self.is_validating:
            avg_force_jac_loss = distutils.all_reduce(batch["force_jac_loss"], average=True )
            metrics['force_jac_loss'] = {}
            metrics['force_jac_loss']['metric'] = avg_force_jac_loss.item()
            metrics['force_jac_loss']['total'] = avg_force_jac_loss.item()
        return metrics