"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import datetime
import errno
import logging
import os
import random
from abc import ABC, abstractmethod
from itertools import chain
from typing import TYPE_CHECKING
import time
from fairchem.core.common.utils import load_config
import numpy as np
import numpy.typing as npt
from src.distill_utils import custom_sigmoid
import torch
import torch.nn as nn
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import lmdb

from fairchem.core import __version__
from fairchem.core.common import distutils, gp_utils
from fairchem.core.common.data_parallel import BalancedBatchSampler, OCPCollater
from fairchem.core.common.registry import registry
from fairchem.core.common.typing import assert_is_instance as aii
from fairchem.core.common.typing import none_throws
from fairchem.core.modules.evaluator import Evaluator
from fairchem.core.modules.exponential_moving_average import ExponentialMovingAverage
from fairchem.core.modules.loss import DDPLoss
from fairchem.core.modules.normalizer import Normalizer
from fairchem.core.modules.scaling.compat import load_scales_compat
from fairchem.core.modules.scaling.util import ensure_fitted
from fairchem.core.modules.scheduler import LRScheduler
from fairchem.core.trainers.ocp_trainer import OCPTrainer
from . import get_jacobian, get_force_jac_loss, print_cuda_memory_usage, get_teacher_jacobian
from . import CombinedDataset, SimpleDataset
from fairchem.core.common import distutils
from fairchem.core.modules.loss import L2MAELoss
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
        self.force_mae =  None
        self.start_time = time.time()
        self.original_fjac_coeff = self.loss_functions[-1][1]['coefficient']
        # Compute teacher MAE
        self.teacher_force_mae = 0
        for datapoint in tqdm(self.val_dataset):
            true_label = datapoint['forces']
            if 'forces' in self.normalizers:
                true_label = self.normalizers['forces'].norm(true_label)
            self.teacher_force_mae += torch.abs(datapoint['teacher_forces'] - true_label).mean().item()
        self.teacher_force_mae /= len(self.val_dataset)
        print("TEACHER FORCE MAE:", self.teacher_force_mae)

    def record_and_save(self, dataloader, file_path, fn):
        # Assuming train_loader is your DataLoader
        data0 = next(iter(dataloader))
        avg_num_atoms = int(sum(data0.natoms) / len(data0.natoms)) # This will hopefully give a good estimate of average atom # for datasets with different sized atoms
        if file_path.endswith('force_jacobians.lmdb'):
            map_size= int((len(dataloader.dataset) * (avg_num_atoms * 3* avg_num_atoms * 3) * 4) * 2 + 1_000_000)
        else:
            map_size = int((len(dataloader.dataset) * (avg_num_atoms * 3) * 4) * 2 + 1_000_000)

        env = lmdb.open(file_path, map_size=map_size)
        env_info = env.info()
        with env.begin(write=True) as txn:
            for batch in tqdm(dataloader):
                batch_ids = [str(int(i)) for i in batch.id]
                batch_output = fn(batch)  # this function needs to output an array where each element correponds to the label for an entire molecule
                print_cuda_memory_usage()
                # Convert tensor to bytes and write to LMDB
                for i in range(len(batch_ids)):
                    txn.put(batch_ids[i].encode(), batch_output[i].detach().cpu().numpy().tobytes())
        env.close()
        logging.info(f"All tensors saved to LMDB:{file_path}")

    def record_labels(self, labels_folder):
        self.model.eval()
        get_forces = lambda data_point: self._forward(data_point)['forces']
        def get_seperated_forces(batch):
            all_forces = self._forward(batch)['forces']
            natoms = batch.natoms
            return [all_forces[sum(natoms[:i]):sum(natoms[:i+1])] for i in range(len(natoms))]
        should_mask = self.output_targets['forces']["train_on_free_atoms"]
        get_seperated_force_jacs = lambda batch: get_teacher_jacobian(self._forward(batch)['forces'], batch, vectorize=self.config["dataset"]["vectorize_teach_jacs"], should_mask=should_mask)
        # def get_seperated_force_jacs(batch):
        #     all_forces = self._forward(batch)['forces']
        #     natoms = batch.natoms
        #     jacs = get_jacobian_old(all_forces, batch.pos)
        #     return [jacs[sum(natoms[:i]):sum(natoms[:i+1]), :, sum(natoms[:i]):sum(natoms[:i+1]), :] for i in range(len(natoms))]
        temp_train_loader = self.get_dataloader(
                self.train_dataset,
                self.get_sampler(
                self.train_dataset,
                self.config["dataset"]["label_force_batch_size"],
                shuffle=False,
            )
        )
        temp_jac_loader = self.get_dataloader(
                self.train_dataset,
                self.get_sampler(
                self.train_dataset,
                self.config["dataset"]["label_jac_batch_size"],
                shuffle=False,
            )
        )
        temp_val_loader = self.get_dataloader(
                self.val_dataset,
                self.get_sampler(
                self.val_dataset,
                self.config["dataset"]["label_force_batch_size"],
                shuffle=False,
            )
        )
        self.record_and_save(temp_train_loader, os.path.join(labels_folder, 'train_forces.lmdb'), get_seperated_forces)
        self.record_and_save(temp_jac_loader, os.path.join(labels_folder, 'force_jacobians.lmdb'), get_seperated_force_jacs )
        self.record_and_save(temp_val_loader, os.path.join(labels_folder, 'val_forces.lmdb'), get_seperated_forces )

    def load_teacher_model_and_record(self, labels_folder):
        model_attributes_holder = self.config['model_attributes']
        model_name_holder = self.config['model']

        checkpoint = torch.load(self.config["dataset"]["teacher_checkpoint_path"], map_location=torch.device("cpu"))
        self.teacher_config = checkpoint["config"]

        # if self.teacher_config['model'].endswith('EfficientGraphAttentionPotential'):
        #     self.teacher_config['model_attributes']['atten_name'] = 'scaled_dot_product'
        if self.config["dataset"].get("teacher_scale_file", None):
            self.teacher_config["model_attributes"]["scale_file"] = self.config["dataset"]["teacher_scale_file"]

        self.config['model_attributes'] = self.teacher_config['model_attributes']
        #Load teacher config from teacher checkpoint
        self.config['model'] =  self.teacher_config['model']
        self.config['model_attributes'].pop('scale_file', None)
        self.normalizers = {}  # This SHOULD be okay since it gets overridden later (in tasks, after datasets), but double check
        self.load_task()
        self.load_model()
        self.load_checkpoint(self.config["dataset"]["teacher_checkpoint_path"])
        self.record_labels(labels_folder)

        self.config['model_attributes'] = model_attributes_holder
        self.config['model'] = model_name_holder

    def insert_teach_datasets(self, main_dataset, dataset_type, indxs=None):
        #dataset_type either equals 'train' or 'val'
        self.is_validating = False
        labels_folder = self.config['dataset']['teacher_labels_folder']
        needs_setup = not os.path.exists(labels_folder)
        if distutils.get_rank() == 0:
            folder_exists = torch.tensor(os.path.exists(labels_folder), dtype=torch.bool).to(self.device)
            distutils.broadcast(folder_exists, src=0)
        else:
            folder_exists = torch.tensor(False, dtype=torch.bool).to(self.device)
            distutils.broadcast(folder_exists, src=0)
        if not folder_exists.item():
            os.makedirs(labels_folder, exist_ok=True)
            self.load_teacher_model_and_record(labels_folder)
        distutils.synchronize()
            
        teacher_force_dataset = SimpleDataset(os.path.join(labels_folder,  f'{dataset_type}_forces.lmdb'  ))
        if indxs is not None:
            teacher_force_dataset = Subset(teacher_force_dataset, torch.tensor(indxs))
        if dataset_type == 'train':
            force_jac_dataset = SimpleDataset(os.path.join(labels_folder, 'force_jacobians.lmdb'))
            if indxs is not None:
                force_jac_dataset = Subset(force_jac_dataset, torch.tensor(indxs))
        else: 
            force_jac_dataset = None
        return CombinedDataset(main_dataset,  teacher_force_dataset, force_jac_dataset)

    def load_datasets(self) -> None:
        self.ocp_collater = OCPCollater(
            self.config["model_attributes"].get("otf_graph", False)
        )
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # load train, val, test datasets
        if self.config["dataset"].get("src", None):
            logging.info(
                f"Loading dataset: {self.config['dataset'].get('format', 'lmdb')}"
            )

            self.train_dataset = registry.get_dataset_class(
                self.config["dataset"].get("format", "lmdb")
            )(self.config["dataset"])
            

            #LOAD IN VAL EARLIER:
            if self.config["val_dataset"].get("use_train_settings", True):
                val_config = self.config["dataset"].copy()
                val_config.pop("split", None) # Aded
                val_config.update(self.config["val_dataset"])
            else:
                val_config = self.config["val_dataset"]

            self.val_dataset = registry.get_dataset_class(
                val_config.get("format", "lmdb")
            )(val_config)
            # END LOAD IN VAL
            # Ryan's code
            train_indxs = val_indxs = None
            if "split" in val_config:
                    logging.info(f"original size {len(self.val_dataset)}, target size {val_config['split']}")
                    # to make sampling deterministic, seed rng
                    if len(self.val_dataset) >= val_config["split"]:
                        val_indxs = np.random.default_rng(seed=0).choice(
                            len(self.val_dataset), 
                            val_config["split"], 
                            replace=False
                        )
                        self.val_dataset = Subset(self.val_dataset, torch.tensor(val_indxs))
                        logging.info("Subsetted validation set.")
                    else:
                        logging.info("Original size must be greater than target size!")
            # END RYAN's CODE
            # RYAN'S CODE:
            if "split" in self.config["dataset"]:
                # to make sampling deterministic, seed rng
                logging.info(f"original size {len(self.train_dataset)}, target size {self.config['dataset']['split']}")
                if len(self.train_dataset) >= self.config["dataset"]["split"]:
                    train_indxs = np.random.default_rng(seed=0).choice(
                        len(self.train_dataset), 
                        self.config["dataset"]["split"], 
                        replace=False
                    )
                    self.train_dataset = Subset(self.train_dataset, torch.tensor(train_indxs))
                    logging.info("Subsetted train set.")
                else:
                    logging.info("Original size must be greater than target size!")
            # END RYAN's CODE
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
            if self.config.get("val_dataset", None):
                ## START ISHAN CODE
                self.val_dataset = self.insert_teach_datasets(self.val_dataset, 'val', val_indxs)
                # END ISHAN CODE
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

            if self.config.get("test_dataset", None):
                if self.config["test_dataset"].get("use_train_settings", True):
                    test_config = self.config["dataset"].copy()
                    test_config.update(self.config["test_dataset"])
                else:
                    test_config = self.config["test_dataset"]

                self.test_dataset = registry.get_dataset_class(
                    test_config.get("format", "lmdb")
                )(test_config)
                self.test_sampler = self.get_sampler(
                    self.test_dataset,
                    self.config["optim"].get(
                        "eval_batch_size", self.config["optim"]["batch_size"]
                    ),
                    shuffle=False,
                )
                self.test_loader = self.get_dataloader(
                    self.test_dataset,
                    self.test_sampler,
                )

        # load relaxation dataset
        if "relax_dataset" in self.config["task"]:
            self.relax_dataset = registry.get_dataset_class("lmdb")(
                self.config["task"]["relax_dataset"]
            )
            self.relax_sampler = self.get_sampler(
                self.relax_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.relax_loader = self.get_dataloader(
                self.relax_dataset,
                self.relax_sampler,
            )

    def update_loss_coefficients(self):
        # self.force_mae, self.teacher_force_mae are good to go
        if self.force_mae == None:
            self.validate()
            print("STUDENT MAE:", self.force_mae)
        if self.step % 20 == 0:
            if self.force_mae < self.teacher_force_mae:
                logging.info("EXCEEDED TEACHER ACCURACY!!")
                # self.loss_functions[-1][1]['coefficient'] = 0

            percent_higher = ((self.force_mae - self.teacher_force_mae) / self.teacher_force_mae) *100
            threshold = 5
            if percent_higher < threshold: 
                self.loss_functions[-1][1]['coefficient'] = 0.25 * self.original_fjac_coeff
                # self.loss_functions[-1][1]['coefficient'] = custom_sigmoid(percent_higher, threshold) * self.original_fjac_coeff
                


    
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

        loss = []
        #ISHAN ADDED CODE!!
        out['teacher_forces'] = out['forces']
        self.update_loss_coefficients()
        should_mask = self.output_targets['forces']["train_on_free_atoms"]
        if self.loss_functions[-1][1]['coefficient'] != 0:
            force_jac_loss = get_force_jac_loss(out, batch, self.config['optim']['force_jac_sample_size'], mask, should_mask, looped=(not self.config['optim']["vectorize_jacs"]))
            if self.config['optim'].get("print_memory_usage", False):
                print_cuda_memory_usage()
        else:
            batch['force_jac_loss'] = torch.tensor(0)
        ## FINISH ISHAN ADDED CODE
        for loss_fn in self.loss_functions:
            target_name, loss_info = loss_fn
            if loss_info['coefficient'] == 0: #Added this, don't bother
                continue
            if target_name == 'force_jacs':
                loss.append(loss_info["coefficient"] * force_jac_loss)
                batch['force_jac_loss'] = force_jac_loss
                continue
            target = batch[target_name]
            pred = out[target_name]

            natoms = batch.natoms
            natoms = torch.repeat_interleave(natoms, natoms)
            if (target_name != 'force_jacs' and target_name != 'teacher_forces'): # added this
                if (
                    self.output_targets[target_name]["level"] == "atom"
                    and self.output_targets[target_name]["train_on_free_atoms"]
                ):
                    target = target[mask]
                    pred = pred[mask]
                    natoms = natoms[mask]

                num_atoms_in_batch = natoms.numel()
                if self.normalizers.get(target_name, False):
                    target = self.normalizers[target_name].norm(target)

                ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
                if self.output_targets[target_name]["level"] == "atom":
                    target = target.view(num_atoms_in_batch, -1)
                else:
                    target = target.view(batch_size, -1)

            mult = loss_info["coefficient"]
            curr_loss = loss_info["fn"](
                        pred,
                        target,
                        natoms=natoms,
                        batch_size=batch_size,
            )
            loss.append(
                mult
                * curr_loss
            )
        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
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