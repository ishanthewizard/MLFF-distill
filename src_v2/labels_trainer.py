"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os, sys
from typing import Callable
import shutil
import gc
import lmdb
from tqdm import tqdm
import contextlib
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from torchtnt.framework.callback import Callback
from torchtnt.framework.fit import fit

from fairchem.core.common import distutils
from fairchem.core.common.utils import get_subdirectories_sorted_by_time
from fairchem.core.components.runner import Runner
from fairchem.core.units.mlip_unit.mlip_unit import (
    convert_train_checkpoint_to_inference_checkpoint,
)

from .distill_utils import get_jacobian, get_force_jac_loss, print_cuda_memory_usage, get_teacher_jacobian

if TYPE_CHECKING:
    from torch.distributed.checkpoint.stateful import Stateful
    from torchtnt.framework import EvalUnit, TrainUnit
    from torchtnt.framework.state import State
    from torchtnt.framework.unit import TTrainUnit
    

class TeacherLabelGenerator(Runner):
    """
    A class to generate labels for training and evaluation datasets in a distributed manner.
    Description:
    -----------
    Parallelized label generator for training and evaluation datasets.
    Using max atom sampler to balance the workload across multiple GPUs.
    """
    def __init__(
        self,
        train_dataloader: torch.utils.data.dataloader,
        eval_dataloader: torch.utils.data.dataloader,
        train_eval_unit: Union[TrainUnit, EvalUnit, Stateful],
        label_folder: str = None,
    ):  
        # initialize the class
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_eval_unit = train_eval_unit
        self.device = self.train_eval_unit.model.device
        
        # Check if the dataloaders are using a deterministic sampler
        if self.train_dataloader.batch_sampler.shuffle == True:
            raise ValueError("TrainSampler should not shuffle, as we need to sample indices in a deterministic way for labeling.")
        if self.eval_dataloader.batch_sampler.shuffle == True:
            raise ValueError("EvalSampler should not shuffle, as we need to sample indices in a deterministic way for labeling.")
        
        # create the label folder if it does not exist
        self.label_folder = label_folder
        if not os.path.exists(self.label_folder):
            os.makedirs(self.label_folder, exist_ok=True)
        
        # create subfolders for indices, train_forces, val_forces, and force_jacobians
        logging.info(f"Creating Label folder at: {self.label_folder}")
        os.makedirs(os.path.join(self.label_folder, "indices"),exist_ok=True)
        os.makedirs(os.path.join(self.label_folder, "train_forces"),exist_ok=True)
        os.makedirs(os.path.join(self.label_folder, "val_forces"),exist_ok=True)
        os.makedirs(os.path.join(self.label_folder, "force_jacobians"),exist_ok=True)
        
        # log the sizes of the dataloaders
        logging.info(f"Train Dataloader size {len(self.train_dataloader)}")
        logging.info(f"Eval Dataloader size {len(self.eval_dataloader)}")

    def run(self) -> None:
        # add your label generation code here
        self.record_labels_parallel(self.label_folder, 'train')
        self.record_labels_parallel(self.label_folder, 'val')
    
    def record_labels_parallel(self, labels_folder: str, dataset_type: str) -> None:
        """
        This function records labels and saves them in separate LMDB files for parallel processing.
        Each worker handles its segment of the dataset.
        """
        ##################
        ### Dataloader ###
        ##################
        # Subset the datasets based on the provided indices
        if dataset_type == 'train':
            dataloader = self.train_dataloader
            # dump the indices to a file
            torch.save(list(self.train_dataloader.batch_sampler), os.path.join(labels_folder, "indices", f"{dataset_type}_indices_{distutils.get_rank():04d}.pt"))
        
        elif dataset_type == 'val':
            dataloader = self.eval_dataloader
            # dump the indices to a file
            torch.save(list(self.eval_dataloader.batch_sampler), os.path.join(labels_folder, "indices", f"{dataset_type}_indices_{distutils.get_rank():04d}.pt"))
        
        else:
            raise ValueError("Invalid dataset type provided")
        
        ####################
        ##### Forces #######
        ####################

        # Define LMDB file path for this particular worker
        lmdb_path = os.path.join(labels_folder, f"{dataset_type}_forces", f"data.{distutils.get_rank():04d}.lmdb")

        # Function to calculate forces
        def get_seperated_forces(batch):
            all_forces = self.train_eval_unit.model(batch)['forces']['forces']
            # sanity check
            # print(all_forces.shape)
            # print("batch.force ", batch.forces.shape)  
            # print("force mae",torch.linalg.vector_norm(all_forces - ((batch.forces)/1.433569), ord=2, dim=-1).mean())
            
            natoms = batch.natoms
            return [all_forces[sum(natoms[:i]):sum(natoms[:i+1])] for i in range(len(natoms))]
        
        # Record and save the data
        self.record_and_save(dataloader, lmdb_path, get_seperated_forces)

        #####################
        ##### Jacobians #####
        #####################
        if dataset_type == 'train':
            # Only for training dataset, save jacobians as well
            jac_dataloader = self.train_dataloader
            
            jac_lmdb_path = os.path.join(labels_folder, "force_jacobians", f"data.{distutils.get_rank():04d}.lmdb")
            # should_mask = self.output_targets['forces']["train_on_free_atoms"]
            should_mask = True
            def get_seperated_force_jacs(batch): 
                batch.pos.requires_grad_()
                jacs = get_teacher_jacobian(
                                            batch, 
                                            # vectorize=self.config["dataset"]["vectorize_teach_jacs"], 
                                            vectorize = False,
                                            should_mask=should_mask, # BUG
                                            # approximation="disabled", # {"disabled","forward","central"}
                                            approximation="forward", # {"disabled","forward","central"}
                                            # detach=True,
                                            forward = self.train_eval_unit.model,
                                            collater = None,
                                            device = self.device
                                            )
                # batch.pos.requires_grad_()
                # jacs_torch = get_teacher_jacobian(
                #                             batch, 
                #                             # vectorize=self.config["dataset"]["vectorize_teach_jacs"], 
                #                             vectorize = False,
                #                             should_mask=should_mask, # BUG
                #                             # approximation="disabled", # {"disabled","forward","central"}
                #                             approximation="disabled", # {"disabled","forward","central"}
                #                             forward = self.train_eval_unit.model,
                #                             collater = None,
                #                             device = self.device
                #                             )
                # breakpoint()
                return jacs
            self.record_and_save(jac_dataloader, jac_lmdb_path, get_seperated_force_jacs)

    def record_and_save(self, dataloader: torch.utils.data.dataloader, file_path: str, fn: Callable) -> None:
        
        map_size= 1099511627776 * 2
        env = lmdb.open(file_path, map_size=map_size)
        
        # Choose the appropriate context manager based on world size
        no_sync_context = self.train_eval_unit.model.no_sync() if distutils.get_world_size() > 1 else contextlib.nullcontext()

        id = 0
        with no_sync_context:
            for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                with env.begin(write=True) as txn:
                    batch_ids = [str(i+id) for i in range(len(batch))]
                    id = int(batch_ids[-1]) + 1  
                    batch_output = fn(batch.to(self.device))
                    for i in range(len(batch_ids)):
                        txn.put(batch_ids[i].encode(), batch_output[i].detach().cpu().numpy().tobytes())
                    del batch_output, batch
                    gc.collect()
                    torch.cuda.empty_cache()

        env.close()
        logging.info(f"All tensors saved to LMDB:{file_path}")
    
    # dummy, just for instantiation
    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        # need the unit to have a save_state protocol
        pass

    def load_state(self, checkpoint_location: str | None) -> None:
        # need the unit to have a load_state protocol
        pass
