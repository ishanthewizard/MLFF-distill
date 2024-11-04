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

sys.path.append("/global/homes/s/sanjeevr/MLFF-distill")
from tqdm import tqdm

logging.basicConfig(level=logging.ERROR)
import argparse
from pathlib import Path
import torch.distributed as dist
from jmp.lightning.data.balanced_batch_sampler import BalancedBatchSampler
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from jmp.configs.pretrain.jmp_l import jmp_l_pt_config_
from jmp.datasets.pretrain_lmdb import PretrainLmdbDataset, PretrainDatasetConfig
from jmp.tasks.pretrain import PretrainConfig, PretrainModel
from jmp.tasks.pretrain.module import (
    NormalizationConfig,
    PretrainDatasetConfig,
    TaskConfig,
)
from src.distill_utils import get_teacher_jacobian


@registry.register_trainer("jmp_labels")
class JMPLabelsTrainer(OCPTrainer):
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

        data_path = os.path.dirname(self.config['dataset']["src"])
        config = self.jmp_l_config(data_path=data_path)
        configs: list[tuple[PretrainConfig, type[PretrainModel]]] = []
        configs.append((config, PretrainModel))
        compute_jmp_labels(config, self.config['dataset']["src"], "/data/shared/ishan_stuff/jmp-s.pt", "train")
        
        self.compute_jmp_labels(self.config['dataset']['teacher_labels_folder'])
        sys.exit(0)


    def record_labels_parallel(self, dataset, batch_size, collate_fn, file_paths, fn):
        """
        This function launches the recording tasks across distributed workers without
        the need for manually spawning processes since they are already distributed.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        dataset_size = len(dataset)
        indices_per_worker = dataset_size // world_size
        start_idx = rank * indices_per_worker
        end_idx = start_idx + indices_per_worker if rank < world_size - 1 else dataset_size

        # Indices for the subset this process will handle
        subset_indices = list(range(start_idx, end_idx))
        subset_dataset = Subset(dataset, subset_indices)
        dataloader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        self.record_and_save(dataloader, file_paths, fn)


    def record_and_save(self, dataloader, file_paths, fn):
        """
        Utility function to record the output of a function applied to all batches in a dataloader and save it to multiple LMDB files.

        Args:
            dataloader (DataLoader): The data loader providing batches.
            file_paths (iterable): An iterable of file paths whose length matches the length of the output of fn.
            fn (callable): A function that takes a batch and returns an iterable of outputs.
        """
        # Ensure that file_paths is an iterable and has the correct length
        if not isinstance(file_paths, (list, tuple)):
            file_paths = [file_paths]

        # Create LMDB environments for each file path
        envs = []
        for file_path in file_paths:
            file_path = file_path.split(".lmdb")[0] + str(dist.get_rank()) + ".lmdb"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            env = lmdb.open(file_path, map_size=1099511627776 * 2)
            envs.append(env)

        no_sync_context = (
            self.model.no_sync()
            if dist.get_world_size() > 1
            else contextlib.nullcontext()
        )
        ids = [0] * len(envs)
        with no_sync_context:
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    batch = batch.cuda()
                    batch_outputs = fn(batch)
                    if not isinstance(batch_outputs, (tuple)):
                        batch_outputs = (batch_outputs,)
                    # Check if the lengths match
                    assert len(batch_outputs) == len(
                        envs
                    ), "The length of batch outputs must match the length of file_paths."

                    for i, (output, env) in enumerate(zip(batch_outputs, envs)):
                        with env.begin(write=True) as txn:
                            for out in output:
                                txn.put(
                                    str(ids[i]).encode(),
                                    out.detach().cpu().numpy().tobytes(),
                                )
                                ids[i] += 1

        for env in envs:
            env.close()
        logging.info(f"All tensors saved to LMDB files.")


    # Let's make the config
    def jmp_l_config(self, data_path):
        config = PretrainConfig.draft()
        jmp_l_pt_config_(config)
        # Set data config
        config.batch_size = 4
        config.num_workers = 8
        config.backbone.otf_graph = True

        # Set the tasks
        config.tasks = [
            TaskConfig(
                name="transition1x",
                train_dataset=PretrainDatasetConfig(
                    src=Path(os.path.join(data_path, "train")),
                ),
                val_dataset=PretrainDatasetConfig(
                    src=Path(os.path.join(data_path, "val")),
                ),
                energy_loss_scale=1.0,
                force_loss_scale=14.0,
                normalization={
                    "y": NormalizationConfig(
                        mean=0.0, std=1.787466168382901
                    ),  # TODO: why is the mean 0
                    "force": NormalizationConfig(mean=0.0, std=0.3591422140598297),
                },
            ),
        ]
        return config.finalize()


    def compute_jmp_labels(self, config, base_dir, data_path, ckpt_path, split="train"):
        if "jmp-s" in ckpt_path:
            model_name = "jmp-small"
        elif "jmp-l" in ckpt_path:
            model_name = "jmp-large"
        ckpt_path = Path(ckpt_path)
        model = PretrainModel(config)
        # Load the checkpoint
        state_dict = torch.load(ckpt_path)
        sd = state_dict["state_dict"].copy()
        new_sd = {}
        # Map key to the right spot since only doing trans1x for now
        for k in sd:
            # TODO: make sure that this is the transition 1x head
            if "output.out_energy.3" in k:
                new_k = k.replace("output.out_energy.3", "output.out_energy.0")
                new_sd[new_k] = sd[k]
            elif "output.out_forces.3" in k:
                new_k = k.replace("output.out_forces.3", "output.out_forces.0")
                new_sd[new_k] = sd[k]
            elif "output.out_forces" in k or "output.out_energy" in k:
                pass
            else:
                new_sd[k] = sd[k]

        model.load_state_dict(new_sd, strict=False)
        model.to("cuda")

        if split == "train":
            dataset = model.train_dataset()
        elif split == "val":
            dataset = model.val_dataset()
        elif split == "test":
            dataset = model.test_dataset()

        sids = []
        energies = []
        forces_all = []
        force_norms = []
        force_maes = []

        # Save the atom embeddings
        atom_embeddings = model.embedding.atom_embedding(torch.arange(120).cuda().long())
        np.save(
            f"{base_dir}/labels/{model_name}_atom_embeddings.npy",
            atom_embeddings.cpu().detach().numpy(),
        )

        # Define the functions to record
        def get_separated_forces_and_node_embeddings(batch):
            _, all_forces, all_node_embeddings = model(batch)
            natoms = batch.natoms
            separated_forces = [
                0.3591422140598297 * all_forces[sum(natoms[:i]) : sum(natoms[: i + 1])]
                for i in range(len(natoms))
            ]
            separated_node_embeddings = [
                all_node_embeddings[sum(natoms[:i]) : sum(natoms[: i + 1])]
                for i in range(len(natoms))
            ]
            return separated_forces, separated_node_embeddings


        def teacher_jacobian_fn(batch):
            
            with torch.enable_grad():
                batch.pos.requires_grad = True
                forces = model(batch)[1]
            
                output = [
                    0.3591422140598297 * jac
                    for jac in get_teacher_jacobian(
                        forces, batch, model, finite_differences=False
                    )
                ]        
            return output

        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=model.collate_fn,
        )

        subgrp_name = args.data_path.split("lmdb_")[-1]
        # Save teacher jacobians, forces, and node embeddings

        if split == "train":
            print("Computing teacher jacobians")
            lmdb_path = f"{base_dir}/labels/{model_name}_{subgrp_name}/force_jacobians/data.lmdb"
            # assert not os.path.exists(lmdb_path), f"{lmdb_path} already exists"
            record_labels_parallel(
                dataset, 1, model.collate_fn, lmdb_path, teacher_jacobian_fn
            )

        print("Computing teacher forces and node embeddings")
        lmdb_paths = [
            f"{base_dir}/labels/{model_name}_{subgrp_name}/{split}_forces/data.lmdb",
            f"{base_dir}/labels{model_name}_{subgrp_name}/{split}_final_node_features/data.lmdb",
        ]

        # # assert that lmdb_paths don't exist
        # for lmdb_path in lmdb_paths:
        #     assert not os.path.exists(lmdb_path), f"{lmdb_path} already exists"

        self.record_labels_parallel(
            dataset,
            32,
            model.collate_fn,
            lmdb_paths,
            get_separated_forces_and_node_embeddings,
        )

        print("Done!")

    
    
    