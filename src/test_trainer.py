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
from torch.utils.data import DataLoader
from tqdm import tqdm

from fairchem.core import __version__
from fairchem.core.common import distutils 
from fairchem.core.common.data_parallel import OCPCollater
from fairchem.core.common.registry import registry
from fairchem.core.datasets.base_dataset import Subset

from fairchem.core.trainers.ocp_trainer import OCPTrainer
from . import get_jacobian, get_force_jac_loss_masked, get_force_jac_loss, print_cuda_memory_usage, get_teacher_jacobian, create_hessian_mask
from . import CombinedDataset, SimpleDataset, Dataset_with_Hessian_Masking
from fairchem.core.common import distutils
from fairchem.core.common.data_parallel import BalancedBatchSampler
if TYPE_CHECKING:
    from collections.abc import Sequence



@registry.register_trainer("test")
class TestTrainer(OCPTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        selected_val_indices = torch.load(self.config['dataset']['subset_test_indices_path'])
        self.val_dataset = Subset(self.val_dataset, indices=selected_val_indices,metadata=None)
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
    
