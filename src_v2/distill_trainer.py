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
    

class distill_trainer(TrainEvalRunner):
    def __init__(
        self,
        train_dataloader: torch.utils.data.dataloader,
        eval_dataloader: torch.utils.data.dataloader,
        train_eval_unit: Union[TrainUnit, EvalUnit, Stateful],
        label_folder: str = None,
    ):  
        
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_eval_unit = train_eval_unit
        self.device = self.train_eval_unit.model.device
        
        self.label_folder = label_folder
        print(next(iter(self.train_dataloader)))
        # make a snapshot of the sampled indices of current rank, will dump it to a file
        self.indices_list_of_current_rank = list(self.train_dataloader.batch_sampler)