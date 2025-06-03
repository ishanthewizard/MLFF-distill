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
from abc import abstractmethod,ABC

import torch
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from torchtnt.framework.callback import Callback
from torchtnt.framework.fit import fit

from fairchem.core.common import distutils
from fairchem.core.common.utils import get_subdirectories_sorted_by_time
from fairchem.core.components.runner import Runner
from fairchem.core.components.train.train_runner import TrainEvalRunner
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
        callbacks: list[Callback] | None = None,
        max_epochs: int | None = 1,
        evaluate_every_n_steps: Optional[int] = None,
        max_steps: int | None = None,
        save_inference_ckpt: bool = True,
    ):  
        super().__init__(
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_dataloader,
                    train_eval_unit=train_eval_unit,
                    callbacks=callbacks,
                    max_epochs=max_epochs,
                    evaluate_every_n_steps=evaluate_every_n_steps,
                    max_steps=max_steps,
                    save_inference_ckpt=save_inference_ckpt,
        )
        
        self.train_dataloader = train_dataloader
        
        debug_dataset = self.train_dataloader.dataset
        breakpoint()
        self.eval_dataloader = eval_dataloader
        self.train_eval_unit = train_eval_unit
        self.device = self.train_eval_unit.model.device
        
        # self.label_folder = label_folder
        print(next(iter(self.train_dataloader)))
        # make a snapshot of the sampled indices of current rank, will dump it to a file
        self.indices_list_of_current_rank = list(self.train_dataloader.batch_sampler)