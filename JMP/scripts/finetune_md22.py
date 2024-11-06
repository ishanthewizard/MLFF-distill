"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
import argparse

from jmp.configs.finetune.jmp_l import jmp_l_ft_config_
from jmp.configs.finetune.jmp_s import jmp_s_ft_config_
from jmp.configs.finetune.md22 import jmp_l_md22_config_
from jmp.tasks.finetune.base import FinetuneConfigBase, FinetuneModelBase
from jmp.tasks.finetune.md22 import MD22Config, MD22Model
from jmp.lightning import Runner, Trainer
from jmp.utils.finetune_state_dict import (
    filter_state_dict,
    retreive_state_dict_for_finetuning,
)


def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    if (ckpt_path := config.meta.get("ckpt_path")) is None:
        raise ValueError("No checkpoint path provided")

    model = model_cls(config)

    # Load the checkpoint
    state_dict = retreive_state_dict_for_finetuning(
        ckpt_path, load_emas=config.meta.get("ema_backbone", False)
    )
    embedding = filter_state_dict(state_dict, "embedding.atom_embedding.")
    backbone = filter_state_dict(state_dict, "backbone.")
    model.load_backbone_state_dict(backbone=backbone, embedding=embedding, strict=True)
    trainer = Trainer(config)
    
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/data/shared/ishan_stuff/jmp-s.pt",
        help="Path to the pretrained checkpoint.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/sanjeevr/MLFF-distill/data/md22",
        help="Path to the training data.",
    )
    parser.add_argument(
        "--molecule",
        type=str,
        default="Ac-Ala3-NHMe",
        help="Molecule for training.",
    )
    parser.add_argument(
        "--direct_forces",
        action="store_true",
        help="Whether to directly predict forces instead of gradient-based.",
    )
   
    args = parser.parse_args()

    # We create a list of all configurations that we want to run.
    configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []

    config = MD22Config.draft()
    if "-l" in args.checkpoint_path:
        jmp_l_ft_config_(config, args.checkpoint_path, disable_force_output_heads = not args.direct_forces)
    else:
        jmp_s_ft_config_(config, args.checkpoint_path, disable_force_output_heads = not args.direct_forces)
    
    # This loads the MD22-specific configuration
    
    jmp_l_md22_config_(config, args.molecule, Path(args.data_path), direct_forces=args.direct_forces)
    
    config = config.finalize()  # Actually construct the config object
    print(config)

    configs.append((config, MD22Model))

    runner = Runner(run)
    runner._run(config, MD22Model)