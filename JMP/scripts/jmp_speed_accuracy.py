import os
from tqdm import tqdm
import argparse
import lmdb
from pathlib import Path
import numpy as np
import torch
import torchmetrics
import time
from torch.utils.data import DataLoader
from jmp.configs.pretrain.jmp_l import jmp_l_pt_config_
from jmp.datasets.pretrain_lmdb import PretrainLmdbDataset, PretrainDatasetConfig
from jmp.tasks.pretrain import PretrainConfig, PretrainModel
from jmp.tasks.pretrain.module import (
    NormalizationConfig,
    PretrainDatasetConfig,
    TaskConfig,
)
from jmp.configs.finetune.jmp_l import jmp_l_ft_config_
from jmp.configs.finetune.jmp_s import jmp_s_ft_config_
from jmp.configs.finetune.md22 import jmp_l_md22_config_
from jmp.tasks.finetune.base import FinetuneConfigBase, FinetuneModelBase
from jmp.tasks.finetune.md22 import MD22Config, MD22Model


def evaluate_speed_accuracy(args):
    

    jmp_config = MD22Config.draft()

    if "-l" in args.checkpoint_path:
        jmp_l_ft_config_(jmp_config, args.checkpoint_path, disable_force_output_heads = not args.direct_forces)
    else:
        jmp_s_ft_config_(jmp_config, args.checkpoint_path, disable_force_output_heads = not args.direct_forces)

    # This loads the MD22-specific configuration
    jmp_l_md22_config_(jmp_config, args.molecule, Path(args.data_path), direct_forces=args.direct_forces)
    jmp_config = jmp_config.finalize()  # Actually construct the config object
    model = MD22Model(jmp_config)

    ckpt_path = Path(args.checkpoint_path)

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
        elif "output.out_forces" in k or "output.out_energy" in k or "task_steps" in k:
            pass
        else:
            new_sd[k] = sd[k]

    model.load_state_dict(new_sd, strict=True)
    model.to("cuda")

    # get normalization constants
    mean_force = jmp_config.normalization['force'].mean
    std_force = jmp_config.normalization['force'].std

    mean_energy = jmp_config.normalization['y'].mean
    std_energy = jmp_config.normalization['y'].std

    

    dataset = model.val_dataset()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=model.collate_fn,
    )
    # Loop through the dataloader and compute speed and accuracy
    force_mae = 0
    energy_mae = 0
    mae = torchmetrics.MeanAbsoluteError().to("cuda")
    start = None
    if args.stop_record_idx == -1:
        args.stop_record_idx = len(dataloader)
    for i, batch in enumerate(tqdm(dataloader)):
        if i > args.stop_record_idx:
            break
        batch = batch.to("cuda")
        if i >= args.start_record_idx and start is None:
                start = time.time()
        with torch.no_grad():
            out = model(batch)
            pred_forces = out["force"]

        # Un-normalize the energies and forces
        # pred_energy = pred_energy * std_energy + mean_energy
        pred_forces = pred_forces * std_force + mean_force

        # Compute the Force MAE and Energy MAE
        force_mae +=  mae(pred_forces, batch.force)
        # energy_mae += (pred_energy - batch.y).abs().mean()

    force_mae /= args.stop_record_idx
    # energy_mae /= len(dataloader)

    print(f"Force MAE (meV/A): {1000*force_mae}")
    # print(f"Energy MAE (meV): {1000*energy_mae}")

    iter_per_s = ((args.stop_record_idx - args.start_record_idx) * args.batch_size) / (time.time() -  start)
    print(f"THROUGHPUT SAMP PER SECOND: {round(iter_per_s)}") #technically this doesn't exactly work for huge batch sizes because te last one could be wrong
    print(f"THROUGHPUT NS PER DAY: {round(iter_per_s * .0864, 1)}") 

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/data/shared/ishan_stuff/nanotube_jmp-l.ckpt",
        help="Path to the JMP finetuned checkpoint.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/shared/ishan_stuff/md22",
        help="Path to the data",
    )

    parser.add_argument(
        "--molecule",
        type=str,
        default="double-walled_nanotube",
        help="Molecule to simulate.",
    )

    parser.add_argument(
        "--direct_forces",
        action="store_true",
        help="Whether to use direct forces.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )

    parser.add_argument(
        "--start_record_idx",
        type=int,
        default=3,
        help="Index to start recording speed.",
    )

    parser.add_argument(
        "--stop_record_idx",
        type=int,
        default=-1,
        help="Index to stop recording speed.",
    )
    

    args = parser.parse_args()
    
    evaluate_speed_accuracy(args)
