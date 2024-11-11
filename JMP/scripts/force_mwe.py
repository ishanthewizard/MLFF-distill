import os
from tqdm import tqdm
import argparse
import lmdb
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from jmp.configs.pretrain.jmp_l import jmp_l_pt_config_
from jmp.datasets.pretrain_lmdb import PretrainLmdbDataset, PretrainDatasetConfig
from jmp.tasks.pretrain import PretrainConfig, PretrainModel
from jmp.tasks.pretrain.module import (
    NormalizationConfig,
    PretrainDatasetConfig,
    TaskConfig,
)


# Let's make the config
def jmp_l_config(data_path):
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
                "y": NormalizationConfig(mean=0.0, std=1.787466168382901),
                "force": NormalizationConfig(mean=0.0, std=0.3591422140598297),
            },
        ),
    ]
    return config.finalize()


def compute_grads(config, data_path, ckpt_path):

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

    dataset = model.train_dataset()

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=model.collate_fn,
    )

    batch = dataloader.__iter__().__next__()
    batch = batch.to("cuda")
    batch.pos.requires_grad = True

    import pdb; pdb.set_trace()
    energy, forces, _ = model(batch)

    # Attempt to compute the gradient of the energy or force w.r.t. to the position
    # This fails with the error: "RuntimeError: One of the differentiated Tensors appears to not have been used in the graph.
    #  Set allow_unused=True if this is the desired behavior."
    assert energy.requires_grad
    assert forces.requires_grad
    assert not energy.is_leaf
    assert not forces.is_leaf
    import pdb; pdb.set_trace()
    energy_grad = torch.autograd.grad(energy.sum(), batch.pos, retain_graph=True)
    forces_grad = torch.autograd.grad(forces.sum(), batch.pos, retain_graph=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/data/shared/ishan_stuff/jmp-s.pt",
        help="Path to the jmp-l.pt or jmp-s.pt checkpoint.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/shared/ishan_stuff/transition1x/lmdb_transitiononly",
        help="Path to the Transition 1x data.",
    )

    args = parser.parse_args()
    config = jmp_l_config(data_path=args.data_path)
    configs: list[tuple[PretrainConfig, type[PretrainModel]]] = []
    configs.append((config, PretrainModel))
    compute_grads(config, args.data_path, args.checkpoint_path)
