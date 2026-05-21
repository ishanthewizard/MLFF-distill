import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    ALLOWED_ATOMIC_NUMBERS,
    PocketLigandDataset,
    SyntheticPocketLigandDataset,
    collate_batch,
)
from .model import ConditionalVAE, vae_loss
from .utils import get_device, save_checkpoint, seed_all, to_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditional ligand VAE")
    p.add_argument("--data-root", type=str, default=None, help="Path to shard directory (.npz files)")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for a smoke test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--max-atoms", type=int, default=128)
    p.add_argument("--max-pocket-points", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_dataloader(args: argparse.Namespace) -> DataLoader:
    if args.synthetic:
        ds = SyntheticPocketLigandDataset(
            n_samples=256, max_atoms=args.max_atoms, max_pocket_points=args.max_pocket_points, seed=args.seed
        )
    else:
        if args.data_root is None:
            raise ValueError("Provide --data-root when not using --synthetic")
        ds = PocketLigandDataset(
            root=args.data_root, max_atoms=args.max_atoms, max_pocket_points=args.max_pocket_points
        )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch)


def train(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = get_device()
    num_elements = len(ALLOWED_ATOMIC_NUMBERS) + 1  # extra id for OOV elements

    model = ConditionalVAE(num_elements=num_elements, latent_dim=args.latent_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    loader = make_dataloader(args)
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for batch in progress:
            batch = to_device(batch, device)
            out = model(batch)
            losses = vae_loss(batch, out)
            optim.zero_grad()
            losses["loss"].backward()
            optim.step()

            global_step += 1
            progress.set_postfix(
                loss=float(losses["loss"]),
                coord=float(losses["coord_loss"]),
                elem=float(losses["elem_loss"]),
                kl=float(losses["kl_loss"]),
            )
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(ckpt_path, {"model": model.state_dict(), "args": vars(args), "step": global_step})


if __name__ == "__main__":
    args = parse_args()
    train(args)
