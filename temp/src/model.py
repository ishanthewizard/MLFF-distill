from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims, activation=nn.ReLU, last_activation=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_activation:
            layers.append(activation())
    return nn.Sequential(*layers)


class PocketEncoder(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 128, out_dim: int = 128):
        super().__init__()
        self.hidden = hidden
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.net = _mlp([in_dim, hidden, hidden, out_dim], activation=nn.SiLU)

    def _ensure_net(self, input_dim: int, device: torch.device) -> None:
        if self.net is not None and input_dim == self.in_dim:
            return
        self.in_dim = input_dim
        self.net = _mlp([input_dim, self.hidden, self.hidden, self.out_dim], activation=nn.SiLU)
        self.net.to(device)

    def forward(self, coords: torch.Tensor, mask: torch.Tensor, features: Optional[torch.Tensor] = None):
        # coords: [B, P, 3], features: [B, P, d], mask: [B, P]
        if features is not None:
            x = torch.cat([coords, features], dim=-1)
        else:
            x = coords
        self._ensure_net(x.shape[-1], x.device)
        x = self.net(x)
        mask = mask.unsqueeze(-1)
        x = x * mask
        pooled = x.sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        return pooled


class LigandEncoder(nn.Module):
    def __init__(self, num_elements: int, emb_dim: int = 64, hidden: int = 256, out_dim: int = 256):
        super().__init__()
        self.emb = nn.Embedding(num_elements, emb_dim)
        self.net = _mlp([emb_dim + 3, hidden, hidden, out_dim], activation=nn.SiLU)

    def forward(self, coords: torch.Tensor, elem_ids: torch.Tensor, mask: torch.Tensor):
        # coords: [B, A, 3], elem_ids: [B, A], mask: [B, A]
        emb = self.emb(elem_ids)
        x = torch.cat([coords, emb], dim=-1)
        x = self.net(x)
        mask = mask.unsqueeze(-1)
        x = x * mask
        pooled = x.sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        return pooled


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        num_elements: int,
        latent_dim: int = 64,
        pocket_dim: int = 128,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.pocket_encoder = PocketEncoder(in_dim=3, hidden=pocket_dim, out_dim=pocket_dim)
        self.ligand_encoder = LigandEncoder(num_elements=num_elements, emb_dim=64, hidden=hidden, out_dim=hidden)
        self.to_mu = nn.Linear(hidden + pocket_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden + pocket_dim, latent_dim)

        self.dec_ctx = _mlp([latent_dim + pocket_dim, hidden, hidden], activation=nn.SiLU)
        self.dec_coords = nn.Linear(hidden, 3)
        self.dec_elem = nn.Linear(hidden, num_elements)

    def encode(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lig = self.ligand_encoder(
            batch["ligand_coords"], batch["ligand_atom_ids"], batch["ligand_mask"]
        )
        pocket = self.pocket_encoder(
            batch["pocket_coords"], batch["pocket_coords_mask"], batch.get("pocket_features")
        )
        h = torch.cat([lig, pocket], dim=-1)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def decode(self, batch: Dict[str, torch.Tensor], z: torch.Tensor) -> Dict[str, torch.Tensor]:
        pocket = self.pocket_encoder(
            batch["pocket_coords"], batch["pocket_coords_mask"], batch.get("pocket_features")
        )
        B, A, _ = batch["ligand_coords"].shape
        # Broadcast latent + pocket context to atoms
        ctx = self.dec_ctx(torch.cat([z, pocket], dim=-1))  # [B, hidden]
        ctx = ctx.unsqueeze(1).expand(-1, A, -1)
        coords_pred = self.dec_coords(ctx)
        elem_logits = self.dec_elem(ctx)
        return {"coords": coords_pred, "elem_logits": elem_logits}

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z, mu, logvar = self.encode(batch)
        dec = self.decode(batch, z)
        dec["mu"] = mu
        dec["logvar"] = logvar
        return dec


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


def vae_loss(
    batch: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    coord_weight: float = 1.0,
    elem_weight: float = 1.0,
    kl_weight: float = 0.01,
) -> Dict[str, torch.Tensor]:
    mask = batch["ligand_mask"].unsqueeze(-1)
    coord_loss = ((out["coords"] - batch["ligand_coords"]) ** 2 * mask).sum(dim=[1, 2]) / (
        mask.sum(dim=[1, 2]) + 1e-6
    )
    elem_logits = out["elem_logits"]
    elem_targets = batch["ligand_atom_ids"]
    elem_mask = batch["ligand_mask"]
    elem_loss = F.cross_entropy(
        elem_logits.reshape(-1, elem_logits.shape[-1]),
        elem_targets.view(-1),
        reduction="none",
    ).view(elem_mask.shape) * elem_mask
    elem_loss = elem_loss.sum(dim=1) / (elem_mask.sum(dim=1) + 1e-6)

    kl = kl_divergence(out["mu"], out["logvar"])
    total = coord_weight * coord_loss + elem_weight * elem_loss + kl_weight * kl
    return {
        "loss": total.mean(),
        "coord_loss": coord_loss.mean(),
        "elem_loss": elem_loss.mean(),
        "kl_loss": kl.mean(),
    }
