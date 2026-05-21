import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


ALLOWED_ATOMIC_NUMBERS = (
    1,   # H
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    15,  # P
    16,  # S
    17,  # Cl
    35,  # Br
    53,  # I
)


def atomic_number_to_id(z: np.ndarray) -> np.ndarray:
    """Map atomic numbers to compact ids; unseen elements become last id."""
    mapping = {z_val: idx for idx, z_val in enumerate(ALLOWED_ATOMIC_NUMBERS)}
    unk_id = len(ALLOWED_ATOMIC_NUMBERS)
    return np.array([mapping.get(int(v), unk_id) for v in z], dtype=np.int64)


@dataclass
class PocketLigandExample:
    ligand_coords: np.ndarray  # [n_atoms, 3]
    ligand_atomic_numbers: np.ndarray  # [n_atoms]
    pocket_coords: np.ndarray  # [n_pocket, 3]
    pocket_features: Optional[np.ndarray] = None  # [n_pocket, d_feat]
    ligand_atom_mask: Optional[np.ndarray] = None  # [n_atoms]


class PocketLigandDataset(Dataset):
    """
    Dataset that reads .npz shards with ligand and pocket info.
    """

    def __init__(
        self,
        root: str,
        max_atoms: int = 128,
        max_pocket_points: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.root = root
        pattern = os.path.join(root, "**", "*.npz")
        self.paths = sorted(glob.glob(pattern, recursive=True))
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise ValueError(f"No .npz files found under {root}")
        self.max_atoms = max_atoms
        self.max_pocket_points = max_pocket_points

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
        sample = np.load(path)

        ligand_coords = np.asarray(sample["ligand_coords"], dtype=np.float32)
        ligand_atomic_numbers = np.asarray(sample["ligand_atomic_numbers"], dtype=np.int64)
        ligand_atom_mask = sample.get("ligand_atom_mask")
        pocket_coords = np.asarray(sample["pocket_coords"], dtype=np.float32)
        pocket_features = sample.get("pocket_features")
        if pocket_features is not None:
            pocket_features = np.asarray(pocket_features, dtype=np.float32)

        return self._to_tensor(
            PocketLigandExample(
                ligand_coords=ligand_coords,
                ligand_atomic_numbers=ligand_atomic_numbers,
                ligand_atom_mask=np.asarray(ligand_atom_mask, dtype=np.float32)
                if ligand_atom_mask is not None
                else None,
                pocket_coords=pocket_coords,
                pocket_features=pocket_features,
            )
        )

    def _to_tensor(self, ex: PocketLigandExample) -> Dict[str, torch.Tensor]:
        atom_ids = atomic_number_to_id(ex.ligand_atomic_numbers)
        n_atoms = ex.ligand_coords.shape[0]
        atom_mask = (
            ex.ligand_atom_mask.astype(np.float32)
            if ex.ligand_atom_mask is not None
            else np.ones(n_atoms, dtype=np.float32)
        )
        # Pad or crop ligands
        max_atoms = self.max_atoms
        padded_coords = np.zeros((max_atoms, 3), dtype=np.float32)
        padded_ids = np.zeros((max_atoms,), dtype=np.int64)
        padded_mask = np.zeros((max_atoms,), dtype=np.float32)
        take = min(n_atoms, max_atoms)
        padded_coords[:take] = ex.ligand_coords[:take]
        padded_ids[:take] = atom_ids[:take]
        padded_mask[:take] = atom_mask[:take]

        # Optionally subsample pocket points
        pocket_coords = ex.pocket_coords
        pocket_features = ex.pocket_features
        if self.max_pocket_points is not None and pocket_coords.shape[0] > self.max_pocket_points:
            choice = np.random.choice(pocket_coords.shape[0], self.max_pocket_points, replace=False)
            pocket_coords = pocket_coords[choice]
            if pocket_features is not None:
                pocket_features = pocket_features[choice]

        out = {
            "ligand_coords": torch.from_numpy(padded_coords),
            "ligand_atom_ids": torch.from_numpy(padded_ids),
            "ligand_mask": torch.from_numpy(padded_mask),
            "pocket_coords": torch.from_numpy(pocket_coords),
        }
        if pocket_features is not None:
            out["pocket_features"] = torch.from_numpy(pocket_features)
        return out


class SyntheticPocketLigandDataset(Dataset):
    """
    Small synthetic dataset for smoke tests and overfit checks.
    """

    def __init__(
        self,
        n_samples: int = 256,
        max_atoms: int = 32,
        max_pocket_points: int = 64,
        seed: int = 0,
    ) -> None:
        self.n_samples = n_samples
        self.max_atoms = max_atoms
        self.max_pocket_points = max_pocket_points
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = self.rng
        n_atoms = rng.integers(low=8, high=self.max_atoms // 2)
        n_pocket = rng.integers(low=16, high=self.max_pocket_points // 2)

        ligand_coords = rng.normal(size=(n_atoms, 3)).astype(np.float32)
        ligand_atomic_numbers = rng.choice(ALLOWED_ATOMIC_NUMBERS, size=n_atoms)
        pocket_coords = rng.normal(loc=0.0, scale=2.0, size=(n_pocket, 3)).astype(np.float32)
        pocket_features = rng.normal(size=(n_pocket, 4)).astype(np.float32)

        ex = PocketLigandExample(
            ligand_coords=ligand_coords,
            ligand_atomic_numbers=ligand_atomic_numbers,
            ligand_atom_mask=None,
            pocket_coords=pocket_coords,
            pocket_features=pocket_features,
        )
        return PocketLigandDataset._to_tensor(self, ex)  # type: ignore[arg-type]


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Simple stack collate; ligands are already padded."""
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        if k in ("pocket_coords", "pocket_features"):
            # Variable length pocket; pad to max length in batch
            lengths = [t.shape[0] for t in tensors]
            max_len = max(lengths)
            if tensors[0].dim() == 2:
                feat_dim = tensors[0].shape[1]
                padded = torch.zeros(len(tensors), max_len, feat_dim, dtype=tensors[0].dtype)
            else:
                raise ValueError(f"Unexpected dim for {k}: {tensors[0].shape}")
            mask = torch.zeros(len(tensors), max_len, dtype=torch.float32)
            for i, t in enumerate(tensors):
                l = t.shape[0]
                padded[i, :l] = t
                mask[i, :l] = 1.0
            out[k] = padded
            out[f"{k}_mask"] = mask
        else:
            out[k] = torch.stack(tensors, dim=0)
    return out
