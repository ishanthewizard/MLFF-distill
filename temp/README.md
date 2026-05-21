Conditional ligand VAE (minimal example)
========================================

Goal
----
Train a conditional VAE that models the joint distribution of ligand atom coordinates and element types given a protein pocket context.

Dataset expectation
-------------------
- Point this project at the PDBBind shard directory: `@yuejian/BADGER/PDBBind/refined_and_v2020_decomp_shards` (or copy a subset under `temp/data/`).
- Each sample is stored as an `.npz` file with keys:
  - `ligand_coords`: float32 array `[n_atoms, 3]` (Angstrom)
  - `ligand_atomic_numbers`: int64 array `[n_atoms]`
  - `pocket_coords`: float32 array `[n_pocket_points, 3]` (e.g., pocket atoms or surface points)
  - Optional `pocket_features`: float32 array `[n_pocket_points, d_feat]` (one-hot residue types, SASA, etc.)
  - Optional `ligand_atom_mask`: float32 array `[n_atoms]` (1.0 for real atoms, 0.0 for padding if pre-padded)

Quickstart
----------
1) Install deps (PyTorch CPU by default):
```
pip install -r requirements.txt
```

2) Run a smoke test with synthetic data:
```
PYTHONPATH=src python -m src.train --synthetic --epochs 2 --batch-size 8
```

3) Train on real shards:
```
PYTHONPATH=src python -m src.train \
  --data-root /path/to/BADGER/PDBBind/refined_and_v2020_decomp_shards \
  --batch-size 16 --epochs 50 --max-atoms 128
```

Notes on modeling choices
-------------------------
- Ligand encoder: embeds element ids, concatenates coordinates, and aggregates with an MLP.
- Pocket encoder: simple point-cloud MLP with mean pooling; optional pocket features are concatenated.
- Latent: diagonal Gaussian posterior conditioned on ligand + pocket context; prior is standard normal.
- Decoder: predicts per-atom coordinate offsets and element logits conditioned on pocket context and latent code.
- Loss: coordinate MSE + element cross-entropy (masked) + KL divergence.

Project layout
--------------
- `requirements.txt` — minimal dependencies
- `src/data.py` — dataset and collate utils, synthetic dataset for smoke tests
- `src/model.py` — conditional VAE definition and loss
- `src/train.py` — training loop, CLI args, checkpointing
- `src/utils.py` — small helpers (masking, seeding, device moves)

Extending
---------
- Swap the simple pocket encoder for a 3D GNN or transformer.
- Replace the MSE coord loss with SE(3)-equivariant likelihoods.
- Add a sampler script to generate ligands conditioned on new pockets.
