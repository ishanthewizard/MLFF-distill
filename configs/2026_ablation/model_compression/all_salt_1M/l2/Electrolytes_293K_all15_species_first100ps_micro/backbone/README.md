# eSCNMDBackbone Config — Parameter Reference

**Model class:** `fairchem.core.models.uma.escn_md.eSCNMDBackbone`  
**Source:** `fairchem/src/fairchem/core/models/uma/escn_md.py`

This is an equivariant spherical-channel network using SO(3)-equivariant message passing on an atomistic graph. The configs in this directory define small-scale variants of the eSEN/eSCN architecture.

---

## Background: Spherical Harmonics & How eSEN Uses Them

### What Are Spherical Harmonics?

Spherical harmonics $Y_l^m(\theta, \phi)$ are a family of functions defined on the surface of a sphere, indexed by two integers:
- **Degree** $l \geq 0$: controls angular frequency / complexity
- **Order** $m$: ranges from $-l$ to $+l$, so degree $l$ has $2l+1$ components

| Degree $l$ | # components | Physical meaning |
|---|---|---|
| $l=0$ | 1 | Scalar (rotationally invariant) |
| $l=1$ | 3 | Vector (like x, y, z) |
| $l=2$ | 5 | Rank-2 tensor |
| $l=\text{lmax}$ | $2\text{lmax}+1$ | Higher-order tensors |

Their key property: **they transform predictably under rotations**. When you rotate a 3D coordinate system by $R$, the degree-$l$ spherical harmonics mix with each other via a $(2l+1)\times(2l+1)$ **Wigner D-matrix** $D^l(R)$. This is what makes them useful for building rotation-equivariant neural networks — the math guarantees exact physical symmetry.

---

### Node Feature Format — Irreps Tensor

Each atom's feature tensor has shape `[N_atoms, (lmax+1)², sphere_channels]`. The middle dimension packs all spherical harmonic coefficients for $l=0, 1, \ldots, \text{lmax}$. For `lmax=2` that is $1+3+5=9$ coefficient slots, each with `sphere_channels` (e.g. 128) independent channels. Think of each atom as carrying a "spherical harmonic expansion" of its local chemical environment.

`CoefficientMapping` (`uma/common/so3.py`) manages the $(l, m)$ index structure and provides a `to_m` permutation matrix that reorders coefficients from **l-first** to **m-first** order (all $l$ with $m=0$, then all $l$ with $m=\pm1$, …). This reordering is essential for the SO(2) convolution trick.

---

### Wigner D-Matrices — The Rotation Machinery

For every interatomic edge ($i \to j$), eSEN computes a **Wigner D-matrix** encoding the rotation that aligns the global frame with the edge's local frame (so the edge direction points along z).

**Step 1 — Euler angles from edge vector** (`uma/common/rotation.py`):

```python
beta  = arccos(xyz[:, 1])             # polar angle (latitude)
alpha = atan2(xyz[:, 0], xyz[:, 2])   # azimuthal angle (longitude)
gamma = rand() * 2π                   # random roll — prevents gauge overfitting
```

**Step 2 — Wigner D-matrix from Euler angles**:

$$D^l(\alpha, \beta, \gamma) = D^l_z(\alpha) \cdot J^l \cdot D^l_z(\beta) \cdot J^l \cdot D^l_z(\gamma)$$

Pre-stored angular-momentum $J$ matrices (borrowed from e3nn 0.4.0) connect the three z-rotation matrices into the full Wigner D-matrix.

**Step 3 — Composed with M-mapping**: The `to_m` permutation is baked into the Wigner matrix upfront (`wigner_and_M_mapping = to_m @ wigner`), so each forward pass only needs two `torch.bmm` calls per layer rather than an explicit reorder.

---

### The eSCN Core Trick: SO(2) Convolution

Instead of expensive full SO(3) tensor products (as in NequIP/MACE), eSEN exploits the following:

1. **Rotate** each source atom's features into the edge's local frame (edge → z-axis)
2. **Apply SO(2) convolution** — in this frame only rotations around z remain; different $m$ orders decouple completely
3. **Rotate back** to the global frame

In the local frame, $m=0$ components are real scalars; each $\pm m$ pair transforms as a 2D vector under z-rotation. `SO2_Convolution` (`uma/nn/so2_layers.py`) applies a separate learnable linear transform per $m$, implemented as a complex multiply:

```python
x_m_r = x_r_0 - x_i_1   # (a+bi)(c+di) real part
x_m_i = x_r_1 + x_i_0   # (a+bi)(c+di) imaginary part
```

Weights are **radially modulated** by distance + atom-type embeddings, making the convolution sensitive to bond distance.

---

### Full Message Passing Flow Per Layer

Inside each `Edgewise` block (`escn_md_block.py`):

```
1. bmm(wigner_and_M_mapping, x_src)    # rotate to edge-local frame
2. SO2Conv_1(x, x_edge)                # radial-modulated, m-decoupled → produces gating scalars
3. GateActivation(x_0_gating, x)       # equivariant nonlinearity
4. SO2Conv_2(x, x_edge)                # second SO2 conv
5. x *= polynomial_envelope(dist)      # smooth cutoff
6. bmm(wigner_and_M_mapping_inv, x)    # rotate back to global frame
7. scatter_add to target atoms
```

---

### Equivariant Nonlinearities

Standard activations (ReLU, SiLU) break equivariance when applied directly to spherical harmonic coefficients. eSEN uses:

- **GateActivation** (`act_type="gate"`): The $l=0$ (scalar) channels gate all higher-$l$ channels via sigmoid. Scalars are invariant, so applying SiLU to them and multiplying onto higher-$l$ tensors preserves equivariance.
- **S2Activation** (`act_type="s2"`): Projects coefficients to a physical grid on the sphere → SiLU pointwise → project back. Equivariant because the grid representation is equivariant and pointwise operations preserve it.
- **GridAtomwise** (`ff_type="grid"`): Same S2-grid idea applied in the atomwise feed-forward block.

---

### Energy & Force Output

Energy is extracted from the **$l=0$ (scalar) component only** — scalars are rotationally invariant by definition:

```python
_input = node_embedding[:, 0:1, :].squeeze(1)   # l=0 only
energy = energy_mlp(_input).sum()
```

Forces are obtained by autograd (`-dE/d pos`), or optionally by a direct force head reading from $l=1$ components.

---

### Equivariance Guarantee — Summary

```
Atom positions (Å)
      ↓
Edge displacement vectors → Euler angles (α, β, γ) → Wigner D-matrices
      ↓
Node features: [N, (lmax+1)², C]   ← packed spherical harmonic coefficients
      ↓
For each layer:
  EquivariantNorm(x)
      ↓  bmm(wigner, x)             rotate to edge-local frame
      ↓  SO2Conv_1(x, x_edge)       m-decoupled, radial-modulated
      ↓  GateActivation             scalar gates higher-l
      ↓  SO2Conv_2(x, x_edge)
      ↓  bmm(wigner_inv, x)         rotate back to global frame
      ↓  scatter_sum to atoms
  EquivariantNorm(x)
      ↓  GridAtomwise or SpectralAtomwise   (equivariant FF)
      ↓
x[l=0 only] → MLP → per-atom energies → sum → total energy E
forces = -dE/dpos  (autograd or direct head)
```

Rotating all atom positions by $R$ is **exactly equivalent** to rotating all node features by the corresponding Wigner D-matrices. The SO(2) convolutions within the local edge frame are equivariant by construction. The entire model is therefore $E(3)$-equivariant without approximation.

---

## Spherical Harmonic / Equivariance Parameters

| Parameter | Role |
|---|---|
| `sphere_channels` | The channel width `C` of every node feature tensor. Node features have shape `[N_atoms, (lmax+1)², C]`. This is the primary width knob. |
| `lmax` | Maximum spherical harmonic (SH) degree `l`. Node tensors have `(lmax+1)²` SH components per channel. Also determines how many Wigner-D matrices are precomputed. For `lmax=2`, this is 9 components. |
| `mmax` | Maximum SH *order* `m` on edges. When `mmax = lmax`, full SO(3) equivariance is preserved on edges. Setting `mmax < lmax` reduces compute cost. |

Two SO3 grids are built internally:
- `SO3_grid["lmax_lmax"]` — used for the *atomwise* feed-forward
- `SO3_grid["lmax_mmax"]` — used for *edge* SO2 convolutions

---

## Graph Construction Parameters

| Parameter | Role |
|---|---|
| `otf_graph` | Build the neighbor graph **on-the-fly** at each forward pass via `generate_graph()`, rather than relying on pre-computed `edge_index` from the dataset. |
| `max_neighbors` | Maximum number of neighbors per atom when the OTF graph is built. |
| `cutoff` | Radial cutoff in Å. Used in: (1) `GaussianSmearing` range, (2) `PolynomialEnvelope` to smoothly zero messages at the boundary, (3) `generate_graph()` radius. |
| `use_pbc` | **Deprecated.** Kept for backward compatibility. |
| `use_pbc_single` | **Deprecated.** Kept for backward compatibility. |

---

## Edge Feature Parameters

| Parameter | Role |
|---|---|
| `edge_channels` | Dimension of the per-atom source and target embeddings appended to each edge. The full edge feature fed into the radial MLP is `num_distance_basis + 2 × edge_channels`. |
| `distance_function` | Only `"gaussian"` is implemented. Encodes the scalar interatomic distance into a `num_distance_basis`-dimensional vector via `GaussianSmearing`. |
| `num_distance_basis` | Number of equally-spaced Gaussian basis functions from 0 to `cutoff`. Higher = finer distance resolution. |

The resulting `edge_channels_list = [num_distance_basis + 2*edge_channels, edge_channels, edge_channels]` defines a 3-layer radial MLP used in `EdgeDegreeEmbedding` and each `Edgewise` SO2 convolution block.

---

## Force / Stress Regression Parameters

| Parameter | Role |
|---|---|
| `regress_forces` | Enables force computation. With `direct_forces=True`, a dedicated force-head module predicts forces directly. |
| `regress_stress` | Enables stress computation. With `direct_forces=True`, the stress head handles it; with autograd, the backbone sets up a Voigt-strain displacement perturbation. |
| `direct_forces` | `True` = forces/stress predicted by dedicated output head modules (faster). `False` = obtained via `torch.autograd.grad` through the energy. |

---

## Architecture / Block Parameters

| Parameter | Role |
|---|---|
| `num_layers` | Number of `eSCNMD_Block` message-passing layers stacked sequentially. |
| `hidden_channels` | Intermediate width inside each block (used in `Edgewise`, atomwise FF, and all output head MLPs). |
| `norm_type` | Selects the equivariant normalization used as pre-norm before each sub-block and as a final post-backbone norm. See options below. |
| `act_type` | Non-linearity in the *edgewise* sub-block only. See options below. |
| `ff_type` | Selects the *atomwise* feed-forward block type. See options below. |

### `norm_type` options

| Value | Class | Description |
|---|---|---|
| `"layer_norm"` | `EquivariantLayerNormArray` | Per-degree normalization; subtracts mean for `L=0`. |
| `"layer_norm_sh"` | `EquivariantLayerNormArraySphericalHarmonics` | `L=0` with standard LayerNorm; `L>0` share one scale across all degrees. |
| `"rms_norm_sh"` | `EquivariantRMSNormArraySphericalHarmonicsV2` | Degree-balanced RMS norm across all SH components, with centering for `L=0` and per-degree learnable affine weights. **(default/recommended)** |

### `act_type` options

| Value | Class | Mechanism |
|---|---|---|
| `"gate"` | `GateActivation` | `L=0` scalars gate the `L>0` vectors via sigmoid; scalars pass through SiLU. **(default)** |
| `"s2"` | `SeparableS2Activation_M` | Projects to S² grid, applies SiLU point-wise, projects back. |

### `ff_type` options

| Value | Class | Mechanism |
|---|---|---|
| `"spectral"` | `SpectralAtomwise` | Two SO3-linear layers with `GateActivation` in between, entirely in coefficient space. No grid projection — cheaper. |
| `"grid"` | `GridAtomwise` | Projects coefficients to S² spatial grid, applies a 3-layer scalar MLP (SiLU), projects back. More expressive but slower. |

### Block structure (per layer)

```
x_res = x
x = norm_1(x)               # pre-norm (norm_type)
x[:, 0, :] += sys_node_emb  # re-inject CSD conditioning
x = Edgewise(x, ...)        # SO2 message passing (act_type selects non-linearity)
x = x + x_res               # residual

x_res = x
x = norm_2(x)               # pre-norm
x = AtomwiseFF(x)           # ff_type selects spectral vs. grid
x = x + x_res               # residual
```

---

## Charge / Spin / Dataset Conditioning Parameters

These control how system-level context (charge, spin, dataset identity) is injected into node features as a `csd_mixed_emb` vector (one per system, shape `[sphere_channels]`).

| Parameter | Role |
|---|---|
| `chg_spin_emb_type` | How charge and spin are embedded. See options below. |
| `cs_emb_grad` | If `True`, charge/spin embedding parameters have `requires_grad=True` (trained). If `False`, frozen. |
| `dataset_list` | List of dataset name strings. Each gets its own `nn.Embedding(1, sphere_channels)` lookup. Required when `use_dataset_embedding=True`. Note: `"mptrj"` and `"salex"` are internally remapped to `"omat"`. |
| `use_dataset_embedding` | `True`: charge + spin + dataset embeddings concatenated → `Linear(3C → C)` + SiLU. `False`: only charge + spin → `Linear(2C → C)` + SiLU. |

### `chg_spin_emb_type` options

| Value | Mechanism |
|---|---|
| `"pos_emb"` | Fourier/positional encoding: `[sin(x·W), cos(x·W)]` with fixed or learned random frequency `W`. Null-spin maps to zeros. |
| `"lin_emb"` | Simple linear projection `Linear(1 → sphere_channels)`. |
| `"rand_emb"` | `nn.Embedding` lookup table. Charge offset by 100 (range -100..+100 → indices 0..200). |

The resulting `csd_mixed_emb` is:
1. Added to the `L=0` component of each node's initial embedding.
2. Re-injected at every block's pre-norm step.

---

## Forward Pass Data Flow

```
Input: atomic_numbers, pos, cell, edge_index, charge, spin, dataset
  │
  ├─ CSD embedding → per-system sphere_channels vector
  ├─ Graph generation (otf_graph, cutoff, max_neighbors)
  ├─ Wigner-D matrices for each edge (rotation into edge-local SH frame)
  ├─ Node init: x[i, L=0, :] = sphere_embedding[Z_i] + csd_mixed_emb[sys_i]
  ├─ EdgeDegreeEmbedding: initial equivariant message pass (populates L>0)
  ├─ × num_layers eSCNMD_Block:
  │     ├─ Pre-norm (norm_type) + CSD re-injection
  │     ├─ Edgewise SO2 message passing (act_type)
  │     ├─ Residual
  │     ├─ Pre-norm (norm_type)
  │     ├─ Atomwise FF (ff_type)
  │     └─ Residual
  ├─ Final norm (norm_type)
  └─ Output: node_embedding [N_atoms, (lmax+1)², sphere_channels] → EFS heads
```

---

## Output Heads

| Head Class | Energy | Forces | Stress | Notes |
|---|---|---|---|---|
| `MLP_EFS_Head` | 3-layer MLP on `L=0` | autograd `-dE/dpos` | autograd via displacement strain | Forces backbone to `direct_forces=False` |
| `MLP_Energy_Head` | 3-layer MLP on `L=0` | — | — | Standalone energy |
| `Linear_Energy_Head` | 1 linear layer on `L=0` | — | — | Lightweight |
| `Linear_Force_Head` | — | SO3_Linear on `L=1` (direct) | — | Direct force readout |
| `MLP_Stress_Head` | — | — | Scalar (isotropic) + SO3_Linear `L=2` (anisotropic) | Decomposes full stress tensor |

---

## Model Compression Analysis (`eSEN_xtra_XTRA_small`)

The `eSEN_xtra_XTRA_small` config is:

```yaml
sphere_channels: 32    lmax: 2       mmax: 2
num_layers: 4          hidden_channels: 32
edge_channels: 32      num_distance_basis: 128
cutoff: 6              max_neighbors: 30
ff_type: spectral      act_type: gate   # already the cheap options
```

`ff_type: spectral` and `act_type: gate` are already the cheapest options and should not be changed. `sphere_channels: 32` is the primary representational capacity and is already at the floor. The remaining parameters are analyzed below.

### Compressible Parameters

#### 1. `mmax: 2 → 1` — Best first move

**What it does:** Controls the maximum order $m$ used *on edges* during SO(2) convolution. Reducing to 1 means edge messages only carry $m=0$ (scalar) and $m=\pm1$ (vector) components — dropping $m=\pm2$ (rank-2 tensor) components from the message computation entirely. Node features still retain `lmax=2` structure; only the *communication* is truncated.

**Speed gain:** The SO(2) convolution loops over $m = 0, 1, \ldots, \text{mmax}$. Going to `mmax=1` removes one SO2 layer per edge per forward pass. The `SO3_grid["lmax_mmax"]` also shrinks. The `edge_split_sizes` for the RadialMLP output is also reduced.

**Performance tradeoff:** The model loses the ability to transmit rank-2 tensor (quadrupole-like) angular information between atoms during message passing. For electrolytes where bond-angle and lone-pair geometry matters (e.g., ion coordination shells), this can degrade force accuracy. However, node features still have `lmax=2`, so the *atomwise* feed-forward still operates on full tensors — only the edges are truncated.

---

#### 2. `num_distance_basis: 128 → 64` — Low-risk compression

**What it does:** The scalar bond distance is encoded into a `num_distance_basis`-dimensional vector via Gaussian smearing, which becomes the primary input to the RadialMLP. The full edge feature has size `num_distance_basis + 2×edge_channels = 128 + 64 = 192`. Reducing to 64 gives `64 + 64 = 128`.

**Speed gain:** The RadialMLP is called once per edge per layer. Halving the input dimension reduces the first linear layer's FLOPs by ~33%.

**Performance tradeoff:** Minimal. 128 Gaussians uniformly spaced over 6Å gives a resolution of ~0.047Å per basis function — far beyond what is physically meaningful. 64 gives ~0.094Å, which is still more than adequate. The model is very unlikely to be bottlenecked by distance resolution at this scale.

---

#### 3. `max_neighbors: 30 → 20` — Moderate edge reduction

**What it does:** Hard cap on neighbors per atom. In a dense electrolyte simulation box, atoms may have 20–40 neighbors within 6Å. Setting this to 20 drops the furthest neighbors first.

**Speed gain:** Proportional to neighbor reduction. If average neighbors drops from ~25 to ~18, that is ~28% fewer edges.

**Performance tradeoff:** The dropped neighbors are the most distant ones (closest to the cutoff), which are also the weakest interactions (polynomial envelope nearly zeros them out anyway). Risk is low in practice, but in very dense regions (e.g., close ion pairs) an important neighbor could occasionally be dropped.

---

#### 4. `num_layers: 4 → 3` — Clean linear speedup

**What it does:** Removes one full `eSCNMD_Block` (one Edgewise SO2 block + one atomwise FF block + two norm layers + two residual connections).

**Speed gain:** Nearly linear — wall-clock time per step scales close to $O(\text{num\_layers})$.

**Performance tradeoff:** Each layer extends the effective receptive field by one hop. At 4 layers and `cutoff=6Å`, the model can reach interactions ~24Å away; at 3 layers, ~18Å. For electrolytes at 0.1M with ion–solvent shells at ~2–4Å, 3 layers is likely still sufficient. The more real risk is reduced *depth* — fewer nonlinear transformations to build complex many-body correlations.

---

#### 5. `cutoff: 6 → 5` — Large speedup, environment-dependent risk

**What it does:** Reduces the neighbor search radius. The number of edges scales as $r_\text{cut}^3$, so the speedup factor is $(5/6)^3 \approx 0.58$ — roughly **40% fewer edges**.

**Speed gain:** Massive, because edge count directly sets the cost of all message-passing operations (Wigner rotations, SO2 convs, RadialMLPs, scatter ops).

**Performance tradeoff:** In electrolytes, the first solvation shell of small ions (Na$^+$, Li$^+$) sits at ~2.3–2.5Å and the second shell at ~4.5–5.5Å. Cutting from 6Å to 5Å risks clipping second-shell interactions. Energies are often still OK but stress tensors and transport properties (MSD, diffusion) can be sensitive to longer-range correlations.

---

#### 6. `hidden_channels: 32 → 16` — Cheap MLP reduction

**What it does:** Reduces the intermediate width inside each block — used in the `Edgewise` SO2 linear layers and the `SpectralAtomwise` FF. This is the "width inside the block" as opposed to `sphere_channels`, which is the "width of the node tensor".

**Speed gain:** Moderate — the SO2 convolution weight matrices shrink from `(sphere_channels, hidden_channels)` to `(sphere_channels, hidden_channels/2)`.

**Performance tradeoff:** With `sphere_channels=32` and `hidden_channels=16`, the intermediate representation is *narrower* than the input — a bottleneck architecture. This can hurt when the model needs to learn complex channel interactions.

---

#### 7. `lmax: 2 → 1` — Most aggressive architectural change

**What it does:** Drops the entire $l=2$ (rank-2 / 5-component) block from all node feature tensors and Wigner D-matrices. The feature tensor shrinks from $(2+1)^2 = 9$ to $(1+1)^2 = 4$ SH slots — a **2.25× reduction** in the feature dimension that propagates everywhere: node tensors, Wigner matrix size, SO2 conv size, norm layers, output heads.

**Speed gain:** Large. The Wigner D-matrix is block-diagonal with blocks $1\times1$, $3\times3$, $5\times5$. Removing the $5\times5$ block saves significant `bmm` compute. All downstream ops see a 2.25× smaller tensor.

**Performance tradeoff:** Severe. The $l=2$ components encode bond-angle and second-order geometry (e.g., octahedral vs. tetrahedral coordination). For electrolytes, where coordination geometry of Na$^+$, Li$^+$ around solvent oxygens is physically meaningful, losing rank-2 descriptors hurts accuracy noticeably — especially for stress and forces in complex coordination environments. Note: if `lmax` is reduced, `mmax` must also be reduced to `≤ lmax`.

---

### Summary Table

| Parameter | Current → Proposed | Speed Impact | Performance Risk | Notes |
|---|---|---|---|---|
| `mmax` | 2 → 1 | High | Low–Medium | Edge truncation only; node tensors unchanged |
| `num_distance_basis` | 128 → 64 | Low | Very low | 128 is overkill at this model/cutoff scale |
| `max_neighbors` | 30 → 20 | Medium | Low | Distant neighbors nearly zeroed by envelope |
| `num_layers` | 4 → 3 | High | Medium | Cuts receptive field depth by one hop |
| `cutoff` | 6 → 5 | Very high | Medium | ~40% fewer edges; clips weak 2nd-shell interactions |
| `hidden_channels` | 32 → 16 | Medium | Medium | Creates a bottleneck vs. `sphere_channels` |
| `lmax` | 2 → 1 | Very high | High | Removes rank-2 tensors entirely; structural change |

**Do not touch:** `sphere_channels` (primary capacity, already at floor), `ff_type: spectral` (already cheapest), `act_type: gate` (already cheapest).

### Recommended "Fast-but-not-broken" Compressed Config

Apply only the low-risk knobs first:

```yaml
mmax: 1                  # was 2 — edge truncation only
num_distance_basis: 64   # was 128 — no meaningful resolution loss
max_neighbors: 20        # was 30 — distant neighbors dropped first
```

This avoids the high-risk architectural changes (`lmax`, `num_layers`, `cutoff`) while still providing a meaningful wall-clock improvement per MD step.
