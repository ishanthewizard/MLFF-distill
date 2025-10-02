import torch

import torch

def get_dense_grad_outputs(
    data,
    num_rows: int,
    density: int = 5,
    random_sign: bool = True,
):
    """
    Build grad_outputs of shape (num_rows, total_atoms, 3).
    For each row and each molecule, select `density` atoms via FPS (on positions) and
    set one random coordinate per selected atom to ±1 (or +1 if random_sign=False).

    Args
    ----
    data.pos : (total_atoms, 3) tensor
    data.natoms : (B,) tensor of atom counts per structure

    Returns
    -------
    grad_outputs : (num_rows, total_atoms, 3) tensor (same device/dtype as data.pos)
    """
    device = data.pos.device
    dtype = data.pos.dtype

    natoms = data.natoms
    total_atoms = int(natoms.sum().item())

    # start indices per molecule
    starts = torch.cumsum(natoms, dim=0) - natoms  # (B,)
    # slices of positions per molecule
    pos_by_mol = [data.pos[s:s+n] for s, n in zip(starts, natoms.tolist())]

    grad_outputs = torch.zeros((num_rows, total_atoms, 3), device=device, dtype=dtype)

    for row in range(num_rows):
        for (start, pos_m) in zip(starts.tolist(), pos_by_mol):
            n = pos_m.shape[0]

            # FPS on positions within this molecule
            # Pairwise distances (n x n)
            dists = torch.cdist(pos_m, pos_m)  # (n, n)

            # pick first index randomly
            current = torch.randint(low=0, high=n, size=(1,), device=device).item()
            selected = [current]
            min_d = dists[:, current]  # (n,)

            # pick remaining k-1 farthest points
            for _ in range(1, density):
                current = int(torch.argmax(min_d).item())
                selected.append(current)
                min_d = torch.minimum(min_d, dists[:, current])

            # set entries in grad_outputs for this row & molecule
            for idx in selected:
                axis = int(torch.randint(0, 3, (1,), device=device).item())
                val = 1.0
                if random_sign:
                    val = 1.0 if torch.randint(0, 2, (1,), device=device).item() == 1 else -1.0
                grad_outputs[row, start + idx, axis] = val

    return grad_outputs

        
    

def get_diverse_idxs(x, natoms, num_samples):
    """
    Selects a diverse subset of atoms from each molecule using farthest point sampling.
    Args:
        x (torch.Tensor): Node embeddings for all atoms in the batch.
                          Expected shape: (total_num_atoms, num_channels, ...),
                          where the scalar features are in x[:, 0, :].
        natoms (torch.Tensor): A tensor where each element is the number of atoms
                               in a molecule. Shape: (num_molecules,).
        num_samples (int): The number of diverse samples to select from each molecule.
    Returns:
        torch.Tensor: A 1D tensor containing the global indices of the selected
                      diverse atoms for the entire batch.
    """
    # Assuming scalar features are at index 0 of the second dimension.
    # Shape of scalar_x: (total_num_atoms, num_sphere_channels)
    scalar_x = x[:, 0, :]

    # Split embeddings by molecule
    cumulative_sums = torch.cumsum(natoms, 0) - natoms
    x_by_mol = [scalar_x[start:start + nat, :]for start, nat in zip(cumulative_sums, natoms)]

    all_diverse_indices = []

    for mol_embeddings in x_by_mol:
        num_atoms_in_mol = mol_embeddings.shape[0]

        if num_atoms_in_mol == 0:
            continue

        # Determine number of points to select for this molecule
        num_to_select = min(num_samples, num_atoms_in_mol)

        if num_to_select < num_samples:
            raise Exception("num_atoms_in_mol < num_samples not currently supported")

        # Farthest Point Sampling (FPS)
        # Step 1: Compute pairwise distance matrix
        dists = torch.cdist(mol_embeddings, mol_embeddings)

        # Step 2: Iteratively select farthest points
        selected_indices = torch.zeros(
            num_to_select, dtype=torch.long, device=x.device
        )

        # Start with a random point
        current_idx = torch.randint(0, num_atoms_in_mol, (1,), device=x.device).item()
        selected_indices[0] = current_idx

        # Initialize distances from all points to the selected set
        min_dists = dists[:, current_idx]

        for j in range(1, num_to_select):
            # Find the point that is farthest from the current set of selected points
            current_idx = torch.argmax(min_dists)
            selected_indices[j] = current_idx

            # Update min_dists with the distances to the new point
            min_dists = torch.minimum(min_dists, dists[:, current_idx])

        # Convert local indices to global indices
        repeated_indices = selected_indices.repeat_interleave(3)
        col = torch.arange(3, device=selected_indices.device).repeat(selected_indices.shape[0])
        final_indices = torch.stack([repeated_indices, col], dim=1)
        
        
        all_diverse_indices.append(final_indices)


    return all_diverse_indices

def make_probe_matrix(
    pos: torch.Tensor,          # (N, 3) positions (for device/dtype & natoms layout)
    natoms: torch.Tensor,       # (B,) number of atoms per structure (sum = N)
    num_probes: int = 30,       # m
    p_active: float = 0.5,      # Bernoulli prob that an atom is active in a column
    remove_rotations: bool = False,  # set True only for molecules / no PBC
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Returns V of shape (3N, m) with V^T V = I (thin-QR applied).
    Construction per column:
      - For each atom, with prob p_active set a random unit 3D direction; else 0
      - Remove per-structure per-axis mean (kills translations)
      - (Optional) Remove rigid rotations for molecules (no PBC)
      - Normalize to unit norm
    Then thin-QR over the 3N x m matrix for orthonormal columns.
    """
    device, dtype = pos.device, pos.dtype
    N = pos.shape[0]
    B = natoms.numel()
    assert natoms.sum().item() == N, "natoms must sum to N"

    # map each atom -> structure id
    struct_ids = torch.repeat_interleave(
        torch.arange(B, device=device), natoms
    )  # (N,)

    # precompute per-structure counts as float for broadcasting
    natoms_f = natoms.to(dtype=dtype).unsqueeze(1)  # (B,1)

    cols = []
    rng = torch.Generator(device=device)
    rng.manual_seed(1234)
    # If you want reproducibility, set a manual seed on rng: rng.manual_seed(1234)

    def _project_translations(w):
        # w: (N,3) per-atom displacement column
        sums = torch.zeros(B, 3, device=device, dtype=dtype)
        sums.index_add_(0, struct_ids, w)          # sum over atoms per structure
        means = sums / natoms_f                    # (B,3)
        return w - means[struct_ids]               # center per structure

    def _project_rotations(w):
        # Optional rigid-rotation removal for isolated molecules (no PBC).
        # Solve min_omega || w - omega x r || over omega (3-vector).
        # Implemented via least-squares using cross-product matrix.
        r = pos  # (N,3)
        # Build A * omega ≈ w, where A_i = [ [0,-z,y],[z,0,-x],[-y,x,0] ]
        # We solve separately per structure, to be safe.
        w_out = w.clone()
        start = 0
        for b, n in enumerate(natoms.tolist()):
            end = start + n
            r_b = r[start:end]  # (n,3)
            w_b = w[start:end]  # (n,3)
            # Assemble normal equations for omega in R^3
            # A_i^T A_i = (||r_i||^2) I - r_i r_i^T ; A_i^T w_i = r_i x w_i
            # Sum over atoms:
            rr = (r_b * r_b).sum(dim=1, keepdim=True)  # (n,1)
            ATA = torch.eye(3, device=device, dtype=dtype) * rr.sum()
            ATA -= (r_b.t() @ r_b)  # 3x3
            ATw = (r_b.cross(w_b)).sum(dim=0)          # 3,
            # Solve ATA * omega = ATw (regularize if needed)
            reg = 1e-12
            omega = torch.linalg.solve(ATA + reg * torch.eye(3, device=device, dtype=dtype), ATw)
            # subtract rotational field
            w_out[start:end] = w_b - torch.cross(omega.expand_as(r_b), r_b, dim=1)
            start = end
        return w_out

    for j in range(num_probes):

        # Mask per atom ~ Bernoulli(p_active)
        mask = (torch.rand((N, 1), generator=rng, device=device) < p_active).to(dtype)

        # Random unit directions on S^2 for active atoms
        dirs = torch.randn((N, 3), generator=rng, device=device, dtype=dtype)
        dirs = dirs / (dirs.norm(dim=1, keepdim=True).clamp_min(eps))  # unit
        w = dirs * mask  # (N,3)

        # Remove per-structure translations
        w = _project_translations(w)

        # Optional: remove rotations for molecules/non-PBC
        if remove_rotations:
            w = _project_rotations(w)

        # Normalize to unit norm
        v = w.reshape(-1)  # (3N,)
        
        # nrm = torch.linalg.norm(v)
        # v = v / nrm # NOTE: for now we WON'T do this, and we'll see what happens hahaha
        
        v = v / 10
        cols.append(v)


    V0 = torch.stack(cols, dim=1)  # (3N, m)
    
    col_norms = torch.linalg.norm(V0, dim=0)  # (m,)

    # Thin QR:
    Q, R = torch.linalg.qr(V0, mode='reduced')  # Q: (3N, m)

    # Optional deterministic sign convention (not required if you rescale by col_norms):
    # Make diagonal of R positive so Q’s column signs are stable
    diag = torch.diag(R)
    sign = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
    Q = Q * sign  # broadcast columnwise

    # Option A: keep your original per-column magnitudes in the probe matrix
    V = Q * col_norms  # scales each column j by col_norms[j]
    return V.reshape(N, 3, -1).permute(2, 0, 1)  # (m, N, 3)