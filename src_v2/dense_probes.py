import torch

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