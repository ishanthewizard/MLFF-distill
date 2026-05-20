"""
Cell distortion augmentation for MLFF training data.

Applies random volume-preserving diagonal stretch to orthorhombic cells.
Output cell stays diagonal (orthorhombic) — no shear.

Recommended max_stretch ranges:
  Liquid systems:          0.05 – 0.10
  Target string collapse:  0.15 – 0.20
  Solid systems:           0.02 – 0.05
"""

import numpy as np


def random_isochoric_F(max_stretch: float, rng=None) -> np.ndarray:
    """Return a (3,3) diagonal volume-preserving deformation gradient.

    Samples e1, e2 uniformly in [-max_stretch, max_stretch], sets
    e3 = -(e1+e2) so tr(eps)=0, then rescales so det(F)=1 exactly.
    """
    if rng is None:
        rng = np.random.default_rng()
    e1, e2 = rng.uniform(-max_stretch, max_stretch, size=2)
    e3 = -(e1 + e2)
    F = np.diag([1.0 + e1, 1.0 + e2, 1.0 + e3])
    # exact volume preservation
    F /= np.linalg.det(F) ** (1.0 / 3.0)
    assert abs(np.linalg.det(F) - 1.0) < 1e-8, f"det(F)={np.linalg.det(F)}"
    return F


def apply_deformation(cell_diag: np.ndarray, positions: np.ndarray, F: np.ndarray):
    """Apply diagonal deformation gradient F to an orthorhombic cell.

    Args:
        cell_diag: (3,1) or (3,) diagonal cell lengths [a, b, c]
        positions: (N, 3) Cartesian coordinates
        F:         (3, 3) diagonal deformation gradient from random_isochoric_F

    Returns:
        new_cell (3, 3), new_positions (N, 3)
    """
    cell_diag = np.asarray(cell_diag, dtype=float).ravel()  # -> (3,)
    h = np.diag(cell_diag)                                  # (3,3) orthorhombic cell matrix

    # fractional coords are invariant under affine cell deformation
    inv_h = np.diag(1.0 / cell_diag)
    frac = positions @ inv_h          # (N,3)
    frac = frac % 1.0                 # wrap into [0,1)

    # deform cell: h_new = h @ F.T  (row-vector convention)
    # for diagonal F and diagonal h, this is just h_new = diag(a*F00, b*F11, c*F22)
    new_cell = h @ F.T                # (3,3) — stays diagonal for diagonal F
    # frac @ new_cell converts fractional -> Cartesian in the new cell
    # (row-vector convention: pos_cartesian = frac @ cell_matrix)
    new_positions = frac @ new_cell   # (N,3) Cartesian

    # verify round-trip: fractional coords of new_positions in new_cell must equal frac
    frac_check = new_positions @ np.linalg.inv(new_cell)
    assert np.allclose(frac_check, frac, atol=1e-10), (
        f"Coordinate round-trip failed: max error={abs(frac_check - frac).max():.2e}"
    )

    vol_orig = np.prod(cell_diag)
    vol_new = abs(np.linalg.det(new_cell))
    assert abs(vol_new / vol_orig - 1.0) < 1e-6, (
        f"Volume not preserved: orig={vol_orig:.4f}, new={vol_new:.4f}"
    )
    return new_cell, new_positions


def diagonal_stretch_only(cell_diag: np.ndarray, positions: np.ndarray, max_stretch: float, rng=None):
    """Stretch-only augmentation; output cell stays (3,3) diagonal (orthorhombic).

    Args:
        cell_diag:   (3,1) or (3,) diagonal cell lengths
        positions:   (N, 3) Cartesian coordinates
        max_stretch: max strain magnitude per axis

    Returns:
        new_cell (3, 3), new_positions (N, 3)
    """
    F = random_isochoric_F(max_stretch, rng=rng)
    return apply_deformation(cell_diag, positions, F)


def augment_configs(cells, positions_list, n_augments, max_stretch, rng=None):
    """Generate augmented configurations by random diagonal stretch.

    Args:
        cells:           list of (3,1) or (3,) cell arrays
        positions_list:  list of (N,3) Cartesian position arrays
        n_augments:      number of augmented copies per input config
        max_stretch:     max strain magnitude (e.g. 0.10)

    Returns:
        list of (new_cell (3,3), new_positions (N,3)) tuples.
        Original (undeformed) configs are included first.
    """
    if rng is None:
        rng = np.random.default_rng()

    results = []
    for cell_diag, positions in zip(cells, positions_list):
        cell_diag = np.asarray(cell_diag, dtype=float).ravel()
        # include original
        results.append((np.diag(cell_diag), positions.copy()))
        for _ in range(n_augments):
            new_cell, new_pos = diagonal_stretch_only(cell_diag, positions, max_stretch, rng=rng)
            results.append((new_cell, new_pos))

    n_orig = len(cells)
    n_aug = len(results) - n_orig
    print(f"Augmentation summary: {n_orig} original + {n_aug} augmented = {len(results)} total configs")
    dets = [abs(np.linalg.det(c)) for c, _ in results]
    vol_orig = [np.prod(np.asarray(c).ravel()) for c in cells]
    print(f"  Volume preservation check: max deviation = {max(abs(d/v - 1.0) for d, v in zip(dets[n_orig:], vol_orig * n_augments)):.2e}")
    return results


def validate_augmentation(n_trials=100, max_stretch=0.10):
    """Validate augmentation on a synthetic cubic cell with random atoms."""
    rng = np.random.default_rng(42)

    cell_diag = np.array([[10.0], [10.0], [10.0]])
    positions = rng.uniform(0, 10, size=(20, 3))
    vol_orig = 1000.0

    dets, angle_devs = [], []
    for _ in range(n_trials):
        new_cell, new_pos = diagonal_stretch_only(cell_diag, positions, max_stretch, rng=rng)

        # volume check
        det = abs(np.linalg.det(new_cell))
        dets.append(det)
        assert abs(det / vol_orig - 1.0) < 1e-6

        # positions within cell
        inv_new = np.linalg.inv(new_cell)
        frac = new_pos @ inv_new
        assert (frac >= -1e-9).all() and (frac <= 1.0 + 1e-9).all(), "Positions outside cell"

        # cell angles (should stay 90° for diagonal F)
        a, b, c = new_cell[0], new_cell[1], new_cell[2]
        for u, v in [(a, b), (a, c), (b, c)]:
            cos_angle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            angle_devs.append(abs(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))) - 90.0))

    dets = np.array(dets)
    print(f"\nValidation ({n_trials} trials, max_stretch={max_stretch}):")
    print(f"  det(F): mean={dets.mean():.8f}, std={dets.std():.2e}, max_dev={abs(dets - 1.0).max():.2e}")
    print(f"  Cell angle deviation from 90°: max={max(angle_devs):.6f} deg (should be ~0 for stretch-only)")
    print("  All volume and bounds checks passed.")

    try:
        import matplotlib.pyplot as plt
        a_lengths = []
        for _ in range(n_trials):
            new_cell, _ = diagonal_stretch_only(cell_diag, positions, max_stretch, rng=rng)
            a_lengths.append(new_cell[0, 0])
        plt.figure(figsize=(5, 3))
        plt.hist(a_lengths, bins=20, edgecolor='black')
        plt.xlabel("Cell length a (Å)")
        plt.title("Distribution of deformed a-axis lengths")
        plt.tight_layout()
        plt.savefig("cell_augmentation_validation.png", dpi=100)
        print("  Saved histogram to cell_augmentation_validation.png")
    except ImportError:
        pass


if __name__ == "__main__":
    validate_augmentation()

    # Example usage
    cells = [np.array([[10.0], [10.0], [10.0]])]
    positions_list = [np.random.rand(50, 3) * 10]

    augmented = augment_configs(
        cells=cells,
        positions_list=positions_list,
        n_augments=10,
        max_stretch=0.10,
    )
    new_cell, new_pos = augmented[1]
    print(f"\nExample augmented cell:\n{new_cell}")
    print(f"Example augmented positions shape: {new_pos.shape}")
