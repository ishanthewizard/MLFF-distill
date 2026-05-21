#!/usr/bin/env python3
"""RMSD computation relative to a reference frame.

All functions are pure: accept paths/arrays, return data structures.
Uses Kabsch algorithm for optimal superposition before RMSD.
"""

from pathlib import Path

import numpy as np
from ase.io.trajectory import Trajectory
from tqdm import tqdm


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute Kabsch-aligned RMSD between two (N, 3) coordinate sets.

    Centers both sets, finds the optimal rotation, and returns
    (rmsd_angstrom, rotated_P).
    """
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)

    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    # handle reflection
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    P_rot = P @ R.T
    rmsd  = float(np.sqrt(((P_rot - Q) ** 2).mean()))
    return rmsd, P_rot


def _resolve_stride(n_tot: int, dt_fs: float,
                    analyze_dt_ps: float | None, n_frames: int) -> int:
    """Return frame stride.

    Priority: analyze_dt_ps (time-based) > n_frames (count-based).
    analyze_dt_ps = desired time gap between evaluated frames in picoseconds.
    """
    if analyze_dt_ps is not None:
        return max(1, round(analyze_dt_ps * 1000.0 / dt_fs))
    return max(1, n_tot // n_frames)


def compute_rmsd(
    traj_path: str | Path,
    n_frames: int = 1000,
    dt_fs: float = 100.0,
    max_ns: float | None = None,
    ref_frame_idx: int = 0,
    atom_mask: list[int] | None = None,
    analyze_dt_ps: float | None = None,
) -> dict:
    """Compute per-frame RMSD relative to a reference frame.

    Uses Kabsch optimal superposition (removes rigid-body translation and rotation).

    Parameters
    ----------
    traj_path      : ASE .traj trajectory file.
    n_frames       : frames to sample uniformly (used when analyze_dt_ps is None).
    dt_fs          : timestep between trajectory frames in femtoseconds.
    max_ns         : truncate trajectory at this simulation length (ns).
    ref_frame_idx  : index of the reference frame (default 0 = first frame).
    atom_mask      : optional list of atom indices; if None all atoms are used.
    analyze_dt_ps  : evaluate one frame every this many picoseconds (preferred
                     over n_frames when given).

    Returns dict with keys:
        times_ns  : ndarray (n,)  — frame timestamps in ns
        rmsd      : ndarray (n,)  — RMSD in Å relative to reference frame
        stride    : int           — frame stride used
        dt_fs     : float         — trajectory frame interval (fs)
    """
    traj_path = Path(traj_path)

    with Trajectory(str(traj_path), mode="r") as _t:
        n_tot = len(_t)
        ref_atoms = _t[ref_frame_idx]

    if max_ns is not None:
        n_tot = min(n_tot, max(1, int(max_ns * 1e6 / dt_fs)))
    stride  = _resolve_stride(n_tot, dt_fs, analyze_dt_ps, n_frames)
    indices = list(range(0, n_tot, stride))

    ref_pos = ref_atoms.get_positions()
    if atom_mask is not None:
        ref_pos = ref_pos[atom_mask]

    times, rmsd_list = [], []
    with Trajectory(str(traj_path), mode="r") as traj:
        for idx in tqdm(indices, desc=f"rmsd {traj_path.name}", unit="frame"):
            at  = traj[idx]
            pos = at.get_positions()
            if atom_mask is not None:
                pos = pos[atom_mask]
            rmsd_val, _ = _kabsch_rmsd(pos, ref_pos)
            rmsd_list.append(rmsd_val)
            times.append(idx * dt_fs * 1e-6)

    return {
        "times_ns": np.array(times),
        "rmsd":     np.array(rmsd_list),
        "stride":   stride,
        "dt_fs":    dt_fs,
    }
