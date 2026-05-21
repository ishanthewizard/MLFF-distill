#!/usr/bin/env python3
"""Subsample a fixed-stride set of frames from a time window in an ASE .traj file."""

from __future__ import annotations

import os
from ase.io.trajectory import Trajectory


def subsample_traj_window(
    traj_path: str,
    dt_per_frame_fs: float,
    window_start_ns: float,
    window_end_ns: float,
    n_samples: int,
) -> list:
    """Return n_samples Atoms from [window_start_ns, window_end_ns) at fixed stride.

    Opens the trajectory in read-only mode so the offset table is used for
    O(1) seeks — only the n_samples frames are actually loaded into memory,
    not the entire window.

    Args:
        traj_path: Path to ASE .traj file.
        dt_per_frame_fs: Simulation time per saved frame in femtoseconds (e.g. 100.0).
        window_start_ns: Start of sampling window in nanoseconds.
        window_end_ns: End of sampling window in nanoseconds.
        n_samples: Number of frames to extract (evenly spaced within the window).

    Returns:
        List of ASE Atoms objects with calculator results (energy + forces).

    Raises:
        FileNotFoundError: If traj_path does not exist.
        ValueError: If no valid frames are found in the window.
    """
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    # 1 ns = 1e6 fs  →  frame_index = time_ns * 1e6 / dt_per_frame_fs
    frame_start = int(window_start_ns * 1e6 / dt_per_frame_fs)
    frame_end = int(window_end_ns * 1e6 / dt_per_frame_fs)

    with Trajectory(traj_path, mode="r") as traj:
        n_total = len(traj)
        # Clip to actual file length
        frame_start = min(frame_start, n_total)
        frame_end = min(frame_end, n_total)
        n_window = frame_end - frame_start

        if n_window <= 0:
            raise ValueError(
                f"Empty window [{window_start_ns}, {window_end_ns}] ns "
                f"(frames {frame_start}:{frame_end}, traj has {n_total} frames) "
                f"in {traj_path}"
            )

        if n_samples >= n_window:
            stride = 1
            print(
                f"  Warning: requested {n_samples} samples but window has only "
                f"{n_window} frames — using all with stride 1"
            )
        else:
            stride = n_window // n_samples

        # Build index list first, then load only those frames
        indices = list(range(frame_start, frame_end, stride))[:n_samples]

        print(
            f"  {os.path.basename(traj_path)}: total={n_total} frames | "
            f"window=[{window_start_ns}, {window_end_ns}] ns "
            f"-> frames {frame_start}:{frame_end} ({n_window} frames) | "
            f"stride={stride} | will load {len(indices)} frames"
        )

        def _has_ef(atoms):
            return (
                atoms.calc is not None
                and "energy" in getattr(atoms.calc, "results", {})
                and "forces" in getattr(atoms.calc, "results", {})
            )

        valid = []
        for i in indices:
            atoms = traj[i]
            if _has_ef(atoms):
                valid.append(atoms)

    dropped = len(indices) - len(valid)
    if dropped:
        print(f"  Dropped {dropped} frames without energy/forces")
    print(
        f"  [traj done] {os.path.basename(traj_path)}: "
        f"loaded {len(indices)} -> {len(valid)} valid frames"
    )

    if not valid:
        raise ValueError(f"No frames with energy+forces found in window for {traj_path}")

    return valid
