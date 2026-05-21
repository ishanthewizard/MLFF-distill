#!/usr/bin/env python3
"""Core density computation functions.

All functions are pure: accept paths/arrays, return data structures.
"""

import sys
from pathlib import Path

import numpy as np
from ase.io.trajectory import Trajectory

_OBS = Path(__file__).resolve().parents[1]
if str(_OBS) not in sys.path:
    sys.path.insert(0, str(_OBS))


def _is_gromacs(path: Path) -> bool:
    return path.suffix.lower() in (".xtc", ".trr")

AMU_TO_KG = 1.66053906660e-27
ANG3_TO_M3 = 1e-30


def compute_density(
    traj_path: str | Path,
    dt_fs: float,
    n_frames: int = 1000,
    skip_ns: float = 0.1,
    window_ns: float = 0.9,
    topology: Path | None = None,
    preloaded_frames: list | None = None,
) -> tuple[float, float]:
    """Compute mean density (g/cm³) and std over a time window.

    Returns (mean_density, std_density).
    """
    traj_path = Path(traj_path)
    start = int(skip_ns * 1e6 / dt_fs)

    def _dens(at):
        return (float(np.sum(at.get_masses())) * AMU_TO_KG /
                (float(at.get_volume()) * ANG3_TO_M3)) / 1000.0

    if preloaded_frames is not None:
        densities = [_dens(at) for at in preloaded_frames]
    elif _is_gromacs(traj_path):
        from gromacs_io import mda_frame_iter, n_frames_gromacs
        tpr = topology or traj_path.with_suffix(".tpr")
        n_tot = n_frames_gromacs(tpr, traj_path)
        end = min(start + int(window_ns * 1e6 / dt_fs), n_tot)
        stride = max(1, (end - start) // n_frames)
        densities = [_dens(at) for at in mda_frame_iter(tpr, traj_path, start, end, stride)]
    else:
        traj = Trajectory(str(traj_path), mode="r")
        try:
            end = min(start + int(window_ns * 1e6 / dt_fs), len(traj))
            stride = max(1, (end - start) // n_frames)
            densities = [_dens(at) for at in traj[start:end:stride]]
        finally:
            traj.close()

    if not densities:
        raise ValueError(f"No frames sampled from {traj_path}")
    d = np.array(densities)
    return float(d.mean()), float(d.std(ddof=1))


def extract_density_timeseries(
    traj_path: str | Path,
    dt_fs: float,
    n_sample: int = 2000,
    max_ns: float | None = None,
    topology: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract density (g/cm³) as a time series.

    Samples n_sample frames uniformly from frame 0 up to max_ns (or full traj if shorter).
    Returns (times_ns, densities).
    """
    traj_path = Path(traj_path)

    def _dens_time(at, idx):
        mass_kg = float(np.sum(at.get_masses())) * AMU_TO_KG
        vol_m3  = float(at.get_volume()) * ANG3_TO_M3
        return (mass_kg / vol_m3) / 1000.0, idx * dt_fs * 1e-6

    if _is_gromacs(traj_path):
        from gromacs_io import mda_frame_iter, n_frames_gromacs
        tpr = topology or traj_path.with_suffix(".tpr")
        n_tot = n_frames_gromacs(tpr, traj_path)
        if max_ns is not None:
            n_tot = min(n_tot, max(1, int(max_ns * 1e6 / dt_fs)))
        stride = max(1, n_tot // n_sample)
        times, densities = [], []
        for i, at in enumerate(mda_frame_iter(tpr, traj_path, 0, n_tot, stride)):
            d, t = _dens_time(at, i * stride)
            densities.append(d); times.append(t)
    else:
        traj = Trajectory(str(traj_path), mode="r")
        try:
            n_tot = len(traj)
            if max_ns is not None:
                n_tot = min(n_tot, max(1, int(max_ns * 1e6 / dt_fs)))
            stride = max(1, n_tot // n_sample)
            times, densities = [], []
            for idx in range(0, n_tot, stride):
                d, t = _dens_time(traj[idx], idx)
                densities.append(d); times.append(t)
        finally:
            traj.close()

    if not densities:
        raise ValueError(f"No frames sampled from {traj_path}")
    return np.array(times), np.array(densities)


def compute_density_tail(
    traj_path: str | Path,
    n_snapshots: int = 1000,
    stride: int = 10,
) -> tuple[float, float]:
    """Compute mean density from the tail of a trajectory (last n_snapshots×stride frames).

    Returns (mean_density, std_density).
    """
    traj = Trajectory(str(traj_path), mode="r")
    try:
        tail = traj[-(n_snapshots * stride):]
        frames = tail[::stride]
        frames = frames[-n_snapshots:]
        densities = []
        for at in frames:
            mass_kg = float(np.sum(at.get_masses())) * AMU_TO_KG
            vol_m3 = float(at.get_volume()) * ANG3_TO_M3
            densities.append((mass_kg / vol_m3) / 1000.0)
    finally:
        traj.close()

    if not densities:
        raise ValueError(f"No frames sampled from {traj_path}")
    d = np.array(densities)
    return float(d.mean()), float(d.std(ddof=1))
