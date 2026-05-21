#!/usr/bin/env python3
"""Core cell-size / cell-volume computation functions.

All functions are pure: accept paths/arrays, return data structures.
"""

from pathlib import Path

import numpy as np
from ase.io.trajectory import Trajectory


def _is_gromacs(path: Path) -> bool:
    return path.suffix.lower() in (".xtc", ".trr")


def _resolve_stride(n_tot: int, dt_fs: float,
                    analyze_dt_ps: float | None, n_frames: int) -> int:
    if analyze_dt_ps is not None:
        return max(1, round(analyze_dt_ps * 1000.0 / dt_fs))
    return max(1, n_tot // n_frames)


def extract_cell_timeseries(
    traj_path: str | Path,
    dt_fs: float,
    n_sample: int = 2000,
    max_ns: float | None = None,
    topology: Path | None = None,
    preloaded_frames: list | None = None,
    analyze_dt_ps: float | None = None,
) -> dict:
    """Extract a, b, c cell lengths (Å) and cell volume (Å³) as time series.

    Samples n_sample frames uniformly (or one frame per analyze_dt_ps) up to
    max_ns (or full trajectory if shorter).

    Returns dict with keys:
        times_ns  : ndarray (n,) — frame timestamps in ns
        a, b, c   : ndarray (n,) — orthorhombic cell lengths in Å
        volume    : ndarray (n,) — cell volume in Å³
        stride    : int          — frame stride used
        dt_fs     : float        — trajectory frame interval (fs)
    """
    traj_path = Path(traj_path)

    def _cell_params(at, idx):
        cp = at.cell.cellpar()          # [a, b, c, alpha, beta, gamma]
        vol = float(at.get_volume())
        t   = idx * dt_fs * 1e-6
        return cp[0], cp[1], cp[2], vol, t

    if preloaded_frames is not None:
        n_tot = len(preloaded_frames)
        if max_ns is not None:
            n_tot = min(n_tot, max(1, int(max_ns * 1e6 / dt_fs)))
        stride = _resolve_stride(n_tot, dt_fs, analyze_dt_ps, n_sample)
        a_list, b_list, c_list, vol_list, t_list = [], [], [], [], []
        for i in range(0, n_tot, stride):
            a, b, c, vol, t = _cell_params(preloaded_frames[i], i)
            a_list.append(a); b_list.append(b); c_list.append(c)
            vol_list.append(vol); t_list.append(t)

    elif _is_gromacs(traj_path):
        import sys
        _OBS = Path(__file__).resolve().parents[1]
        if str(_OBS) not in sys.path:
            sys.path.insert(0, str(_OBS))
        from gromacs_io import mda_frame_iter, n_frames_gromacs
        tpr = topology or traj_path.with_suffix(".tpr")
        n_tot = n_frames_gromacs(tpr, traj_path)
        if max_ns is not None:
            n_tot = min(n_tot, max(1, int(max_ns * 1e6 / dt_fs)))
        stride = _resolve_stride(n_tot, dt_fs, analyze_dt_ps, n_sample)
        a_list, b_list, c_list, vol_list, t_list = [], [], [], [], []
        for i, at in enumerate(mda_frame_iter(tpr, traj_path, 0, n_tot, stride)):
            a, b, c, vol, t = _cell_params(at, i * stride)
            a_list.append(a); b_list.append(b); c_list.append(c)
            vol_list.append(vol); t_list.append(t)

    else:
        with Trajectory(str(traj_path), mode="r") as traj:
            n_tot = len(traj)
            if max_ns is not None:
                n_tot = min(n_tot, max(1, int(max_ns * 1e6 / dt_fs)))
            stride = _resolve_stride(n_tot, dt_fs, analyze_dt_ps, n_sample)
            a_list, b_list, c_list, vol_list, t_list = [], [], [], [], []
            for idx in range(0, n_tot, stride):
                a, b, c, vol, t = _cell_params(traj[idx], idx)
                a_list.append(a); b_list.append(b); c_list.append(c)
                vol_list.append(vol); t_list.append(t)

    if not t_list:
        raise ValueError(f"No frames sampled from {traj_path}")

    return {
        "times_ns": np.array(t_list),
        "a":        np.array(a_list),
        "b":        np.array(b_list),
        "c":        np.array(c_list),
        "volume":   np.array(vol_list),
        "stride":   stride,
        "dt_fs":    dt_fs,
    }
