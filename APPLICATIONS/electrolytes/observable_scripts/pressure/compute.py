#!/usr/bin/env python3
"""Instantaneous pressure extraction from ASE trajectories.

Pressure is derived from the stored stress tensor:
    P (GPa) = -(σ_xx + σ_yy + σ_zz) / 3  ×  160.21766208

where stress is the Voigt 6-vector [xx, yy, zz, yz, xz, xy] in eV/Å³.
"""

from pathlib import Path

import numpy as np
from ase.io.trajectory import Trajectory

_EV_ANG3_TO_GPA = 160.21766208


def _resolve_stride(n_tot: int, dt_fs: float,
                    analyze_dt_ps: float | None, n_frames: int) -> int:
    if analyze_dt_ps is not None:
        return max(1, round(analyze_dt_ps * 1000.0 / dt_fs))
    return max(1, n_tot // n_frames)


def extract_pressure(
    traj_path: str | Path,
    n_sample: int = 3000,
    dt_fs: float = 100.0,
    max_ns: float | None = None,
    analyze_dt_ps: float | None = None,
) -> dict:
    """Extract instantaneous pressure (GPa) from a trajectory.

    Returns dict with keys:
        times_ns  : ndarray (n,) — frame timestamps in ns
        pressure  : ndarray (n,) — isotropic pressure in GPa
        pxx, pyy, pzz : ndarray (n,) — diagonal stress components in GPa
        stride    : int   — frame stride used
        dt_fs     : float — trajectory frame interval (fs)
    """
    traj_path = Path(traj_path)
    traj = Trajectory(str(traj_path), mode="r")
    n_tot = len(traj)
    if max_ns is not None:
        n_tot = min(n_tot, max(1, int(max_ns * 1e6 / dt_fs)))
    stride = _resolve_stride(n_tot, dt_fs, analyze_dt_ps, n_sample)
    indices = range(0, n_tot, stride)

    times, pressure, pxx, pyy, pzz = [], [], [], [], []
    try:
        for idx in indices:
            at = traj[idx]
            if at.calc is None or "stress" not in at.calc.results:
                continue
            s = at.calc.results["stress"]   # eV/Å³, Voigt [xx,yy,zz,yz,xz,xy]
            p_iso = -(s[0] + s[1] + s[2]) / 3.0 * _EV_ANG3_TO_GPA
            times.append(idx * dt_fs * 1e-6)
            pressure.append(p_iso)
            pxx.append(-s[0] * _EV_ANG3_TO_GPA)
            pyy.append(-s[1] * _EV_ANG3_TO_GPA)
            pzz.append(-s[2] * _EV_ANG3_TO_GPA)
    finally:
        traj.close()

    return {
        "times_ns": np.array(times),
        "pressure": np.array(pressure),
        "pxx":      np.array(pxx),
        "pyy":      np.array(pyy),
        "pzz":      np.array(pzz),
        "stride":   stride,
        "dt_fs":    dt_fs,
    }
